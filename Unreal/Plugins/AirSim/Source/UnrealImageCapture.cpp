#include "UnrealImageCapture.h"
#include "Engine/World.h"
#include "ImageUtils.h"

#include "RenderRequest.h"
#include "SyntheticCameraProcessor.h"
#include "AirBlueprintLib.h"
#include "common/ClockFactory.hpp"

UnrealImageCapture::UnrealImageCapture(const common_utils::UniqueValueMap<std::string, APIPCamera*>* cameras)
    : cameras_(cameras)
{
    //TODO: explore screenshot option
    //addScreenCaptureHandler(camera->GetWorld());
}

UnrealImageCapture::~UnrealImageCapture()
{
}

void UnrealImageCapture::getImages(const std::vector<msr::airlib::ImageCaptureBase::ImageRequest>& requests,
                                   std::vector<msr::airlib::ImageCaptureBase::ImageResponse>& responses) const
{
    if (cameras_->valsSize() == 0) {
        for (unsigned int i = 0; i < requests.size(); ++i) {
            responses.push_back(ImageResponse());
            responses[responses.size() - 1].message = "camera is not set";
        }
        return;
    }

    bool has_synthetic = false;
    for (const auto& request : requests) {
        if (request.image_type == ImageType::ThermalIR || request.image_type == ImageType::NightVision) {
            has_synthetic = true;
            break;
        }
    }
    if (!has_synthetic) {
        getSceneCaptureImage(requests, responses, false);
        return;
    }

    //Synthetic types (ThermalIR, NightVision) are composed on CPU from hidden
    //sub-requests rendered in the SAME RenderRequest batch as the user's normal
    //requests: one render pass, same frame, lockstep-safe.
    std::vector<ImageRequest> render_requests;
    struct Slot
    {
        bool synthetic = false;
        size_t normal_idx = 0; //normal requests: index into render_requests
        size_t sub_a = 0; //ThermalIR: Segmentation; NightVision: Scene
        size_t sub_b = 0; //ThermalIR: DepthPlanar;  NightVision: Segmentation
    };
    std::vector<Slot> slots(requests.size());

    //pass 1: forward all normal requests untouched (order within render_requests is
    //free; the 1:1 output order is restored from slots below)
    for (size_t i = 0; i < requests.size(); ++i) {
        const ImageRequest& request = requests.at(i);
        if (request.image_type == ImageType::ThermalIR || request.image_type == ImageType::NightVision)
            continue;
        render_requests.push_back(request);
        slots[i].normal_idx = render_requests.size() - 1;
    }

    //pass 2: append the hidden sub-requests, deduped against identical captures
    //already in the batch (user-requested or from another synthetic request)
    auto find_or_append = [&render_requests](const std::string& camera_name, ImageType image_type, bool pixels_as_float) -> size_t {
        for (size_t j = 0; j < render_requests.size(); ++j) {
            const ImageRequest& existing = render_requests[j];
            if (existing.camera_name == camera_name && existing.image_type == image_type &&
                existing.pixels_as_float == pixels_as_float && (pixels_as_float || !existing.compress) &&
                existing.annotation_name.empty())
                return j;
        }
        render_requests.push_back(ImageRequest(camera_name, image_type, pixels_as_float, false, ""));
        return render_requests.size() - 1;
    };
    for (size_t i = 0; i < requests.size(); ++i) {
        const ImageRequest& request = requests.at(i);
        if (request.image_type == ImageType::ThermalIR) {
            slots[i].synthetic = true;
            slots[i].sub_a = find_or_append(request.camera_name, ImageType::Segmentation, false);
            slots[i].sub_b = find_or_append(request.camera_name, ImageType::DepthPlanar, true);
        }
        else if (request.image_type == ImageType::NightVision) {
            slots[i].synthetic = true;
            slots[i].sub_a = find_or_append(request.camera_name, ImageType::Scene, false);
            slots[i].sub_b = find_or_append(request.camera_name, ImageType::Segmentation, false);
        }
    }

    //single batched render for the union of underlying captures
    std::vector<ImageResponse> render_responses;
    getSceneCaptureImage(render_requests, render_responses, false);
    if (render_responses.size() != render_requests.size())
        return; //no viewport (getSceneCaptureImage bailed out)

    //assemble responses 1:1 with the incoming request order; hidden sub-responses
    //are consumed here and never surfaced
    for (size_t i = 0; i < requests.size(); ++i) {
        const ImageRequest& request = requests.at(i);
        if (!slots[i].synthetic) {
            responses.push_back(render_responses.at(slots[i].normal_idx));
            continue;
        }

        const ImageResponse& sub_a = render_responses.at(slots[i].sub_a);
        const ImageResponse& sub_b = render_responses.at(slots[i].sub_b);
        //both pipelines consume a segmentation capture; use it as the metadata source
        const ImageResponse& seg_response = (request.image_type == ImageType::ThermalIR) ? sub_a : sub_b;

        ImageResponse response;
        response.camera_name = request.camera_name;
        response.image_type = request.image_type;
        response.pixels_as_float = false;
        response.annotation_name = "";
        response.width = seg_response.width;
        response.height = seg_response.height;
        response.time_stamp = seg_response.time_stamp;
        response.camera_position = seg_response.camera_position;
        response.camera_orientation = seg_response.camera_orientation;

        const size_t expected_size = static_cast<size_t>(seg_response.width) * seg_response.height * 3;

        std::vector<uint8_t> composed;
        if (expected_size == 0) {
            response.message = "Can't compose synthetic image because underlying captures failed";
        }
        else if (request.image_type == ImageType::ThermalIR) {
            if (sub_a.image_data_uint8.size() != expected_size)
                response.message = "Can't compose synthetic image because the segmentation capture failed";
            else
                SyntheticCameraProcessor::instance().composeThermalIR(
                    sub_a.image_data_uint8, sub_b.image_data_float, seg_response.width, seg_response.height, composed);
        }
        else { //NightVision
            //a scene capture of a different resolution than the segmentation is
            //tolerated: the processor then falls back to the pure thermal map,
            //matching the ROS node's behavior on mismatched topic sizes
            if (sub_b.image_data_uint8.size() != expected_size)
                response.message = "Can't compose synthetic image because the segmentation capture failed";
            else
                SyntheticCameraProcessor::instance().composeNightVision(
                    sub_a.image_data_uint8, sub_b.image_data_uint8, seg_response.width, seg_response.height, composed);
        }

        if (!composed.empty() && request.compress) {
            //reuse the same PNG path RenderRequest uses for compressed uint8 captures
            TArray<FColor> bmp;
            bmp.SetNumUninitialized(response.width * response.height);
            for (int32 p = 0; p < response.width * response.height; ++p)
                bmp[p] = FColor(composed[p * 3 + 0], composed[p * 3 + 1], composed[p * 3 + 2], 255);
            TArray<uint8_t> png;
            UAirBlueprintLib::CompressImageArray(response.width, response.height, bmp, png);
            response.image_data_uint8 = std::vector<uint8_t>(png.GetData(), png.GetData() + png.Num());
            response.compress = true;
        }
        else {
            response.image_data_uint8 = std::move(composed);
            response.compress = false;
        }
        responses.push_back(response);
    }
}

void UnrealImageCapture::getSceneCaptureImage(const std::vector<msr::airlib::ImageCaptureBase::ImageRequest>& requests,
                                              std::vector<msr::airlib::ImageCaptureBase::ImageResponse>& responses, bool use_safe_method) const
{
    std::vector<std::shared_ptr<RenderRequest::RenderParams>> render_params;
    std::vector<std::shared_ptr<RenderRequest::RenderResult>> render_results;

    

    bool visibilityChanged = false;
    for (unsigned int i = 0; i < requests.size(); ++i) {
        APIPCamera* camera = cameras_->at(requests.at(i).camera_name);
        //TODO: may be we should have these methods non-const?
        if (requests[i].image_type == ImageType::Annotation) {
            if (camera->GetAnnotationNameExist(requests[i].annotation_name)){
                visibilityChanged = const_cast<UnrealImageCapture*>(this)->updateCameraVisibility(camera, requests[i]) || visibilityChanged;
			}
        }
        else {
            visibilityChanged = const_cast<UnrealImageCapture*>(this)->updateCameraVisibility(camera, requests[i]) || visibilityChanged;
        }
    }

    if (use_safe_method && visibilityChanged) {
        // We don't do game/render thread synchronization for safe method.
        // We just blindly sleep for 200ms (the old way)
        std::this_thread::sleep_for(std::chrono::duration<double>(0.2));
    }

    UGameViewportClient* gameViewport = nullptr;
    for (unsigned int i = 0; i < requests.size(); ++i) {

        APIPCamera* camera = cameras_->at(requests.at(i).camera_name);

        if (gameViewport == nullptr) {
            gameViewport = camera->GetWorld()->GetGameViewport();
        }

        responses.push_back(ImageResponse());
        ImageResponse& response = responses.at(i);          
        UTextureRenderTarget2D* textureTarget = nullptr;
        USceneCaptureComponent2D* capture = nullptr;
        if (requests[i].image_type == ImageType::Annotation) {
            if (camera->GetAnnotationNameExist(requests[i].annotation_name)) {
                capture = camera->getCaptureComponent(requests[i].image_type, false, requests[i].annotation_name);
                if (capture == nullptr) {
                    response.message = "Can't take screenshot because none camera type is not active";
                }
                else if (capture->TextureTarget == nullptr) {
                    response.message = "Can't take screenshot because texture target is null";
                }
                else
                    textureTarget = capture->TextureTarget;
            }
            else {
                response.message = "Can't take screenshot because none annotation name does not exist for this camera";
            }
        }
        else {
            capture = camera->getCaptureComponent(requests[i].image_type, false, requests[i].annotation_name);
            if (capture == nullptr) {
                response.message = "Can't take screenshot because none camera type is not active";
            }
            else if (capture->TextureTarget == nullptr) {
                response.message = "Can't take screenshot because texture target is null";
            }
            else
                textureTarget = capture->TextureTarget;
        }
        
        bool disable_gamma = false;
        if (requests[i].image_type == ImageCaptureBase::ImageType::Segmentation || requests[i].image_type == ImageCaptureBase::ImageType::Annotation)disable_gamma = true;
        render_params.push_back(std::make_shared<RenderRequest::RenderParams>(capture, textureTarget, requests[i].pixels_as_float, requests[i].compress, disable_gamma));
    }

    if (nullptr == gameViewport) {
        return;
    }

    auto query_camera_pose_cb = [this, &requests, &responses]() {
        size_t count = requests.size();
        for (size_t i = 0; i < count; i++) {
            const ImageRequest& request = requests.at(i);
            APIPCamera* camera = cameras_->at(request.camera_name);
            ImageResponse& response = responses.at(i);
            auto camera_pose = camera->getPose();
            response.camera_position = camera_pose.position;
            response.camera_orientation = camera_pose.orientation;
        }
    };
    RenderRequest render_request{ gameViewport, std::move(query_camera_pose_cb) };

    render_request.getScreenshot(render_params.data(), render_results, render_params.size(), use_safe_method);

    for (unsigned int i = 0; i < requests.size(); ++i) {
        const ImageRequest& request = requests.at(i);
        ImageResponse& response = responses.at(i);

        response.camera_name = request.camera_name;
        response.time_stamp = render_results[i]->time_stamp;
        response.image_data_uint8 = std::vector<uint8_t>(render_results[i]->image_data_uint8.GetData(), render_results[i]->image_data_uint8.GetData() + render_results[i]->image_data_uint8.Num());
        response.image_data_float = std::vector<float>(render_results[i]->image_data_float.GetData(), render_results[i]->image_data_float.GetData() + render_results[i]->image_data_float.Num());

        if (use_safe_method) {
            // Currently, we don't have a way to synthronize image capturing and camera pose when safe method is used,
            APIPCamera* camera = cameras_->at(request.camera_name);
            msr::airlib::Pose pose = camera->getPose();
            response.camera_position = pose.position;
            response.camera_orientation = pose.orientation;
        }
        response.pixels_as_float = request.pixels_as_float;
        response.compress = request.compress;
        response.width = render_results[i]->width;
        response.height = render_results[i]->height;
        response.image_type = request.image_type;
		response.annotation_name = request.annotation_name;
    }
}

bool UnrealImageCapture::updateCameraVisibility(APIPCamera* camera, const msr::airlib::ImageCaptureBase::ImageRequest& request)
{
    bool visibilityChanged = false;
    if (!camera->getCameraTypeEnabled(request.image_type, request.annotation_name)) {
        camera->setCameraTypeEnabled(request.image_type, true, request.annotation_name);
        visibilityChanged = true;
    }

    return visibilityChanged;
}

bool UnrealImageCapture::getScreenshotScreen(ImageType image_type, std::vector<uint8_t>& compressedPng)
{
    FScreenshotRequest::RequestScreenshot(false); // This is an async operation
    return true;
}

void UnrealImageCapture::addScreenCaptureHandler(UWorld* world)
{
    static bool is_installed = false;

    if (!is_installed) {
        UGameViewportClient* ViewportClient = world->GetGameViewport();
        ViewportClient->OnScreenshotCaptured().Clear();
        ViewportClient->OnScreenshotCaptured().AddLambda(
            [this](int32 SizeX, int32 SizeY, const TArray<FColor>& Bitmap) {
                // Make sure that all alpha values are opaque.
                TArray<FColor>& RefBitmap = const_cast<TArray<FColor>&>(Bitmap);
                for (auto& Color : RefBitmap)
                    Color.A = 255;

                TArray<uint8_t> last_compressed_png;
                FImageUtils::CompressImageArray(SizeX, SizeY, RefBitmap, last_compressed_png);
                last_compressed_png_ = std::vector<uint8_t>(last_compressed_png.GetData(), last_compressed_png.GetData() + last_compressed_png.Num());
            });

        is_installed = true;
    }
}
