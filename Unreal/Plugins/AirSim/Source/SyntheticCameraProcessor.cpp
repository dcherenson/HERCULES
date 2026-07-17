#include "SyntheticCameraProcessor.h"

#include "SimMode/SimModeBase.h"
#include "common/AirSimSettings.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <set>

namespace
{

typedef msr::airlib::AirSimSettings AirSimSettings;

//OpenCV COLORMAP_INFERNO (matplotlib inferno), stored as RGB
const uint8_t kInfernoRgb[256][3] = {
    { 0, 0, 4 }, { 1, 0, 5 }, { 1, 1, 6 }, { 1, 1, 8 }, { 2, 1, 10 }, { 2, 2, 12 }, { 2, 2, 14 }, { 3, 2, 16 },
    { 4, 3, 18 }, { 4, 3, 20 }, { 5, 4, 23 }, { 6, 4, 25 }, { 7, 5, 27 }, { 8, 5, 29 }, { 9, 6, 31 }, { 10, 7, 34 },
    { 11, 7, 36 }, { 12, 8, 38 }, { 13, 8, 41 }, { 14, 9, 43 }, { 16, 9, 45 }, { 17, 10, 48 }, { 18, 10, 50 }, { 20, 11, 52 },
    { 21, 11, 55 }, { 22, 11, 57 }, { 24, 12, 60 }, { 25, 12, 62 }, { 27, 12, 65 }, { 28, 12, 67 }, { 30, 12, 69 }, { 31, 12, 72 },
    { 33, 12, 74 }, { 35, 12, 76 }, { 36, 12, 79 }, { 38, 12, 81 }, { 40, 11, 83 }, { 41, 11, 85 }, { 43, 11, 87 }, { 45, 11, 89 },
    { 47, 10, 91 }, { 49, 10, 92 }, { 50, 10, 94 }, { 52, 10, 95 }, { 54, 9, 97 }, { 56, 9, 98 }, { 57, 9, 99 }, { 59, 9, 100 },
    { 61, 9, 101 }, { 62, 9, 102 }, { 64, 10, 103 }, { 66, 10, 104 }, { 68, 10, 104 }, { 69, 10, 105 }, { 71, 11, 106 }, { 73, 11, 106 },
    { 74, 12, 107 }, { 76, 12, 107 }, { 77, 13, 108 }, { 79, 13, 108 }, { 81, 14, 108 }, { 82, 14, 109 }, { 84, 15, 109 }, { 85, 15, 109 },
    { 87, 16, 110 }, { 89, 16, 110 }, { 90, 17, 110 }, { 92, 18, 110 }, { 93, 18, 110 }, { 95, 19, 110 }, { 97, 19, 110 }, { 98, 20, 110 },
    { 100, 21, 110 }, { 101, 21, 110 }, { 103, 22, 110 }, { 105, 22, 110 }, { 106, 23, 110 }, { 108, 24, 110 }, { 109, 24, 110 }, { 111, 25, 110 },
    { 113, 25, 110 }, { 114, 26, 110 }, { 116, 26, 110 }, { 117, 27, 110 }, { 119, 28, 109 }, { 120, 28, 109 }, { 122, 29, 109 }, { 124, 29, 109 },
    { 125, 30, 109 }, { 127, 30, 108 }, { 128, 31, 108 }, { 130, 32, 108 }, { 132, 32, 107 }, { 133, 33, 107 }, { 135, 33, 107 }, { 136, 34, 106 },
    { 138, 34, 106 }, { 140, 35, 105 }, { 141, 35, 105 }, { 143, 36, 105 }, { 144, 37, 104 }, { 146, 37, 104 }, { 147, 38, 103 }, { 149, 38, 103 },
    { 151, 39, 102 }, { 152, 39, 102 }, { 154, 40, 101 }, { 155, 41, 100 }, { 157, 41, 100 }, { 159, 42, 99 }, { 160, 42, 99 }, { 162, 43, 98 },
    { 163, 44, 97 }, { 165, 44, 96 }, { 166, 45, 96 }, { 168, 46, 95 }, { 169, 46, 94 }, { 171, 47, 94 }, { 173, 48, 93 }, { 174, 48, 92 },
    { 176, 49, 91 }, { 177, 50, 90 }, { 179, 50, 90 }, { 180, 51, 89 }, { 182, 52, 88 }, { 183, 53, 87 }, { 185, 53, 86 }, { 186, 54, 85 },
    { 188, 55, 84 }, { 189, 56, 83 }, { 191, 57, 82 }, { 192, 58, 81 }, { 193, 58, 80 }, { 195, 59, 79 }, { 196, 60, 78 }, { 198, 61, 77 },
    { 199, 62, 76 }, { 200, 63, 75 }, { 202, 64, 74 }, { 203, 65, 73 }, { 204, 66, 72 }, { 206, 67, 71 }, { 207, 68, 70 }, { 208, 69, 69 },
    { 210, 70, 68 }, { 211, 71, 67 }, { 212, 72, 66 }, { 213, 74, 65 }, { 215, 75, 63 }, { 216, 76, 62 }, { 217, 77, 61 }, { 218, 78, 60 },
    { 219, 80, 59 }, { 221, 81, 58 }, { 222, 82, 56 }, { 223, 83, 55 }, { 224, 85, 54 }, { 225, 86, 53 }, { 226, 87, 52 }, { 227, 89, 51 },
    { 228, 90, 49 }, { 229, 92, 48 }, { 230, 93, 47 }, { 231, 94, 46 }, { 232, 96, 45 }, { 233, 97, 43 }, { 234, 99, 42 }, { 235, 100, 41 },
    { 235, 102, 40 }, { 236, 103, 38 }, { 237, 105, 37 }, { 238, 106, 36 }, { 239, 108, 35 }, { 239, 110, 33 }, { 240, 111, 32 }, { 241, 113, 31 },
    { 241, 115, 29 }, { 242, 116, 28 }, { 243, 118, 27 }, { 243, 120, 25 }, { 244, 121, 24 }, { 245, 123, 23 }, { 245, 125, 21 }, { 246, 126, 20 },
    { 246, 128, 19 }, { 247, 130, 18 }, { 247, 132, 16 }, { 248, 133, 15 }, { 248, 135, 14 }, { 248, 137, 12 }, { 249, 139, 11 }, { 249, 140, 10 },
    { 249, 142, 9 }, { 250, 144, 8 }, { 250, 146, 7 }, { 250, 148, 7 }, { 251, 150, 6 }, { 251, 151, 6 }, { 251, 153, 6 }, { 251, 155, 6 },
    { 251, 157, 7 }, { 252, 159, 7 }, { 252, 161, 8 }, { 252, 163, 9 }, { 252, 165, 10 }, { 252, 166, 12 }, { 252, 168, 13 }, { 252, 170, 15 },
    { 252, 172, 17 }, { 252, 174, 18 }, { 252, 176, 20 }, { 252, 178, 22 }, { 252, 180, 24 }, { 251, 182, 26 }, { 251, 184, 29 }, { 251, 186, 31 },
    { 251, 188, 33 }, { 251, 190, 35 }, { 250, 192, 38 }, { 250, 194, 40 }, { 250, 196, 42 }, { 250, 198, 45 }, { 249, 199, 47 }, { 249, 201, 50 },
    { 249, 203, 53 }, { 248, 205, 55 }, { 248, 207, 58 }, { 247, 209, 61 }, { 247, 211, 64 }, { 246, 213, 67 }, { 246, 215, 70 }, { 245, 217, 73 },
    { 245, 219, 76 }, { 244, 221, 79 }, { 244, 223, 83 }, { 244, 225, 86 }, { 243, 227, 90 }, { 243, 229, 93 }, { 242, 230, 97 }, { 242, 232, 101 },
    { 242, 234, 105 }, { 241, 236, 109 }, { 241, 237, 113 }, { 241, 239, 117 }, { 241, 241, 121 }, { 242, 242, 125 }, { 242, 244, 130 }, { 243, 245, 134 },
    { 243, 246, 138 }, { 244, 248, 142 }, { 245, 249, 146 }, { 246, 250, 150 }, { 248, 251, 154 }, { 249, 252, 157 }, { 250, 253, 161 }, { 252, 255, 164 },
};

//Planck's law constants for spectral radiance in [W / (m^2 sr um)] with lambda in um
constexpr double kPlanckC1 = 1.19104e8;
constexpr double kPlanckC2 = 1.43879e4;

inline uint8_t saturateU8(double v)
{
    double r = std::round(v);
    if (r <= 0.0) return 0;
    if (r >= 255.0) return 255;
    return static_cast<uint8_t>(r);
}

inline uint8_t saturateU8i(int v)
{
    return static_cast<uint8_t>(std::min(255, std::max(0, v)));
}

//Rec.601 luma, same coefficients cv_bridge / cv::cvtColor use for color->mono8
inline uint8_t rgbToGray(uint8_t r, uint8_t g, uint8_t b)
{
    return saturateU8(0.299 * r + 0.587 * g + 0.114 * b);
}

// Pack RGB into a 32-bit key: 0x00RRGGBB (same layout as the ROS node)
inline uint32_t makeColorKey(uint8_t r, uint8_t g, uint8_t b)
{
    return (static_cast<uint32_t>(r) << 16) |
           (static_cast<uint32_t>(g) << 8) |
           static_cast<uint32_t>(b);
}

void grayFromRgb(const std::vector<uint8_t>& rgb, size_t pixel_count, std::vector<uint8_t>& gray)
{
    gray.resize(pixel_count);
    for (size_t i = 0; i < pixel_count; ++i)
        gray[i] = rgbToGray(rgb[i * 3 + 0], rgb[i * 3 + 1], rgb[i * 3 + 2]);
}

//port of cv::addWeighted for CV_8U
void addWeighted(const std::vector<uint8_t>& a, double alpha,
                 const std::vector<uint8_t>& b, double beta,
                 std::vector<uint8_t>& dst)
{
    dst.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        dst[i] = saturateU8(a[i] * alpha + b[i] * beta);
}

//automatic gain control by 2%/98% histogram percentiles
//(port of thermal_image_node.cpp lines 155-196)
void agcStretch(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst)
{
    int hist[256] = { 0 };
    for (uint8_t v : src)
        ++hist[v];

    int total = static_cast<int>(src.size());
    int low_count = static_cast<int>(0.02 * total);
    int high_count = static_cast<int>(0.98 * total);

    int cumsum = 0;
    int p_low = 0, p_high = 255;
    for (int i = 0; i < 256; ++i) {
        cumsum += hist[i];
        if (cumsum >= low_count) {
            p_low = i;
            break;
        }
    }
    cumsum = 0;
    for (int i = 255; i >= 0; --i) {
        cumsum += hist[i];
        if (cumsum >= total - high_count) {
            p_high = i;
            break;
        }
    }
    if (p_high <= p_low)
        p_high = p_low + 1; // avoid divide by zero

    double alpha_gain = 255.0 / (p_high - p_low);
    double beta_offset = -alpha_gain * p_low;

    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        dst[i] = saturateU8(src[i] * alpha_gain + beta_offset);
}

//standard CLAHE (contrast limited adaptive histogram equalization), 8x8 tiles,
//clip limit 2.0 — same algorithm as cv::createCLAHE(2.0, cv::Size(8, 8))
void clahe(const std::vector<uint8_t>& src, int width, int height, std::vector<uint8_t>& dst)
{
    constexpr int kTilesX = 8, kTilesY = 8;
    constexpr double kClipLimit = 2.0;
    constexpr int kHistSize = 256;

    //extend the image to a tile-divisible size with reflected border (like OpenCV)
    const int tile_w = (width + kTilesX - 1) / kTilesX;
    const int tile_h = (height + kTilesY - 1) / kTilesY;
    const int ext_w = tile_w * kTilesX;
    const int ext_h = tile_h * kTilesY;

    auto sample = [&](int x, int y) -> uint8_t {
        //BORDER_REFLECT_101 on the right/bottom extension
        if (x >= width) x = 2 * width - x - 2;
        if (y >= height) y = 2 * height - y - 2;
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        return src[static_cast<size_t>(y) * width + x];
    };

    const int tile_area = tile_w * tile_h;
    int clip = std::max(static_cast<int>(kClipLimit * tile_area / kHistSize), 1);
    const double lut_scale = static_cast<double>(kHistSize - 1) / tile_area;

    //per-tile clipped-equalization LUTs
    std::vector<uint8_t> tile_luts(static_cast<size_t>(kTilesX) * kTilesY * kHistSize);
    for (int ty = 0; ty < kTilesY; ++ty) {
        for (int tx = 0; tx < kTilesX; ++tx) {
            int hist[kHistSize] = { 0 };
            for (int y = ty * tile_h; y < (ty + 1) * tile_h; ++y)
                for (int x = tx * tile_w; x < (tx + 1) * tile_w; ++x)
                    ++hist[sample(x, y)];

            //clip histogram and redistribute the excess (OpenCV scheme)
            int clipped = 0;
            for (int i = 0; i < kHistSize; ++i) {
                if (hist[i] > clip) {
                    clipped += hist[i] - clip;
                    hist[i] = clip;
                }
            }
            int redist_batch = clipped / kHistSize;
            int residual = clipped - redist_batch * kHistSize;
            for (int i = 0; i < kHistSize; ++i)
                hist[i] += redist_batch;
            if (residual != 0) {
                int residual_step = std::max(kHistSize / residual, 1);
                for (int i = 0; i < kHistSize && residual > 0; i += residual_step, --residual)
                    ++hist[i];
            }

            uint8_t* lut = &tile_luts[(static_cast<size_t>(ty) * kTilesX + tx) * kHistSize];
            int sum = 0;
            for (int i = 0; i < kHistSize; ++i) {
                sum += hist[i];
                lut[i] = saturateU8(sum * lut_scale);
            }
        }
    }

    //bilinear interpolation between the four neighboring tile LUTs
    dst.resize(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        double tyf = static_cast<double>(y) / tile_h - 0.5;
        int ty1 = static_cast<int>(std::floor(tyf));
        int ty2 = ty1 + 1;
        double ya = tyf - ty1;
        ty1 = std::max(ty1, 0);
        ty2 = std::min(ty2, kTilesY - 1);

        for (int x = 0; x < width; ++x) {
            double txf = static_cast<double>(x) / tile_w - 0.5;
            int tx1 = static_cast<int>(std::floor(txf));
            int tx2 = tx1 + 1;
            double xa = txf - tx1;
            tx1 = std::max(tx1, 0);
            tx2 = std::min(tx2, kTilesX - 1);

            uint8_t v = src[static_cast<size_t>(y) * width + x];
            const uint8_t* lut11 = &tile_luts[(static_cast<size_t>(ty1) * kTilesX + tx1) * kHistSize];
            const uint8_t* lut12 = &tile_luts[(static_cast<size_t>(ty1) * kTilesX + tx2) * kHistSize];
            const uint8_t* lut21 = &tile_luts[(static_cast<size_t>(ty2) * kTilesX + tx1) * kHistSize];
            const uint8_t* lut22 = &tile_luts[(static_cast<size_t>(ty2) * kTilesX + tx2) * kHistSize];

            double res = lut11[v] * (1.0 - xa) * (1.0 - ya) +
                         lut12[v] * xa * (1.0 - ya) +
                         lut21[v] * (1.0 - xa) * ya +
                         lut22[v] * xa * ya;
            dst[static_cast<size_t>(y) * width + x] = saturateU8(res);
        }
    }
}

//separable Gaussian blur, sigma 5.0 -> 31-tap kernel (what cv::GaussianBlur picks
//for ksize=0, sigma=5 on 8-bit input), BORDER_REFLECT_101
void gaussianBlurSigma5(const std::vector<uint8_t>& src, int width, int height, std::vector<uint8_t>& dst)
{
    constexpr int kRadius = 15; //31 taps
    constexpr double kSigma = 5.0;

    double kernel[2 * kRadius + 1];
    double ksum = 0.0;
    for (int i = -kRadius; i <= kRadius; ++i) {
        kernel[i + kRadius] = std::exp(-0.5 * (static_cast<double>(i) * i) / (kSigma * kSigma));
        ksum += kernel[i + kRadius];
    }
    for (double& k : kernel)
        k /= ksum;

    auto reflect = [](int p, int len) -> int {
        //BORDER_REFLECT_101
        while (p < 0 || p >= len) {
            if (p < 0) p = -p;
            if (p >= len) p = 2 * len - p - 2;
        }
        return p;
    };

    std::vector<double> tmp(static_cast<size_t>(width) * height);
    //horizontal pass
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double acc = 0.0;
            for (int k = -kRadius; k <= kRadius; ++k)
                acc += kernel[k + kRadius] * src[static_cast<size_t>(y) * width + reflect(x + k, width)];
            tmp[static_cast<size_t>(y) * width + x] = acc;
        }
    }
    //vertical pass
    dst.resize(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double acc = 0.0;
            for (int k = -kRadius; k <= kRadius; ++k)
                acc += kernel[k + kRadius] * tmp[static_cast<size_t>(reflect(y + k, height)) * width + x];
            dst[static_cast<size_t>(y) * width + x] = saturateU8(acc);
        }
    }
}

//port of cv::normalize(src, dst, 0, max_out, cv::NORM_MINMAX) for CV_8U
void normalizeMinMax(std::vector<uint8_t>& img, double max_out)
{
    uint8_t mn = 255, mx = 0;
    for (uint8_t v : img) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    double scale = (mx > mn) ? max_out / (mx - mn) : 0.0;
    for (uint8_t& v : img)
        v = saturateU8((v - mn) * scale);
}

//spectral radiance integral over the LWIR band with a Gaussian sensor response
//centered at 11 um (sigma 1 um); band/resp supplied by the caller so the two
//pipelines keep their (slightly different) band construction
double planckBandRadiance(const std::vector<double>& band, const std::vector<double>& resp,
                          double temperature_K, double emissivity)
{
    double sum = 0.0;
    for (size_t i = 0; i < band.size(); ++i) {
        double lambda = band[i];
        double Ld = emissivity * resp[i] *
                    (kPlanckC1 / (std::pow(lambda, 5) *
                                  (std::exp(kPlanckC2 / (lambda * temperature_K)) - 1.0)));
        sum += Ld * 0.01; // d-lambda = 0.01 um
    }
    return sum;
}

std::string toLower(const std::string& s)
{
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

} //namespace

SyntheticCameraProcessor& SyntheticCameraProcessor::instance()
{
    static SyntheticCameraProcessor processor;
    return processor;
}

//port of thermal_image_node.cpp initLUT (lines 301-349): per-label random
//temperature/emissivity, Planck LWIR integral, normalized to [0,255].
//Uses a fixed seed (settings) instead of std::random_device for reproducibility,
//and extends the LUT incrementally when labels first appear mid-run (the ROS node
//mapped unknown labels to 0 forever).
void SyntheticCameraProcessor::ensureNightVisionLut(const std::vector<uint8_t>& seg_gray)
{
    const auto& cfg = AirSimSettings::singleton().synthetic_camera.night_vision;

    if (!nvg_rng_initialized_) {
        nvg_rng_.seed(static_cast<uint64_t>(cfg.seed));
        nvg_rng_initialized_ = true;
    }

    std::set<uint8_t> labels(seg_gray.begin(), seg_gray.end());

    // synthetic camera response (mu=11, sigma=1); note: node A uses lambda < 14.0
    const double mu = 11.0, sigma = 1.0;
    std::vector<double> band, resp;
    for (double lambda = 8.0; lambda < 14.0; lambda += 0.01) {
        band.push_back(lambda);
        resp.push_back(std::exp(-0.5 * std::pow((lambda - mu) / sigma, 2)));
    }

    std::uniform_real_distribution<double> dT(cfg.temp_min, cfg.temp_max);
    std::uniform_real_distribution<double> dE(cfg.eps_min, cfg.eps_max);

    bool changed = false;
    for (uint8_t L : labels) {
        if (nvg_rad_cache_.count(L))
            continue;
        double T = dT(nvg_rng_);
        double eps = dE(nvg_rng_);
        nvg_rad_cache_[L] = planckBandRadiance(band, resp, T, eps);
        changed = true;
    }
    if (!changed && !nvg_lut_.empty())
        return;

    double max_rad = 0.0;
    for (const auto& kv : nvg_rad_cache_)
        max_rad = std::max(max_rad, kv.second);
    if (max_rad <= 0.0)
        max_rad = 1.0;

    nvg_lut_.clear();
    for (const auto& kv : nvg_rad_cache_)
        nvg_lut_[kv.first] = static_cast<uint8_t>(std::round((kv.second / max_rad) * 255.0));
}

//port of thermal_image_node.cpp "nightvision" branch (lines 90-273)
void SyntheticCameraProcessor::composeNightVision(const std::vector<uint8_t>& scene_rgb,
                                                  const std::vector<uint8_t>& seg_rgb,
                                                  int width, int height,
                                                  std::vector<uint8_t>& out_rgb)
{
    std::lock_guard<std::mutex> lock(mutex_);

    const auto& cfg = AirSimSettings::singleton().synthetic_camera.night_vision;
    const size_t pixel_count = static_cast<size_t>(width) * height;

    //the ROS node consumed the segmentation topic as mono8; reproduce that with luma
    std::vector<uint8_t> seg_gray;
    grayFromRgb(seg_rgb, pixel_count, seg_gray);

    ensureNightVisionLut(seg_gray);

    //flat thermal map from the per-label LUT
    std::vector<uint8_t> thermo(pixel_count);
    for (size_t i = 0; i < pixel_count; ++i) {
        auto it = nvg_lut_.find(seg_gray[i]);
        thermo[i] = (it != nvg_lut_.end() ? it->second : 0);
    }

    //blend with the scene grayscale; if the scene capture has a different
    //resolution (or is missing) fall back to the pure thermal map, exactly like
    //the ROS node did when the scene topic size didn't match the segmentation
    const bool have_scene = (scene_rgb.size() == pixel_count * 3);
    std::vector<uint8_t> gray;
    std::vector<uint8_t> blended;
    if (have_scene) {
        grayFromRgb(scene_rgb, pixel_count, gray);
        addWeighted(thermo, cfg.blend_alpha, gray, 1.0 - cfg.blend_alpha, blended);
    }
    else {
        blended = thermo;
    }

    //base luminance from the real scene, falling back to blended
    const std::vector<uint8_t>& luminance = have_scene ? gray : blended;

    //small thermal fusion on top of the luminance (for "fused" goggles)
    const double thermal_blend = 0.15; // 0 = pure scene, 1 = pure thermal
    std::vector<uint8_t> fused;
    addWeighted(luminance, 1.0 - thermal_blend, blended, thermal_blend, fused);

    //1) automatic gain control by histogram percentiles
    std::vector<uint8_t> agc;
    agcStretch(fused, agc);

    //2) CLAHE for local contrast
    std::vector<uint8_t> clahe_out;
    clahe(agc, width, height, clahe_out);

    //3) gain-dependent shot noise (darker scenes produce more grain)
    double mean_val = 0.0;
    for (uint8_t v : fused)
        mean_val += v;
    mean_val /= std::max<size_t>(pixel_count, 1);
    double norm_mean = std::max(1.0, mean_val);
    double gain_factor = std::min(std::max((80.0 / norm_mean) * cfg.nvg_gain, 0.5), 4.0);
    double noise_sigma = 3.0 * gain_factor;

    //deterministic noise stream (the ROS node used an unseeded cv::randn); the
    //generator advances across frames so the grain is not static
    if (!noise_rng_initialized_) {
        noise_rng_.seed(static_cast<uint64_t>(cfg.seed) ^ 0x9E3779B97F4A7C15ull);
        noise_rng_initialized_ = true;
    }
    std::normal_distribution<double> noise_dist(0.0, noise_sigma);
    std::vector<uint8_t> noisy(pixel_count);
    for (size_t i = 0; i < pixel_count; ++i) {
        int n = static_cast<int>(std::lround(noise_dist(noise_rng_)));
        noisy[i] = saturateU8i(static_cast<int>(clahe_out[i]) + n);
    }

    //4) highlight bloom/halo around the brightest pixels
    std::vector<uint8_t> bright_mask(pixel_count);
    for (size_t i = 0; i < pixel_count; ++i)
        bright_mask[i] = (noisy[i] > 220) ? 255 : 0;
    std::vector<uint8_t> glow;
    gaussianBlurSigma5(bright_mask, width, height, glow);
    normalizeMinMax(glow, 40.0);
    std::vector<uint8_t> halo(pixel_count);
    for (size_t i = 0; i < pixel_count; ++i)
        halo[i] = saturateU8i(static_cast<int>(noisy[i]) + glow[i]);

    //5) radial vignette (port of thermal_image_node.cpp lines 235-256)
    std::vector<uint8_t> vignet(pixel_count);
    {
        const float cx = width / 2.f, cy = height / 2.f;
        const float maxD = std::hypot(width / 2.f, height / 2.f);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float d = std::hypot(x - cx, y - cy) / maxD;
                float v = 1.0f - d * d;
                v = std::min(std::max(v, 0.35f), 1.0f); // a bit stronger vignette
                size_t i = static_cast<size_t>(y) * width + x;
                vignet[i] = saturateU8(halo[i] * static_cast<double>(v));
            }
        }
    }

    //6) green phosphor conversion (port of lines 258-272); output is RGB8
    out_rgb.resize(pixel_count * 3);
    for (size_t i = 0; i < pixel_count; ++i) {
        uint8_t g = saturateU8(vignet[i] * 0.8 + 15.0); // main green channel
        uint8_t b = saturateU8(g * 0.03); // tiny blue bleed
        out_rgb[i * 3 + 0] = 0; // no red
        out_rgb[i * 3 + 1] = g;
        out_rgb[i * 3 + 2] = b;
    }
}

//port of thermal_image_segmentation_based_node.cpp classifyLabel (lines 149-221)
SyntheticCameraProcessor::ThermalProfile SyntheticCameraProcessor::classifyLabel(
    const std::string& label, const std::string& object_name) const
{
    const auto& cfg = AirSimSettings::singleton().synthetic_camera.thermal_ir;

    std::string combined = label + " " + object_name;
    std::string l = toLower(combined);

    ThermalProfile p;
    // Default neutral
    p.base_temp_K = 295.0; // ~22 C
    p.emissivity = 0.90;
    p.is_animal = false;
    p.is_fire = false;
    p.is_kangaroo = false;

    if (l.find("tree") != std::string::npos ||
        l.find("bush") != std::string::npos ||
        l.find("grass") != std::string::npos ||
        l.find("plant") != std::string::npos ||
        l.find("vegetation") != std::string::npos) {
        // Vegetation, slightly cooler
        p.base_temp_K = 293.0; // ~20 C
        p.emissivity = 0.97;
    }
    else if (l.find("road") != std::string::npos ||
             l.find("ground") != std::string::npos ||
             l.find("dirt") != std::string::npos ||
             l.find("rock") != std::string::npos ||
             l.find("soil") != std::string::npos) {
        // Soil / asphalt / rock
        p.base_temp_K = 300.0; // ~27 C
        p.emissivity = 0.93;
    }
    else if (l.find("car") != std::string::npos ||
             l.find("truck") != std::string::npos ||
             l.find("vehicle") != std::string::npos ||
             l.find("husky") != std::string::npos) {
        // Vehicles / metal, warmed by engine
        p.base_temp_K = 305.0; // ~32 C
        p.emissivity = 0.90;
    }
    else if (l.find("fire") != std::string::npos ||
             l.find("flame") != std::string::npos ||
             l.find("torch") != std::string::npos) {
        // Fire source (very hot, high emissivity)
        p.base_temp_K = 1000.0;
        p.emissivity = 0.98;
        p.is_fire = true;
    }
    else if (l.find("animal") != std::string::npos ||
             l.find("kangaroo") != std::string::npos ||
             l.find("deer") != std::string::npos ||
             l.find("human") != std::string::npos ||
             l.find("person") != std::string::npos) {
        // Warm body: kangaroo, etc.
        p.base_temp_K = 315.0; // ~42 C
        p.emissivity = 0.98;
        p.is_animal = true;

        if (l.find("kangaroo") != std::string::npos)
            p.is_kangaroo = true;
    }

    //user-configured per-object overrides (settings.json SyntheticCameraSettings ->
    //ThermalIR -> Overrides) take priority over the keyword classification; first
    //matching entry wins
    for (const auto& o : cfg.overrides) {
        if (l.find(toLower(o.match)) != std::string::npos) {
            p.base_temp_K = o.temp_K;
            p.emissivity = o.emissivity;
            p.is_animal = o.is_animal;
            p.is_fire = o.is_fire;
            p.is_kangaroo = o.is_kangaroo;
            break;
        }
    }

    p.base_temp_K = std::min(std::max(p.base_temp_K, cfg.temp_min), cfg.temp_max);
    p.emissivity = std::min(std::max(p.emissivity, cfg.eps_min), cfg.eps_max);
    return p;
}

//native replacement of the ROS node's loadLabelMap CSV: zip the index-aligned
//instance segmentation name/color lists from the sim (same pairing the
//CSV-generation scripts use via listInstanceSegmentationObjects +
//getInstanceSegmentationColorMap)
void SyntheticCameraProcessor::ensureColorProfileMap()
{
    if (profile_map_initialized_)
        return;

    ASimModeBase* simmode = ASimModeBase::getSimMode();
    if (simmode == nullptr)
        return; //retry on the next request

    std::vector<std::string> names = simmode->GetAllInstanceSegmentationMeshIDs();
    std::vector<msr::airlib::Vector3r> colors = simmode->GetInstanceSegmentationColorMap();
    if (names.empty())
        return; //annotator not populated yet; retry on the next request

    size_t count = std::min(names.size(), colors.size());
    for (size_t i = 0; i < count; ++i) {
        uint8_t r = static_cast<uint8_t>(colors[i].x());
        uint8_t g = static_cast<uint8_t>(colors[i].y());
        uint8_t b = static_cast<uint8_t>(colors[i].z());
        //the mesh id carries the object name (e.g. bp_kangaroo23_...); the ROS CSV
        //split it into label/object_name columns which classifyLabel re-joined anyway
        color_to_profile_[makeColorKey(r, g, b)] = classifyLabel(names[i], "");
    }

    profile_map_initialized_ = true;
}

//port of thermal_image_segmentation_based_node.cpp initLUT (lines 647-777):
//radiance per color, normalized, animal/fire boosted. Unlike the ROS node (which
//froze the LUT on the first frame, mapping later-appearing colors to 0), the
//radiance cache grows as new segmentation colors are encountered and the LUT is
//renormalized, so objects entering view mid-run still render.
void SyntheticCameraProcessor::ensureThermalIrLut(const std::vector<uint8_t>& seg_rgb, int width, int height)
{
    //precomputeSpectralResponse (lines 291-304); note: node B uses lambda <= 14.0
    if (band_.empty()) {
        const double mu = 11.0, sigma = 1.0;
        for (double lambda = 8.0; lambda <= 14.0; lambda += 0.01) {
            band_.push_back(lambda);
            resp_.push_back(std::exp(-0.5 * std::pow((lambda - mu) / sigma, 2)));
        }
    }

    const size_t pixel_count = static_cast<size_t>(width) * height;
    std::set<uint32_t> colors;
    for (size_t i = 0; i < pixel_count; ++i)
        colors.insert(makeColorKey(seg_rgb[i * 3 + 0], seg_rgb[i * 3 + 1], seg_rgb[i * 3 + 2]));

    //until the native color->profile map is available every color classifies as
    //neutral; recompute from scratch each request so profiles apply once it loads
    if (!profile_map_initialized_)
        flir_rad_cache_.clear();

    bool changed = false;
    for (uint32_t key : colors) {
        if (flir_rad_cache_.count(key))
            continue;

        ThermalProfile prof; //default neutral if not in the native map
        auto itp = color_to_profile_.find(key);
        if (itp != color_to_profile_.end())
            prof = itp->second;

        double sum = planckBandRadiance(band_, resp_, prof.base_temp_K, prof.emissivity);

        // Optionally clamp fire radiance so it does not completely dominate
        if (prof.is_fire) {
            const double max_fire_factor = 3.0;
            sum = std::min(sum, max_fire_factor * 1e6); // arbitrary large scale
        }

        flir_rad_cache_[key] = sum;
        changed = true;
    }
    if (!changed && !flir_lut_.empty())
        return;

    double max_rad = 0.0;
    for (const auto& kv : flir_rad_cache_)
        max_rad = std::max(max_rad, kv.second);
    if (max_rad <= 0.0)
        max_rad = 1.0;

    flir_lut_.clear();
    for (const auto& kv : flir_rad_cache_) {
        uint8_t cnt = static_cast<uint8_t>(std::round((kv.second / max_rad) * 255.0));
        auto itp = color_to_profile_.find(kv.first);
        if (itp != color_to_profile_.end()) {
            // Ensure animals (kangaroos, etc.) live near the top of the brightness range
            if (itp->second.is_animal && cnt < 220)
                cnt = 220;
            if (itp->second.is_fire)
                cnt = 255;
        }
        flir_lut_[kv.first] = cnt;
    }
}

//port of thermal_image_segmentation_based_node.cpp "flir" branch (segCallback
//lines 350-644 minus the use_fire_rgb path, which is off by default): pure
//thermal map (no scene blend, per the node's comment at line 558) -> Inferno
//colormap -> kangaroo recolor
void SyntheticCameraProcessor::composeThermalIR(const std::vector<uint8_t>& seg_rgb,
                                                const std::vector<float>& depth_m,
                                                int width, int height,
                                                std::vector<uint8_t>& out_rgb)
{
    std::lock_guard<std::mutex> lock(mutex_);

    const auto& cfg = AirSimSettings::singleton().synthetic_camera.thermal_ir;
    const size_t pixel_count = static_cast<size_t>(width) * height;

    ensureColorProfileMap();
    ensureThermalIrLut(seg_rgb, width, height);

    const bool use_depth = (depth_m.size() == pixel_count);

    std::vector<uint8_t> thermo(pixel_count);
    std::vector<uint8_t> kangaroo_mask(pixel_count, 0);

    for (size_t i = 0; i < pixel_count; ++i) {
        uint32_t key = makeColorKey(seg_rgb[i * 3 + 0], seg_rgb[i * 3 + 1], seg_rgb[i * 3 + 2]);

        auto it = flir_lut_.find(key);
        uint8_t base_cnt = (it != flir_lut_.end() ? it->second : 0);

        auto itp = color_to_profile_.find(key);
        if (itp != color_to_profile_.end() && itp->second.is_kangaroo) {
            // Extra boost before colormap so they sit near the very hot end
            if (base_cnt < 245)
                base_cnt = 245;
            kangaroo_mask[i] = 255;
        }

        if (use_depth) {
            float d = depth_m[i];
            if (std::isfinite(d) && d > 0.0f) {
                double atten = 1.0 / (1.0 + cfg.depth_attenuation * d);
                double val = base_cnt * atten;
                thermo[i] = static_cast<uint8_t>(std::min(std::max(val, 0.0), 255.0));
            }
            else {
                thermo[i] = base_cnt;
            }
        }
        else {
            thermo[i] = base_cnt;
        }
    }

    //FLIR-style view: pure thermal map -> Inferno colormap, then kangaroo recolor
    //(bright orange, RGB (255,165,0)); truncating casts match the ROS node
    out_rgb.resize(pixel_count * 3);
    for (size_t i = 0; i < pixel_count; ++i) {
        const uint8_t* c = kInfernoRgb[thermo[i]];
        uint8_t r = c[0], g = c[1], b = c[2];
        if (kangaroo_mask[i]) {
            r = static_cast<uint8_t>(0.2 * r + 0.8 * 255.0);
            g = static_cast<uint8_t>(0.2 * g + 0.8 * 165.0);
            b = static_cast<uint8_t>(0.2 * b + 0.8 * 0.0);
        }
        out_rgb[i * 3 + 0] = r;
        out_rgb[i * 3 + 1] = g;
        out_rgb[i * 3 + 2] = b;
    }
}
