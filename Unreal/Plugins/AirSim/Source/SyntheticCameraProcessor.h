#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// CPU-side composition of the synthetic camera image types (ImageType::ThermalIR and
// ImageType::NightVision) from underlying captures rendered in the same simGetImages
// batch. The pixel pipelines are faithful ports of the two HERCULES ROS2
// post-processing nodes:
//   NightVision: ros2/src/hercules-ros2/src/thermal_image_node.cpp ("nightvision" mode)
//   ThermalIR:   ros2/src/hercules-ros2/src/thermal_image_segmentation_based_node.cpp ("flir" mode)
// with the FLIR label->temperature map built natively from the instance segmentation
// API instead of a CSV file.
//
// All uint8 image buffers are tightly packed interleaved RGB8, the byte order
// RenderRequest produces for uncompressed uint8 captures. Depth is planar meters.
// LUTs and the color->profile map are built lazily on first use and cached, like the
// ROS nodes; tuning comes from AirSimSettings::singleton().synthetic_camera.
class SyntheticCameraProcessor
{
public:
    static SyntheticCameraProcessor& instance();

    // scene_rgb may be empty or a different resolution than width*height (the ROS
    // node's fallback then applies: pure thermal map, no scene luminance)
    void composeNightVision(const std::vector<uint8_t>& scene_rgb,
                            const std::vector<uint8_t>& seg_rgb,
                            int width, int height,
                            std::vector<uint8_t>& out_rgb);

    // depth_m may be empty (no attenuation is applied then, like the ROS node
    // before its first depth message)
    void composeThermalIR(const std::vector<uint8_t>& seg_rgb,
                          const std::vector<float>& depth_m,
                          int width, int height,
                          std::vector<uint8_t>& out_rgb);

private:
    struct ThermalProfile
    {
        double base_temp_K = 295.0;
        double emissivity = 0.90;
        bool is_animal = false;
        bool is_fire = false;
        bool is_kangaroo = false;
    };

    SyntheticCameraProcessor() = default;

    void ensureNightVisionLut(const std::vector<uint8_t>& seg_gray);
    void ensureColorProfileMap();
    void ensureThermalIrLut(const std::vector<uint8_t>& seg_rgb, int width, int height);
    ThermalProfile classifyLabel(const std::string& label, const std::string& object_name) const;

    std::mutex mutex_;

    //NightVision state: seg-gray label -> radiance cache and normalized thermal count.
    //Labels unseen so far get a radiance on first encounter (deterministic draw from
    //nvg_rng_) and the LUT is renormalized, so late-appearing objects still show up.
    std::map<uint8_t, double> nvg_rad_cache_;
    std::map<uint8_t, uint8_t> nvg_lut_;
    bool nvg_rng_initialized_ = false;
    std::mt19937_64 nvg_rng_;
    //noise stream persists across frames so the grain varies per frame while the
    //whole sequence stays reproducible for a fixed seed
    bool noise_rng_initialized_ = false;
    std::mt19937_64 noise_rng_;

    //ThermalIR state: 0x00RRGGBB color key -> profile / radiance cache / thermal
    //count; same incremental scheme as NVG for colors that first appear mid-run
    bool profile_map_initialized_ = false;
    std::unordered_map<uint32_t, ThermalProfile> color_to_profile_;
    std::unordered_map<uint32_t, double> flir_rad_cache_;
    std::unordered_map<uint32_t, uint8_t> flir_lut_;
    std::vector<double> band_;
    std::vector<double> resp_;
};
