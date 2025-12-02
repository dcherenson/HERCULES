#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include <random>
#include <set>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

using std::placeholders::_1;

struct ThermalProfile
{
    double base_temp_K; // nominal surface temperature
    double emissivity;  // 0..1
};

class ThermalImageNode : public rclcpp::Node
{
public:
    ThermalImageNode()
        : Node("thermal_image_node"),
          initialized_(false)
    {
        // --- Topics ---
        declare_parameter("scene_topic",
                          std::string("/hercules_node/Drone1/front_center_Scene/image"));
        declare_parameter("seg_topic",
                          std::string("/hercules_node/Drone1/front_center_Segmentation/image"));
        declare_parameter("depth_topic",
                          std::string("/hercules_node/Drone1/front_center_DepthPlanar/image"));
        declare_parameter("thermal_topic",
                          std::string("/hercules_node/Drone1/front_center_ThermalIR/image"));

        declare_parameter("label_map_csv",
                          std::string("/home/sgarimella34/multi-robot-coordination/Cosys-AirSim/PythonClient/segmentation/label_color_map_ausenv.csv"));

        get_parameter("scene_topic", scene_topic_);
        get_parameter("seg_topic", seg_topic_);
        get_parameter("depth_topic", depth_topic_);
        get_parameter("thermal_topic", thermal_topic_);
        get_parameter("label_map_csv", label_map_csv_);

        // --- Thermal sim parameters (global floor/ceiling, mainly for sanity) ---
        declare_parameter("temp_min", 280.0);
        declare_parameter("temp_max", 1300.0); // allow fire-like objects
        declare_parameter("eps_min", 0.80);
        declare_parameter("eps_max", 0.99);
        get_parameter("temp_min", temp_min_);
        get_parameter("temp_max", temp_max_);
        get_parameter("eps_min", eps_min_);
        get_parameter("eps_max", eps_max_);

        // Depth attenuation factor (how fast intensity decays with distance)
        declare_parameter("depth_attenuation", 0.03); // 1 / (1 + k * d)
        get_parameter("depth_attenuation", depth_attenuation_);

        // --- Blending weight & colormap (still used for NVG) ---
        declare_parameter("thermal_weight", 0.25);
        declare_parameter("use_colormap", false);
        get_parameter("thermal_weight", alpha_);
        get_parameter("use_colormap", use_cmap_);

        // --- View mode: thermal | nightvision | flir ---
        declare_parameter("view_mode", std::string("nightvision"));
        get_parameter("view_mode", view_mode_);

        // --- NVG gain (multiplier for night-vision intensifier) ---
        declare_parameter("nvg_gain", 1.0);
        get_parameter("nvg_gain", nvg_gain_);

        // --- Precompute spectral response and load label map ---
        precomputeSpectralResponse();
        loadLabelMap(label_map_csv_);

        // --- ROS interfaces ---
        image_pub_ = create_publisher<sensor_msgs::msg::Image>(
            thermal_topic_, 10);

        scene_sub_ = create_subscription<sensor_msgs::msg::Image>(
            scene_topic_, 10,
            std::bind(&ThermalImageNode::sceneCallback, this, _1));

        seg_sub_ = create_subscription<sensor_msgs::msg::Image>(
            seg_topic_, 10,
            std::bind(&ThermalImageNode::segCallback, this, _1));

        depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
            depth_topic_, 10,
            std::bind(&ThermalImageNode::depthCallback, this, _1));

        RCLCPP_INFO(get_logger(),
                    "ThermalImageNode:\n"
                    " scene:      '%s'\n"
                    " segmentation:'%s'\n"
                    " depth:      '%s'\n"
                    " output:     '%s'\n"
                    " T-range:    [%.1f,%.1f] K\n"
                    " eps-range:  [%.2f,%.2f]\n"
                    " depth_att:  %.3f\n"
                    " view_mode:  %s",
                    scene_topic_.c_str(),
                    seg_topic_.c_str(),
                    depth_topic_.c_str(),
                    thermal_topic_.c_str(),
                    temp_min_, temp_max_,
                    eps_min_, eps_max_,
                    depth_attenuation_,
                    view_mode_.c_str());
    }

private:
    // Small helper: lowercase copy of a string
    static std::string toLower(const std::string &s)
    {
        std::string out = s;
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); });
        return out;
    }

    // Decide emissivity and base temperature from a semantic label string
    ThermalProfile classifyLabel(const std::string &label) const
    {
        std::string l = toLower(label);
        ThermalProfile p;

        // Very rough priors, you can tweak as you learn your environment better.
        if (l.find("tree") != std::string::npos ||
            l.find("bush") != std::string::npos ||
            l.find("grass") != std::string::npos ||
            l.find("plant") != std::string::npos)
        {
            // Vegetation
            p.base_temp_K = 295.0; // ~22 C
            p.emissivity = 0.97;
        }
        else if (l.find("road") != std::string::npos ||
                 l.find("ground") != std::string::npos ||
                 l.find("dirt") != std::string::npos ||
                 l.find("rock") != std::string::npos)
        {
            // Soil / asphalt / rock
            p.base_temp_K = 305.0; // warmer in sun
            p.emissivity = 0.93;
        }
        else if (l.find("car") != std::string::npos ||
                 l.find("truck") != std::string::npos ||
                 l.find("vehicle") != std::string::npos ||
                 l.find("husky") != std::string::npos)
        {
            // Vehicles / metal
            p.base_temp_K = 315.0; // engine / cabin warmer
            p.emissivity = 0.90;
        }
        else if (l.find("fire") != std::string::npos ||
                 l.find("flame") != std::string::npos ||
                 l.find("torch") != std::string::npos)
        {
            // Fire source (very hot, high emissivity)
            p.base_temp_K = 1000.0;
            p.emissivity = 0.98;
        }
        else if (l.find("animal") != std::string::npos ||
                 l.find("kangaroo") != std::string::npos ||
                 l.find("deer") != std::string::npos ||
                 l.find("human") != std::string::npos)
        {
            // Warm body
            p.base_temp_K = 310.0;
            p.emissivity = 0.98;
        }
        else
        {
            // Default neutral stuff
            p.base_temp_K = 295.0;
            p.emissivity = 0.90;
        }

        // Clamp to global min/max in case you extend these later
        p.base_temp_K = std::clamp(p.base_temp_K, temp_min_, temp_max_);
        p.emissivity = std::clamp(p.emissivity, eps_min_, eps_max_);
        return p;
    }

    // Load CSV: Label, ObjectName, SegmentationID, R, G, B
    void loadLabelMap(const std::string &csv_path)
    {
        std::ifstream file(csv_path);
        if (!file.is_open())
        {
            RCLCPP_WARN(get_logger(),
                        "Could not open label map CSV: %s", csv_path.c_str());
            return;
        }

        std::string line;
        // skip header
        std::getline(file, line);

        size_t count = 0;
        while (std::getline(file, line))
        {
            if (line.empty())
                continue;

            std::stringstream ss(line);
            std::string label, object_name, seg_id_str, r_str, g_str, b_str;

            if (!std::getline(ss, label, ','))
                continue;
            if (!std::getline(ss, object_name, ','))
                continue;
            if (!std::getline(ss, seg_id_str, ','))
                continue;
            // R,G,B exist but we do not need them for thermal; we can still parse
            std::getline(ss, r_str, ',');
            std::getline(ss, g_str, ',');
            std::getline(ss, b_str, ',');

            int seg_id = 0;
            try
            {
                seg_id = std::stoi(seg_id_str);
            }
            catch (...)
            {
                continue;
            }

            ThermalProfile prof = classifyLabel(label);
            id_to_profile_[static_cast<uint8_t>(seg_id)] = prof;
            ++count;
        }

        RCLCPP_INFO(get_logger(),
                    "Loaded %zu thermal profiles from %s",
                    count, csv_path.c_str());
    }

    // Precompute spectral band and sensor response for LWIR 8–14 µm
    void precomputeSpectralResponse()
    {
        const double mu = 11.0;   // center wavelength
        const double sigma = 1.0; // spread

        band_.clear();
        resp_.clear();

        for (double lambda = 8.0; lambda <= 14.0; lambda += 0.01)
        {
            band_.push_back(lambda);
            resp_.push_back(std::exp(-0.5 * std::pow((lambda - mu) / sigma, 2)));
        }
    }

    void sceneCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Keep last grayscale scene for NVG / potential blending
        auto cvb = cv_bridge::toCvShare(msg, "bgr8");
        cv::cvtColor(cvb->image, last_gray_, cv::COLOR_BGR2GRAY);
    }

    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            auto cvb = cv_bridge::toCvShare(msg, msg->encoding);

            if (msg->encoding == "32FC1")
            {
                last_depth_ = cvb->image.clone(); // meters
            }
            else if (msg->encoding == "16UC1")
            {
                // e.g. millimeters -> meters
                cv::Mat tmp;
                cvb->image.convertTo(tmp, CV_32F, 0.001);
                last_depth_ = tmp;
            }
            else
            {
                RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 5000,
                    "Unexpected depth encoding: %s", msg->encoding.c_str());
            }
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(get_logger(), "depthCallback cv_bridge exception: %s", e.what());
        }
    }

    void segCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // 1) segmentation mask (mono8: each value is a SegmentationID)
        cv::Mat seg = cv_bridge::toCvShare(msg, "mono8")->image;

        // 2) init LUT on first frame (per-label radiance from CSV + physics)
        if (!initialized_)
        {
            initLUT(seg);
            initialized_ = true;
            RCLCPP_INFO(
                get_logger(),
                "LUT initialized for %zu labels", lut_.size());
        }

        // 3) build flat thermal map [0–255] with depth attenuation
        cv::Mat thermo(seg.size(), CV_8UC1);

        bool use_depth =
            !last_depth_.empty() &&
            last_depth_.size() == seg.size() &&
            last_depth_.type() == CV_32FC1;

        for (int y = 0; y < seg.rows; ++y)
        {
            const uint8_t *seg_row = seg.ptr<uint8_t>(y);
            uint8_t *t_row = thermo.ptr<uint8_t>(y);

            const float *d_row = use_depth ? last_depth_.ptr<float>(y) : nullptr;

            for (int x = 0; x < seg.cols; ++x)
            {
                uint8_t lbl = seg_row[x];
                auto it = lut_.find(lbl);
                uint8_t base_cnt = (it != lut_.end() ? it->second : 0);

                if (use_depth)
                {
                    float d = d_row[x];
                    if (std::isfinite(d) && d > 0.0f)
                    {
                        double atten = 1.0 / (1.0 + depth_attenuation_ * d);
                        double val = base_cnt * atten;
                        t_row[x] = static_cast<uint8_t>(std::clamp(val, 0.0, 255.0));
                    }
                    else
                    {
                        t_row[x] = base_cnt;
                    }
                }
                else
                {
                    t_row[x] = base_cnt;
                }
            }
        }

        // 4) build "blended" only for NVG; for FLIR we use pure thermo
        cv::Mat blended;
        if (!last_gray_.empty() && last_gray_.size() == thermo.size())
        {
            cv::addWeighted(thermo, alpha_,
                            last_gray_, 1.0 - alpha_,
                            0.0, blended);
        }
        else
        {
            blended = thermo;
        }

        // 5) visualization
        cv::Mat output;
        std::string encoding;

        if (view_mode_ == "nightvision")
        {
            // keep your existing NVG pipeline here (omitted for brevity) ...
            // output = nv_bgr; encoding = "rgb8";
            // [YOUR NIGHT VISION CODE GOES HERE]
            // For now just fall back to green display of blended:
            cv::Mat nv_bgr;
            cv::cvtColor(blended, nv_bgr, cv::COLOR_GRAY2BGR);
            std::vector<cv::Mat> ch(3);
            cv::split(nv_bgr, ch);
            ch[1] = ch[1] * 0.9f + 15; // green
            ch[0] = ch[1] * 0.05f;
            ch[2] = cv::Mat::zeros(ch[2].size(), ch[2].type());
            cv::merge(ch, nv_bgr);
            output = nv_bgr;
            encoding = "rgb8";
        }
        else if (view_mode_ == "flir")
        {
            // FLIR-style view: pure thermal map -> Inferno colormap
            cv::applyColorMap(thermo, output, cv::COLORMAP_INFERNO);
            encoding = "rgb8";
        }
        else // "thermal" grayscale
        {
            if (use_cmap_)
            {
                cv::applyColorMap(thermo, output, cv::COLORMAP_INFERNO);
                encoding = "rgb8";
            }
            else
            {
                output = thermo;
                encoding = "mono8";
            }
        }

        // 6) publish
        auto out_msg = cv_bridge::CvImage(
                           msg->header, encoding, output)
                           .toImageMsg();
        image_pub_->publish(*out_msg);
    }

    // Build LUT: segID -> thermal count [0..255] using CSV + Planck integral
    void initLUT(const cv::Mat &seg)
    {
        std::set<uint8_t> labels;
        for (int y = 0; y < seg.rows; ++y)
        {
            const uint8_t *row = seg.ptr<uint8_t>(y);
            for (int x = 0; x < seg.cols; ++x)
                labels.insert(row[x]);
        }

        const double c1 = 1.19104e8;
        const double c2 = 1.43879e4;

        double max_rad = 0.0;
        std::unordered_map<uint8_t, double> rads;

        for (uint8_t L : labels)
        {
            // lookup profile; if not found, fall back to a default
            ThermalProfile prof;
            auto itp = id_to_profile_.find(L);
            if (itp != id_to_profile_.end())
            {
                prof = itp->second;
            }
            else
            {
                prof.base_temp_K = 295.0;
                prof.emissivity = 0.9;
            }

            double T = prof.base_temp_K;
            double eps = prof.emissivity;

            // integrate LWIR radiance over [8,14] µm
            double sum = 0.0;
            for (size_t i = 0; i < band_.size(); ++i)
            {
                double lambda = band_[i];
                double sensor = resp_[i];

                double Ld = eps * sensor *
                            (c1 / (std::pow(lambda, 5) *
                                   (std::exp(c2 / (lambda * T)) - 1.0)));

                sum += Ld * 0.01; // Δλ = 0.01 µm
            }
            rads[L] = sum;
            max_rad = std::max(max_rad, sum);
        }

        if (max_rad <= 0.0)
            max_rad = 1.0;

        // normalize to [0,255]
        for (auto &kv : rads)
        {
            uint8_t cnt = static_cast<uint8_t>(
                std::round((kv.second / max_rad) * 255.0));
            lut_[kv.first] = cnt;
        }
    }

    // state
    bool initialized_;
    std::unordered_map<uint8_t, uint8_t> lut_;                  // segID -> grayscale
    std::unordered_map<uint8_t, ThermalProfile> id_to_profile_; // segID -> thermal profile
    cv::Mat last_gray_;
    cv::Mat last_depth_;
    std::vector<double> band_;
    std::vector<double> resp_;

    // params & ROS
    std::string scene_topic_, seg_topic_, depth_topic_, thermal_topic_;
    std::string label_map_csv_;
    double temp_min_, temp_max_, eps_min_, eps_max_;
    double alpha_, nvg_gain_;
    double depth_attenuation_;
    bool use_cmap_;
    std::string view_mode_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
        scene_sub_,
        seg_sub_,
        depth_sub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ThermalImageNode>());
    rclcpp::shutdown();
    return 0;
}
