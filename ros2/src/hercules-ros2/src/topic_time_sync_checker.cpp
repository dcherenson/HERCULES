#include <rclcpp/rclcpp.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

// Standard ROS clock topic
#include "rosgraph_msgs/msg/clock.hpp" // for /clock

// Navigation and odometry
#include <nav_msgs/msg/odometry.hpp> // for nav_msgs::msg::Odometry

// GPS
#include "sensor_msgs/msg/nav_sat_fix.hpp" // for sensor_msgs::msg::NavSatFix

// IMU & Magnetometer
#include "sensor_msgs/msg/imu.hpp"            // for sensor_msgs::msg::Imu
#include "sensor_msgs/msg/magnetic_field.hpp" // for sensor_msgs::msg::MagneticField

// LiDAR point clouds
#include "sensor_msgs/msg/point_cloud2.hpp" // for sensor_msgs::msg::PointCloud2

// Camera images & info
#include "sensor_msgs/msg/image.hpp"       // for sensor_msgs::msg::Image
#include "sensor_msgs/msg/camera_info.hpp" // for sensor_msgs::msg::CameraInfo

// AirSim-specific messages
#include "airsim_interfaces/msg/altimeter.hpp"                  // for airsim_interfaces::msg::Altimeter
#include "airsim_interfaces/msg/string_array.hpp"               // for airsim_interfaces::msg::StringArray
#include "airsim_interfaces/msg/instance_segmentation_list.hpp" // for airsim_interfaces::msg::InstanceSegmentationList
#include "airsim_interfaces/msg/gps_yaw.hpp"                    // for airsim_interfaces::msg::GPSYaw

#include <deque>
#include <numeric>
#include <vector>
#include <algorithm>
#include <cmath>

// A node that subscribes to two topics (e.g. camera and IMU),
// computes the time difference between each camera frame and its
// closest preceding IMU sample, and reports sync stats in real time.
class TopicTimeSyncChecker : public rclcpp::Node
{
public:
    TopicTimeSyncChecker()
        : Node("topic_time_sync_checker")
    {
        // Declare launch parameters
        declare_parameter<std::string>("camera_topic", "/camera/image_raw");
        declare_parameter<std::string>("imu_topic", "/imu/data");
        declare_parameter<double>("sync_tolerance", 0.005);

        declare_parameter<int>("imu_buffer_size", 2000);

        camera_topic_ = get_parameter("camera_topic").as_string();
        imu_topic_ = get_parameter("imu_topic").as_string();
        sync_tolerance_ = get_parameter("sync_tolerance").as_double();
        imu_buffer_size_ = get_parameter("imu_buffer_size").as_int();

        // Subscribe to IMU: store last N stamps in a deque
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, rclcpp::SensorDataQoS(),
            std::bind(&TopicTimeSyncChecker::imu_callback, this, std::placeholders::_1));

        // Subscribe to Camera images: on each frame, compute offset
        cam_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            camera_topic_, rclcpp::SensorDataQoS(),
            std::bind(&TopicTimeSyncChecker::cam_callback, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(),
                    "Checking sync between '%s' and '%s' with tolerance ±%.3f s",
                    camera_topic_.c_str(), imu_topic_.c_str(), sync_tolerance_);
    }

    ~TopicTimeSyncChecker() override
    {
        // Print summary when node shuts down
        if (offsets_.empty())
        {
            RCLCPP_WARN(get_logger(), "No camera–IMU pairs processed.");
            return;
        }
        double sum = std::accumulate(offsets_.begin(), offsets_.end(), 0.0);
        auto [min_it, max_it] = std::minmax_element(offsets_.begin(), offsets_.end());
        double avg = sum / offsets_.size();
        RCLCPP_INFO(get_logger(), "===== Sync Summary =====");
        RCLCPP_INFO(get_logger(), "Pairs: %zu", offsets_.size());
        RCLCPP_INFO(get_logger(), "Min offset: %.6f s", *min_it);
        RCLCPP_INFO(get_logger(), "Max offset: %.6f s", *max_it);
        RCLCPP_INFO(get_logger(), "Avg offset: %.6f s", avg);
        RCLCPP_INFO(get_logger(), "Sync tolerance: ±%.6f s", sync_tolerance_);
        RCLCPP_INFO(get_logger(), "Out-of-tolerance count: %zu", violations_);
        RCLCPP_INFO(get_logger(), "========================");
    }

private:
    // IMU callback: store header stamps
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        rclcpp::Time ts = msg->header.stamp;
        imu_buffer_.push_back(ts);
        if (imu_buffer_.size() > imu_buffer_size_)
        {
            imu_buffer_.pop_front();
        }
    }

    // Camera callback: match to the latest preceding IMU stamp
    void cam_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        rclcpp::Time cam_ts = msg->header.stamp;
        // Find the last IMU stamp ≤ cam_ts
        auto it = std::upper_bound(
            imu_buffer_.begin(), imu_buffer_.end(), cam_ts);
        if (it == imu_buffer_.begin())
        {
            RCLCPP_WARN(get_logger(),
                        "No IMU data older than camera stamp %.6f", cam_ts.seconds());
            return;
        }
        --it;
        double offset = (cam_ts - *it).seconds(); // image_ts – imu_ts
        offsets_.push_back(offset);
        if (std::fabs(offset) > sync_tolerance_)
        {
            violations_++;
            RCLCPP_WARN(get_logger(),
                        "Offset %.6f s exceeds ±%.6f s tolerance", offset, sync_tolerance_);
        }
    }

    // Parameters
    std::string camera_topic_, imu_topic_;
    double sync_tolerance_;
    int imu_buffer_size_;

    // State
    std::deque<rclcpp::Time> imu_buffer_;
    std::vector<double> offsets_;
    size_t violations_{0};

    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam_sub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TopicTimeSyncChecker>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
