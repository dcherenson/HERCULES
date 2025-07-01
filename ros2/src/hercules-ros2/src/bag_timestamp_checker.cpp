#include <rclcpp/rclcpp.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include "rclcpp/rclcpp.hpp"

// Standard ROS clock topic
#include "rosgraph_msgs/msg/clock.hpp"              // for /clock

// Navigation and odometry
#include <nav_msgs/msg/odometry.hpp>               // for nav_msgs::msg::Odometry

// GPS
#include "sensor_msgs/msg/nav_sat_fix.hpp"           // for sensor_msgs::msg::NavSatFix

// IMU & Magnetometer
#include "sensor_msgs/msg/imu.hpp"                  // for sensor_msgs::msg::Imu
#include "sensor_msgs/msg/magnetic_field.hpp"       // for sensor_msgs::msg::MagneticField

// LiDAR point clouds
#include "sensor_msgs/msg/point_cloud2.hpp"         // for sensor_msgs::msg::PointCloud2

// Camera images & info
#include "sensor_msgs/msg/image.hpp"                // for sensor_msgs::msg::Image
#include "sensor_msgs/msg/camera_info.hpp"          // for sensor_msgs::msg::CameraInfo

// AirSim-specific messages
#include "airsim_interfaces/msg/altimeter.hpp"          // for airsim_interfaces::msg::Altimeter
#include "airsim_interfaces/msg/string_array.hpp"       // for airsim_interfaces::msg::StringArray
#include "airsim_interfaces/msg/instance_segmentation_list.hpp" // for airsim_interfaces::msg::InstanceSegmentationList
#include "airsim_interfaces/msg/gps_yaw.hpp"            // for airsim_interfaces::msg::GPSYaw


#include <vector>
#include <algorithm>
#include <cmath>

class TimestampChecker : public rclcpp::Node
{
public:
    TimestampChecker()
        : Node("bag_timestamp_checker")
    {
        // Declare parameters: topic name, expected period, and tolerance
        this->declare_parameter<std::string>("topic", "/hercules_node/Drone1/imu/imu"); // default topic :contentReference[oaicite:0]{index=0}
        this->declare_parameter<double>("expected_period", 0.05);                       // default 0.05 s :contentReference[oaicite:1]{index=1}
        this->declare_parameter<double>("tolerance", 1e-6);                             // default tolerance :contentReference[oaicite:2]{index=2}

        topic_ = this->get_parameter("topic").as_string();
        expected_period_ = this->get_parameter("expected_period").as_double();
        tolerance_ = this->get_parameter("tolerance").as_double();

        // Create a subscription to the chosen topic (sensor_msgs::msg::Imu here) :contentReference[oaicite:3]{index=3}
        sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            topic_, 10,
            std::bind(&TimestampChecker::callback, this, std::placeholders::_1));

        // Initialize counters and last_stamp to zero time (ROS epoch)
        last_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        count_ = 0;
        violation_chrono_ = violation_repeat_ = 0;
    }

    ~TimestampChecker()
    {
        // On shutdown, print a clear summary :contentReference[oaicite:4]{index=4}
        RCLCPP_INFO(this->get_logger(), "===== Timestamp Check Summary =====");
        RCLCPP_INFO(this->get_logger(), "Total messages: %zu", count_);
        RCLCPP_INFO(this->get_logger(), "Chronological violations (stamp ≤ prev): %zu", violation_chrono_);
        RCLCPP_INFO(this->get_logger(), "Repeated stamps (stamp == prev): %zu", violation_repeat_);
        if (!dts_.empty())
        {
            auto [min_it, max_it] = std::minmax_element(dts_.begin(), dts_.end());
            double sum = std::accumulate(dts_.begin(), dts_.end(), 0.0);
            double avg = sum / dts_.size();
            RCLCPP_INFO(this->get_logger(), "Expected period: %f s", expected_period_);
            RCLCPP_INFO(this->get_logger(), "Observed dt — min: %f, max: %f, avg: %f",
                        *min_it, *max_it, avg);
        }
        RCLCPP_INFO(this->get_logger(), "==================================");
    }

private:
    void callback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        rclcpp::Time stamp = msg->header.stamp; // use ROS timestamp :contentReference[oaicite:5]{index=5}
        if (count_ > 0)
        {
            if (stamp <= last_stamp_)
            {
                if (stamp < last_stamp_)
                    violation_chrono_++;
                else
                    violation_repeat_++;
            }
            double dt = (stamp - last_stamp_).seconds(); // Duration.seconds() gives float seconds :contentReference[oaicite:6]{index=6}
            dts_.push_back(dt);
            // Warn if dt deviates from expected beyond tolerance
            if (std::fabs(dt - expected_period_) > tolerance_)
            {
                RCLCPP_WARN(this->get_logger(),
                            "Unexpected dt: %f s (expected %f ± %f)", dt, expected_period_, tolerance_);
            }
        }
        last_stamp_ = stamp;
        ++count_;
    }

    std::string topic_;
    double expected_period_, tolerance_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;
    rclcpp::Time last_stamp_;
    size_t count_, violation_chrono_, violation_repeat_;
    std::vector<double> dts_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv); // initialize ROS :contentReference[oaicite:7]{index=7}
    auto node = std::make_shared<TimestampChecker>();
    rclcpp::spin(node); // spin until shutdown :contentReference[oaicite:8]{index=8}
    rclcpp::shutdown();
    return 0;
}
