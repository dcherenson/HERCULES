#include <rclcpp/rclcpp.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include "rclcpp/rclcpp.hpp"

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

#include <vector>
#include <algorithm>
#include <cmath>

class TimestampChecker : public rclcpp::Node
{
public:
    TimestampChecker()
        : Node("bag_timestamp_checker")
    {
        // parameters: topic, expected period, tolerance, message type
        declare_parameter<std::string>("topic", "/hercules_node/Drone1/imu/imu");
        declare_parameter<double>("expected_period", 0.05);
        declare_parameter<double>("tolerance", 1e-6);
        declare_parameter<std::string>("message_type", "sensor_msgs/msg/Imu");

        topic_ = get_parameter("topic").as_string();
        expected_period_ = get_parameter("expected_period").as_double();
        tolerance_ = get_parameter("tolerance").as_double();
        message_type_ = get_parameter("message_type").as_string();

        // choose subscription based on message_type_
        if (message_type_ == "sensor_msgs/msg/Imu")
        {
            sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
                topic_, 10,
                std::bind(&TimestampChecker::imu_callback, this, std::placeholders::_1));
        }
        else if (message_type_ == "nav_msgs/msg/Odometry")
        {
            sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
                topic_, 10,
                std::bind(&TimestampChecker::odom_callback, this, std::placeholders::_1));
        }
        else
        {
            RCLCPP_ERROR(get_logger(),
                         "Unsupported message_type '%s'. Valid: sensor_msgs/msg/Imu, nav_msgs/msg/Odometry",
                         message_type_.c_str());
            rclcpp::shutdown();
            return;
        }

        last_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        count_ = violation_chrono_ = violation_repeat_ = 0;
    }

    ~TimestampChecker() override
    {
        RCLCPP_INFO(get_logger(), "===== Timestamp Check Summary =====");
        RCLCPP_INFO(get_logger(), "Total messages: %zu", count_);
        RCLCPP_INFO(get_logger(), "Chrono violations: %zu", violation_chrono_);
        RCLCPP_INFO(get_logger(), "Repeated stamps: %zu", violation_repeat_);
        if (!dts_.empty())
        {
            auto [min_it, max_it] = std::minmax_element(dts_.begin(), dts_.end());
            double sum = std::accumulate(dts_.begin(), dts_.end(), 0.0);
            RCLCPP_INFO(get_logger(), "Expected period: %f s", expected_period_);
            RCLCPP_INFO(get_logger(), "Observed dt — min: %f, max: %f, avg: %f",
                        *min_it, *max_it, sum / dts_.size());
        }
        RCLCPP_INFO(get_logger(), "==================================");
    }

private:
    // common processing
    void process_stamp(const rclcpp::Time &stamp)
    {
        if (count_ > 0)
        {
            if (stamp <= last_stamp_)
            {
                if (stamp < last_stamp_)
                    violation_chrono_++;
                else
                    violation_repeat_++;
            }
            double dt = (stamp - last_stamp_).seconds();
            dts_.push_back(dt);
            if (std::fabs(dt - expected_period_) > tolerance_)
            {
                RCLCPP_WARN(get_logger(),
                            "Unexpected dt: %f s (expected %f ± %f)", dt, expected_period_, tolerance_);
            }
        }
        last_stamp_ = stamp;
        ++count_;
    }

    // Imu callback
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        process_stamp(msg->header.stamp);
    }

    // Odometry callback
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        process_stamp(msg->header.stamp);
    }

    // params & state
    std::string topic_, message_type_;
    double expected_period_, tolerance_;
    size_t count_, violation_chrono_, violation_repeat_;
    rclcpp::Time last_stamp_;
    std::vector<double> dts_;

    // subscriptions
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TimestampChecker>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}