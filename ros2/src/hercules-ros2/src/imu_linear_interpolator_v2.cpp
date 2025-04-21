// Copyright (c) 2025
// Author: Claude
// Description: ROS 2 node for linear interpolation of IMU data to achieve higher frequencies

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <memory>
#include <deque>
#include <algorithm>
#include <chrono>
#include <functional>

using namespace std::chrono_literals;

class IMUInterpolatorNode : public rclcpp::Node
{
public:
    explicit IMUInterpolatorNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
        : Node("imu_interpolator", options)
    {
        // Declare parameters
        this->declare_parameter<double>("target_frequency", 500.0); // Default: 500 Hz
        this->declare_parameter<int>("buffer_size", 10);            // Size of the message buffer
        this->declare_parameter<std::string>("input_topic", "/hercules_node/Drone1/imu/imu");
        this->declare_parameter<std::string>("output_topic", "/hercules_node/Drone1/imu/imu_interpolated");

        // Get parameters
        target_frequency_ = this->get_parameter("target_frequency").as_double();
        buffer_size_ = this->get_parameter("buffer_size").as_int();
        input_topic_ = this->get_parameter("input_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();

        // Calculate period between messages in nanoseconds
        period_ns_ = static_cast<int64_t>(1e9 / target_frequency_);

        // Initialize the buffer
        imu_buffer_.resize(0);

        // Create publisher and subscriber
        publisher_ = this->create_publisher<sensor_msgs::msg::Imu>(
            output_topic_, rclcpp::QoS(rclcpp::KeepLast(100)).reliable());

        subscriber_ = this->create_subscription<sensor_msgs::msg::Imu>(
            input_topic_, rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(),
            std::bind(&IMUInterpolatorNode::imu_callback, this, std::placeholders::_1));

        // Create high-frequency timer for publishing at target rate
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / target_frequency_),
            std::bind(&IMUInterpolatorNode::timer_callback, this));

        // Initialize timestamps
        next_publish_time_ns_ = 0;
        last_source_time_ns_ = 0;
        has_initialized_ = false;

        RCLCPP_INFO(this->get_logger(), "IMU Interpolator started");
        RCLCPP_INFO(this->get_logger(), "Input topic: %s", input_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Output topic: %s", output_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Target frequency: %.1f Hz", target_frequency_);
    }

private:
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Add message to buffer
        imu_buffer_.push_back(*msg);

        // Sort buffer by timestamp
        std::sort(imu_buffer_.begin(), imu_buffer_.end(),
                  [](const sensor_msgs::msg::Imu &a, const sensor_msgs::msg::Imu &b)
                  {
                      int64_t a_ns = a.header.stamp.sec * 1000000000LL + a.header.stamp.nanosec;
                      int64_t b_ns = b.header.stamp.sec * 1000000000LL + b.header.stamp.nanosec;
                      return a_ns < b_ns;
                  });

        // Keep buffer size limited
        while (imu_buffer_.size() > static_cast<size_t>(buffer_size_))
        {
            imu_buffer_.pop_front();
        }

        // Convert new message timestamp to nanoseconds
        int64_t msg_time_ns = msg->header.stamp.sec * 1000000000LL + msg->header.stamp.nanosec;

        // Update last received timestamp if this is a newer message
        if (msg_time_ns > last_source_time_ns_)
        {
            last_source_time_ns_ = msg_time_ns;
        }

        // Initialize next_publish_time_ if not set yet
        if (!has_initialized_ && !imu_buffer_.empty())
        {
            // Start publishing just after the first message timestamp
            int64_t first_msg_ns = imu_buffer_.front().header.stamp.sec * 1000000000LL +
                                   imu_buffer_.front().header.stamp.nanosec;
            next_publish_time_ns_ = first_msg_ns + period_ns_;
            has_initialized_ = true;
        }

        RCLCPP_DEBUG(this->get_logger(), "Received IMU message with timestamp %d.%09d",
                     msg->header.stamp.sec, msg->header.stamp.nanosec);
    }

    void timer_callback()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Need at least 2 messages to interpolate and must be initialized
        if (imu_buffer_.size() < 2 || !has_initialized_)
        {
            return;
        }

        // If we've already published up to the most recent message time, don't publish more
        // This prevents extrapolation beyond our actual data
        if (next_publish_time_ns_ >= last_source_time_ns_)
        {
            return;
        }

        // Find the two messages that bracket our next publish time
        size_t i = 0;
        while (i < imu_buffer_.size() - 1)
        {
            int64_t t1_ns = imu_buffer_[i].header.stamp.sec * 1000000000LL +
                            imu_buffer_[i].header.stamp.nanosec;
            int64_t t2_ns = imu_buffer_[i + 1].header.stamp.sec * 1000000000LL +
                            imu_buffer_[i + 1].header.stamp.nanosec;

            if (t1_ns <= next_publish_time_ns_ && next_publish_time_ns_ <= t2_ns)
            {
                // Found our bracket
                break;
            }
            i++;
        }

        // If we couldn't find a bracket, we can't interpolate
        if (i >= imu_buffer_.size() - 1)
        {
            return;
        }

        // Get the two adjacent messages
        const sensor_msgs::msg::Imu &imu1 = imu_buffer_[i];
        const sensor_msgs::msg::Imu &imu2 = imu_buffer_[i + 1];

        // Convert timestamps to nanoseconds for precise interpolation
        int64_t t1_ns = imu1.header.stamp.sec * 1000000000LL + imu1.header.stamp.nanosec;
        int64_t t2_ns = imu2.header.stamp.sec * 1000000000LL + imu2.header.stamp.nanosec;

        // Check if time difference is too small to avoid division by zero
        if (t2_ns - t1_ns < 1)
        {
            return;
        }

        // Calculate interpolation factor (0 to 1)
        double alpha = static_cast<double>(next_publish_time_ns_ - t1_ns) / static_cast<double>(t2_ns - t1_ns);

        // Clamp alpha between 0 and 1 to avoid extrapolation
        alpha = std::max(0.0, std::min(1.0, alpha));

        // Create and publish interpolated message
        publish_interpolated_message(imu1, imu2, alpha, next_publish_time_ns_);

        // Increment next publish time for the next callback
        next_publish_time_ns_ += period_ns_;
    }

    void publish_interpolated_message(const sensor_msgs::msg::Imu &imu1,
                                      const sensor_msgs::msg::Imu &imu2,
                                      double alpha,
                                      int64_t timestamp_ns)
    {
        // Create interpolated message
        sensor_msgs::msg::Imu interpolated_msg;

        // Set header with the interpolated timestamp
        interpolated_msg.header.stamp.sec = timestamp_ns / 1000000000LL;
        interpolated_msg.header.stamp.nanosec = timestamp_ns % 1000000000LL;
        interpolated_msg.header.frame_id = imu1.header.frame_id;

        // Linear interpolation of orientation quaternion
        interpolated_msg.orientation.w = (1.0 - alpha) * imu1.orientation.w + alpha * imu2.orientation.w;
        interpolated_msg.orientation.x = (1.0 - alpha) * imu1.orientation.x + alpha * imu2.orientation.x;
        interpolated_msg.orientation.y = (1.0 - alpha) * imu1.orientation.y + alpha * imu2.orientation.y;
        interpolated_msg.orientation.z = (1.0 - alpha) * imu1.orientation.z + alpha * imu2.orientation.z;

        // Normalize the quaternion
        double norm = std::sqrt(
            interpolated_msg.orientation.w * interpolated_msg.orientation.w +
            interpolated_msg.orientation.x * interpolated_msg.orientation.x +
            interpolated_msg.orientation.y * interpolated_msg.orientation.y +
            interpolated_msg.orientation.z * interpolated_msg.orientation.z);

        if (norm > 1e-10)
        {
            interpolated_msg.orientation.w /= norm;
            interpolated_msg.orientation.x /= norm;
            interpolated_msg.orientation.y /= norm;
            interpolated_msg.orientation.z /= norm;
        }

        // Linear interpolation of angular velocity
        interpolated_msg.angular_velocity.x = (1.0 - alpha) * imu1.angular_velocity.x + alpha * imu2.angular_velocity.x;
        interpolated_msg.angular_velocity.y = (1.0 - alpha) * imu1.angular_velocity.y + alpha * imu2.angular_velocity.y;
        interpolated_msg.angular_velocity.z = (1.0 - alpha) * imu1.angular_velocity.z + alpha * imu2.angular_velocity.z;

        // Linear interpolation of linear acceleration
        interpolated_msg.linear_acceleration.x = (1.0 - alpha) * imu1.linear_acceleration.x + alpha * imu2.linear_acceleration.x;
        interpolated_msg.linear_acceleration.y = (1.0 - alpha) * imu1.linear_acceleration.y + alpha * imu2.linear_acceleration.y;
        interpolated_msg.linear_acceleration.z = (1.0 - alpha) * imu1.linear_acceleration.z + alpha * imu2.linear_acceleration.z;

        // Copy covariance matrices (or interpolate them if needed)
        for (size_t j = 0; j < 9; j++)
        {
            interpolated_msg.orientation_covariance[j] = (1.0 - alpha) * imu1.orientation_covariance[j] + alpha * imu2.orientation_covariance[j];
            interpolated_msg.angular_velocity_covariance[j] = (1.0 - alpha) * imu1.angular_velocity_covariance[j] + alpha * imu2.angular_velocity_covariance[j];
            interpolated_msg.linear_acceleration_covariance[j] = (1.0 - alpha) * imu1.linear_acceleration_covariance[j] + alpha * imu2.linear_acceleration_covariance[j];
        }

        // Publish the interpolated message
        publisher_->publish(interpolated_msg);

        RCLCPP_DEBUG(this->get_logger(), "Published interpolated IMU message with timestamp %d.%09d",
                     interpolated_msg.header.stamp.sec, interpolated_msg.header.stamp.nanosec);
    }

    // Member variables
    double target_frequency_;
    int buffer_size_;
    std::string input_topic_;
    std::string output_topic_;
    std::deque<sensor_msgs::msg::Imu> imu_buffer_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subscriber_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::mutex mutex_;
    int64_t period_ns_;            // Period between messages in nanoseconds
    int64_t next_publish_time_ns_; // Next publish time in nanoseconds
    int64_t last_source_time_ns_;  // Last source time in nanoseconds
    bool has_initialized_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<IMUInterpolatorNode>());
    rclcpp::shutdown();
    return 0;
}