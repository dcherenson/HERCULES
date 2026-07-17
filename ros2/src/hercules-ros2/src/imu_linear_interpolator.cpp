#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <mutex>
#include <Eigen/Geometry>
#include <chrono>

class ImuInterpolator : public rclcpp::Node
{
public:
    ImuInterpolator()
        : Node("imu_interpolator")
    {
        // parameters
        this->declare_parameter<std::string>("input_topic", "/hercules_node/Drone1/imu/imu");
        this->declare_parameter<std::string>("output_topic", "/hercules_node/Drone1/imu/imu_interpolated");
        this->declare_parameter<double>("frequency", 500.0);

        input_topic_ = this->get_parameter("input_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        frequency_ = this->get_parameter("frequency").as_double();
        dt_sec_ = 1.0 / frequency_;

        auto qos = rclcpp::SensorDataQoS().keep_last(1000);

        sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            input_topic_, qos,
            [this](sensor_msgs::msg::Imu::SharedPtr m)
            { imuCallback(m); });

        pub_ = this->create_publisher<sensor_msgs::msg::Imu>(
            output_topic_, qos);

        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(dt_sec_),
            std::bind(&ImuInterpolator::timerCallback, this));
    }

private:
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!prev_msg_)
        {
            prev_msg_ = msg;
            return;
        }
        // new segment
        next_msg_ = msg;
        prev_time_ = rclcpp::Time(prev_msg_->header.stamp);
        next_time_ = rclcpp::Time(next_msg_->header.stamp);
        next_pub_time_ = prev_time_;
        segment_ready_ = true;

        // publish raw
        pub_->publish(*msg);

        // shift window
        prev_msg_ = next_msg_;
    }

    void timerCallback()
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!segment_ready_)
            return;

        // advance
        next_pub_time_ = next_pub_time_ + rclcpp::Duration::from_seconds(dt_sec_);
        if (next_pub_time_ > next_time_)
        {
            segment_ready_ = false;
            return;
        }

        // fraction
        double total = (next_time_ - prev_time_).seconds();
        double alpha = (next_pub_time_ - prev_time_).seconds() / total;

        sensor_msgs::msg::Imu out;
        out.header.frame_id = prev_msg_->header.frame_id;
        out.header.stamp = next_pub_time_;

        // orientation SLERP
        Eigen::Quaterniond q1{
            prev_msg_->orientation.w,
            prev_msg_->orientation.x,
            prev_msg_->orientation.y,
            prev_msg_->orientation.z};
        Eigen::Quaterniond q2{
            next_msg_->orientation.w,
            next_msg_->orientation.x,
            next_msg_->orientation.y,
            next_msg_->orientation.z};
        Eigen::Quaterniond qi = q1.slerp(alpha, q2);
        out.orientation.x = qi.x();
        out.orientation.y = qi.y();
        out.orientation.z = qi.z();
        out.orientation.w = qi.w();
        out.orientation_covariance = prev_msg_->orientation_covariance;

        // angular velocity
        out.angular_velocity.x = prev_msg_->angular_velocity.x +
                                 alpha * (next_msg_->angular_velocity.x - prev_msg_->angular_velocity.x);
        out.angular_velocity.y = prev_msg_->angular_velocity.y +
                                 alpha * (next_msg_->angular_velocity.y - prev_msg_->angular_velocity.y);
        out.angular_velocity.z = prev_msg_->angular_velocity.z +
                                 alpha * (next_msg_->angular_velocity.z - prev_msg_->angular_velocity.z);
        out.angular_velocity_covariance = prev_msg_->angular_velocity_covariance;

        // linear acceleration
        out.linear_acceleration.x = prev_msg_->linear_acceleration.x +
                                    alpha * (next_msg_->linear_acceleration.x - prev_msg_->linear_acceleration.x);
        out.linear_acceleration.y = prev_msg_->linear_acceleration.y +
                                    alpha * (next_msg_->linear_acceleration.y - prev_msg_->linear_acceleration.y);
        out.linear_acceleration.z = prev_msg_->linear_acceleration.z +
                                    alpha * (next_msg_->linear_acceleration.z - prev_msg_->linear_acceleration.z);
        out.linear_acceleration_covariance = prev_msg_->linear_acceleration_covariance;

        pub_->publish(out);
    }

    // members
    std::string input_topic_, output_topic_;
    double frequency_, dt_sec_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    sensor_msgs::msg::Imu::SharedPtr prev_msg_, next_msg_;
    rclcpp::Time prev_time_, next_time_, next_pub_time_;
    bool segment_ready_{false};
    std::mutex mtx_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ImuInterpolator>();
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    rclcpp::shutdown();
    return 0;
}
