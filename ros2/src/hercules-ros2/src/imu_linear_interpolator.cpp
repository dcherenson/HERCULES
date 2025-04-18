#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <deque>

class IMUInterpolator : public rclcpp::Node
{
public:
    IMUInterpolator()
        : Node("imu_interpolator"), target_frequency_(1000.0)
    {
        using std::placeholders::_1;

        this->declare_parameter("input_topic", "/hercules_node/Drone1/imu/imu");
        this->declare_parameter("output_topic", "/hercules_node/Drone1/imu/imu_interpolated");
        this->get_parameter("input_topic", input_topic_);
        this->get_parameter("output_topic", output_topic_);

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            input_topic_, 100,
            std::bind(&IMUInterpolator::imu_callback, this, _1));

        imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>(output_topic_, 100);

        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / target_frequency_),
            std::bind(&IMUInterpolator::timer_callback, this));
    }

private:
    std::string input_topic_, output_topic_;
    double target_frequency_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer_;
    const size_t max_buffer_size_ = 200;

    void imu_callback(sensor_msgs::msg::Imu::SharedPtr msg)
    {
        imu_buffer_.push_back(msg);
        if (imu_buffer_.size() > max_buffer_size_)
        {
            imu_buffer_.pop_front();
        }
    }

    void timer_callback()
    {
        if (imu_buffer_.size() < 2)
            return;

        rclcpp::Time now = this->now();

        // Find the two IMU messages surrounding "now"
        for (size_t i = 1; i < imu_buffer_.size(); ++i)
        {
            rclcpp::Time t0 = imu_buffer_[i - 1]->header.stamp;
            rclcpp::Time t1 = imu_buffer_[i]->header.stamp;

            if (t0 <= now && now <= t1)
            {
                double alpha = (now - t0).seconds() / (t1 - t0).seconds();

                auto interp_msg = std::make_shared<sensor_msgs::msg::Imu>();

                // ✅ True interpolated timestamp
                rclcpp::Duration dt = rclcpp::Duration::from_seconds(alpha * (t1 - t0).seconds());
                interp_msg->header.stamp = t0 + dt;
                interp_msg->header.frame_id = imu_buffer_[i]->header.frame_id;

                // Interpolate values
                interpolate_vector3(imu_buffer_[i - 1]->linear_acceleration,
                                    imu_buffer_[i]->linear_acceleration,
                                    alpha,
                                    interp_msg->linear_acceleration);

                interpolate_vector3(imu_buffer_[i - 1]->angular_velocity,
                                    imu_buffer_[i]->angular_velocity,
                                    alpha,
                                    interp_msg->angular_velocity);

                interpolate_quaternion(imu_buffer_[i - 1]->orientation,
                                       imu_buffer_[i]->orientation,
                                       alpha,
                                       interp_msg->orientation);

                imu_pub_->publish(*interp_msg);
                return;
            }
        }
    }

    void interpolate_vector3(const geometry_msgs::msg::Vector3 &v0,
                             const geometry_msgs::msg::Vector3 &v1,
                             double alpha,
                             geometry_msgs::msg::Vector3 &out)
    {
        out.x = v0.x + alpha * (v1.x - v0.x);
        out.y = v0.y + alpha * (v1.y - v0.y);
        out.z = v0.z + alpha * (v1.z - v0.z);
    }

    void interpolate_quaternion(const geometry_msgs::msg::Quaternion &q0,
                                const geometry_msgs::msg::Quaternion &q1,
                                double alpha,
                                geometry_msgs::msg::Quaternion &out)
    {
        // Simple LERP with normalization
        out.w = q0.w + alpha * (q1.w - q0.w);
        out.x = q0.x + alpha * (q1.x - q0.x);
        out.y = q0.y + alpha * (q1.y - q0.y);
        out.z = q0.z + alpha * (q1.z - q0.z);

        double norm = std::sqrt(out.w * out.w + out.x * out.x +
                                out.y * out.y + out.z * out.z);
        out.w /= norm;
        out.x /= norm;
        out.y /= norm;
        out.z /= norm;
    }
};
