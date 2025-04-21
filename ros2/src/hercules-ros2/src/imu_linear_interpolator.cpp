#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <mutex>
#include <Eigen/Geometry>

class ImuInterpolator : public rclcpp::Node
{
public:
    ImuInterpolator()
        : Node("imu_interpolator")
    {
        // declare & read params
        this->declare_parameter<std::string>("input_topic", "/hercules_node/Drone1/imu/imu");
        this->declare_parameter<std::string>("output_topic", "/hercules_node/Drone1/imu/imu_interpolated");
        this->declare_parameter<double>("frequency", 500.0);

        input_topic_ = this->get_parameter("input_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        frequency_ = this->get_parameter("frequency").as_double();
        dt_sec_ = 1.0 / frequency_;

        sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            input_topic_, 10,
            std::bind(&ImuInterpolator::imuCallback, this, std::placeholders::_1));
        pub_ = this->create_publisher<sensor_msgs::msg::Imu>(output_topic_, 10);
    }

private:
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (prev_)
        {
            interpolateAndPublish(prev_, msg);
        }
        // always forward the raw message
        pub_->publish(*msg);
        prev_ = msg;
    }

    void interpolateAndPublish(
        const sensor_msgs::msg::Imu::SharedPtr &m1,
        const sensor_msgs::msg::Imu::SharedPtr &m2)
    {
        rclcpp::Time t1 = m1->header.stamp;
        rclcpp::Time t2 = m2->header.stamp;
        double segment = (t2 - t1).seconds();
        int count = static_cast<int>(std::floor(segment / dt_sec_)) - 1;
        if (count < 1)
        {
            return;
        }

        for (int i = 1; i <= count; ++i)
        {
            double α = (dt_sec_ * i) / segment;
            rclcpp::Time ti = t1 + rclcpp::Duration::from_seconds(dt_sec_ * i);

            sensor_msgs::msg::Imu out;
            out.header = m1->header;
            out.header.stamp = ti;

            // orientation slerp
            Eigen::Quaterniond q1{m1->orientation.w,
                                  m1->orientation.x,
                                  m1->orientation.y,
                                  m1->orientation.z};
            Eigen::Quaterniond q2{m2->orientation.w,
                                  m2->orientation.x,
                                  m2->orientation.y,
                                  m2->orientation.z};
            Eigen::Quaterniond qi = q1.slerp(α, q2);
            out.orientation.x = qi.x();
            out.orientation.y = qi.y();
            out.orientation.z = qi.z();
            out.orientation.w = qi.w();
            out.orientation_covariance = m1->orientation_covariance;

            // angular velocity
            out.angular_velocity.x = m1->angular_velocity.x +
                                     α * (m2->angular_velocity.x - m1->angular_velocity.x);
            out.angular_velocity.y = m1->angular_velocity.y +
                                     α * (m2->angular_velocity.y - m1->angular_velocity.y);
            out.angular_velocity.z = m1->angular_velocity.z +
                                     α * (m2->angular_velocity.z - m1->angular_velocity.z);
            out.angular_velocity_covariance = m1->angular_velocity_covariance;

            // linear acceleration
            out.linear_acceleration.x = m1->linear_acceleration.x +
                                        α * (m2->linear_acceleration.x - m1->linear_acceleration.x);
            out.linear_acceleration.y = m1->linear_acceleration.y +
                                        α * (m2->linear_acceleration.y - m1->linear_acceleration.y);
            out.linear_acceleration.z = m1->linear_acceleration.z +
                                        α * (m2->linear_acceleration.z - m1->linear_acceleration.z);
            out.linear_acceleration_covariance = m1->linear_acceleration_covariance;

            pub_->publish(out);
        }
    }

    // parameters & state
    std::string input_topic_, output_topic_;
    double frequency_, dt_sec_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_;
    sensor_msgs::msg::Imu::SharedPtr prev_;
    std::mutex mtx_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImuInterpolator>());
    rclcpp::shutdown();
    return 0;
}
