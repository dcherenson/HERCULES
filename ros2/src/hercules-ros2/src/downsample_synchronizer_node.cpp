#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class DownSyncNode : public rclcpp::Node
{
public:
    DownSyncNode()
        : Node("downsync_node"),
          imu_period_(0, 0),
          cam_period_(0, 0)
    {
        // Declare parameters
        declare_parameter<std::string>("imu_input_topic", "/imu/data_raw");
        declare_parameter<std::string>("imu_output_topic", "/imu/downsampled");
        declare_parameter<double>("imu_rate_hz", 2000.0);

        declare_parameter<std::string>("cam_input_topic", "/camera/image_raw");
        declare_parameter<std::string>("cam_output_topic", "/camera/image_gray");
        declare_parameter<double>("cam_rate_hz", 20.0);

        // Read parameters
        imu_in_topic_ = get_parameter("imu_input_topic").as_string();
        imu_out_topic_ = get_parameter("imu_output_topic").as_string();
        double imu_hz = get_parameter("imu_rate_hz").as_double();

        cam_in_topic_ = get_parameter("cam_input_topic").as_string();
        cam_out_topic_ = get_parameter("cam_output_topic").as_string();
        double cam_hz = get_parameter("cam_rate_hz").as_double();

        // Compute periods
        imu_period_ = rclcpp::Duration::from_seconds(1.0 / imu_hz);
        cam_period_ = rclcpp::Duration::from_seconds(1.0 / cam_hz);

        last_imu_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        last_cam_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

        // Publishers
        imu_pub_ = create_publisher<sensor_msgs::msg::Imu>(imu_out_topic_, 10);
        cam_pub_ = create_publisher<sensor_msgs::msg::Image>(cam_out_topic_, 10);

        // Subscribers
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            imu_in_topic_, rclcpp::SensorDataQoS(),
            std::bind(&DownSyncNode::imu_callback, this, std::placeholders::_1));

        cam_sub_ = create_subscription<sensor_msgs::msg::Image>(
            cam_in_topic_, rclcpp::SensorDataQoS(),
            std::bind(&DownSyncNode::cam_callback, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(),
                    "DownSyncNode: IMU %s→%s @ %.1f Hz, Camera %s→%s @ %.1f Hz",
                    imu_in_topic_.c_str(), imu_out_topic_.c_str(), imu_hz,
                    cam_in_topic_.c_str(), cam_out_topic_.c_str(), cam_hz);
    }

private:
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        // Wrap header timestamp
        rclcpp::Time now(msg->header.stamp);
        // Downsample: publish only if enough time has elapsed
        if ((now - last_imu_stamp_) >= imu_period_)
        {
            imu_pub_->publish(*msg);
            last_imu_stamp_ = now;
        }
    }

    void cam_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Wrap header timestamp
        rclcpp::Time now(msg->header.stamp);
        // Downsample: skip if called too soon
        if ((now - last_cam_stamp_) < cam_period_)
        {
            return;
        }
        last_cam_stamp_ = now;

        // Convert to grayscale
        cv_bridge::CvImageConstPtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        cv::Mat gray;
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

        // Publish grayscale image, preserving header
        auto out_msg = cv_bridge::CvImage(
                           msg->header,
                           sensor_msgs::image_encodings::MONO8,
                           gray)
                           .toImageMsg();

        cam_pub_->publish(*out_msg);
    }

    // Parameters
    std::string imu_in_topic_, imu_out_topic_;
    std::string cam_in_topic_, cam_out_topic_;
    rclcpp::Duration imu_period_, cam_period_;

    // State
    rclcpp::Time last_imu_stamp_, last_cam_stamp_;

    // ROS pubs & subs
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr cam_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam_sub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DownSyncNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
