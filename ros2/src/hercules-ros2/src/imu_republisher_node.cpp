#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <string>
#include <vector>
#include <memory>

class ImuRepublisher : public rclcpp::Node
{
public:
    ImuRepublisher()
        : Node("imu_republisher")
    {
        // Declare and get parameter: list of IMU topics to process
        this->declare_parameter<std::vector<std::string>>(
            "imu_topics", std::vector<std::string>());

        auto imu_topics = this->get_parameter("imu_topics").as_string_array();

        if (imu_topics.empty())
        {
            RCLCPP_ERROR(this->get_logger(),
                         "Parameter 'imu_topics' is empty. Please provide at least one IMU topic.");
            throw std::runtime_error("No IMU topics specified");
        }

        // For each topic, create a subscription and a matching publisher
        for (const auto &topic : imu_topics)
        {
            // Extract robot name: assume topic of form /.../<RobotName>/imu/imu
            auto parts = splitString(topic, '/');
            if (parts.size() < 3u)
            {
                RCLCPP_WARN(this->get_logger(),
                            "Topic '%s' did not split into enough parts; skipping", topic.c_str());
                continue;
            }
            std::string robot = parts[2]; // index 0 is empty string before first '/'

            // Build the output topic name
            std::string out_topic = topic + "/with_frame";

            // Create publisher and subscription for this topic
            auto pub = this->create_publisher<sensor_msgs::msg::Imu>(out_topic, 10);
            auto sub = this->create_subscription<sensor_msgs::msg::Imu>(
                topic, 10,
                [pub, robot](sensor_msgs::msg::Imu::UniquePtr msg)
                {
                    // Set frame_id
                    msg->header.frame_id = robot + "/ground_truth/odom_local";
                    // Republish
                    pub->publish(std::move(msg));
                });

            // RCLCPP_INFO(this->get_logger(),
            //             "Republishing '%s' → '%s' with frame_id '%s'",
            //             topic.c_str(), out_topic.c_str(),
            //             (robot + "/ground_truth/odom_local").c_str());

            // Keep subscriptions/publishers alive
            subscriptions_.push_back(sub);
            publishers_.push_back(pub);
        }
    }

private:
    // Utility to split a string by delimiter
    static std::vector<std::string> splitString(
        const std::string &str, char delim)
    {
        std::vector<std::string> elems;
        std::string item;
        std::stringstream ss(str);
        while (std::getline(ss, item, delim))
        {
            elems.push_back(item);
        }
        return elems;
    }

    std::vector<rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr> subscriptions_;
    std::vector<rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr> publishers_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ImuRepublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
