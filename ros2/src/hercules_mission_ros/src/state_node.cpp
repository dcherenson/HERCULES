#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <hercules_interfaces/msg/ground_truth_state.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

#include "hercules_mission_ros/ros_state_adapter.hpp"
#include "hercules_mission_ros/state_cache.hpp"

namespace hercules_mission_ros {
namespace {

using SteadyClock = std::chrono::steady_clock;

double steadySeconds() {
  return std::chrono::duration<double>(
      SteadyClock::now().time_since_epoch()).count();
}

hercules_mission_core::VehicleType parseVehicleType(const std::string& value) {
  if (value == "drone") return hercules_mission_core::VehicleType::kDrone;
  if (value == "ugv") return hercules_mission_core::VehicleType::kUgv;
  if (value == "target_ugv") {
    return hercules_mission_core::VehicleType::kTargetUgv;
  }
  throw std::invalid_argument("unsupported vehicle type: " + value);
}

struct AgentChannel {
  std::string id;
  hercules_mission_core::VehicleType type;
  Eigen::Vector3d origin;
  StateCache cache;
  rclcpp::Publisher<hercules_interfaces::msg::GroundTruthState>::SharedPtr publisher;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription;

  AgentChannel(std::string agent_id,
               hercules_mission_core::VehicleType vehicle_type,
               const Eigen::Vector3d& vehicle_origin,
               double freshness_timeout)
      : id(std::move(agent_id)),
        type(vehicle_type),
        origin(vehicle_origin),
        cache(freshness_timeout) {}
};

}  // namespace

class MissionStateNode : public rclcpp::Node {
 public:
  MissionStateNode() : Node("mission_state_adapter") {
    const auto agent_ids = declare_parameter<std::vector<std::string>>(
        "agent_ids", std::vector<std::string>{});
    const auto vehicle_types = declare_parameter<std::vector<std::string>>(
        "vehicle_types", std::vector<std::string>{});
    const auto odom_topics = declare_parameter<std::vector<std::string>>(
        "odom_topics", std::vector<std::string>{});
    const auto origins = declare_parameter<std::vector<double>>(
        "vehicle_origins_ned", std::vector<double>{});
    const std::string topic_prefix = declare_parameter<std::string>(
        "state_topic_prefix", "/hercules_mission/ground_truth");
    const double freshness_timeout = declare_parameter<double>(
        "freshness_timeout_sec", 0.5);

    if (agent_ids.empty() || vehicle_types.size() != agent_ids.size() ||
        odom_topics.size() != agent_ids.size() ||
        origins.size() != 3 * agent_ids.size()) {
      throw std::invalid_argument(
          "agent_ids, vehicle_types, odom_topics, and three origin values per "
          "agent are required");
    }

    channels_.reserve(agent_ids.size());
    for (std::size_t index = 0; index < agent_ids.size(); ++index) {
      const Eigen::Vector3d origin(
          origins[3 * index], origins[3 * index + 1], origins[3 * index + 2]);
      channels_.push_back(std::make_unique<AgentChannel>(
          agent_ids[index], parseVehicleType(vehicle_types[index]), origin,
          freshness_timeout));
      AgentChannel& channel = *channels_.back();
      channel.publisher = create_publisher<
          hercules_interfaces::msg::GroundTruthState>(
          topic_prefix + "/" + channel.id, rclcpp::QoS(10));
      channel.subscription = create_subscription<nav_msgs::msg::Odometry>(
          odom_topics[index], rclcpp::QoS(10),
          [this, index](nav_msgs::msg::Odometry::ConstSharedPtr message) {
            receive(index, *message);
          });
      RCLCPP_INFO(get_logger(),
                  "%s: %s -> %s/%s, origin NED [%.3f, %.3f, %.3f]",
                  channel.id.c_str(), odom_topics[index].c_str(),
                  topic_prefix.c_str(), channel.id.c_str(),
                  origin.x(), origin.y(), origin.z());
    }

    freshness_timer_ = create_wall_timer(
        std::chrono::milliseconds(250), [this]() { checkFreshness(); });
  }

 private:
  void receive(std::size_t index, const nav_msgs::msg::Odometry& message) {
    AgentChannel& channel = *channels_.at(index);
    const CanonicalState state = fromWrapperOdometry(
        wrapperDataFromMessage(message), channel.origin, channel.id,
        channel.type);
    const ReceiveStatus status = channel.cache.receive(state, steadySeconds());
    if (status == ReceiveStatus::kAdvanced) {
      channel.publisher->publish(groundTruthMessage(state));
      return;
    }
    if (status == ReceiveStatus::kRejected) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Rejected %s state: %s or regressing timestamp",
          channel.id.c_str(), conversionErrorName(state.error));
    }
  }

  void checkFreshness() {
    const double now = steadySeconds();
    for (const auto& channel : channels_) {
      if (channel->cache.messages() > 0 && !channel->cache.fresh(now)) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "%s canonical state is stale or has not advanced twice",
            channel->id.c_str());
      }
    }
  }

  std::vector<std::unique_ptr<AgentChannel>> channels_;
  rclcpp::TimerBase::SharedPtr freshness_timer_;
};

}  // namespace hercules_mission_ros

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int result = 0;
  try {
    rclcpp::spin(std::make_shared<hercules_mission_ros::MissionStateNode>());
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    result = 1;
  }
  rclcpp::shutdown();
  return result;
}
