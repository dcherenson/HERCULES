#include "hercules_mission_ros/ros_state_adapter.hpp"

namespace hercules_mission_ros {

WrapperOdometryData wrapperDataFromMessage(
    const nav_msgs::msg::Odometry& message) {
  WrapperOdometryData result;
  result.stamp_ns = static_cast<int64_t>(message.header.stamp.sec) *
                        1000000000LL +
                    static_cast<int64_t>(message.header.stamp.nanosec);
  const auto& position = message.pose.pose.position;
  const auto& orientation = message.pose.pose.orientation;
  const auto& velocity = message.twist.twist.linear;
  const auto& angular_velocity = message.twist.twist.angular;
  result.position = {position.x, position.y, position.z};
  result.orientation = Eigen::Quaterniond(
      orientation.w, orientation.x, orientation.y, orientation.z);
  result.velocity = {velocity.x, velocity.y, velocity.z};
  result.angular_velocity = {
      angular_velocity.x, angular_velocity.y, angular_velocity.z};
  return result;
}

hercules_interfaces::msg::GroundTruthState groundTruthMessage(
    const CanonicalState& state) {
  hercules_interfaces::msg::GroundTruthState message;
  message.header.stamp.sec = static_cast<int32_t>(state.stamp_ns / 1000000000LL);
  message.header.stamp.nanosec = static_cast<uint32_t>(
      state.stamp_ns % 1000000000LL);
  if (state.valid) message.header.frame_id = kCanonicalFrame;
  message.agent_id = state.agent_id;
  switch (state.vehicle_type) {
    case hercules_mission_core::VehicleType::kDrone:
      message.vehicle_type = "drone";
      break;
    case hercules_mission_core::VehicleType::kUgv:
      message.vehicle_type = "ugv";
      break;
    case hercules_mission_core::VehicleType::kTargetUgv:
      message.vehicle_type = "target_ugv";
      break;
  }
  message.position = {
      state.position.x(), state.position.y(), state.position.z()};
  message.velocity = {
      state.velocity.x(), state.velocity.y(), state.velocity.z()};
  message.orientation = {
      state.orientation.w(), state.orientation.x(), state.orientation.y(),
      state.orientation.z()};
  message.yaw = state.yaw;
  message.yaw_rate = state.yaw_rate;
  message.valid = state.valid;
  return message;
}

}  // namespace hercules_mission_ros
