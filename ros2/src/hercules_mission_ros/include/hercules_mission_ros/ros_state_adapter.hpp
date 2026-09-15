#pragma once

#include <nav_msgs/msg/odometry.hpp>
#include <hercules_interfaces/msg/ground_truth_state.hpp>

#include "hercules_mission_ros/state_adapter.hpp"

namespace hercules_mission_ros {

WrapperOdometryData wrapperDataFromMessage(
    const nav_msgs::msg::Odometry& message);
hercules_interfaces::msg::GroundTruthState groundTruthMessage(
    const CanonicalState& state);

}  // namespace hercules_mission_ros
