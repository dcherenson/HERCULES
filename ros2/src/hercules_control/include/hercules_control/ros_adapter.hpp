#pragma once
#include "hercules_control/core.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <airsim_interfaces/msg/vel_cmd.hpp>
#include <airsim_interfaces/msg/car_controls.hpp>

namespace hercules_control {
State from_odometry(const nav_msgs::msg::Odometry &message);
airsim_interfaces::msg::VelCmd to_drone_message(const VelocityCommand &command);
airsim_interfaces::msg::CarControls to_ground_message(const GroundCommand &command);
}
