#include "hercules_control/ros_adapter.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hercules_control {
State from_odometry(const nav_msgs::msg::Odometry &m) {
  State s;
  s.stamp_ns = int64_t(m.header.stamp.sec) * 1000000000LL + m.header.stamp.nanosec;
  const auto &p = m.pose.pose.position;
  const auto &q = m.pose.pose.orientation;
  const auto &v = m.twist.twist.linear;
  const auto &w = m.twist.twist.angular;
  s.position = {p.x, p.y, p.z};
  s.orientation = Eigen::Quaterniond(q.w, q.x, q.y, q.z);
  s.velocity = {v.x, v.y, v.z}; s.angular_velocity = {w.x, w.y, w.z};
  return s;
}
airsim_interfaces::msg::VelCmd to_drone_message(const VelocityCommand &c) {
  if (!c.velocity.allFinite() || !std::isfinite(c.yaw_rate)) throw std::invalid_argument("Nonfinite velocity command");
  airsim_interfaces::msg::VelCmd m;
  // Existing world-velocity wrapper forwards native NED components unchanged.
  m.twist.linear.x = std::clamp(c.velocity.x(), -0.2, 0.2);
  m.twist.linear.y = -std::clamp(c.velocity.y(), -0.2, 0.2);
  m.twist.linear.z = -std::clamp(c.velocity.z(), -0.2, 0.2);
  m.twist.angular.z = -std::clamp(c.yaw_rate, -0.2, 0.2);
  return m;
}
airsim_interfaces::msg::CarControls to_ground_message(const GroundCommand &c) {
  if (!std::isfinite(c.throttle) || !std::isfinite(c.steering) || !std::isfinite(c.brake))
    throw std::invalid_argument("Nonfinite ground command");
  airsim_interfaces::msg::CarControls m;
  m.throttle = std::clamp(c.throttle, 0.0, 0.15);
  m.brake = std::clamp(c.brake, 0.0, 1.0);
  m.steering = std::clamp(c.steering, -0.2, 0.2);
  m.manual = false; m.manual_gear = 0; m.gear_immediate = false; m.handbrake = false;
  return m;
}
}
