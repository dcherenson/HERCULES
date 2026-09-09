#include "hercules_mission_ros/mission_actuation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hercules_mission_ros {
namespace {
double clip(double value, double low, double high) {
  return std::max(low, std::min(high, value));
}
}  // namespace

Eigen::Vector3d uavVelocityCommand(const Eigen::Vector3d& measured_velocity,
                                   const Eigen::Vector3d& nominal_acceleration,
                                   double dt, double component_limit) {
  if (dt <= 0.0 || component_limit <= 0.0) {
    throw std::invalid_argument("UAV adapter limits must be positive");
  }
  Eigen::Vector3d command = measured_velocity + dt * nominal_acceleration;
  for (int axis = 0; axis < 3; ++axis) {
    command[axis] = clip(command[axis], -component_limit, component_limit);
  }
  return command;
}

CarCommand ugvCarCommand(double desired_speed, double desired_yaw_rate,
                         double measured_speed, double max_yaw_rate,
                         bool target_vehicle) {
  if (max_yaw_rate <= 0.0) {
    throw std::invalid_argument("UGV yaw-rate limit must be positive");
  }
  CarCommand result;
  const double speed = std::max(0.0, desired_speed);
  const double measured = std::max(0.0, measured_speed);
  if (speed < 0.05) return stoppedCarCommand();
  const double speed_error = speed - measured;
  result.throttle = clip(0.02 * speed + 0.10 * std::max(0.0, speed_error),
                         0.0, 0.08);
  if (target_vehicle) result.throttle = std::max(0.02, result.throttle);
  result.brake = 0.0;
  if (measured > speed + 0.10) {
    result.brake = clip((measured - speed - 0.10) / 0.50, 0.0, 1.0);
    result.throttle = 0.0;
  }
  result.steering = clip(desired_yaw_rate / max_yaw_rate, -1.0, 1.0);
  return result;
}

CarCommand stoppedCarCommand() {
  CarCommand result;
  result.handbrake = true;
  return result;
}

}  // namespace hercules_mission_ros
