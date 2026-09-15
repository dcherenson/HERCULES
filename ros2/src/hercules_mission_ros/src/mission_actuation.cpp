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

Eigen::Vector3d pythonCbfUavVelocityCommand(
    const Eigen::Vector3d& measured_velocity,
    const Eigen::Vector3d& nominal_acceleration, double dt,
    double velocity_limit, double ceiling_z, double position_z) {
  Eigen::Vector3d result = uavVelocityCommand(measured_velocity, nominal_acceleration,
                                               dt, velocity_limit);
  result.z() = std::max(result.z(), (ceiling_z - position_z) / dt);
  result.z() = clip(result.z(), -velocity_limit, velocity_limit);
  return result;
}

CarCommand pythonCbfUgvCarCommand(double desired_speed, double desired_yaw_rate,
                                  double measured_speed, double cbf_yaw_rate_limit,
                                  bool target_vehicle, double speed_limit) {
  if (speed_limit <= 0.0) {
    throw std::invalid_argument("UGV speed limit must be positive");
  }
  double speed = std::max(0.0, desired_speed);
  double yaw_rate = desired_yaw_rate;
  if (speed < 0.05 && std::abs(yaw_rate) > 0.05)
    speed = std::min(0.3, speed_limit);
  if (speed < 0.05) return stoppedCarCommand();
  return ugvCarCommand(speed, yaw_rate, measured_speed, cbf_yaw_rate_limit, target_vehicle);
}

}  // namespace hercules_mission_ros
