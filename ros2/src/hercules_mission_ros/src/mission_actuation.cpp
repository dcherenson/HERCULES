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

CarCommand paperUgvCarCommand(double speed, double yaw_rate, double measured_speed,
                             double yaw_rate_limit) {
  if (std::abs(speed) < 0.03 && std::abs(yaw_rate) < 0.03) return stoppedCarCommand();
  CarCommand command;
  // The old nominal target adapter capped throttle at 0.08, which cannot
  // track a short research patrol. Use velocity feedback for the 10 s loop.
  command.manual_gear = speed < 0.0 ? -1 : 1;
  const double error = std::abs(speed) - measured_speed;
  command.throttle = clip(0.15 * std::abs(speed) + 0.4 * error, 0.0, 0.85);
  command.brake = error < -0.15 ? clip(-0.6 * error, 0.0, 1.0) : 0.0;
  if (command.brake > 0.0) command.throttle = 0.0;
  // The running AirSim Husky needs signed throttle as well as reverse gear.
  if (speed < 0.0) command.throttle = -command.throttle;
  command.steering = clip(yaw_rate / yaw_rate_limit, -1.0, 1.0);
  return command;
}

Eigen::Vector2d ugvAccelerationCommand(const Eigen::Vector3d& velocity, double yaw,
                                     const Eigen::Vector3d& acceleration, double dt,
                                     double speed_limit, double yaw_rate_limit) {
  if (!velocity.allFinite() || !acceleration.allFinite() || !std::isfinite(yaw) ||
      !std::isfinite(dt) || dt <= 0.0 || !std::isfinite(speed_limit) || speed_limit <= 0.0 ||
      !std::isfinite(yaw_rate_limit) || yaw_rate_limit <= 0.0)
    throw std::invalid_argument("invalid UGV acceleration command");
  const Eigen::Vector2d desired = velocity.head<2>() + dt * acceleration.head<2>();
  const double speed = std::min(speed_limit, desired.norm());
  if (speed < 1e-8) return Eigen::Vector2d::Zero();
  const double longitudinal = desired.dot(Eigen::Vector2d(std::cos(yaw), std::sin(yaw)));
  const double direction = longitudinal < 0.0 ? -1.0 : 1.0;
  const double heading = std::atan2(direction * desired.y(), direction * desired.x()) - yaw;
  const double error = std::atan2(std::sin(heading), std::cos(heading));
  return {direction * speed * std::max(0.0, std::cos(error)),
          clip(error / dt, -yaw_rate_limit, yaw_rate_limit)};
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
