#pragma once

#include <Eigen/Core>

namespace hercules_mission_ros {

struct CarCommand {
  double throttle{0.0};
  double brake{1.0};
  double steering{0.0};
  bool handbrake{false};
  int manual_gear{1};
};

// Production mission adapter. Canonical mission state and wrapper world-frame
// commands are both native AirSim NED, so no axis conversion belongs here.
Eigen::Vector3d uavVelocityCommand(const Eigen::Vector3d& measured_velocity,
                                   const Eigen::Vector3d& nominal_acceleration,
                                   double dt, double component_limit);
CarCommand ugvCarCommand(double desired_speed, double desired_yaw_rate,
                         double measured_speed, double max_yaw_rate,
                         bool target_vehicle = false);
Eigen::Vector3d pythonCbfUavVelocityCommand(const Eigen::Vector3d& measured_velocity,
                                             const Eigen::Vector3d& nominal_acceleration,
                                             double dt, double velocity_limit,
                                             double ceiling_z, double position_z);
CarCommand pythonCbfUgvCarCommand(double desired_speed, double desired_yaw_rate,
                                  double measured_speed, double cbf_yaw_rate_limit,
                                  bool target_vehicle = false,
                                  double speed_limit = 3.0);
CarCommand stoppedCarCommand();
CarCommand paperUgvCarCommand(double speed, double yaw_rate, double measured_speed,
                             double yaw_rate_limit);

// Map the planar acceleration requested by Wang's QP onto the Husky's
// speed/turn-rate interface. Tracking error remains in the calibrated residual.
Eigen::Vector2d ugvAccelerationCommand(const Eigen::Vector3d& velocity, double yaw,
                                     const Eigen::Vector3d& acceleration, double dt,
                                     double speed_limit, double yaw_rate_limit);

}  // namespace hercules_mission_ros
