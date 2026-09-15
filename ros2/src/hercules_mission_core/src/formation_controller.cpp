#include "hercules_mission_core/formation_controller.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "hercules_mission_core/target_motion.hpp"

namespace hercules_mission_core {

FormationController::FormationController(FormationConfig config)
    : config_(std::move(config)) {}

double FormationController::speedLimit(const std::string& agent_id) const {
  if (agent_id == config_.leader_id && config_.has_leader_max_speed) {
    return config_.leader_max_speed;
  }
  return config_.max_speed;
}

Eigen::Vector3d FormationController::targetNominalControl(
    const AgentState& agent, const TargetEstimate& target_estimate,
    double fallback_heading, double target_z,
    double ugv_circumradius) const {
  if (!target_estimate.active) return Eigen::Vector3d::Zero();
  const Eigen::Vector3d target_position(
      target_estimate.position.x(), target_estimate.position.y(), target_z);
  const Eigen::Vector3d target_velocity(
      target_estimate.velocity.x(), target_estimate.velocity.y(), 0.0);
  const SlotReference reference = targetCenteredSlot(
      agent.agent_id, VehicleType::kDrone, target_position, target_velocity,
      fallback_heading, config_.uav_altitude, target_z, ugv_circumradius);
  Eigen::Vector3d desired_velocity = reference.velocity +
      config_.position_gain * (reference.position - agent.position);
  const double speed = desired_velocity.norm();
  const double speed_limit = speedLimit(agent.agent_id);
  if (speed > speed_limit) desired_velocity *= speed_limit / speed;
  return config_.velocity_gain * (desired_velocity - agent.velocity);
}

Eigen::Vector2d FormationController::targetNominalUnicycleControl(
    const AgentState& agent, const TargetEstimate& target_estimate,
    double fallback_heading, double target_z,
    double ugv_circumradius) const {
  if (!target_estimate.active) return Eigen::Vector2d::Zero();
  const Eigen::Vector3d target_position(
      target_estimate.position.x(), target_estimate.position.y(), target_z);
  const Eigen::Vector3d target_velocity(
      target_estimate.velocity.x(), target_estimate.velocity.y(), 0.0);
  const SlotReference reference = targetCenteredSlot(
      agent.agent_id, VehicleType::kUgv, target_position, target_velocity,
      fallback_heading, config_.uav_altitude, target_z, ugv_circumradius);
  const Eigen::Vector2d delta = reference.position.head<2>() - agent.position.head<2>();
  const double distance = delta.norm();
  if (distance <= 0.5) return Eigen::Vector2d::Zero();
  const double desired_heading = std::atan2(delta.y(), delta.x());
  const double heading_error = wrapAngle(desired_heading - agent.yaw);
  double speed = std::min(speedLimit(agent.agent_id), config_.position_gain * distance);
  speed *= std::max(0.25, std::cos(heading_error));
  const double yaw_rate = std::clamp(
      config_.ugv_heading_gain * heading_error,
      -config_.ugv_max_yaw_rate, config_.ugv_max_yaw_rate);
  return Eigen::Vector2d(speed, yaw_rate);
}

}  // namespace hercules_mission_core
