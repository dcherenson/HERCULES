#include "hercules_mission_ros/state_adapter.hpp"

#include <cmath>

#include "hercules_mission_core/target_motion.hpp"

namespace hercules_mission_ros {
namespace {

constexpr double kQuaternionNormTolerance = 0.05;

CanonicalState invalidState(const WrapperOdometryData& wrapper,
                            const std::string& agent_id,
                            hercules_mission_core::VehicleType vehicle_type,
                            ConversionError error) {
  CanonicalState result;
  result.stamp_ns = wrapper.stamp_ns;
  result.agent_id = agent_id;
  result.vehicle_type = vehicle_type;
  result.error = error;
  return result;
}

}  // namespace

double yawFromNedQuaternion(const Eigen::Quaterniond& orientation) {
  const double sin_yaw = 2.0 *
      (orientation.w() * orientation.z() +
       orientation.x() * orientation.y());
  const double cos_yaw = 1.0 - 2.0 *
      (orientation.y() * orientation.y() +
       orientation.z() * orientation.z());
  return hercules_mission_core::wrapAngle(std::atan2(sin_yaw, cos_yaw));
}

CanonicalState fromWrapperOdometry(
    const WrapperOdometryData& wrapper,
    const Eigen::Vector3d& vehicle_origin_ned,
    const std::string& agent_id,
    hercules_mission_core::VehicleType vehicle_type) {
  if (agent_id.empty()) {
    return invalidState(wrapper, agent_id, vehicle_type,
                        ConversionError::kEmptyAgentId);
  }
  if (wrapper.stamp_ns <= 0) {
    return invalidState(wrapper, agent_id, vehicle_type,
                        ConversionError::kNonpositiveTimestamp);
  }
  if (!vehicle_origin_ned.allFinite()) {
    return invalidState(wrapper, agent_id, vehicle_type,
                        ConversionError::kNonfiniteOrigin);
  }
  if (!wrapper.position.allFinite()) {
    return invalidState(wrapper, agent_id, vehicle_type,
                        ConversionError::kNonfinitePosition);
  }
  if (!wrapper.velocity.allFinite()) {
    return invalidState(wrapper, agent_id, vehicle_type,
                        ConversionError::kNonfiniteVelocity);
  }
  if (!wrapper.angular_velocity.allFinite()) {
    return invalidState(wrapper, agent_id, vehicle_type,
                        ConversionError::kNonfiniteAngularVelocity);
  }
  if (!wrapper.orientation.coeffs().allFinite() ||
      std::abs(wrapper.orientation.norm() - 1.0) >=
          kQuaternionNormTolerance) {
    return invalidState(wrapper, agent_id, vehicle_type,
                        ConversionError::kInvalidQuaternion);
  }

  CanonicalState result;
  result.stamp_ns = wrapper.stamp_ns;
  result.agent_id = agent_id;
  result.vehicle_type = vehicle_type;

  // Exact inverse of airsim_ros_wrapper.cpp: X is unchanged; Y and Z are
  // negated. AirSim local NED is aligned with world NED, so origin is only a
  // translation and must not rotate the local position.
  const Eigen::Vector3d local_ned(
      wrapper.position.x(), -wrapper.position.y(), -wrapper.position.z());
  result.position = vehicle_origin_ned + local_ned;
  result.velocity = Eigen::Vector3d(
      wrapper.velocity.x(), -wrapper.velocity.y(), -wrapper.velocity.z());
  result.angular_velocity = Eigen::Vector3d(
      wrapper.angular_velocity.x(), -wrapper.angular_velocity.y(),
      -wrapper.angular_velocity.z());
  result.orientation = Eigen::Quaterniond(
      wrapper.orientation.w(), wrapper.orientation.x(),
      -wrapper.orientation.y(), -wrapper.orientation.z());
  result.orientation.normalize();
  result.yaw = yawFromNedQuaternion(result.orientation);
  result.yaw_rate = result.angular_velocity.z();
  result.valid = true;
  return result;
}

hercules_mission_core::AgentState CanonicalState::agentState() const {
  return {agent_id, position, velocity, yaw, vehicle_type};
}

const char* conversionErrorName(ConversionError error) {
  switch (error) {
    case ConversionError::kNone: return "none";
    case ConversionError::kEmptyAgentId: return "empty_agent_id";
    case ConversionError::kNonpositiveTimestamp: return "nonpositive_timestamp";
    case ConversionError::kNonfiniteOrigin: return "nonfinite_origin";
    case ConversionError::kNonfinitePosition: return "nonfinite_position";
    case ConversionError::kNonfiniteVelocity: return "nonfinite_velocity";
    case ConversionError::kNonfiniteAngularVelocity:
      return "nonfinite_angular_velocity";
    case ConversionError::kInvalidQuaternion: return "invalid_quaternion";
  }
  return "unknown";
}

}  // namespace hercules_mission_ros
