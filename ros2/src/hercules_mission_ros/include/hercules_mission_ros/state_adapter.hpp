#pragma once

#include <cstdint>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "hercules_mission_core/formation_controller.hpp"

namespace hercules_mission_ros {

inline constexpr char kCanonicalFrame[] = "airsim_world_ned";

enum class ConversionError {
  kNone,
  kEmptyAgentId,
  kNonpositiveTimestamp,
  kNonfiniteOrigin,
  kNonfinitePosition,
  kNonfiniteVelocity,
  kNonfiniteAngularVelocity,
  kInvalidQuaternion,
};

// Plain transport data extracted from the existing wrapper odometry. The
// wrapper axes are X unchanged, Y negated, and Z negated from AirSim NED.
struct WrapperOdometryData {
  int64_t stamp_ns{0};
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_velocity{Eigen::Vector3d::Zero()};
};

struct CanonicalState {
  int64_t stamp_ns{0};
  std::string agent_id;
  hercules_mission_core::VehicleType vehicle_type{
      hercules_mission_core::VehicleType::kDrone};
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_velocity{Eigen::Vector3d::Zero()};
  double yaw{0.0};
  double yaw_rate{0.0};
  bool valid{false};
  ConversionError error{ConversionError::kNone};

  hercules_mission_core::AgentState agentState() const;
};

CanonicalState fromWrapperOdometry(
    const WrapperOdometryData& wrapper,
    const Eigen::Vector3d& vehicle_origin_ned,
    const std::string& agent_id,
    hercules_mission_core::VehicleType vehicle_type);

double yawFromNedQuaternion(const Eigen::Quaterniond& orientation);
const char* conversionErrorName(ConversionError error);

}  // namespace hercules_mission_ros
