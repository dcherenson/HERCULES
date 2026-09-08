#pragma once

#include <optional>
#include <string>

#include <Eigen/Core>

namespace hercules_mission_core {

inline constexpr double kTargetSlotHeadingSpeedThreshold = 2.5;

enum class VehicleType { kDrone, kUgv, kTargetUgv };

struct SlotReference {
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  double heading{0.0};
};

Eigen::Vector3d uavTargetSlot(const std::string& agent_id);
double ugvTargetSlotPhase(const std::string& agent_id);

SlotReference targetCenteredSlot(
    const std::string& agent_id, VehicleType vehicle_type,
    const Eigen::Vector3d& target_position,
    const Eigen::Vector3d& target_velocity, double fallback_heading,
    double uav_altitude, std::optional<double> target_z = std::nullopt,
    double ugv_circumradius = 6.0);

}  // namespace hercules_mission_core
