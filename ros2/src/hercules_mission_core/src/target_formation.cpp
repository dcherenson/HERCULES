#include "hercules_mission_core/target_formation.hpp"

#include <cmath>
#include <map>

#include "hercules_mission_core/target_motion.hpp"

namespace hercules_mission_core {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

}  // namespace

Eigen::Vector3d uavTargetSlot(const std::string& agent_id) {
  static const std::map<std::string, Eigen::Vector3d> slots{
      {"Drone1", Eigen::Vector3d(-2.0, -2.0, 0.0)},
      {"Drone2", Eigen::Vector3d(2.0, -2.0, 0.0)},
      {"SimpleFlight", Eigen::Vector3d(0.0, 0.0, 0.0)},
      {"Drone4", Eigen::Vector3d(-2.0, 2.0, 0.0)},
      {"Drone5", Eigen::Vector3d(2.0, 2.0, 0.0)},
  };
  const auto found = slots.find(agent_id);
  return found == slots.end() ? Eigen::Vector3d::Zero() : found->second;
}

double ugvTargetSlotPhase(const std::string& agent_id) {
  if (agent_id == "Husky2") return 2.0 * kPi / 3.0;
  if (agent_id == "Husky3") return -2.0 * kPi / 3.0;
  return 0.0;
}

SlotReference targetCenteredSlot(
    const std::string& agent_id, VehicleType vehicle_type,
    const Eigen::Vector3d& target_position,
    const Eigen::Vector3d& target_velocity, double fallback_heading,
    double uav_altitude, std::optional<double> target_z,
    double ugv_circumradius) {
  const double planar_speed = target_velocity.head<2>().norm();
  const double heading = planar_speed > kTargetSlotHeadingSpeedThreshold
                             ? std::atan2(target_velocity.y(), target_velocity.x())
                             : fallback_heading;
  SlotReference output;
  output.heading = heading;

  if (vehicle_type == VehicleType::kDrone) {
    Eigen::Vector3d offset = uavTargetSlot(agent_id);
    const double x = offset.x();
    const double y = offset.y();
    offset.x() = std::cos(heading) * x - std::sin(heading) * y;
    offset.y() = std::sin(heading) * x + std::cos(heading) * y;
    output.position = target_position + offset;
    output.position.z() = uav_altitude;
    output.velocity = target_velocity;
    output.velocity.z() = 0.0;
    return output;
  }

  const double angle = heading + ugvTargetSlotPhase(agent_id);
  output.position = target_position;
  output.position.head<2>() +=
      ugv_circumradius * Eigen::Vector2d(std::cos(angle), std::sin(angle));
  output.position.z() = target_z.value_or(target_position.z());
  output.velocity = target_velocity;
  output.velocity.z() = 0.0;
  return output;
}

}  // namespace hercules_mission_core
