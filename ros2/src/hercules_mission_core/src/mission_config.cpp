#include "hercules_mission_core/mission_config.hpp"

#include <cmath>

namespace hercules_mission_core {

RuralTargetTrackingConfig ruralTargetTrackingConfig() {
  RuralTargetTrackingConfig config;
  config.source_revision = "2ef27d8ddc7f019f4b1af0196f7fbb4d91ce595f";
  config.agents = {
      {"Drone1", VehicleType::kDrone},
      {"Drone2", VehicleType::kDrone},
      {"SimpleFlight", VehicleType::kDrone},
      {"Drone4", VehicleType::kDrone},
      {"Drone5", VehicleType::kDrone},
      {"Husky1", VehicleType::kUgv},
      {"Husky2", VehicleType::kUgv},
      {"Husky3", VehicleType::kUgv},
  };
  config.target = {"Target1", VehicleType::kTargetUgv};
  config.target_motion.longitudinal_span = 10.0;
  config.target_motion.lateral_span = 8.0;
  config.target_motion.speed = 0.10;
  config.target_motion.sample_count = 64;
  config.target_motion.waypoint_radius = 1.0;
  config.target_motion.heading_gain = 2.0;
  config.target_motion.max_yaw_rate = 1.5;
  config.target_motion.minimum_alignment = 0.75;
  config.target_motion.direction = 1;
  config.formation.position_gain = 0.5;
  config.formation.velocity_gain = 3.0;
  config.formation.max_speed = 1.0;
  config.formation.leader_max_speed = 1.0;
  config.formation.has_leader_max_speed = true;
  config.formation.ugv_heading_gain = 1.0;
  config.formation.ugv_max_yaw_rate = 1.0;
  config.formation.uav_altitude = -5.0;
  return config;
}

Eigen::Vector3d ruralTargetStartAnchor(
    const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
    double route_heading, const RuralTargetTrackingConfig& config) {
  Eigen::Vector3d anchor = targetCenterBeforeGoal(
      start, goal, config.target_route_fraction);
  const auto [route_forward, camera_right] = routeBasis(route_heading);
  anchor.head<2>() += config.target_camera_right_offset * camera_right;
  anchor.head<2>() -= config.target_toward_robots_offset * route_forward;
  anchor.z() = config.target_ground_z;
  return anchor;
}

}  // namespace hercules_mission_core
