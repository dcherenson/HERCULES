#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include "hercules_mission_core/formation_controller.hpp"
#include "hercules_mission_core/target_motion.hpp"

namespace hercules_mission_core {

struct MissionAgent {
  std::string id;
  VehicleType type;
};

struct TrackingConfig {
  double window_seconds{5.0};
  double admm_rho{1.0};
  int admm_max_iterations{20};
  double admm_tolerance{1e-3};
  double process_noise{0.20};
  double measurement_std{0.25};
};

struct RuralTargetTrackingConfig {
  std::string source_revision;
  std::vector<MissionAgent> agents;
  MissionAgent target;
  double control_dt{0.1};
  double tracking_rate{4.0};
  double communication_range{10.0};
  double uav_altitude{-5.0};
  double target_ugv_circumradius{5.0};
  double initial_heading_offset_deg{90.0};
  double target_route_fraction{0.25};
  double target_camera_right_offset{5.0};
  double target_toward_robots_offset{5.0};
  double target_ground_z{-1.0};
  int target_start_sample_index{5};
  FigureEightConfig target_motion;
  FormationConfig formation;
  TrackingConfig tracking;
};

RuralTargetTrackingConfig ruralTargetTrackingConfig();
Eigen::Vector3d ruralTargetStartAnchor(
    const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
    double route_heading, const RuralTargetTrackingConfig& config);

}  // namespace hercules_mission_core
