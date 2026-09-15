#pragma once

#include <string>

#include <Eigen/Core>

#include "hercules_mission_core/target_formation.hpp"

namespace hercules_mission_core {

struct AgentState {
  std::string agent_id;
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  double yaw{0.0};
  VehicleType vehicle_type{VehicleType::kDrone};
};

struct TargetEstimate {
  Eigen::Vector2d position{Eigen::Vector2d::Zero()};
  Eigen::Vector2d velocity{Eigen::Vector2d::Zero()};
  bool active{false};
};

struct FormationConfig {
  std::string leader_id{"Husky1"};
  double position_gain{1.0};
  double velocity_gain{3.0};
  double max_speed{3.0};
  double leader_max_speed{3.0};
  bool has_leader_max_speed{false};
  double ugv_max_yaw_rate{1.5};
  double ugv_heading_gain{2.0};
  double uav_altitude{-5.0};
};

class FormationController {
 public:
  explicit FormationController(FormationConfig config = {});

  Eigen::Vector3d targetNominalControl(
      const AgentState& agent, const TargetEstimate& target_estimate,
      double fallback_heading, double target_z = 0.0,
      double ugv_circumradius = 6.0) const;

  Eigen::Vector2d targetNominalUnicycleControl(
      const AgentState& agent, const TargetEstimate& target_estimate,
      double fallback_heading, double target_z = 0.0,
      double ugv_circumradius = 6.0) const;

  const FormationConfig& config() const { return config_; }

 private:
  double speedLimit(const std::string& agent_id) const;

  FormationConfig config_;
};

}  // namespace hercules_mission_core
