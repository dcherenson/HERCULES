#pragma once

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace hercules_cbf {

enum class VehicleType { kDrone, kUgv };
enum class Method { kMestres, kWang };

struct AgentState {
  std::string agent_id;
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  double yaw{0.0};
  std::optional<Eigen::Vector3d> acceleration;
  VehicleType vehicle_type{VehicleType::kDrone};
  double timestamp{0.0};
  double yaw_rate{0.0};
};

struct ObstacleProxy {
  std::string obstacle_id;
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  double radius{0.0};
  std::string source{"unknown"};
  double timestamp{0.0};
  int point_count{0};
  bool is_planar{false};
  std::optional<Eigen::Vector3d> velocity;

  void clampRadius() { radius = std::max(0.0, radius); }
};

struct CBFConfig {
  Method method{Method::kMestres};
  double k1{2.0};
  double k2{2.0};
  double alpha{2.0};
  double uncertainty_radius{0.0};
  double uav_radius{1.0};
  double ugv_radius{1.25};
  double obstacle_margin{0.0};
  double uav_acceleration_limit{6.0};
  double ugv_acceleration_limit{3.0};
  double uav_velocity_limit{3.0};
  double ugv_speed_limit{3.0};
  double ugv_yaw_rate_limit{1.5};
  double lookahead_distance{1.0};
  double uav_altitude_floor{-1.0};
  int distributed_rounds{20};
  double distributed_tolerance{1e-3};
  double solver_eps_abs{1e-5};
  double solver_eps_rel{1e-5};
  int solver_max_iter{4000};
};

struct CBFRequest {
  AgentState ego;
  Eigen::VectorXd nominal_control;
  std::vector<AgentState> neighbors;
  std::vector<ObstacleProxy> obstacles;
  std::optional<double> uncertainty_radius;
  std::optional<std::pair<Eigen::VectorXd, Eigen::VectorXd>> control_bounds;
  bool sensor_valid{true};
};

struct ConstraintSet {
  std::vector<Eigen::VectorXd> rows;
  std::vector<double> rhs;
  std::vector<double> barriers;
  std::vector<double> robust_terms;
  std::vector<std::string> row_labels;
  Eigen::VectorXd lower;
  Eigen::VectorXd upper;
};

struct SolverDiagnostics {
  std::string solver{"none"};
  std::string raw_status;
  int iterations{0};
  double primal_residual{std::numeric_limits<double>::quiet_NaN()};
  double dual_residual{std::numeric_limits<double>::quiet_NaN()};
};

struct CBFResult {
  Eigen::VectorXd safe_control;
  bool success{false};
  std::string status;
  std::string message;
  double minimum_barrier{std::numeric_limits<double>::infinity()};
  std::vector<double> robust_terms;
  int active_constraints{0};
  int constraint_count{0};
  int distributed_rounds{0};
  double solve_time_ms{0.0};
  bool fallback{false};
  double maximum_row_violation{0.0};
  double maximum_bound_violation{0.0};
  SolverDiagnostics solver;
};

}  // namespace hercules_cbf
