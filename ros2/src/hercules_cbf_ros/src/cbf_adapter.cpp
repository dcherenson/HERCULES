#include "hercules_cbf_ros/cbf_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <builtin_interfaces/msg/time.hpp>

namespace hercules_cbf_ros {
namespace {
hercules_cbf::VehicleType vehicleType(const std::string& value) {
  return value == "ugv" || value == "target_ugv"
      ? hercules_cbf::VehicleType::kUgv : hercules_cbf::VehicleType::kDrone;
}

template <typename Values>
bool finiteValues(const Values& values) {
  for (const auto value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

double timeSeconds(const builtin_interfaces::msg::Time& stamp) {
  return static_cast<double>(stamp.sec) +
         1e-9 * static_cast<double>(stamp.nanosec);
}

bool finiteStateMessage(const hercules_interfaces::msg::GroundTruthState& message) {
  return finiteValues(message.position) && finiteValues(message.velocity) &&
         finiteValues(message.orientation) && std::isfinite(message.yaw) &&
         std::isfinite(message.yaw_rate);
}

bool finiteObstacleMessage(const hercules_interfaces::msg::ObstacleProxy& message) {
  if (!finiteValues(message.center) || !std::isfinite(message.radius) ||
      message.radius < 0.0) return false;
  return !message.has_velocity || finiteValues(message.velocity);
}

void copyVector(std::vector<double>& destination, const Eigen::VectorXd& source) {
  destination.clear();
  destination.reserve(static_cast<std::size_t>(source.size()));
  for (Eigen::Index index = 0; index < source.size(); ++index)
    destination.push_back(source[index]);
}
}

hercules_cbf::AgentState stateFromRos(
    const hercules_interfaces::msg::GroundTruthState& message) {
  hercules_cbf::AgentState state;
  state.agent_id = message.agent_id;
  state.vehicle_type = vehicleType(message.vehicle_type);
  state.position = Eigen::Vector3d(message.position[0], message.position[1], message.position[2]);
  state.velocity = Eigen::Vector3d(message.velocity[0], message.velocity[1], message.velocity[2]);
  state.yaw = message.yaw;
  state.yaw_rate = message.yaw_rate;
  state.timestamp = timeSeconds(message.header.stamp);
  return state;
}

hercules_cbf::ObstacleProxy obstacleFromRos(
    const hercules_interfaces::msg::ObstacleProxy& message) {
  hercules_cbf::ObstacleProxy obstacle;
  obstacle.obstacle_id = message.proxy_id;
  obstacle.source = message.source;
  obstacle.center = Eigen::Vector3d(message.center[0], message.center[1], message.center[2]);
  obstacle.radius = std::max(0.0, message.radius);
  obstacle.point_count = static_cast<int>(message.point_count);
  obstacle.is_planar = message.is_planar;
  obstacle.timestamp = timeSeconds(message.stamp);
  if (message.has_velocity) {
    obstacle.velocity = Eigen::Vector3d(message.velocity[0], message.velocity[1], message.velocity[2]);
  }
  return obstacle;
}

void appendTargetProxy(const hercules_interfaces::msg::TargetEstimate& estimate,
                       const std::string& agent_id, double ugv_radius,
                       std::vector<hercules_cbf::ObstacleProxy>& obstacles,
                       double target_z) {
  if (!estimate.active) return;
  if (!finiteValues(estimate.position) || !finiteValues(estimate.velocity) ||
      !finiteValues(estimate.state_covariance) || !std::isfinite(target_z) ||
      !std::isfinite(ugv_radius) || ugv_radius < 0.0) return;
  // The wire representation is row-major, but a tracker or transport bridge
  // can leave the two off-diagonal entries very slightly different.  Python's
  // target-proxy helper symmetrizes before taking the largest eigenvalue; do
  // the same here rather than treating the array as an arbitrary matrix.
  const double c00 = estimate.state_covariance[0];
  const double c01 = 0.5 * (estimate.state_covariance[1] + estimate.state_covariance[4]);
  const double c11 = estimate.state_covariance[5];
  const double trace = c00 + c11;
  const double determinant = c00 * c11 - c01 * c01;
  const double discriminant = std::max(0.0, trace * trace * 0.25 - determinant);
  const double eigen_max = std::max(0.0, trace * 0.5 + std::sqrt(discriminant));
  hercules_cbf::ObstacleProxy proxy;
  proxy.obstacle_id = "target_" +
      (estimate.target_id.empty() ? std::string("Target1") : estimate.target_id);
  proxy.source = "target_tracking";
  proxy.center = Eigen::Vector3d(estimate.position[0], estimate.position[1], target_z);
  proxy.radius = ugv_radius + 2.0 * std::sqrt(eigen_max);
  proxy.velocity = Eigen::Vector3d(estimate.velocity[0], estimate.velocity[1], 0.0);
  proxy.timestamp = static_cast<double>(estimate.stamp.sec) + 1e-9 * estimate.stamp.nanosec;
  proxy.is_planar = true;
  obstacles.push_back(proxy);
  (void)agent_id;
}

void appendTargetProxy(const hercules_interfaces::msg::TargetEstimate& estimate,
                       const std::string& agent_id, double ugv_radius,
                       std::vector<hercules_interfaces::msg::ObstacleProxy>& obstacles,
                       double target_z) {
  std::vector<hercules_cbf::ObstacleProxy> native;
  appendTargetProxy(estimate, agent_id, ugv_radius, native, target_z);
  for (const auto& input : native) {
    hercules_interfaces::msg::ObstacleProxy output;
    output.proxy_id = input.obstacle_id;
    output.source = input.source;
    output.center = {input.center.x(), input.center.y(), input.center.z()};
    output.radius = input.radius;
    output.stamp = estimate.stamp;
    output.point_count = static_cast<uint32_t>(input.point_count);
    output.is_planar = input.is_planar;
    if (input.velocity) {
      output.has_velocity = true;
      output.velocity = {input.velocity->x(), input.velocity->y(), input.velocity->z()};
    }
    obstacles.push_back(output);
  }
}

AdapterResult filterRequest(
    const hercules_interfaces::msg::GroundTruthState& ego,
    const Eigen::VectorXd& nominal,
    const std::vector<hercules_interfaces::msg::GroundTruthState>& neighbors,
    const std::vector<hercules_interfaces::msg::ObstacleProxy>& obstacles,
    const hercules_cbf::CBFConfig& config, bool sensor_valid,
    const std::string& selected_method, const std::string& effective_method) {
  hercules_cbf::CBFRequest request;
  request.ego = stateFromRos(ego);
  request.nominal_control = nominal;
  request.sensor_valid = sensor_valid && finiteStateMessage(ego);
  for (const auto& neighbor : neighbors) {
    request.sensor_valid = request.sensor_valid && finiteStateMessage(neighbor);
    request.neighbors.push_back(stateFromRos(neighbor));
  }
  for (const auto& obstacle : obstacles) {
    request.sensor_valid = request.sensor_valid && finiteObstacleMessage(obstacle);
    request.obstacles.push_back(obstacleFromRos(obstacle));
  }
  AdapterResult output;
  output.result = hercules_cbf::filter(request, config);
  auto& diagnostic = output.diagnostics;
  diagnostic.header = ego.header;
  diagnostic.agent_id = ego.agent_id;
  diagnostic.enabled = true;
  diagnostic.selected_method = selected_method;
  diagnostic.effective_method = effective_method;
  diagnostic.control_kind = hercules_cbf::isUnicycle(request, config) ? "unicycle" : "double_integrator";
  diagnostic.control_dimension = static_cast<uint32_t>(output.result.safe_control.size());
  copyVector(diagnostic.nominal_control, nominal);
  copyVector(diagnostic.safe_control, output.result.safe_control);
  diagnostic.success = output.result.success;
  diagnostic.fallback = output.result.fallback;
  diagnostic.solver_status = output.result.status;
  diagnostic.solver_iterations = output.result.solver.iterations;
  diagnostic.primal_residual = output.result.solver.primal_residual;
  diagnostic.dual_residual = output.result.solver.dual_residual;
  diagnostic.minimum_barrier = output.result.minimum_barrier;
  diagnostic.constraint_count = output.result.constraint_count;
  diagnostic.active_constraints = output.result.active_constraints;
  diagnostic.distributed_rounds = output.result.distributed_rounds;
  diagnostic.filter_time_ms = output.result.solve_time_ms;
  diagnostic.solver_time_ms = output.result.solve_time_ms;
  diagnostic.robust_terms = output.result.robust_terms;
  const auto constraints = request.sensor_valid
      ? hercules_cbf::buildConstraints(request, config)
      : hercules_cbf::ConstraintSet{};
  diagnostic.row_labels = constraints.row_labels;
  diagnostic.maximum_row_violation = output.result.maximum_row_violation;
  diagnostic.maximum_bound_violation = output.result.maximum_bound_violation;
  diagnostic.final_feasible = diagnostic.maximum_row_violation <= 1e-6 &&
                              diagnostic.maximum_bound_violation <= 1e-6;
  if (output.result.safe_control.size() > 0 &&
      nominal.size() >= output.result.safe_control.size() &&
      output.result.safe_control.array().isFinite().all() &&
      nominal.array().isFinite().all()) {
    diagnostic.intervention_norm =
        (output.result.safe_control - nominal.head(output.result.safe_control.size())).norm();
  } else {
    // A malformed request is represented in diagnostics instead of allowing
    // Eigen::head() to assert at the adapter boundary.
    diagnostic.intervention_norm = 0.0;
  }
  return output;
}

}  // namespace hercules_cbf_ros
