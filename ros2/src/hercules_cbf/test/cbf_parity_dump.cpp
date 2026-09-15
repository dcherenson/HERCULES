#include <iomanip>
#include <iostream>
#include <string>

#include "hercules_cbf/filter.hpp"

namespace hc = hercules_cbf;

namespace {
void vec(const Eigen::VectorXd& value) {
  std::cout << '[';
  for (Eigen::Index i = 0; i < value.size(); ++i) { if (i) std::cout << ','; std::cout << value[i]; }
  std::cout << ']';
}
void rows(const hc::ConstraintSet& value) {
  std::cout << '[';
  for (std::size_t i = 0; i < value.rows.size(); ++i) { if (i) std::cout << ','; vec(value.rows[i]); }
  std::cout << ']';
}
void doubles(const std::vector<double>& value) {
  std::cout << '[';
  for (std::size_t i = 0; i < value.size(); ++i) { if (i) std::cout << ','; std::cout << value[i]; }
  std::cout << ']';
}
void strings(const std::vector<std::string>& value) {
  std::cout << '[';
  for (std::size_t i = 0; i < value.size(); ++i) { if (i) std::cout << ','; std::cout << '"' << value[i] << '"'; }
  std::cout << ']';
}

void emit(const std::string& name, const hc::CBFRequest& request, const hc::CBFConfig& config) {
  const auto constraints = hc::buildConstraints(request, config);
  const auto result = hc::filter(request, config);
  std::cout << "{\"name\":\"" << name << "\",\"rows\":"; rows(constraints);
  std::cout << ",\"rhs\":"; doubles(constraints.rhs);
  std::cout << ",\"barriers\":"; doubles(constraints.barriers);
  std::cout << ",\"robust_terms\":"; doubles(constraints.robust_terms);
  std::cout << ",\"labels\":"; strings(constraints.row_labels);
  std::cout << ",\"lower\":"; vec(constraints.lower);
  std::cout << ",\"upper\":"; vec(constraints.upper);
  std::cout << ",\"safe_control\":"; vec(result.safe_control);
  std::cout << ",\"success\":" << (result.success ? "true" : "false")
            << ",\"status\":\"" << result.status << "\",\"fallback\":"
            << (result.fallback ? "true" : "false")
            << ",\"active_constraints\":" << result.active_constraints
            << ",\"constraint_count\":" << result.constraint_count
            << ",\"distributed_rounds\":" << result.distributed_rounds
            << ",\"maximum_row_violation\":" << result.maximum_row_violation << '}';
}

hc::AgentState state(const std::string& id, hc::VehicleType type, Eigen::Vector3d p,
                     Eigen::Vector3d v = Eigen::Vector3d::Zero(), double yaw = 0.0) {
  hc::AgentState result; result.agent_id = id; result.vehicle_type = type;
  result.position = p; result.velocity = v; result.yaw = yaw; return result;
}
hc::ObstacleProxy obstacle(const std::string& id, Eigen::Vector3d c, double r,
                           std::optional<Eigen::Vector3d> v = std::nullopt) {
  hc::ObstacleProxy result; result.obstacle_id = id; result.center = c; result.radius = r; result.velocity = v; return result;
}
hc::CBFRequest droneRequest(Eigen::Vector3d p, Eigen::Vector3d v, Eigen::Vector3d u) {
  hc::CBFRequest result; result.ego = state("Drone1", hc::VehicleType::kDrone, p, v); result.nominal_control = u; return result;
}
}

int main() {
  std::cout << std::setprecision(17) << '[';
  bool first = true;
  auto add = [&](const std::string& name, hc::CBFRequest request, hc::CBFConfig config = {}) {
    if (!first) {
      std::cout << ',';
    }
    first = false;
    emit(name, request, config);
  };
  hc::CBFConfig wang; wang.method = hc::Method::kWang;
  auto neighbor = state("Drone2", hc::VehicleType::kDrone, {3, 1, -5}, {0.2, -0.1, 0.0});
  neighbor.acceleration = Eigen::Vector3d(0.3, -0.2, 0.1);
  auto req = droneRequest({0, 0, -5}, {1, -0.5, 0.2}, {0.4, -0.2, 0.1}); req.neighbors = {neighbor}; add("uav_neighbor", req, wang);
  req = droneRequest({0, 0, -5}, {2, 0, 0}, {4, 0, 0}); req.obstacles = {obstacle("wall", {2, 0, -5}, 1)}; add("uav_obstacle_static", req, wang);
  req.obstacles = {obstacle("wall", {2, 0, -5}, 1, Eigen::Vector3d(-1, 0, 0))}; add("uav_obstacle_moving", req, wang);
  req = droneRequest({0, 0, 0}, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()); req.uncertainty_radius = 0.25; add("altitude_uncertainty", req, wang);

  hc::CBFConfig mestres; mestres.method = hc::Method::kMestres; mestres.lookahead_distance = 1.0;
  req = {}; req.ego = state("Husky1", hc::VehicleType::kUgv, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.4); req.nominal_control = Eigen::Vector2d(1, 0); add("ugv_no_obstacle", req, mestres);
  auto ugv_neighbor = state("Husky2", hc::VehicleType::kUgv, {2, 1, 0}, {0.1, 0.2, 0});
  req.neighbors = {ugv_neighbor}; add("ugv_neighbor", req, mestres);
  req.neighbors.clear(); req.obstacles = {obstacle("target", {3, 0, 0}, 0.5, Eigen::Vector3d(-2, 0, 0))}; add("ugv_moving_obstacle", req, mestres);

  req = droneRequest({0, 0, -5}, {1, 0, 0}, Eigen::Vector3d::Zero()); req.neighbors = {neighbor}; add("wang_split", req, wang);
  req = droneRequest({0, 0, -5}, Eigen::Vector3d::Zero(), {2, -3, 4}); req.control_bounds = std::make_pair((Eigen::Vector3d() << -1, -2, -3).finished(), (Eigen::Vector3d() << 1, 2, 3).finished()); add("custom_bounds", req, wang);
  req = droneRequest({0, 0, -5}, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()); req.obstacles = {obstacle("occupied", {0, 0, -5}, 2)}; add("infeasible", req, wang);
  req.sensor_valid = false; add("invalid_sensor", req, wang);
  req = droneRequest({0, 0, -5}, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()); req.neighbors = {state("Husky1", hc::VehicleType::kUgv, {0, 0, -5})}; add("cross_type_exclusion", req, wang);
  req = droneRequest({0, 0, -5}, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()); req.obstacles = {obstacle("target_Target1", {3, -2, -1}, 2.25, Eigen::Vector3d(.5, 1, 0))}; add("moving_target_proxy", req, wang);
  std::cout << "]\n";
}
