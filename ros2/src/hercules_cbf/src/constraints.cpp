#include "hercules_cbf/constraints.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace hercules_cbf {
namespace {

Eigen::Vector3d optionalOrZero(const std::optional<Eigen::Vector3d>& value) {
  return value.value_or(Eigen::Vector3d::Zero());
}

double radiusFor(VehicleType type, const CBFConfig& config) {
  return type == VehicleType::kDrone ? config.uav_radius : config.ugv_radius;
}

void addDoublePair(const AgentState& ego, const Eigen::Vector3d& center,
                   const Eigen::Vector3d& velocity, double other_radius,
                   const Eigen::Vector3d& other_acceleration, bool split,
                   const std::string& label, const CBFConfig& config,
                   double uncertainty_value, ConstraintSet& result) {
  const Eigen::Vector3d dp = ego.position - center;
  const Eigen::Vector3d dv = ego.velocity - velocity;
  const double clearance = radiusFor(ego.vehicle_type, config) + other_radius + config.obstacle_margin;
  const double h = dp.dot(dp) - clearance * clearance;
  const double h_dot = 2.0 * dp.dot(dv);
  const double psi = h_dot + config.k1 * h;
  const double base = 2.0 * dv.dot(dv) + config.k1 * h_dot + config.k2 * psi;
  const double robust = (Eigen::VectorXd(6) <<
      2.0 * dv + 2.0 * config.k1 * dp, 2.0 * dp).finished().norm() * uncertainty_value;
  double b = -base + 2.0 * dp.dot(other_acceleration) + robust;
  if (split) b *= 0.5;
  result.rows.push_back(2.0 * dp);
  result.rhs.push_back(b);
  result.barriers.push_back(h);
  result.robust_terms.push_back(robust);
  result.row_labels.push_back(label);
}

void addUnicycleCircle(const AgentState& ego, const Eigen::Vector3d& center,
                       const Eigen::Vector3d& velocity, double other_radius,
                       const std::string& label, const CBFConfig& config,
                       double uncertainty_value, ConstraintSet& result) {
  const double c = std::cos(ego.yaw);
  const double s = std::sin(ego.yaw);
  const double lookahead = config.lookahead_distance;
  const Eigen::Vector2d q = ego.position.head<2>() + lookahead * Eigen::Vector2d(c, s);
  Eigen::Matrix2d control_map;
  control_map << c, -lookahead * s, s, lookahead * c;
  const Eigen::Vector2d delta = q - center.head<2>();
  const double clearance = radiusFor(ego.vehicle_type, config) + other_radius + config.obstacle_margin;
  const double h = delta.dot(delta) - clearance * clearance;
  const double robust = (2.0 * delta).norm() * uncertainty_value;
  const double b = -config.alpha * h + 2.0 * delta.dot(velocity.head<2>()) + robust;
  result.rows.push_back(2.0 * delta.transpose() * control_map);
  result.rhs.push_back(b);
  result.barriers.push_back(h);
  result.robust_terms.push_back(robust);
  result.row_labels.push_back(label);
}

}  // namespace

int controlDimension(const CBFRequest& request, const CBFConfig& config) {
  return isUnicycle(request, config) ? 2 : 3;
}

bool isUnicycle(const CBFRequest& request, const CBFConfig& config) {
  return request.ego.vehicle_type == VehicleType::kUgv && config.method == Method::kMestres;
}

double uncertainty(const CBFRequest& request, const CBFConfig& config) {
  return std::max(0.0, request.uncertainty_radius.value_or(config.uncertainty_radius));
}

ConstraintSet buildConstraints(const CBFRequest& request, const CBFConfig& config) {
  ConstraintSet result;
  const bool unicycle = isUnicycle(request, config);
  const double robust = uncertainty(request, config);
  if (unicycle) {
    for (const auto& neighbor : request.neighbors) {
      if (neighbor.vehicle_type != request.ego.vehicle_type) continue;
      addUnicycleCircle(request.ego, neighbor.position, neighbor.velocity,
                        radiusFor(neighbor.vehicle_type, config),
                        "neighbor:" + neighbor.agent_id, config, robust, result);
    }
    for (auto obstacle : request.obstacles) {
      obstacle.clampRadius();
      addUnicycleCircle(request.ego, obstacle.center,
                        obstacle.velocity.value_or(Eigen::Vector3d::Zero()), obstacle.radius,
                        "obstacle:" + obstacle.obstacle_id, config, robust, result);
    }
  } else {
    for (const auto& neighbor : request.neighbors) {
      if (neighbor.vehicle_type != request.ego.vehicle_type) continue;
      addDoublePair(request.ego, neighbor.position, neighbor.velocity,
                    radiusFor(neighbor.vehicle_type, config), optionalOrZero(neighbor.acceleration),
                    config.method == Method::kWang, "neighbor:" + neighbor.agent_id,
                    config, robust, result);
    }
    for (auto obstacle : request.obstacles) {
      obstacle.clampRadius();
      addDoublePair(request.ego, obstacle.center,
                    obstacle.velocity.value_or(Eigen::Vector3d::Zero()), obstacle.radius,
                    Eigen::Vector3d::Zero(), false, "obstacle:" + obstacle.obstacle_id,
                    config, robust, result);
    }
    if (request.ego.vehicle_type == VehicleType::kDrone) {
      const double h = config.uav_altitude_floor - request.ego.position.z();
      result.rows.emplace_back((Eigen::Vector3d() << 0.0, 0.0, -1.0).finished());
      result.rhs.push_back(-config.alpha * h + robust);
      result.barriers.push_back(h);
      result.robust_terms.push_back(robust);
      result.row_labels.push_back("altitude");
    }
  }
  const int dimension = unicycle ? 2 : 3;
  if (request.control_bounds) {
    result.lower = request.control_bounds->first.head(dimension);
    result.upper = request.control_bounds->second.head(dimension);
  } else if (unicycle) {
    result.lower = (Eigen::Vector2d() << 0.0, -config.ugv_yaw_rate_limit).finished();
    result.upper = (Eigen::Vector2d() << config.ugv_speed_limit, config.ugv_yaw_rate_limit).finished();
  } else {
    const double limit = request.ego.vehicle_type == VehicleType::kDrone
        ? config.uav_acceleration_limit : config.ugv_acceleration_limit;
    result.lower = Eigen::VectorXd::Constant(dimension, -limit);
    result.upper = Eigen::VectorXd::Constant(dimension, limit);
  }
  return result;
}

}  // namespace hercules_cbf
