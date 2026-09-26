#include "hercules_cbf/constraints.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace hercules_cbf {
namespace {

Eigen::Vector3d optionalOrZero(const std::optional<Eigen::Vector3d>& value) {
  return value.value_or(Eigen::Vector3d::Zero());
}

bool finiteVector(const Eigen::Vector3d& value) {
  return value.array().isFinite().all();
}

double radiusFor(VehicleType type, const CBFConfig& config) {
  return std::max(0.0, type == VehicleType::kDrone ? config.uav_radius : config.ugv_radius);
}

double accelerationLimitFor(VehicleType type, const CBFConfig& config) {
  return type == VehicleType::kDrone
      ? config.uav_acceleration_limit : config.ugv_acceleration_limit;
}

double accelerationLimitFor(const AgentState& state, const CBFConfig& config) {
  if (state.acceleration_limit && std::isfinite(*state.acceleration_limit) &&
      *state.acceleration_limit > 0.0) {
    return *state.acceleration_limit;
  }
  return accelerationLimitFor(state.vehicle_type, config);
}

double configuredMargin(VehicleType type, const CBFConfig& config) {
  return std::max(0.0, type == VehicleType::kDrone ? config.uav_margin : config.ugv_margin);
}

double stateMargin(const AgentState& state, const CBFConfig& config) {
  // The state value is used for locally received class packets.  Taking the
  // maximum with the mission value keeps a packet with an omitted/default
  // field from accidentally disabling the configured robust bound.
  const double local = std::isfinite(state.margin) ? std::max(0.0, state.margin) : 0.0;
  return std::max(local, configuredMargin(state.vehicle_type, config));
}

void markInvalid(ConstraintSet& result, const std::string& reason) {
  result.valid = false;
  if (result.invalid_reason.empty()) result.invalid_reason = reason;
}

struct WangGeometry {
  Eigen::Vector2d q{Eigen::Vector2d::Zero()};
  Eigen::Vector2d w{Eigen::Vector2d::Zero()};
  double distance{0.0};
  double root{0.0};
  double alpha_sum{0.0};
  double dot{0.0};
};

bool wangGeometry(const Eigen::Vector3d& position_i,
                  const Eigen::Vector3d& position_k,
                  const Eigen::Vector3d& velocity_i,
                  const Eigen::Vector3d& velocity_k,
                  double acceleration_limit_i,
                  double acceleration_limit_k,
                  double safe_distance,
                  WangGeometry* geometry) {
  if (!finiteVector(position_i) || !finiteVector(position_k) ||
      !finiteVector(velocity_i) || !finiteVector(velocity_k) ||
      !std::isfinite(acceleration_limit_i) || !std::isfinite(acceleration_limit_k) ||
      !std::isfinite(safe_distance) || acceleration_limit_i <= 0.0 ||
      acceleration_limit_k < 0.0 || safe_distance < 0.0) {
    return false;
  }
  geometry->q = position_i.head<2>() - position_k.head<2>();
  geometry->w = velocity_i.head<2>() - velocity_k.head<2>();
  geometry->distance = geometry->q.norm();
  geometry->alpha_sum = acceleration_limit_i + acceleration_limit_k;
  const double gap = geometry->distance - safe_distance;
  if (!std::isfinite(geometry->distance) || geometry->distance <= 0.0 ||
      !std::isfinite(geometry->alpha_sum) || geometry->alpha_sum <= 0.0 ||
      !std::isfinite(gap) || gap <= 0.0) {
    return false;
  }
  geometry->root = std::sqrt(2.0 * geometry->alpha_sum * gap);
  geometry->dot = geometry->q.dot(geometry->w);
  return std::isfinite(geometry->root) && geometry->root > 0.0 &&
      std::isfinite(geometry->dot);
}

double wangBarrierFromGeometry(const WangGeometry& geometry) {
  return geometry.root + geometry.dot / geometry.distance;
}

double wangRhsFromGeometry(const WangGeometry& geometry, double gamma) {
  if (!std::isfinite(gamma) || gamma < 0.0) return std::numeric_limits<double>::quiet_NaN();
  const double closing_rate = geometry.dot / geometry.distance;
  const double barrier = wangBarrierFromGeometry(geometry);
  return gamma * barrier * geometry.distance - closing_rate * closing_rate +
      geometry.w.dot(geometry.w) + geometry.alpha_sum * geometry.dot / geometry.root;
}

bool wangValues(const AgentState& ego, const AgentState& other,
                double safe_distance, const CBFConfig& config,
                WangGeometry* geometry, double* barrier, double* rhs,
                double* tightening) {
  if (!wangGeometry(ego.position, other.position, ego.velocity, other.velocity,
                    accelerationLimitFor(ego, config),
                    accelerationLimitFor(other, config), safe_distance,
                    geometry)) {
    return false;
  }
  if (ego.learned_acceleration && !finiteVector(*ego.learned_acceleration)) return false;
  if (other.learned_acceleration && !finiteVector(*other.learned_acceleration)) return false;
  *barrier = wangBarrierFromGeometry(*geometry);
  *rhs = wangRhsFromGeometry(*geometry, config.wang_gamma);
  const Eigen::Vector2d learned_i = optionalOrZero(ego.learned_acceleration).head<2>();
  const Eigen::Vector2d learned_k = optionalOrZero(other.learned_acceleration).head<2>();
  *rhs += geometry->q.dot(learned_i - learned_k);
  *tightening = geometry->distance *
      (stateMargin(ego, config) + stateMargin(other, config));
  return std::isfinite(*barrier) && std::isfinite(*rhs) && std::isfinite(*tightening);
}

void addDoublePair(const AgentState& ego, const Eigen::Vector3d& center,
                   const Eigen::Vector3d& velocity, double other_radius,
                   const Eigen::Vector3d& other_acceleration, bool split,
                   const std::string& label, const CBFConfig& config,
                   double uncertainty_value, ConstraintSet& result) {
  const Eigen::Vector3d dp = ego.position - center;
  const Eigen::Vector3d dv = ego.velocity - velocity;
  const double clearance = radiusFor(ego.vehicle_type, config) + other_radius +
      std::max(0.0, config.obstacle_margin);
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
  const double clearance = radiusFor(ego.vehicle_type, config) + other_radius +
      std::max(0.0, config.obstacle_margin);
  const double h = delta.dot(delta) - clearance * clearance;
  const double robust = (2.0 * delta).norm() * uncertainty_value;
  const double b = -config.alpha * h + 2.0 * delta.dot(velocity.head<2>()) + robust;
  result.rows.push_back(2.0 * delta.transpose() * control_map);
  result.rhs.push_back(b);
  result.barriers.push_back(h);
  result.robust_terms.push_back(robust);
  result.row_labels.push_back(label);
}

void addWangPair(const AgentState& ego, const AgentState& neighbor,
                 double safe_distance, const std::string& label,
                 const CBFConfig& config, ConstraintSet& result) {
  WangGeometry geometry;
  double barrier = 0.0;
  double rhs = 0.0;
  double tightening = 0.0;
  if (!wangValues(ego, neighbor, safe_distance, config, &geometry, &barrier,
                  &rhs, &tightening)) {
    markInvalid(result, "invalid_wang_neighbor_geometry:" + neighbor.agent_id);
    return;
  }
  const double weight = strategyBAuthorityWeight(
      accelerationLimitFor(ego, config), accelerationLimitFor(neighbor, config));
  if (!std::isfinite(weight)) {
    markInvalid(result, "invalid_wang_authority");
    return;
  }
  // The reference uses -q^T u_i <= eta_i (b_hat-rho).  ConstraintSet rows
  // use the equivalent lower-bound convention row^T u >= rhs.
  Eigen::Vector3d row = Eigen::Vector3d::Zero();
  row.head<2>() = geometry.q;
  result.rows.push_back(row);
  result.rhs.push_back(-weight * (rhs - tightening));
  result.barriers.push_back(barrier);
  result.robust_terms.push_back(tightening);
  result.row_labels.push_back(label);
}

void addWangObstacle(const AgentState& ego, const ObstacleProxy& obstacle,
                     double safe_distance, const std::string& label,
                     const CBFConfig& config, ConstraintSet& result) {
  AgentState fixed;
  fixed.agent_id = obstacle.obstacle_id;
  fixed.position = obstacle.center;
  fixed.velocity = obstacle.velocity.value_or(Eigen::Vector3d::Zero());
  fixed.vehicle_type = ego.vehicle_type;
  fixed.margin = 0.0;
  fixed.learned_acceleration = Eigen::Vector3d::Zero();
  WangGeometry geometry;
  double rhs = 0.0;
  double ignored_tightening = 0.0;
  if (!wangGeometry(ego.position, fixed.position, ego.velocity, fixed.velocity,
                    accelerationLimitFor(ego, config), 0.0,
                    safe_distance, &geometry) ||
      (ego.learned_acceleration && !finiteVector(*ego.learned_acceleration))) {
    markInvalid(result, "invalid_wang_obstacle_geometry:" + obstacle.obstacle_id);
    return;
  }
  rhs = wangRhsFromGeometry(geometry, config.wang_gamma) +
      geometry.q.dot(optionalOrZero(ego.learned_acceleration).head<2>());
  ignored_tightening = geometry.distance * stateMargin(ego, config);
  if (!std::isfinite(rhs) || !std::isfinite(ignored_tightening)) {
    markInvalid(result, "invalid_wang_obstacle_model:" + obstacle.obstacle_id);
    return;
  }
  Eigen::Vector3d row = Eigen::Vector3d::Zero();
  row.head<2>() = geometry.q;
  result.rows.push_back(row);
  result.rhs.push_back(-(rhs - ignored_tightening));
  result.barriers.push_back(wangBarrierFromGeometry(geometry));
  result.robust_terms.push_back(ignored_tightening);
  result.row_labels.push_back(label);
}

void addWangWall(const AgentState& ego, double wall_y, bool lower_wall,
                 const std::string& label, const CBFConfig& config,
                 ConstraintSet& result) {
  const double radius = radiusFor(ego.vehicle_type, config) +
      std::max(0.0, config.obstacle_margin);
  // m points from the wall into the admissible corridor.  With
  // h=sqrt(2*a*g)+m^T v, the local robust inequality is
  // m^T u >= -gamma*h - a*(m^T v)/sqrt(2*a*g) - m^T d_hat + r.
  const Eigen::Vector2d inward = lower_wall
      ? Eigen::Vector2d(0.0, 1.0) : Eigen::Vector2d(0.0, -1.0);
  const double clearance = lower_wall
      ? ego.position.y() - wall_y - radius
      : wall_y - ego.position.y() - radius;
  const double alpha = accelerationLimitFor(ego, config);
  if (!std::isfinite(wall_y) || !std::isfinite(clearance) || clearance <= 0.0 ||
      !std::isfinite(alpha) || alpha <= 0.0 ||
      (ego.learned_acceleration && !finiteVector(*ego.learned_acceleration))) {
    markInvalid(result, "invalid_wang_corridor_geometry:" + label);
    return;
  }
  const double root = std::sqrt(2.0 * alpha * clearance);
  const double barrier = root + inward.dot(ego.velocity.head<2>());
  const double learned = inward.dot(optionalOrZero(ego.learned_acceleration).head<2>());
  const double rhs = -config.wang_gamma * barrier -
      alpha * inward.dot(ego.velocity.head<2>()) / root - learned +
      stateMargin(ego, config);
  if (!std::isfinite(root) || !std::isfinite(barrier) || !std::isfinite(rhs)) {
    markInvalid(result, "invalid_wang_corridor_model:" + label);
    return;
  }
  Eigen::Vector3d row = Eigen::Vector3d::Zero();
  row.head<2>() = inward;
  result.rows.push_back(row);
  result.rhs.push_back(rhs);
  result.barriers.push_back(barrier);
  result.robust_terms.push_back(stateMargin(ego, config));
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

double wangBarrier(const Eigen::Vector3d& position_i,
                   const Eigen::Vector3d& position_k,
                   const Eigen::Vector3d& velocity_i,
                   const Eigen::Vector3d& velocity_k,
                   double acceleration_limit_i,
                   double acceleration_limit_k,
                   double safe_distance) {
  WangGeometry geometry;
  if (!wangGeometry(position_i, position_k, velocity_i, velocity_k,
                    acceleration_limit_i, acceleration_limit_k, safe_distance,
                    &geometry)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return wangBarrierFromGeometry(geometry);
}

double wangRhs(const Eigen::Vector3d& position_i,
               const Eigen::Vector3d& position_k,
               const Eigen::Vector3d& velocity_i,
               const Eigen::Vector3d& velocity_k,
               double acceleration_limit_i,
               double acceleration_limit_k,
               double safe_distance,
               double gamma) {
  WangGeometry geometry;
  if (!wangGeometry(position_i, position_k, velocity_i, velocity_k,
                    acceleration_limit_i, acceleration_limit_k, safe_distance,
                    &geometry)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return wangRhsFromGeometry(geometry, gamma);
}

double strategyBAuthorityWeight(double own_acceleration_limit,
                                double other_acceleration_limit) {
  if (!std::isfinite(own_acceleration_limit) ||
      !std::isfinite(other_acceleration_limit) || own_acceleration_limit < 0.0 ||
      other_acceleration_limit < 0.0 ||
      own_acceleration_limit + other_acceleration_limit <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return own_acceleration_limit /
      (own_acceleration_limit + other_acceleration_limit);
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
  } else if (config.method == Method::kWang) {
    const double ego_radius = radiusFor(request.ego.vehicle_type, config);
    for (const auto& neighbor : request.neighbors) {
      if (neighbor.vehicle_type != request.ego.vehicle_type) continue;
      const double safe_distance = ego_radius + radiusFor(neighbor.vehicle_type, config) +
          std::max(0.0, config.obstacle_margin);
      addWangPair(request.ego, neighbor, safe_distance,
                  "neighbor:" + neighbor.agent_id, config, result);
    }
    for (auto obstacle : request.obstacles) {
      obstacle.clampRadius();
      const double safe_distance = ego_radius + obstacle.radius +
          std::max(0.0, config.obstacle_margin);
      addWangObstacle(request.ego, obstacle, safe_distance,
                      "obstacle:" + obstacle.obstacle_id, config, result);
    }
    if (std::isfinite(config.wang_corridor_y_min) &&
        std::isfinite(config.wang_corridor_y_max) &&
        config.wang_corridor_y_min >= config.wang_corridor_y_max) {
      markInvalid(result, "invalid_wang_corridor_bounds");
    } else {
      if (std::isfinite(config.wang_corridor_y_min)) {
        addWangWall(request.ego, config.wang_corridor_y_min, true,
                    "wall:y_min", config, result);
      }
      if (std::isfinite(config.wang_corridor_y_max)) {
        addWangWall(request.ego, config.wang_corridor_y_max, false,
                    "wall:y_max", config, result);
      }
    }
  } else {
    for (const auto& neighbor : request.neighbors) {
      if (neighbor.vehicle_type != request.ego.vehicle_type) continue;
      addDoublePair(request.ego, neighbor.position, neighbor.velocity,
                    radiusFor(neighbor.vehicle_type, config), optionalOrZero(neighbor.acceleration),
                    false, "neighbor:" + neighbor.agent_id,
                    config, robust, result);
    }
    for (auto obstacle : request.obstacles) {
      obstacle.clampRadius();
      addDoublePair(request.ego, obstacle.center,
                    obstacle.velocity.value_or(Eigen::Vector3d::Zero()), obstacle.radius,
                    Eigen::Vector3d::Zero(), false, "obstacle:" + obstacle.obstacle_id,
                    config, robust, result);
    }
  }
  if (!unicycle && request.ego.vehicle_type == VehicleType::kDrone) {
    const double h = config.uav_altitude_floor - request.ego.position.z();
    result.rows.emplace_back((Eigen::Vector3d() << 0.0, 0.0, -1.0).finished());
    result.rhs.push_back(-config.alpha * h + robust);
    result.barriers.push_back(h);
    result.robust_terms.push_back(robust);
    result.row_labels.push_back("altitude");
  }
  const int dimension = unicycle ? 2 : 3;
  if (request.control_bounds) {
    result.lower = request.control_bounds->first.head(dimension);
    result.upper = request.control_bounds->second.head(dimension);
  } else if (unicycle) {
    result.lower = (Eigen::Vector2d() << 0.0, -config.ugv_yaw_rate_limit).finished();
    result.upper = (Eigen::Vector2d() << config.ugv_speed_limit, config.ugv_yaw_rate_limit).finished();
  } else {
    const double limit = accelerationLimitFor(request.ego, config);
    result.lower = Eigen::VectorXd::Constant(dimension, -limit);
    result.upper = Eigen::VectorXd::Constant(dimension, limit);
  }
  return result;
}

}  // namespace hercules_cbf
