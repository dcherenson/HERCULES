#include "hercules_localization/gs_ci.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Geometry>

#include "hercules_localization/covariance.hpp"

namespace hercules_localization {
namespace {

PoseEstimate sanitizedEstimate(const PoseEstimate& estimate, double floor) {
  PoseEstimate result = estimate;
  result.mean = normalizePose(estimate.mean);
  result.covariance = regularizeCovariance(estimate.covariance, floor);
  if (!std::isfinite(result.timestamp)) result.timestamp = 0.0;
  return result;
}

bool finiteMeasurement(const RangeBearingMeasurement& measurement) {
  if (!measurement.valid || !std::isfinite(measurement.range) ||
      !std::isfinite(measurement.bearing) || measurement.range <= 0.0 ||
      !measurement.covariance.allFinite()) return false;
  const Eigen::Matrix2d symmetric =
      0.5 * (measurement.covariance + measurement.covariance.transpose());
  if ((measurement.covariance - measurement.covariance.transpose()).norm() >
      1e-8 * (1.0 + measurement.covariance.norm())) return false;
  const Eigen::LLT<Eigen::Matrix2d> factor(symmetric);
  return factor.info() == Eigen::Success &&
         factor.matrixL().toDenseMatrix().diagonal().array().all() > 0.0;
}

bool finitePoseCovariance(const PoseCovariance& covariance) {
  if (!covariance.allFinite()) return false;
  const PoseCovariance symmetric = 0.5 * (covariance + covariance.transpose());
  if ((covariance - covariance.transpose()).norm() >
      1e-8 * (1.0 + covariance.norm())) return false;
  const Eigen::LLT<PoseCovariance> factor(symmetric);
  return factor.info() == Eigen::Success &&
         factor.matrixL().toDenseMatrix().diagonal().array().all() > 0.0;
}

PoseEstimate poseCandidate(const PoseEstimate& current,
                           const RelativeObservation& observation,
                           const LocalizationConfig& config) {
  PoseEstimate candidate = sanitizedEstimate(current, config.covariance_floor);
  const Pose2D neighbor = normalizePose(observation.neighbor.mean);
  const RangeBearing measurement = observation.measurement.value();
  const Position2D relative = polarToCartesian(measurement);
  candidate.mean.position = neighbor.position - rotate(relative, current.mean.yaw);
  candidate.mean.yaw = current.mean.yaw;
  candidate.mean = normalizePose(candidate.mean);

  const double floor = config.covariance_floor;
  const double measurement_floor =
      std::isfinite(config.measurement_covariance_floor) &&
              config.measurement_covariance_floor > 0.0
          ? config.measurement_covariance_floor
          : floor;
  Eigen::Matrix2d measurement_covariance =
      regularizeCovariance(observation.measurement.covariance, measurement_floor);
  const double margin = std::isfinite(config.robustness_margin)
                            ? std::max(0.0, config.robustness_margin)
                            : 0.0;
  const double inflation = std::isfinite(config.covariance_inflation)
                               ? std::max(0.0, config.covariance_inflation)
                               : 0.0;
  measurement_covariance = inflateCovariance(
      measurement_covariance, std::hypot(margin, inflation), measurement_floor);

  // Candidate position = neighbor position - R(yaw) * polar(range,bearing).
  // The observer yaw is retained by the candidate, but its uncertainty still
  // contributes to the position covariance through this Jacobian.
  const Eigen::Matrix2d rotation =
      (Eigen::Rotation2Dd(current.mean.yaw)).toRotationMatrix();
  Eigen::Matrix<double, 2, 2> measurement_jacobian =
      -rotation * polarJacobian(measurement);
  const Position2D rotated_relative = rotation * relative;
  Eigen::Vector2d yaw_jacobian;
  yaw_jacobian << rotated_relative.y(), -rotated_relative.x();

  const PoseCovariance current_covariance =
      regularizeCovariance(current.covariance, floor);
  const PoseCovariance neighbor_covariance =
      config.lambda * regularizeCovariance(observation.neighbor.covariance, floor);
  PoseCovariance covariance = PoseCovariance::Zero();
  covariance(0, 0) = neighbor_covariance(0, 0);
  covariance(0, 1) = neighbor_covariance(0, 1);
  covariance(1, 0) = neighbor_covariance(1, 0);
  covariance(1, 1) = neighbor_covariance(1, 1);
  covariance.topLeftCorner<2, 2>() +=
      measurement_jacobian * measurement_covariance * measurement_jacobian.transpose();
  covariance.topLeftCorner<2, 2>() +=
      current_covariance(2, 2) * (yaw_jacobian * yaw_jacobian.transpose());
  // The heading is not observed by a scalar range/bearing measurement without
  // an orientation-bearing model.  Carry its prior uncertainty and preserve
  // cross-covariances instead of inventing information about it.
  covariance(2, 2) = current_covariance(2, 2);
  covariance(0, 2) = current_covariance(0, 2);
  covariance(1, 2) = current_covariance(1, 2);
  covariance(2, 0) = covariance(0, 2);
  covariance(2, 1) = covariance(1, 2);
  candidate.covariance = regularizeCovariance(covariance, floor);
  candidate.timestamp = std::max(
      current.timestamp,
      std::max(std::isfinite(observation.measurement.timestamp)
                   ? observation.measurement.timestamp
                   : current.timestamp,
               std::isfinite(observation.neighbor.timestamp)
                   ? observation.neighbor.timestamp
                   : current.timestamp));
  candidate.valid = true;
  return candidate;
}

PoseEstimate fuseAtWeight(const PoseEstimate& first, const PoseEstimate& second,
                          double weight, double floor) {
  PoseEstimate result;
  const PoseEstimate a = sanitizedEstimate(first, floor);
  const PoseEstimate b = sanitizedEstimate(second, floor);
  const PoseVector a_state = a.vector();
  PoseVector b_state = b.vector();
  b_state.z() = a_state.z() + angleDifference(b_state.z(), a_state.z());
  const PoseCovariance a_inverse = safeInverse(a.covariance, floor);
  const PoseCovariance b_inverse = safeInverse(b.covariance, floor);
  const PoseCovariance information =
      weight * a_inverse + (1.0 - weight) * b_inverse;
  const PoseCovariance covariance = safeInverse(information, floor);
  const PoseVector information_vector =
      weight * a_inverse * a_state + (1.0 - weight) * b_inverse * b_state;
  result.mean = normalizePose(Pose2D::fromVector(covariance * information_vector));
  result.covariance = regularizeCovariance(covariance, floor);
  result.timestamp = std::max(a.timestamp, b.timestamp);
  result.valid = a.valid || b.valid;
  result.agent_id = !a.agent_id.empty() ? a.agent_id : b.agent_id;
  return result;
}

PoseEstimate fuseBatch(const std::vector<PoseEstimate>& beliefs,
                        const std::vector<double>& requested_weights,
                        double floor) {
  PoseEstimate result;
  if (beliefs.empty()) return result;
  const std::size_t count = beliefs.size();
  std::vector<double> weights(count, 0.0);
  double weight_sum = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    const double value = index < requested_weights.size()
                             ? std::max(0.0, requested_weights[index])
                             : 0.0;
    weights[index] = std::isfinite(value) ? value : 0.0;
    weight_sum += weights[index];
  }
  if (weight_sum <= 0.0) {
    weights[0] = 1.0;
    weight_sum = 1.0;
  }
  for (double& value : weights) value /= weight_sum;

  const PoseEstimate anchor = sanitizedEstimate(beliefs.front(), floor);
  const PoseVector anchor_state = anchor.vector();
  PoseCovariance information = PoseCovariance::Zero();
  PoseVector information_vector = PoseVector::Zero();
  for (std::size_t index = 0; index < count; ++index) {
    const PoseEstimate belief = sanitizedEstimate(beliefs[index], floor);
    const PoseCovariance inverse = safeInverse(belief.covariance, floor);
    PoseVector state = belief.vector();
    state.z() = anchor_state.z() + angleDifference(state.z(), anchor_state.z());
    information += weights[index] * inverse;
    information_vector += weights[index] * inverse * state;
  }
  const PoseCovariance covariance = safeInverse(information, floor);
  result.mean = normalizePose(Pose2D::fromVector(covariance * information_vector));
  result.covariance = regularizeCovariance(covariance, floor);
  result.timestamp = 0.0;
  result.valid = true;
  for (const auto& belief : beliefs) {
    result.timestamp = std::max(result.timestamp, belief.timestamp);
    result.agent_id = result.agent_id.empty() ? belief.agent_id : result.agent_id;
  }
  return result;
}

double objective(const PoseEstimate& estimate, CovarianceIntersectionCost cost,
                 double floor) {
  if (cost == CovarianceIntersectionCost::kTrace) {
    return estimate.covariance.trace();
  }
  return logDeterminant(estimate.covariance, floor);
}

}  // namespace

CovarianceIntersectionResult covarianceIntersection(
    const PoseEstimate& first, const PoseEstimate& second,
    const LocalizationConfig& config) {
  CovarianceIntersectionResult result;
  if (!first.valid && !second.valid) {
    result.status = "invalid_inputs";
    return result;
  }
  if ((first.valid && (!first.mean.position.allFinite() || !std::isfinite(first.mean.yaw))) ||
      (second.valid && (!second.mean.position.allFinite() || !std::isfinite(second.mean.yaw)))) {
    result.status = "nonfinite_inputs";
    return result;
  }
  if ((first.valid && !finitePoseCovariance(first.covariance)) ||
      (second.valid && !finitePoseCovariance(second.covariance))) {
    result.status = "invalid_covariance";
    return result;
  }
  if (!first.valid) {
    result.estimate = sanitizedEstimate(second, config.covariance_floor);
    result.weight = 0.0;
    result.objective = objective(result.estimate, config.ci_cost,
                                 config.covariance_floor);
    result.success = true;
    result.status = "first_invalid";
    return result;
  }
  if (!second.valid) {
    result.estimate = sanitizedEstimate(first, config.covariance_floor);
    result.weight = 1.0;
    result.objective = objective(result.estimate, config.ci_cost,
                                 config.covariance_floor);
    result.success = true;
    result.status = "second_invalid";
    return result;
  }

  const double floor = config.covariance_floor;
  const int requested_points = std::max(3, config.ci_grid_points);
  const PoseEstimate a = sanitizedEstimate(first, floor);
  const PoseEstimate b = sanitizedEstimate(second, floor);
  const bool equal_covariance =
      (a.covariance - b.covariance).norm() <= 1e-12 *
                                               (1.0 + a.covariance.norm());
  double best_weight = equal_covariance ? 0.5 : 0.0;
  PoseEstimate best = fuseAtWeight(a, b, best_weight, floor);
  double best_objective = objective(best, config.ci_cost, floor);

  for (int index = 0; index < requested_points; ++index) {
    const double weight = static_cast<double>(index) /
                          static_cast<double>(requested_points - 1);
    if (equal_covariance && std::abs(weight - 0.5) < 0.5) continue;
    const PoseEstimate candidate = fuseAtWeight(a, b, weight, floor);
    const double candidate_objective = objective(candidate, config.ci_cost, floor);
    if (candidate_objective < best_objective - 1e-14 ||
        (std::abs(candidate_objective - best_objective) <= 1e-14 &&
         std::abs(weight - 0.5) < std::abs(best_weight - 0.5))) {
      best_weight = weight;
      best = candidate;
      best_objective = candidate_objective;
    }
  }

  // Refine only around an interior grid point.  End points are retained as
  // candidates because the optimum of the CI objective often lies there.
  if (best_weight > 0.0 && best_weight < 1.0 && !equal_covariance) {
    const double half_step = 1.0 / static_cast<double>(requested_points - 1);
    double left = std::max(0.0, best_weight - half_step);
    double right = std::min(1.0, best_weight + half_step);
    constexpr double golden = 0.6180339887498948482;
    double x1 = right - golden * (right - left);
    double x2 = left + golden * (right - left);
    double f1 = objective(fuseAtWeight(a, b, x1, floor), config.ci_cost, floor);
    double f2 = objective(fuseAtWeight(a, b, x2, floor), config.ci_cost, floor);
    for (int iteration = 0; iteration < 32; ++iteration) {
      if (f1 > f2) {
        left = x1;
        x1 = x2;
        f1 = f2;
        x2 = left + golden * (right - left);
        f2 = objective(fuseAtWeight(a, b, x2, floor), config.ci_cost, floor);
      } else {
        right = x2;
        x2 = x1;
        f2 = f1;
        x1 = right - golden * (right - left);
        f1 = objective(fuseAtWeight(a, b, x1, floor), config.ci_cost, floor);
      }
    }
    const double refined_weight = 0.5 * (left + right);
    const PoseEstimate refined = fuseAtWeight(a, b, refined_weight, floor);
    const double refined_objective = objective(refined, config.ci_cost, floor);
    if (refined_objective < best_objective) {
      best_weight = refined_weight;
      best = refined;
      best_objective = refined_objective;
    }
  }

  result.estimate = best;
  result.weight = best_weight;
  result.objective = best_objective;
  result.success = true;
  result.status = "ok";
  return result;
}

CovarianceIntersectionResult covarianceIntersectionFuse(
    const PoseEstimate& first, const PoseEstimate& second,
    const LocalizationConfig& config) {
  return covarianceIntersection(first, second, config);
}

LocalizationResult gsCiUpdate(
    const PoseEstimate& prior,
    const std::vector<RelativeObservation>& observations,
    const LocalizationConfig& config) {
  LocalizationResult result;
  result.estimate = sanitizedEstimate(prior, config.covariance_floor);
  if (!std::isfinite(config.ci_self_weight) || config.ci_self_weight < 0.0 ||
      config.ci_self_weight > 1.0) {
    result.status = "invalid_ci_weight";
    result.message = "ci_self_weight must be finite and in [0, 1]";
    return result;
  }
  if (!std::isfinite(config.lambda) || config.lambda < 0.0 || config.lambda > 1.0) {
    result.status = "invalid_lambda";
    result.message = "lambda must be finite and in [0, 1]";
    return result;
  }
  if (!prior.valid) {
    result.status = "invalid_prior";
    result.message = "the local prior estimate is marked invalid";
    return result;
  }
  if (!prior.mean.position.allFinite() || !std::isfinite(prior.mean.yaw)) {
    result.status = "nonfinite_prior";
    result.message = "the local prior pose contains nonfinite values";
    return result;
  }
  if (!finitePoseCovariance(prior.covariance)) {
    result.status = "invalid_prior_covariance";
    result.message = "the local prior covariance is not SPD";
    return result;
  }
  if (observations.empty()) {
    result.success = true;
    result.status = "no_measurements";
    result.message = "GS-CI had no observations";
    return result;
  }

  PoseEstimate current = result.estimate;
  const int max_iterations = std::max(0, config.max_iterations);
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    const PoseEstimate before = current;
    std::vector<PoseEstimate> beliefs{current};
    std::vector<double> weights{config.ci_self_weight};
    std::size_t valid_observations = 0;
    for (const RelativeObservation& observation : observations) {
      if (!observation.neighbor.valid || !finiteMeasurement(observation.measurement)) {
        if (iteration == 0) ++result.rejected_measurements;
        continue;
      }
      if (!finitePoseCovariance(observation.neighbor.covariance)) {
        if (iteration == 0) ++result.rejected_measurements;
        continue;
      }
      const PoseEstimate candidate = poseCandidate(current, observation, config);
      beliefs.push_back(candidate);
      ++valid_observations;
      const RangeBearing predicted =
          predictRangeBearing(current.mean, observation.neighbor.mean,
                              config.range_epsilon);
      const double innovation_norm = std::hypot(
          observation.measurement.range - predicted.range,
          angleDifference(observation.measurement.bearing, predicted.bearing));
      result.maximum_innovation =
          std::max(result.maximum_innovation, innovation_norm);
    }
    if (valid_observations > 0) {
      const double peer_weight =
          (1.0 - config.ci_self_weight) / static_cast<double>(valid_observations);
      weights.resize(beliefs.size(), peer_weight);
      current = fuseBatch(beliefs, weights, config.covariance_floor);
      if (iteration == 0) {
        result.used_measurements += static_cast<int>(valid_observations);
        result.ci_weights.push_back(config.ci_self_weight);
        for (std::size_t index = 0; index < valid_observations; ++index) {
          result.ci_weights.push_back(peer_weight);
        }
      }
    }
    result.iterations = iteration + 1;
    const PoseVector delta = current.vector() - before.vector();
    const double pose_change =
        std::hypot(std::hypot(delta.x(), delta.y()),
                   angleDifference(current.mean.yaw, before.mean.yaw));
    const double covariance_change =
        (current.covariance - before.covariance).norm();
    if (pose_change + covariance_change <= config.convergence_tolerance) break;
  }
  result.estimate = current;
  result.success = true;
  result.status = result.used_measurements > 0 ? "ok" : "no_valid_measurements";
  result.message = result.used_measurements > 0
                       ? "GS-CI update complete"
                       : "all observations were rejected";
  return result;
}

LocalizationResult gaussSeidelCovarianceIntersection(
    const PoseEstimate& prior,
    const std::vector<RelativeObservation>& observations,
    const LocalizationConfig& config) {
  return gsCiUpdate(prior, observations, config);
}

GsCiLocalizer::GsCiLocalizer(PoseEstimate initial, LocalizationConfig config)
    : estimate_(sanitizedEstimate(initial, config.covariance_floor)),
      config_(std::move(config)) {}

void GsCiLocalizer::reset(const PoseEstimate& estimate) {
  estimate_ = sanitizedEstimate(estimate, config_.covariance_floor);
}

LocalizationResult GsCiLocalizer::update(
    const std::vector<RelativeObservation>& observations) {
  const LocalizationResult result = gsCiUpdate(estimate_, observations, config_);
  estimate_ = result.estimate;
  return result;
}

LocalizationResult GsCiLocalizer::update(
    const PoseEstimate& prior,
    const std::vector<RelativeObservation>& observations) {
  reset(prior);
  return update(observations);
}

}  // namespace hercules_localization
