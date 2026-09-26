#include "hercules_localization/recursive_localization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <Eigen/Cholesky>

#include "hercules_localization/covariance.hpp"

namespace hercules_localization {
namespace {

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

bool finiteCovariance(const PoseCovariance& covariance) {
  if (!covariance.allFinite()) return false;
  const PoseCovariance symmetric = 0.5 * (covariance + covariance.transpose());
  if ((covariance - covariance.transpose()).norm() >
      1e-8 * (1.0 + covariance.norm())) return false;
  const Eigen::LLT<PoseCovariance> factor(symmetric);
  return factor.info() == Eigen::Success &&
         factor.matrixL().toDenseMatrix().diagonal().array().all() > 0.0;
}

PoseEstimate sanitizedEstimate(const PoseEstimate& estimate, double floor) {
  PoseEstimate result = estimate;
  result.mean = normalizePose(estimate.mean);
  result.covariance = regularizeCovariance(estimate.covariance, floor);
  if (!std::isfinite(result.timestamp)) result.timestamp = 0.0;
  return result;
}

double timestampFor(const PoseEstimate& prior, const RelativeObservation& observation) {
  double timestamp = prior.timestamp;
  if (std::isfinite(observation.measurement.timestamp)) {
    timestamp = std::max(timestamp, observation.measurement.timestamp);
  }
  if (std::isfinite(observation.neighbor.timestamp)) {
    timestamp = std::max(timestamp, observation.neighbor.timestamp);
  }
  return timestamp;
}

}  // namespace

ObservationUpdate updateFromRangeBearing(const PoseEstimate& prior,
                                         const RelativeObservation& observation,
                                         const LocalizationConfig& config) {
  ObservationUpdate result;
  const double floor = config.covariance_floor;
  result.estimate = sanitizedEstimate(prior, floor);

  if (!prior.valid) {
    result.status = "invalid_prior";
    return result;
  }
  if (!prior.mean.position.allFinite() || !std::isfinite(prior.mean.yaw)) {
    result.status = "nonfinite_prior";
    return result;
  }
  if (!std::isfinite(config.lambda) || config.lambda < 0.0 || config.lambda > 1.0) {
    result.status = "invalid_lambda";
    return result;
  }
  if (!observation.neighbor.valid) {
    result.status = "invalid_neighbor";
    return result;
  }
  if (!observation.neighbor.mean.position.allFinite() ||
      !std::isfinite(observation.neighbor.mean.yaw)) {
    result.status = "nonfinite_neighbor";
    return result;
  }
  if (!finiteMeasurement(observation.measurement)) {
    result.status = "invalid_measurement";
    return result;
  }
  if (!finiteCovariance(observation.neighbor.covariance)) {
    result.status = "invalid_neighbor_covariance";
    return result;
  }

  const Pose2D& observer = result.estimate.mean;
  const Pose2D neighbor = normalizePose(observation.neighbor.mean);
  const RangeBearing predicted = predictRangeBearing(observer, neighbor, config.range_epsilon);
  result.innovation << observation.measurement.range - predicted.range,
      angleDifference(observation.measurement.bearing, predicted.bearing);
  if (!result.innovation.allFinite()) {
    result.status = "nonfinite_innovation";
    return result;
  }

  const Eigen::Matrix<double, 2, 3> h_observer =
      rangeBearingJacobianObserver(observer, neighbor, config.range_epsilon);
  const Eigen::Matrix<double, 2, 3> h_neighbor =
      rangeBearingJacobianTarget(observer, neighbor, config.range_epsilon);
  const Eigen::Matrix3d prior_covariance =
      regularizeCovariance(result.estimate.covariance, floor);
  const Eigen::Matrix3d neighbor_covariance =
      regularizeCovariance(observation.neighbor.covariance, floor);

  const double measurement_floor = std::isfinite(config.measurement_covariance_floor) &&
                                           config.measurement_covariance_floor > 0.0
                                       ? config.measurement_covariance_floor
                                       : floor;
  Eigen::Matrix2d measurement_covariance =
      regularizeCovariance(observation.measurement.covariance, measurement_floor);
  measurement_covariance = measurementCovarianceForConfig(
      measurement_covariance, config, measurement_floor);
  // Treat the neighbor state as uncertain rather than as an independent exact
  // anchor.  This is the key consistency property of the recursive update.
  const Eigen::Matrix2d projected_neighbor = config.lambda *
      (h_neighbor * neighbor_covariance * h_neighbor.transpose());
  const Eigen::Matrix2d effective_measurement_raw =
      measurement_covariance + projected_neighbor;
  const Eigen::Matrix2d effective_measurement =
      regularizeCovariance(effective_measurement_raw, measurement_floor);
  const Eigen::Matrix2d innovation_covariance_raw =
      h_observer * prior_covariance * h_observer.transpose() +
      effective_measurement;
  result.innovation_covariance =
      regularizeCovariance(innovation_covariance_raw, floor);
  result.mahalanobis_squared = mahalanobisSquared(
      result.innovation, result.innovation_covariance, floor);

  const double gate = config.innovation_gate;
  if (std::isfinite(gate) && gate > 0.0 && result.mahalanobis_squared > gate) {
    result.status = "innovation_gated";
    return result;
  }

  const Eigen::Matrix2d inverse_innovation =
      safeInverse(result.innovation_covariance, floor);
  const Eigen::Matrix<double, 3, 2> gain =
      prior_covariance * h_observer.transpose() * inverse_innovation;
  const PoseVector updated_vector =
      result.estimate.vector() + gain * result.innovation;
  result.estimate.mean = normalizePose(Pose2D::fromVector(updated_vector));
  const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d residual_map = identity - gain * h_observer;
  // Joseph form remains positive semidefinite when the innovation covariance
  // has been regularized and is safer than (I-KH)P for ill-conditioned data.
  if (config.joseph_form) {
    const Eigen::Matrix3d joseph_covariance =
        residual_map * prior_covariance * residual_map.transpose() +
        gain * effective_measurement * gain.transpose();
    result.estimate.covariance =
        regularizeCovariance(joseph_covariance, floor);
  } else {
    const Eigen::Matrix3d simple_covariance = residual_map * prior_covariance;
    result.estimate.covariance = regularizeCovariance(simple_covariance, floor);
  }
  result.estimate.timestamp = timestampFor(prior, observation);
  result.estimate.valid = true;
  result.accepted = true;
  result.valid = true;
  result.status = "accepted";
  return result;
}

LocalizationResult recursiveDecentralizedUpdate(
    const PoseEstimate& prior,
    const std::vector<RelativeObservation>& observations,
    const LocalizationConfig& config) {
  LocalizationResult result;
  result.estimate = sanitizedEstimate(prior, config.covariance_floor);
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

  PoseEstimate current = result.estimate;
  for (const RelativeObservation& observation : observations) {
    const ObservationUpdate update = updateFromRangeBearing(current, observation, config);
    if (update.accepted) {
      current = update.estimate;
      ++result.used_measurements;
      result.maximum_innovation =
          std::max(result.maximum_innovation, update.innovation.norm());
      result.maximum_mahalanobis_squared = std::max(
          result.maximum_mahalanobis_squared, update.mahalanobis_squared);
    } else {
      ++result.rejected_measurements;
    }
  }
  result.estimate = current;
  result.success = true;
  result.status = observations.empty() ? "no_measurements"
                                       : (result.used_measurements > 0
                                              ? "ok"
                                              : "no_valid_measurements");
  result.message = result.used_measurements == 0 && !observations.empty()
                       ? "all observations were rejected"
                       : "recursive update complete";
  result.iterations = result.used_measurements;
  return result;
}

LocalizationResult recursiveDecentralizedLocalization(
    const PoseEstimate& prior,
    const std::vector<RelativeObservation>& observations,
    const LocalizationConfig& config) {
  return recursiveDecentralizedUpdate(prior, observations, config);
}

LocalizationResult recursiveDecentralizedUpdate(
    const PoseEstimate& prior,
    const std::vector<RecursivePairTransaction>& transactions,
    const LocalizationConfig& config) {
  std::vector<RelativeObservation> observations;
  observations.reserve(transactions.size());
  for (const auto& transaction : transactions) {
    RelativeObservation observation;
    observation.measurement = transaction.measurement;
    observation.neighbor = transaction.observed;
    if (observation.measurement.neighbor_id.empty()) {
      observation.measurement.neighbor_id = transaction.observed_id;
    }
    observations.push_back(std::move(observation));
  }
  return recursiveDecentralizedUpdate(prior, observations, config);
}

RecursiveDecentralizedLocalizer::RecursiveDecentralizedLocalizer(
    PoseEstimate initial, LocalizationConfig config)
    : estimate_(sanitizedEstimate(initial, config.covariance_floor)),
      config_(std::move(config)) {}

RecursiveDecentralizedState::RecursiveDecentralizedState(
    std::string agent_id, PoseEstimate initial, LocalizationConfig config)
    : agent_id_(std::move(agent_id)),
      estimate_(sanitizedEstimate(initial, config.covariance_floor)),
      config_(std::move(config)) {}

void RecursiveDecentralizedState::reset(const PoseEstimate& estimate) {
  estimate_ = sanitizedEstimate(estimate, config_.covariance_floor);
  factors_.clear();
  processed_events_.clear();
  latest_sequences_.clear();
}

void RecursiveDecentralizedState::initializePeer(
    const std::string& peer_id, const Eigen::Matrix3d& factor) {
  if (peer_id.empty() || peer_id == agent_id_) return;
  factors_[peer_id] = factor.allFinite() ? factor : Eigen::Matrix3d::Zero();
}

void RecursiveDecentralizedState::propagate(
    const Twist2D& twist, double dt, const Eigen::Matrix3d& process_covariance) {
  const Eigen::Matrix3d transition = motionJacobian(estimate_.mean, twist, dt);
  const double process_floor = std::isfinite(config_.process_covariance_floor) &&
                                       config_.process_covariance_floor > 0.0
                                   ? config_.process_covariance_floor
                                   : config_.covariance_floor;
  const Eigen::Matrix3d process =
      regularizeCovariance(process_covariance, process_floor);
  estimate_ = hercules_localization::propagate(
      estimate_, twist, dt, process, config_.covariance_floor);
  // Ordinary private motion follows the paper's factor propagation rule
  // sigma_ij+ = G_i sigma_ij.  Lambda belongs only to the nonparticipant
  // correction after a pair update (equation 26), not to motion propagation.
  for (auto& item : factors_) {
    item.second = transition * item.second;
    if (!item.second.allFinite()) item.second.setZero();
  }
}

void RecursiveDecentralizedState::propagateWorldDelta(
    const PoseVector& world_delta, const Eigen::Matrix3d& process_covariance,
    double timestamp) {
  if (!world_delta.allFinite() || !process_covariance.allFinite()) {
    estimate_.valid = false;
    return;
  }
  estimate_.mean.position += world_delta.head<2>();
  estimate_.mean.yaw = wrapAngle(estimate_.mean.yaw + world_delta.z());
  const Eigen::Matrix3d propagated_covariance = estimate_.covariance +
      regularizeCovariance(process_covariance,
                           config_.process_covariance_floor);
  estimate_.covariance = regularizeCovariance(
      propagated_covariance, config_.covariance_floor);
  if (std::isfinite(timestamp)) estimate_.timestamp = timestamp;
}

bool RecursiveDecentralizedState::applyPrivatePositionMeasurement(
    const Eigen::Vector2d& position, const Eigen::Matrix2d& covariance,
    double timestamp) {
  if (!estimate_.valid || !position.allFinite() || !covariance.allFinite()) return false;
  const Eigen::Matrix<double, 2, 3> h =
      (Eigen::Matrix<double, 2, 3>() << 1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0).finished();
  const Eigen::Matrix3d prior = regularizeCovariance(
      estimate_.covariance, config_.covariance_floor);
  const Eigen::Matrix2d noise = regularizeCovariance(
      covariance, config_.measurement_covariance_floor);
  const Eigen::Matrix2d innovation_covariance = regularizeCovariance(
      Eigen::Matrix2d(h * prior * h.transpose() + noise),
      config_.covariance_floor);
  const Eigen::Matrix<double, 3, 2> gain = prior * h.transpose() *
      safeInverse(innovation_covariance, config_.covariance_floor);
  const Eigen::Vector2d innovation = position - estimate_.mean.position;
  estimate_.mean = Pose2D::fromVector(estimate_.vector() + gain * innovation);
  estimate_.mean.yaw = wrapAngle(estimate_.mean.yaw);
  const Eigen::Matrix3d multiplier = Eigen::Matrix3d::Identity() - gain * h;
  estimate_.covariance = regularizeCovariance(
      Eigen::Matrix3d(multiplier * prior * multiplier.transpose() +
                      gain * noise * gain.transpose()),
      config_.covariance_floor);
  for (auto& [peer_id, factor] : factors_) {
    (void)peer_id;
    factor = multiplier * factor;
  }
  if (std::isfinite(timestamp)) estimate_.timestamp = timestamp;
  return true;
}

bool RecursiveDecentralizedState::replaceInitialYaw(double yaw, double variance) {
  if (!std::isfinite(yaw) || !std::isfinite(variance) || variance <= 0.0 ||
      !estimate_.valid) {
    return false;
  }
  estimate_.mean.yaw = wrapAngle(yaw);
  estimate_.covariance(0, 2) = estimate_.covariance(2, 0) = 0.0;
  estimate_.covariance(1, 2) = estimate_.covariance(2, 1) = 0.0;
  estimate_.covariance(2, 2) = variance;
  estimate_.covariance = regularizeCovariance(
      estimate_.covariance, config_.covariance_floor);
  return true;
}

RecursiveTransactionResult RecursiveDecentralizedState::applyPairTransaction(
    const RecursivePairTransaction& transaction) {
  RecursiveTransactionResult output;
  output.result.estimate = estimate_;
  if (!std::isfinite(config_.lambda) || config_.lambda < 0.0 ||
      config_.lambda > 1.0) {
    output.rejected = true;
    output.status = "invalid_lambda";
    output.result.status = output.status;
    return output;
  }
  const std::string observer = transaction.observer_id;
  const std::string observed = transaction.observed_id.empty()
                                   ? transaction.measurement.neighbor_id
                                   : transaction.observed_id;
  if (observer.empty() || (!agent_id_.empty() && observer != agent_id_) ||
      observed.empty() || observed == agent_id_) {
    output.rejected = true;
    output.status = "invalid_pair";
    output.result.status = output.status;
    return output;
  }
  const std::string event_id = transaction.event_id;
  const auto sequence = latest_sequences_.find(observed);
  const bool duplicate_event = !event_id.empty() && processed_events_.count(event_id);
  const bool duplicate_sequence = sequence != latest_sequences_.end() &&
                                  transaction.sequence != 0 &&
                                  transaction.sequence <= sequence->second;
  if (duplicate_event || duplicate_sequence) {
    output.accepted = true;
    output.duplicate = true;
    output.status = "duplicate";
    output.result.success = true;
    output.result.status = output.status;
    return output;
  }

  RelativeObservation observation;
  observation.measurement = transaction.measurement;
  observation.measurement.neighbor_id = observed;
  observation.neighbor = transaction.observed;
  if (!estimate_.valid || !transaction.observed.valid ||
      !finiteCovariance(estimate_.covariance) ||
      !finiteCovariance(transaction.observed.covariance) ||
      !finiteMeasurement(transaction.measurement)) {
    output.rejected = true;
    output.status = "invalid_pair_state";
    output.result.status = output.status;
    return output;
  }

  if (factors_.find(observed) == factors_.end()) initializePeer(observed);
  const Eigen::Matrix3d local_factor =
      transaction.correlation_factor.norm() > 0.0
          ? transaction.correlation_factor
          : factors_[observed];
  const Eigen::Matrix3d reciprocal_factor = transaction.reciprocal_factor;
  if (!local_factor.allFinite() || !reciprocal_factor.allFinite()) {
    output.rejected = true;
    output.status = "invalid_correlation_factor";
    output.result.status = output.status;
    return output;
  }
  const Eigen::Matrix3d cross_covariance =
      local_factor * reciprocal_factor.transpose();
  Eigen::Matrix<double, 6, 6> joint_covariance;
  joint_covariance << estimate_.covariance, cross_covariance,
      cross_covariance.transpose(), transaction.observed.covariance;
  const Eigen::LLT<Eigen::Matrix<double, 6, 6>> joint_factor(joint_covariance);
  if (joint_factor.info() != Eigen::Success) {
    output.rejected = true;
    output.status = "non_spd_pair_covariance";
    output.result.status = output.status;
    return output;
  }

  const RangeBearing predicted = predictRangeBearing(
      estimate_.mean, transaction.observed.mean, config_.range_epsilon);
  Eigen::Vector2d innovation;
  innovation << transaction.measurement.range - predicted.range,
      angleDifference(transaction.measurement.bearing, predicted.bearing);
  Eigen::Matrix<double, 2, 6> measurement_jacobian;
  measurement_jacobian <<
      rangeBearingJacobianObserver(estimate_.mean, transaction.observed.mean,
                                   config_.range_epsilon),
      rangeBearingJacobianTarget(estimate_.mean, transaction.observed.mean,
                                 config_.range_epsilon);
  const double measurement_floor =
      config_.measurement_covariance_floor > 0.0
          ? config_.measurement_covariance_floor
          : config_.covariance_floor;
  Eigen::Matrix2d measurement_covariance = regularizeCovariance(
      transaction.measurement.covariance, measurement_floor);
  measurement_covariance = measurementCovarianceForConfig(
      measurement_covariance, config_, measurement_floor);
  const Eigen::Matrix2d innovation_covariance = regularizeCovariance(
      Eigen::Matrix2d(measurement_jacobian * joint_covariance *
                      measurement_jacobian.transpose() + measurement_covariance),
      config_.covariance_floor);
  const double mahalanobis = mahalanobisSquared(
      innovation, innovation_covariance, config_.covariance_floor);
  if (std::isfinite(config_.innovation_gate) && config_.innovation_gate > 0.0 &&
      mahalanobis > config_.innovation_gate) {
    output.rejected = true;
    output.status = "innovation_gated";
    output.result.status = output.status;
    return output;
  }
  const Eigen::Matrix<double, 6, 2> gain = joint_covariance *
      measurement_jacobian.transpose() *
      safeInverse(innovation_covariance, config_.covariance_floor);
  Eigen::Matrix<double, 6, 1> joint_mean;
  joint_mean << estimate_.vector(), transaction.observed.vector();
  joint_mean += gain * innovation;
  joint_mean(2) = wrapAngle(joint_mean(2));
  joint_mean(5) = wrapAngle(joint_mean(5));
  const Eigen::Matrix<double, 6, 6> residual =
      Eigen::Matrix<double, 6, 6>::Identity() - gain * measurement_jacobian;
  const Eigen::Matrix<double, 6, 6> posterior =
      residual * joint_covariance * residual.transpose() +
      gain * measurement_covariance * gain.transpose();
  const Eigen::MatrixXd posterior_matrix = posterior;
  const Eigen::Ref<const Eigen::MatrixXd> posterior_reference(posterior_matrix);
  const Eigen::MatrixXd posterior_dynamic = regularizeCovariance(
      posterior_reference, config_.covariance_floor);
  const Eigen::Matrix3d prior_local_covariance = estimate_.covariance;
  estimate_.mean = Pose2D::fromVector(joint_mean.head<3>());
  estimate_.covariance = posterior_dynamic.topLeftCorner<3, 3>();
  estimate_.timestamp = timestampFor(estimate_, observation);
  estimate_.valid = true;
  output.observed_estimate = transaction.observed;
  output.observed_estimate.mean = Pose2D::fromVector(joint_mean.tail<3>());
  output.observed_estimate.covariance = posterior_dynamic.bottomRightCorner<3, 3>();
  output.observed_estimate.timestamp = estimate_.timestamp;
  output.observed_estimate.valid = true;
  output.correlation_factor = posterior_dynamic.topRightCorner<3, 3>();
  output.reciprocal_factor = Eigen::Matrix3d::Identity();
  factors_[observed] = output.correlation_factor;
  const Eigen::Matrix3d equation_26_multiplier = config_.lambda *
      estimate_.covariance * safeInverse(prior_local_covariance,
                                         config_.covariance_floor);
  for (auto &[peer_id, factor] : factors_) {
    if (peer_id != observed) factor = equation_26_multiplier * factor;
  }
  output.result.estimate = estimate_;
  output.result.success = true;
  output.result.used_measurements = 1;
  output.result.maximum_innovation = innovation.norm();
  output.result.maximum_mahalanobis_squared = mahalanobis;
  output.result.status = "accepted";
  output.result.message = "exact two-agent EKF update complete";
  if (!event_id.empty()) processed_events_.insert(event_id);
  if (transaction.sequence != 0) latest_sequences_[observed] = transaction.sequence;
  output.accepted = true;
  output.status = "accepted";
  return output;
}

RecursiveTransactionResult RecursiveDecentralizedState::installPairPosterior(
    const std::string& peer_id, const PoseEstimate& posterior,
    const Eigen::Matrix3d& peer_factor, const std::string& event_id,
    std::uint64_t sequence_number) {
  RecursiveTransactionResult output;
  output.result.estimate = estimate_;
  if (!std::isfinite(config_.lambda) || config_.lambda < 0.0 ||
      config_.lambda > 1.0) {
    output.rejected = true;
    output.status = "invalid_lambda";
    output.result.status = output.status;
    return output;
  }
  if (peer_id.empty() || peer_id == agent_id_ || !posterior.valid ||
      !posterior.mean.position.allFinite() || !std::isfinite(posterior.mean.yaw) ||
      !finiteCovariance(posterior.covariance) || !peer_factor.allFinite()) {
    output.rejected = true;
    output.status = "invalid_pair_posterior";
    output.result.status = output.status;
    return output;
  }
  const auto latest = latest_sequences_.find(peer_id);
  const bool duplicate_event = !event_id.empty() && processed_events_.count(event_id);
  const bool duplicate_sequence = latest != latest_sequences_.end() && sequence_number != 0U &&
                                  sequence_number <= latest->second;
  if (duplicate_event || duplicate_sequence) {
    output.accepted = true;
    output.duplicate = true;
    output.status = "duplicate";
    output.result.success = true;
    output.result.status = output.status;
    output.result.estimate = estimate_;
    return output;
  }

  const Eigen::Matrix3d prior_covariance = regularizeCovariance(
      estimate_.covariance, config_.covariance_floor);
  const PoseEstimate installed = sanitizedEstimate(
      posterior, config_.covariance_floor);
  const Eigen::Matrix3d equation_26_multiplier = config_.lambda *
      installed.covariance * safeInverse(prior_covariance,
                                          config_.covariance_floor);
  for (auto& [other_id, factor] : factors_) {
    if (other_id != peer_id) factor = equation_26_multiplier * factor;
  }
  factors_[peer_id] = peer_factor;
  estimate_ = installed;
  if (!event_id.empty()) processed_events_.insert(event_id);
  if (sequence_number != 0U) latest_sequences_[peer_id] = sequence_number;

  output.accepted = true;
  output.status = "installed";
  output.result.success = true;
  output.result.status = output.status;
  output.result.message = "pair posterior installed";
  output.result.estimate = estimate_;
  output.correlation_factor = peer_factor;
  return output;
}

void RecursiveDecentralizedLocalizer::reset(const PoseEstimate& estimate) {
  estimate_ = sanitizedEstimate(estimate, config_.covariance_floor);
}

LocalizationResult RecursiveDecentralizedLocalizer::update(
    const std::vector<RelativeObservation>& observations) {
  const LocalizationResult result =
      recursiveDecentralizedUpdate(estimate_, observations, config_);
  estimate_ = result.estimate;
  return result;
}

LocalizationResult RecursiveDecentralizedLocalizer::update(
    const PoseEstimate& prior,
    const std::vector<RelativeObservation>& observations) {
  reset(prior);
  return update(observations);
}

}  // namespace hercules_localization
