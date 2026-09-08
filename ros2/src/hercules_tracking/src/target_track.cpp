#include "hercules_tracking/target_track.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "hercules_tracking/linear_algebra.hpp"

namespace hercules_tracking {
namespace {

void addFactor(Eigen::MatrixXd& information, Eigen::VectorXd& information_vector,
               const std::map<int, Eigen::MatrixXd>& factors,
               const Eigen::Ref<const Eigen::VectorXd>& value,
               const Eigen::Ref<const Eigen::MatrixXd>& covariance) {
  const Eigen::MatrixXd inverse = safeInverse(covariance);
  for (const auto& [first_index, first_matrix] : factors) {
    information_vector.segment(first_index * kStateDim, kStateDim) +=
        first_matrix.transpose() * inverse * value;
    for (const auto& [second_index, second_matrix] : factors) {
      information.block(first_index * kStateDim, second_index * kStateDim,
                        kStateDim, kStateDim) +=
          first_matrix.transpose() * inverse * second_matrix;
    }
  }
}

double latestSupport(const std::optional<double>& direct, const std::optional<double>& handoff,
                     const std::optional<double>& consensus, bool include_consensus) {
  double value = -std::numeric_limits<double>::infinity();
  if (direct) value = std::max(value, *direct);
  if (handoff) value = std::max(value, *handoff);
  if (include_consensus && consensus) value = std::max(value, *consensus);
  return value;
}

Eigen::VectorXd alignTrajectory(const TrackMessage& message, const std::vector<double>& local_times,
                                Eigen::Index local_state_count) {
  const Eigen::Index neighbor_count = message.trajectory.size() / kStateDim;
  if (message.trajectory.size() == 0 || message.trajectory.size() % kStateDim != 0) {
    throw std::invalid_argument("neighbor trajectory must contain complete states");
  }
  if (static_cast<Eigen::Index>(message.times.size()) == neighbor_count && !local_times.empty()) {
    Eigen::VectorXd aligned(local_state_count * kStateDim);
    for (Eigen::Index local = 0; local < local_state_count; ++local) {
      const double time = local_times[static_cast<std::size_t>(local)];
      auto upper = std::lower_bound(message.times.begin(), message.times.end(), time);
      if (upper == message.times.begin()) {
        aligned.segment(local * kStateDim, kStateDim) = message.trajectory.head(kStateDim);
      } else if (upper == message.times.end()) {
        aligned.segment(local * kStateDim, kStateDim) = message.trajectory.tail(kStateDim);
      } else {
        const auto right = static_cast<Eigen::Index>(upper - message.times.begin());
        const Eigen::Index left = right - 1;
        const double span = message.times[static_cast<std::size_t>(right)] -
                            message.times[static_cast<std::size_t>(left)];
        const double weight = span == 0.0 ? 1.0 :
            (time - message.times[static_cast<std::size_t>(left)]) / span;
        aligned.segment(local * kStateDim, kStateDim) =
            (1.0 - weight) * message.trajectory.segment(left * kStateDim, kStateDim) +
            weight * message.trajectory.segment(right * kStateDim, kStateDim);
      }
    }
    return aligned;
  }
  if (neighbor_count == local_state_count) return message.trajectory;
  Eigen::VectorXd aligned(local_state_count * kStateDim);
  for (Eigen::Index index = 0; index < local_state_count; ++index) {
    aligned.segment(index * kStateDim, kStateDim) = message.trajectory.tail(kStateDim);
  }
  return aligned;
}

}  // namespace

std::pair<Eigen::MatrixXd, Eigen::VectorXd> assembleWindowInformation(
    const std::vector<double>& times,
    const std::vector<std::optional<TargetMeasurement>>& measurements,
    const State& prior_mean, const StateMatrix& prior_covariance,
    double process_noise_spectral_density, int active_tracker_count,
    const std::optional<StateMatrix>& handoff_information,
    const std::optional<State>& handoff_information_vector) {
  if (times.empty() || times.size() != measurements.size()) {
    throw std::invalid_argument("times and measurements must have equal nonzero length");
  }
  if (!prior_mean.allFinite() || !prior_covariance.allFinite()) {
    throw std::invalid_argument("prior must be finite");
  }
  for (double time : times) {
    if (!std::isfinite(time)) throw std::invalid_argument("times must be finite");
  }
  const Eigen::Index dimension = static_cast<Eigen::Index>(times.size()) * kStateDim;
  Eigen::MatrixXd information = Eigen::MatrixXd::Zero(dimension, dimension);
  Eigen::VectorXd information_vector = Eigen::VectorXd::Zero(dimension);
  const int scale = std::max(1, active_tracker_count);

  addFactor(information, information_vector, {{0, StateMatrix::Identity()}}, prior_mean,
            prior_covariance * static_cast<double>(scale));
  if (handoff_information) {
    information.topLeftCorner(kStateDim, kStateDim) += positiveDefinite(*handoff_information);
    if (handoff_information_vector) information_vector.head(kStateDim) += *handoff_information_vector;
  }

  for (std::size_t index = 1; index < times.size(); ++index) {
    const double delta = std::max(1e-6, times[index] - times[index - 1]);
    const StateMatrix transition = constantVelocityTransition(delta);
    const StateMatrix process = constantAccelerationProcessNoise(
        delta, process_noise_spectral_density) * static_cast<double>(scale);
    addFactor(information, information_vector,
              {{static_cast<int>(index - 1), -transition},
               {static_cast<int>(index), StateMatrix::Identity()}},
              State::Zero(), process);
  }

  const auto measurement_matrix = positionMeasurementMatrix();
  for (std::size_t index = 0; index < measurements.size(); ++index) {
    if (!measurements[index] || !measurements[index]->valid) continue;
    addFactor(information, information_vector,
              {{static_cast<int>(index), measurement_matrix}},
              measurements[index]->position, positiveDefinite(measurements[index]->covariance));
  }
  information = positiveDefinite(information);
  return {information, information_vector};
}

TargetTrack::TargetTrack(std::string target_id, TrackConfig config)
    : target_id_(std::move(target_id)), config_(config) {
  if (target_id_.empty()) throw std::invalid_argument("target_id must not be empty");
}

bool TargetTrack::ensureInitialized(const std::optional<TargetMeasurement>& measurement,
                                    double timestamp) {
  if (prior_mean_) return true;
  if (!measurement || !measurement->valid) return false;
  prior_mean_ = State(measurement->position.x(), measurement->position.y(), 0.0, 0.0);
  const double velocity_variance = std::max(1.0, 4.0 * config_.measurement_std * config_.measurement_std);
  prior_covariance_ = StateMatrix::Zero();
  prior_covariance_(0, 0) = std::max(measurement->covariance(0, 0), 1e-4);
  prior_covariance_(1, 1) = std::max(measurement->covariance(1, 1), 1e-4);
  prior_covariance_(2, 2) = velocity_variance;
  prior_covariance_(3, 3) = velocity_variance;
  latest_time_ = timestamp;
  return true;
}

void TargetTrack::appendSample(double timestamp,
                               const std::optional<TargetMeasurement>& measurement) {
  double sample_timestamp = timestamp;
  if (measurement && measurement->valid && std::isfinite(measurement->timestamp)) {
    sample_timestamp = std::min(timestamp, measurement->timestamp);
  }
  auto found = std::lower_bound(times_.begin(), times_.end(), sample_timestamp);
  const std::size_t index = static_cast<std::size_t>(found - times_.begin());
  // np.isclose(..., atol=1e-9) retains NumPy's default rtol=1e-5.
  const double timestamp_tolerance = 1e-9 + 1e-5 * std::abs(sample_timestamp);
  if (found != times_.end() && std::abs(*found - sample_timestamp) <= timestamp_tolerance) {
    measurements_[index] = measurement;
  } else {
    times_.insert(found, sample_timestamp);
    measurements_.insert(measurements_.begin() + static_cast<std::ptrdiff_t>(index), measurement);
  }
  latest_time_ = std::max(latest_time_.value_or(timestamp), timestamp);
  if (measurement && measurement->valid) {
    last_direct_observation_ = std::max(last_direct_observation_.value_or(measurement->timestamp),
                                        measurement->timestamp);
  }
  const double cutoff = timestamp - std::max(0.0, config_.window_seconds);
  while (times_.size() > 1 && times_[1] < cutoff) {
    times_.erase(times_.begin());
    measurements_.erase(measurements_.begin());
    if (x_.size() >= 2 * kStateDim && covariance_.rows() >= 2 * kStateDim) {
      prior_mean_ = x_.segment(kStateDim, kStateDim);
      prior_covariance_ = covariance_.block(kStateDim, kStateDim, kStateDim, kStateDim);
    }
  }
}

void TargetTrack::clearForFreshObservation() {
  prior_mean_.reset();
  times_.clear();
  measurements_.clear();
  x_.resize(0);
  dual_.resize(0);
  information_.resize(0, 0);
  information_vector_.resize(0);
}

bool TargetTrack::begin(double timestamp, const std::optional<TargetMeasurement>& measurement,
                        int active_count) {
  if (!std::isfinite(timestamp)) throw std::invalid_argument("timestamp must be finite");
  if (measurement && measurement->valid) consensus_covariance_.reset();
  const double support = latestSupport(last_direct_observation_, last_handoff_time_,
                                       last_consensus_support_, true);
  if (active_ && std::isfinite(support) && timestamp - support > config_.window_seconds) active_ = false;
  if (!active_ && prior_mean_) {
    if (!measurement || !measurement->valid) return false;
    clearForFreshObservation();
  }
  if (!ensureInitialized(measurement, timestamp)) return false;
  appendSample(timestamp, measurement);
  auto assembled = assembleWindowInformation(
      times_, measurements_, *prior_mean_, prior_covariance_,
      config_.process_noise_spectral_density, active_count,
      pending_handoff_information_, pending_handoff_vector_);
  information_ = std::move(assembled.first);
  information_vector_ = std::move(assembled.second);
  pending_handoff_information_.reset();
  pending_handoff_vector_.reset();
  x_ = solveBlockTridiagonal(information_, information_vector_);
  dual_ = Eigen::VectorXd::Zero(x_.size());
  covariance_ = pseudoInverse(information_);
  active_ = true;
  admm_iterations_ = 0;
  consensus_residual_ = std::numeric_limits<double>::infinity();
  return true;
}

State TargetTrack::latestState() const {
  return x_.size() >= kStateDim ? State(x_.tail(kStateDim)) : State::Zero();
}

TrackMessage TargetTrack::message() const {
  TrackMessage output;
  output.target_id = target_id_;
  output.trajectory = x_;
  output.times = times_;
  output.state = latestState();
  if (consensus_covariance_) output.state_covariance = *consensus_covariance_;
  else if (covariance_.rows() >= kStateDim) output.state_covariance = covariance_.bottomRightCorner(kStateDim, kStateDim);
  output.active = active_ && x_.size() > 0;
  output.last_direct_observation = last_direct_observation_;
  return output;
}

double TargetTrack::consensusStep(const std::vector<TrackMessage>& neighbor_messages) {
  if (x_.size() == 0 || information_.size() == 0 || information_vector_.size() == 0) return 0.0;
  const Eigen::VectorXd previous = x_;
  if (dual_.size() == 0) dual_ = Eigen::VectorXd::Zero(previous.size());
  std::vector<Eigen::VectorXd> trajectories;
  for (const auto& neighbor : neighbor_messages) {
    if (neighbor.target_id != target_id_ || neighbor.trajectory.size() == 0 || !neighbor.active) continue;
    trajectories.push_back(alignTrajectory(neighbor, times_, previous.size() / kStateDim));
  }
  if (!trajectories.empty()) {
    const double degree = static_cast<double>(trajectories.size());
    Eigen::MatrixXd system = information_ +
        2.0 * config_.rho * degree * Eigen::MatrixXd::Identity(previous.size(), previous.size());
    Eigen::VectorXd sum = Eigen::VectorXd::Zero(previous.size());
    for (const auto& trajectory : trajectories) sum += previous + trajectory;
    const Eigen::VectorXd vector = information_vector_ - dual_ + config_.rho * sum;
    x_ = solveBlockTridiagonal(system, vector);
    Eigen::VectorXd dual_sum = Eigen::VectorXd::Zero(previous.size());
    for (const auto& trajectory : trajectories) dual_sum += x_ - trajectory;
    dual_ += config_.rho * dual_sum;
  } else {
    x_ = solveBlockTridiagonal(information_, information_vector_);
  }
  ++admm_iterations_;
  consensus_residual_ = (x_ - previous).norm();
  return consensus_residual_;
}

TargetEstimate TargetTrack::finalize() const {
  TargetEstimate estimate;
  estimate.target_id = target_id_;
  const State state = latestState();
  estimate.position = state.head<2>();
  estimate.velocity = state.tail<2>();
  if (consensus_covariance_) estimate.state_covariance = *consensus_covariance_;
  else if (covariance_.rows() >= kStateDim) estimate.state_covariance = covariance_.bottomRightCorner(kStateDim, kStateDim);
  estimate.covariance = estimate.state_covariance.topLeftCorner<2, 2>();
  estimate.timestamp = latest_time_.value_or(0.0);
  estimate.active = active_;
  estimate.admm_iterations = admm_iterations_;
  estimate.consensus_residual = consensus_residual_;
  estimate.last_direct_observation = last_direct_observation_;
  for (auto it = measurements_.rbegin(); it != measurements_.rend(); ++it) {
    if (*it && (*it)->valid) {
      estimate.measurement_residual = estimate.position - (*it)->position;
      break;
    }
  }
  return estimate;
}

TargetEstimate TargetTrack::predicted(double timestamp) const {
  TargetEstimate estimate = finalize();
  if (!estimate.active) return estimate;
  const double delta = std::max(0.0, timestamp - estimate.timestamp);
  State state;
  state << estimate.position, estimate.velocity;
  state = constantVelocityTransition(delta) * state;
  estimate.position = state.head<2>();
  estimate.velocity = state.tail<2>();
  // Preserved Python behavior: only reported XY covariance receives the XY
  // process block; state_covariance is intentionally left unchanged.
  estimate.covariance += constantAccelerationProcessNoise(
      delta, config_.process_noise_spectral_density).topLeftCorner<2, 2>();
  estimate.timestamp = timestamp;
  return estimate;
}

std::optional<HandoffMessage> TargetTrack::handoffMessage(double timestamp) {
  if (!active_) return std::nullopt;
  const double support = latestSupport(last_direct_observation_, last_handoff_time_,
                                       std::nullopt, false);
  if (!std::isfinite(support) || timestamp - support < config_.window_seconds) return std::nullopt;
  if (last_handoff_sent_ && timestamp - *last_handoff_sent_ < config_.window_seconds) return std::nullopt;
  const TargetEstimate estimate = finalize();
  HandoffMessage message;
  message.target_id = target_id_;
  message.information_matrix = safeInverse(positiveDefinite(estimate.state_covariance));
  State state;
  state << estimate.position, estimate.velocity;
  message.information_vector = message.information_matrix * state;
  message.timestamp = timestamp;
  last_handoff_sent_ = timestamp;
  return message;
}

bool TargetTrack::acceptHandoff(const HandoffMessage& message) {
  if (message.target_id != target_id_ || !message.information_matrix.allFinite() ||
      !message.information_vector.allFinite() || !std::isfinite(message.timestamp)) return false;
  const StateMatrix covariance = safeInverse(message.information_matrix);
  if (!prior_mean_) {
    prior_mean_ = covariance * message.information_vector;
    prior_covariance_ = covariance;
    times_ = {message.timestamp};
    measurements_ = {std::nullopt};
    latest_time_ = message.timestamp;
    x_ = *prior_mean_;
    dual_ = Eigen::VectorXd::Zero(kStateDim);
    information_ = message.information_matrix;
    information_vector_ = message.information_vector;
  }
  if (message.state_covariance) consensus_covariance_ = positiveDefinite(*message.state_covariance);
  active_ = true;
  last_handoff_time_ = message.timestamp;
  if (!pending_handoff_information_) {
    pending_handoff_information_ = message.information_matrix;
    pending_handoff_vector_ = message.information_vector;
  } else {
    *pending_handoff_information_ += message.information_matrix;
    *pending_handoff_vector_ += message.information_vector;
  }
  handoffs_.emplace_back(message.source_id, message.timestamp);
  return true;
}

void TargetTrack::seed(const TrackMessage& message, double timestamp) {
  if (active_ || message.state.size() != kStateDim) return;
  prior_mean_ = message.state;
  prior_covariance_ = StateMatrix::Identity() * 16.0;
  latest_time_ = timestamp;
  last_handoff_time_ = timestamp;
  last_consensus_support_ = timestamp;
  active_ = true;
  times_ = {timestamp};
  measurements_ = {std::nullopt};
  auto assembled = assembleWindowInformation(times_, measurements_, *prior_mean_, prior_covariance_,
                                             config_.process_noise_spectral_density, 1);
  information_ = std::move(assembled.first);
  information_vector_ = std::move(assembled.second);
  x_ = *prior_mean_;
  dual_ = Eigen::VectorXd::Zero(kStateDim);
}

void TargetTrack::setConsensusSupport(double timestamp, const StateMatrix& covariance) {
  last_consensus_support_ = timestamp;
  consensus_covariance_ = positiveDefinite(covariance);
}

}  // namespace hercules_tracking
