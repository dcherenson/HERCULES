#include "hercules_localization/global_ci.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <set>

#include <Eigen/Cholesky>

#include "hercules_localization/covariance.hpp"
#include "hercules_localization/math.hpp"

namespace hercules_localization {
namespace {

Eigen::MatrixXd regularizeDynamic(const Eigen::MatrixXd& matrix, double floor) {
  Eigen::Ref<const Eigen::MatrixXd> reference(matrix);
  return regularizeCovariance(reference, floor);
}

Eigen::MatrixXd inverseDynamic(const Eigen::MatrixXd& matrix, double floor) {
  Eigen::Ref<const Eigen::MatrixXd> reference(matrix);
  return safeInverse(reference, floor);
}

std::size_t stateDimension(const GlobalCiBelief& belief) {
  return 2U * belief.ordered_agent_ids.size() + 1U;
}

bool orderedIdsValid(const std::vector<std::string>& ids) {
  std::set<std::string> unique;
  for (const auto& id : ids) {
    if (id.empty() || !unique.insert(id).second) return false;
  }
  return !ids.empty();
}

bool covarianceSpd(const Eigen::MatrixXd& covariance, double floor) {
  if (covariance.rows() == 0 || covariance.rows() != covariance.cols() ||
      !covariance.allFinite()) {
    return false;
  }
  const Eigen::MatrixXd symmetric = 0.5 * (covariance + covariance.transpose());
  if ((covariance - covariance.transpose()).norm() >
      1e-8 * (1.0 + covariance.norm())) {
    return false;
  }
  // Validation must reject singular/indefinite packets.  The configurable
  // floor is applied later, during numerical regularization; adding it here
  // would silently turn a zero or negative covariance into an accepted wire
  // belief and violate the transport contract.
  (void)floor;
  const Eigen::LLT<Eigen::MatrixXd> factor(symmetric);
  return factor.info() == Eigen::Success &&
         factor.matrixL().toDenseMatrix().diagonal().array().all() > 0.0;
}

bool validFloor(double floor) {
  return std::isfinite(floor) && floor > 0.0;
}

GlobalCiBelief mappedForReceiver(const GlobalCiBelief& peer,
                                 const GlobalCiBelief& self,
                                 double floor) {
  GlobalCiBelief mapped = peer;
  const std::size_t dimension = stateDimension(self);
  const std::size_t positions_dimension = dimension - 1U;
  mapped.mean = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
  mapped.covariance = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(dimension),
                                            static_cast<Eigen::Index>(dimension));
  mapped.mean.head(static_cast<Eigen::Index>(positions_dimension)) =
      peer.mean.head(static_cast<Eigen::Index>(positions_dimension));
  mapped.mean(static_cast<Eigen::Index>(positions_dimension)) =
      self.mean(static_cast<Eigen::Index>(positions_dimension));
  mapped.covariance.topLeftCorner(static_cast<Eigen::Index>(positions_dimension),
                                  static_cast<Eigen::Index>(positions_dimension)) =
      peer.covariance.topLeftCorner(static_cast<Eigen::Index>(positions_dimension),
                                    static_cast<Eigen::Index>(positions_dimension));
  // Remove the sender yaw and insert the receiver yaw explicitly.  Keeping
  // the receiver cross-covariance preserves the local block while avoiding
  // a false correlation between a sender's private heading and this state.
  mapped.covariance.topRightCorner(static_cast<Eigen::Index>(positions_dimension), 1) =
      self.covariance.topRightCorner(static_cast<Eigen::Index>(positions_dimension), 1);
  mapped.covariance.bottomLeftCorner(1, static_cast<Eigen::Index>(positions_dimension)) =
      self.covariance.bottomLeftCorner(1, static_cast<Eigen::Index>(positions_dimension));
  mapped.covariance(static_cast<Eigen::Index>(positions_dimension),
                    static_cast<Eigen::Index>(positions_dimension)) =
      self.covariance(static_cast<Eigen::Index>(positions_dimension),
                      static_cast<Eigen::Index>(positions_dimension));
  mapped.covariance = regularizeDynamic(mapped.covariance, floor);
  return mapped;
}

// A duplicate sender can carry different payloads.  Sorting by sender alone
// would still leave the selected payload dependent on vector arrival order,
// so use a complete, deterministic tie-breaker after preferring the newest
// timestamp/sequence.
bool canonicalBeliefLess(const GlobalCiBelief& first,
                         const GlobalCiBelief& second) {
  if (first.sender_id != second.sender_id) return first.sender_id < second.sender_id;
  if (first.timestamp != second.timestamp) return first.timestamp > second.timestamp;
  if (first.sequence != second.sequence) return first.sequence > second.sequence;
  for (Eigen::Index index = 0; index < first.mean.size(); ++index) {
    if (first.mean(index) != second.mean(index)) {
      return first.mean(index) < second.mean(index);
    }
  }
  for (Eigen::Index row = 0; row < first.covariance.rows(); ++row) {
    for (Eigen::Index col = 0; col < first.covariance.cols(); ++col) {
      if (first.covariance(row, col) != second.covariance(row, col)) {
        return first.covariance(row, col) < second.covariance(row, col);
      }
    }
  }
  return false;
}

bool findAgent(const GlobalCiBelief& belief, const std::string& id,
               std::size_t* index) {
  const auto found = std::find(belief.ordered_agent_ids.begin(),
                               belief.ordered_agent_ids.end(), id);
  if (found == belief.ordered_agent_ids.end()) return false;
  if (index != nullptr) {
    *index = static_cast<std::size_t>(
        std::distance(belief.ordered_agent_ids.begin(), found));
  }
  return true;
}

bool validMeasurementCovariance(const MeasurementCovariance& covariance,
                                double floor) {
  return covarianceSpd(covariance, floor);
}

// Solve S * X = rhs without forming an unregularized inverse.  The LLT path
// is exact for the normal SPD case; safeInverse is a finite fallback for
// round-off-corrupted innovation matrices.
Eigen::MatrixXd solveInnovation(const Eigen::Matrix2d& covariance,
                                const Eigen::Ref<const Eigen::MatrixXd>& rhs,
                                double floor) {
  const Eigen::Matrix2d regularized = regularizeCovariance(covariance, floor);
  const Eigen::LLT<Eigen::Matrix2d> factor(regularized);
  if (factor.info() == Eigen::Success) {
    const Eigen::MatrixXd solution = factor.solve(rhs);
    if (solution.allFinite()) return solution;
  }
  return safeInverse(regularized, floor) * rhs;
}

bool validPosterior(const Eigen::VectorXd& mean,
                    const Eigen::MatrixXd& covariance) {
  return mean.allFinite() && covariance.allFinite() &&
         covariance.rows() == covariance.cols();
}

void stampUpdate(GlobalCiBelief* belief, double timestamp) {
  if (belief == nullptr || !std::isfinite(timestamp)) return;
  belief->timestamp = std::max(belief->timestamp, timestamp);
}

GlobalCiResult privatePositionUpdateImpl(
    const GlobalCiBelief& prior, const std::string& receiver_id,
    const Position2D& measured_position,
    const MeasurementCovariance& measurement_covariance, double timestamp,
    double covariance_floor, double measurement_floor) {
  GlobalCiResult result;
  result.belief = prior;
  if (!isValidGlobalCiBelief(prior, covariance_floor)) {
    result.status = "invalid_prior_belief";
    return result;
  }
  std::size_t receiver_index = 0U;
  if (!findAgent(prior, receiver_id, &receiver_index)) {
    result.status = "unknown_receiver";
    return result;
  }
  if (!measured_position.allFinite() ||
      !validMeasurementCovariance(measurement_covariance, measurement_floor) ||
      (!std::isfinite(timestamp) && !std::isnan(timestamp))) {
    // NaN is the documented sentinel meaning "preserve the prior timestamp";
    // infinities and other non-finite values are rejected.
    result.status = "invalid_private_position";
    return result;
  }
  if (!validFloor(covariance_floor) || !validFloor(measurement_floor)) {
    result.status = "invalid_covariance_floor";
    return result;
  }

  const std::size_t dimension = stateDimension(prior);
  const Eigen::Index state_size = static_cast<Eigen::Index>(dimension);
  const Eigen::Index offset = static_cast<Eigen::Index>(2U * receiver_index);
  const Eigen::Index yaw_offset = state_size - 1;
  const Eigen::MatrixXd covariance =
      regularizeDynamic(prior.covariance, covariance_floor);
  const Eigen::Matrix2d noise =
      regularizeCovariance(measurement_covariance, measurement_floor);
  Eigen::MatrixXd selector = Eigen::MatrixXd::Zero(2, state_size);
  selector(0, offset) = 1.0;
  selector(1, offset + 1) = 1.0;
  const Eigen::Vector2d innovation =
      measured_position - prior.mean.segment<2>(offset);
  const Eigen::Matrix2d raw_innovation_covariance =
      selector * covariance * selector.transpose() + noise;
  const Eigen::Matrix2d innovation_covariance = regularizeCovariance(
      raw_innovation_covariance, covariance_floor);
  const Eigen::MatrixXd gain =
      solveInnovation(innovation_covariance,
                      (covariance * selector.transpose()).eval().transpose(),
                      covariance_floor)
          .transpose();
  if (!gain.allFinite() || !innovation.allFinite()) {
    result.status = "invalid_private_position_update";
    return result;
  }

  const Eigen::VectorXd posterior_mean = prior.mean + gain * innovation;
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(state_size, state_size);
  const Eigen::MatrixXd residual = identity - gain * selector;
  const Eigen::MatrixXd posterior_covariance = regularizeDynamic(
      (residual * covariance * residual.transpose() + gain * noise * gain.transpose())
          .eval(),
      covariance_floor);
  if (!validPosterior(posterior_mean, posterior_covariance)) {
    result.status = "invalid_private_position_update";
    return result;
  }

  result.belief.mean = posterior_mean;
  result.belief.mean(yaw_offset) = wrapAngle(result.belief.mean(yaw_offset));
  result.belief.covariance = posterior_covariance;
  stampUpdate(&result.belief, timestamp);
  result.belief.valid = true;
  result.success = true;
  result.status = "ok";
  return result;
}

GlobalCiResult rangeBearingUpdateImpl(
    const GlobalCiBelief& prior, const std::string& observer_id,
    const std::string& observed_id, const RangeBearingMeasurement& measurement,
    const LocalizationConfig& config) {
  GlobalCiResult result;
  result.belief = prior;
  if (!isValidGlobalCiBelief(prior, config.covariance_floor)) {
    result.status = "invalid_prior_belief";
    return result;
  }
  std::size_t observer_index = 0U;
  std::size_t observed_index = 0U;
  if (!findAgent(prior, observer_id, &observer_index) ||
      !findAgent(prior, observed_id, &observed_index) ||
      observer_id == observed_id) {
    result.status = "invalid_relative_agents";
    return result;
  }
  // There is one yaw slot and it belongs to the sender.  Refusing a
  // non-sender observer prevents accidentally applying that yaw to another
  // agent's position block.
  if (observer_id != prior.sender_id) {
    result.status = "observer_yaw_unavailable";
    return result;
  }
  if (!measurement.valid || !std::isfinite(measurement.range) ||
      !std::isfinite(measurement.bearing) ||
      measurement.range <= config.range_epsilon ||
      !validMeasurementCovariance(measurement.covariance,
                                  config.measurement_covariance_floor) ||
      (!measurement.neighbor_id.empty() && measurement.neighbor_id != observed_id)) {
    result.status = "invalid_range_bearing";
    return result;
  }
  if (!validFloor(config.covariance_floor) ||
      !validFloor(config.measurement_covariance_floor) ||
      !std::isfinite(config.range_epsilon) || config.range_epsilon <= 0.0) {
    result.status = "invalid_localization_config";
    return result;
  }

  const std::size_t dimension = stateDimension(prior);
  const Eigen::Index state_size = static_cast<Eigen::Index>(dimension);
  const Eigen::Index observer_offset =
      static_cast<Eigen::Index>(2U * observer_index);
  const Eigen::Index observed_offset =
      static_cast<Eigen::Index>(2U * observed_index);
  const Eigen::Index yaw_offset = state_size - 1;
  const Eigen::Vector2d observer_position =
      prior.mean.segment<2>(observer_offset);
  const Eigen::Vector2d observed_position =
      prior.mean.segment<2>(observed_offset);
  const Pose2D observer(observer_position, prior.mean(yaw_offset));
  const Pose2D observed(observed_position, 0.0);
  const RangeBearing predicted =
      predictRangeBearing(observer, observed, config.range_epsilon);
  if (!std::isfinite(predicted.range) || !std::isfinite(predicted.bearing)) {
    result.status = "invalid_range_bearing_prediction";
    return result;
  }

  Eigen::Vector2d innovation;
  innovation << measurement.range - predicted.range,
      angleDifference(measurement.bearing, predicted.bearing);
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(2, state_size);
  const Eigen::Matrix<double, 2, 3> observer_jacobian =
      rangeBearingJacobianObserver(observer, observed, config.range_epsilon);
  const Eigen::Matrix<double, 2, 3> observed_jacobian =
      rangeBearingJacobianTarget(observer, observed, config.range_epsilon);
  jacobian.block<2, 2>(0, observer_offset) = observer_jacobian.leftCols<2>();
  jacobian.block<2, 2>(0, observed_offset) = observed_jacobian.leftCols<2>();
  jacobian.col(yaw_offset) = observer_jacobian.col(2);

  const Eigen::MatrixXd covariance =
      regularizeDynamic(prior.covariance, config.covariance_floor);
  const Eigen::Matrix2d noise = measurementCovarianceForConfig(
      measurement.covariance, config, config.measurement_covariance_floor);
  const Eigen::Matrix2d raw_innovation_covariance =
      jacobian * covariance * jacobian.transpose() + noise;
  const Eigen::Matrix2d innovation_covariance = regularizeCovariance(
      raw_innovation_covariance, config.covariance_floor);
  const Eigen::MatrixXd gain =
      solveInnovation(innovation_covariance,
                      (covariance * jacobian.transpose()).eval().transpose(),
                      config.covariance_floor)
          .transpose();
  if (!gain.allFinite() || !innovation.allFinite()) {
    result.status = "invalid_range_bearing_update";
    return result;
  }

  const Eigen::VectorXd posterior_mean = prior.mean + gain * innovation;
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(state_size, state_size);
  const Eigen::MatrixXd residual = identity - gain * jacobian;
  const Eigen::MatrixXd posterior_covariance = regularizeDynamic(
      (residual * covariance * residual.transpose() + gain * noise * gain.transpose())
          .eval(),
      config.covariance_floor);
  if (!validPosterior(posterior_mean, posterior_covariance)) {
    result.status = "invalid_range_bearing_update";
    return result;
  }

  result.belief.mean = posterior_mean;
  result.belief.mean(yaw_offset) = wrapAngle(result.belief.mean(yaw_offset));
  result.belief.covariance = posterior_covariance;
  stampUpdate(&result.belief, measurement.timestamp);
  result.belief.valid = true;
  result.success = true;
  result.status = "ok";
  return result;
}

}  // namespace

GlobalCiBelief initializeGlobalCiBelief(
    const std::string& sender_id,
    const std::vector<std::string>& ordered_agent_ids,
    const PoseEstimate& local_pose, double remote_position_variance,
    double covariance_floor) {
  GlobalCiBelief result;
  result.sender_id = sender_id;
  result.ordered_agent_ids = ordered_agent_ids;
  result.timestamp = local_pose.timestamp;
  result.valid = false;
  if (!orderedIdsValid(ordered_agent_ids) || sender_id.empty() ||
      !findAgent(result, sender_id, nullptr) || !local_pose.valid ||
      !local_pose.mean.position.allFinite() || !std::isfinite(local_pose.mean.yaw) ||
      !std::isfinite(local_pose.timestamp) || !local_pose.covariance.allFinite() ||
      !validFloor(covariance_floor) ||
      !std::isfinite(remote_position_variance) || remote_position_variance <= 0.0) {
    return result;
  }

  const std::size_t dimension = stateDimension(result);
  result.mean = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
  result.covariance = Eigen::MatrixXd::Identity(
      static_cast<Eigen::Index>(dimension), static_cast<Eigen::Index>(dimension)) *
      remote_position_variance;
  const std::size_t sender_index = static_cast<std::size_t>(
      std::distance(ordered_agent_ids.begin(),
                    std::find(ordered_agent_ids.begin(), ordered_agent_ids.end(), sender_id)));
  const Eigen::Index position_offset = static_cast<Eigen::Index>(2U * sender_index);
  const Eigen::Index yaw_offset = static_cast<Eigen::Index>(dimension - 1U);
  const Eigen::Matrix3d local_covariance =
      regularizeCovariance(local_pose.covariance, covariance_floor);
  result.mean.segment<2>(position_offset) = local_pose.mean.position;
  result.mean(yaw_offset) = wrapAngle(local_pose.mean.yaw);
  result.covariance.block<2, 2>(position_offset, position_offset) =
      local_covariance.topLeftCorner<2, 2>();
  result.covariance.block<2, 1>(position_offset, yaw_offset) =
      local_covariance.topRightCorner<2, 1>();
  result.covariance.block<1, 2>(yaw_offset, position_offset) =
      local_covariance.bottomLeftCorner<1, 2>();
  result.covariance(yaw_offset, yaw_offset) = local_covariance(2, 2);
  result.covariance = regularizeDynamic(result.covariance, covariance_floor);
  // isValidGlobalCiBelief validates the public validity flag as part of the
  // wire contract, so mark the fully constructed candidate before asking it
  // to validate itself.
  result.valid = true;
  result.valid = isValidGlobalCiBelief(result, covariance_floor);
  return result;
}

GlobalCiBelief initializeGlobalCiBelief(
    const PoseEstimate& local_pose,
    const std::vector<std::string>& ordered_agent_ids,
    double remote_position_variance, double covariance_floor) {
  return initializeGlobalCiBelief(local_pose.agent_id, ordered_agent_ids,
                                  local_pose, remote_position_variance,
                                  covariance_floor);
}

GlobalCiBelief createGlobalCiBelief(
    const std::string& sender_id,
    const std::vector<std::string>& ordered_agent_ids,
    const PoseEstimate& local_pose, double remote_position_variance,
    double covariance_floor) {
  return initializeGlobalCiBelief(sender_id, ordered_agent_ids, local_pose,
                                  remote_position_variance, covariance_floor);
}

GlobalCiBelief createGlobalCiBelief(
    const PoseEstimate& local_pose,
    const std::vector<std::string>& ordered_agent_ids,
    double remote_position_variance, double covariance_floor) {
  return initializeGlobalCiBelief(local_pose, ordered_agent_ids,
                                  remote_position_variance, covariance_floor);
}

bool isValidGlobalCiBelief(const GlobalCiBelief& belief, double covariance_floor) {
  const std::size_t dimension = stateDimension(belief);
  const bool sender_is_ordered = std::find(
      belief.ordered_agent_ids.begin(), belief.ordered_agent_ids.end(), belief.sender_id) !=
      belief.ordered_agent_ids.end();
  return belief.valid && orderedIdsValid(belief.ordered_agent_ids) &&
         sender_is_ordered &&
         belief.mean.size() == static_cast<Eigen::Index>(dimension) &&
         belief.covariance.rows() == static_cast<Eigen::Index>(dimension) &&
         belief.covariance.cols() == static_cast<Eigen::Index>(dimension) &&
         belief.mean.allFinite() && std::isfinite(belief.timestamp) &&
         covarianceSpd(belief.covariance, covariance_floor);
}

GlobalCiBelief propagateGlobalCiBelief(
    const GlobalCiBelief& prior, const std::string& receiver_id,
    const PoseVector& world_delta, const PoseCovariance& local_process_covariance,
    double dt, double unknown_motion_variance, double covariance_floor) {
  GlobalCiBelief result = prior;
  if (!isValidGlobalCiBelief(prior, covariance_floor) || !world_delta.allFinite() ||
      !covarianceSpd(local_process_covariance, covariance_floor) ||
      !std::isfinite(dt) || dt < 0.0 || !std::isfinite(unknown_motion_variance) ||
      unknown_motion_variance < 0.0) {
    result.valid = false;
    return result;
  }
  const auto iterator = std::find(prior.ordered_agent_ids.begin(),
                                  prior.ordered_agent_ids.end(), receiver_id);
  if (iterator == prior.ordered_agent_ids.end()) {
    result.valid = false;
    return result;
  }
  const std::size_t index = static_cast<std::size_t>(
      std::distance(prior.ordered_agent_ids.begin(), iterator));
  const Eigen::Index position_offset = static_cast<Eigen::Index>(2U * index);
  result.mean.segment<2>(position_offset) += world_delta.head<2>();
  const Eigen::Index yaw_offset = static_cast<Eigen::Index>(prior.mean.size() - 1);
  result.mean(yaw_offset) = wrapAngle(result.mean(yaw_offset) + world_delta.z());
  const PoseCovariance process = regularizeCovariance(
      local_process_covariance, covariance_floor);
  result.covariance.block<2, 2>(position_offset, position_offset) +=
      process.topLeftCorner<2, 2>();
  result.covariance(position_offset, yaw_offset) += process(0, 2);
  result.covariance(position_offset + 1, yaw_offset) += process(1, 2);
  result.covariance(yaw_offset, position_offset) += process(2, 0);
  result.covariance(yaw_offset, position_offset + 1) += process(2, 1);
  result.covariance(yaw_offset, yaw_offset) += process(2, 2);
  for (std::size_t peer_index = 0; peer_index < prior.ordered_agent_ids.size(); ++peer_index) {
    if (peer_index == index) continue;
    const Eigen::Index offset = static_cast<Eigen::Index>(2U * peer_index);
    result.covariance.block<2, 2>(offset, offset).diagonal().array() +=
        unknown_motion_variance * std::max(1.0, dt);
  }
  result.covariance = regularizeDynamic(result.covariance, covariance_floor);
  result.timestamp += dt;
  return result;
}

GlobalCiResult globalCiPrivatePositionUpdate(
    const GlobalCiBelief& prior, const std::string& receiver_id,
    const Position2D& measured_position,
    const MeasurementCovariance& measurement_covariance, double timestamp,
    double covariance_floor) {
  return privatePositionUpdateImpl(
      prior, receiver_id, measured_position, measurement_covariance, timestamp,
      covariance_floor, covariance_floor);
}

GlobalCiResult globalCiPrivatePositionUpdate(
    const GlobalCiBelief& prior, const std::string& receiver_id,
    const Position2D& measured_position,
    const MeasurementCovariance& measurement_covariance, double timestamp,
    const LocalizationConfig& config) {
  return privatePositionUpdateImpl(
      prior, receiver_id, measured_position, measurement_covariance, timestamp,
      config.covariance_floor, config.measurement_covariance_floor);
}

GlobalCiResult globalCiRangeBearingUpdate(
    const GlobalCiBelief& prior, const std::string& observer_id,
    const std::string& observed_id,
    const RangeBearingMeasurement& measurement,
    const LocalizationConfig& config) {
  return rangeBearingUpdateImpl(prior, observer_id, observed_id, measurement,
                                config);
}

GlobalCiResult globalCiPrivatePositionEkfUpdate(
    const GlobalCiBelief& prior, const std::string& receiver_id,
    const Position2D& measured_position,
    const MeasurementCovariance& measurement_covariance, double timestamp,
    double covariance_floor) {
  return globalCiPrivatePositionUpdate(
      prior, receiver_id, measured_position, measurement_covariance, timestamp,
      covariance_floor);
}

GlobalCiResult globalCiRangeBearingEkfUpdate(
    const GlobalCiBelief& prior, const std::string& observer_id,
    const std::string& observed_id,
    const RangeBearingMeasurement& measurement,
    const LocalizationConfig& config) {
  return globalCiRangeBearingUpdate(prior, observer_id, observed_id,
                                    measurement, config);
}

GlobalCiResult globalCiUpdate(const GlobalCiBelief& self,
                              const std::vector<GlobalCiBelief>& peers,
                              const LocalizationConfig& config) {
  GlobalCiResult result;
  result.belief = self;
  if (!isValidGlobalCiBelief(self, config.covariance_floor)) {
    result.status = "invalid_self_belief";
    return result;
  }
  const std::size_t dimension = stateDimension(self);
  std::vector<GlobalCiBelief> candidates;
  candidates.reserve(peers.size());
  for (const auto& peer : peers) {
    const bool stale = std::isfinite(config.communication_timeout_sec) &&
                       config.communication_timeout_sec > 0.0 &&
                       std::isfinite(peer.timestamp) &&
                       peer.timestamp + config.communication_timeout_sec < self.timestamp;
    if (!isValidGlobalCiBelief(peer, config.covariance_floor) ||
        peer.ordered_agent_ids != self.ordered_agent_ids ||
        peer.sender_id == self.sender_id || stale) {
      ++result.rejected_peers;
      continue;
    }
    candidates.push_back(peer);
  }
  std::sort(candidates.begin(), candidates.end(), canonicalBeliefLess);

  std::vector<GlobalCiBelief> beliefs{self};
  std::set<std::string> accepted_senders{self.sender_id};
  for (const auto& peer : candidates) {
    // Sorted candidates make both the accepted payload and the rejection of
    // duplicate senders independent of the wire/vector arrival order.
    if (!accepted_senders.insert(peer.sender_id).second) {
      ++result.rejected_peers;
      continue;
    }
    beliefs.push_back(mappedForReceiver(peer, self, config.covariance_floor));
    ++result.accepted_peers;
  }

  const double self_weight = config.ci_self_weight;
  if (!std::isfinite(self_weight) || self_weight < 0.0 || self_weight > 1.0) {
    result.status = "invalid_ci_weight";
    return result;
  }
  const double peer_weight = beliefs.size() > 1U
      ? (1.0 - self_weight) / static_cast<double>(beliefs.size() - 1U)
      : 0.0;
  result.weights.assign(beliefs.size(), peer_weight);
  result.weights[0] = self_weight;
  // With no accepted peer, CI must be the identity operation.  Retaining a
  // configured 0.8 self coefficient alone would make the weights sum to 0.8
  // and inflate the covariance every time an invalid/stale packet arrived.
  if (beliefs.size() == 1U) result.weights[0] = 1.0;

  Eigen::MatrixXd information = Eigen::MatrixXd::Zero(
      static_cast<Eigen::Index>(dimension), static_cast<Eigen::Index>(dimension));
  Eigen::VectorXd information_vector = Eigen::VectorXd::Zero(
      static_cast<Eigen::Index>(dimension));
  const double anchor_yaw = self.mean(static_cast<Eigen::Index>(dimension - 1U));
  for (std::size_t index = 0; index < beliefs.size(); ++index) {
    const Eigen::MatrixXd covariance = regularizeDynamic(
        beliefs[index].covariance, config.covariance_floor);
    const Eigen::MatrixXd inverse = inverseDynamic(covariance, config.covariance_floor);
    Eigen::VectorXd state = beliefs[index].mean;
    state(static_cast<Eigen::Index>(dimension - 1U)) =
        anchor_yaw + angleDifference(
                        state(static_cast<Eigen::Index>(dimension - 1U)), anchor_yaw);
    information += result.weights[index] * inverse;
    information_vector += result.weights[index] * inverse * state;
  }
  result.belief.mean = inverseDynamic(information, config.covariance_floor) *
                       information_vector;
  result.belief.mean(static_cast<Eigen::Index>(dimension - 1U)) =
      wrapAngle(result.belief.mean(static_cast<Eigen::Index>(dimension - 1U)));
  result.belief.covariance = regularizeDynamic(
      inverseDynamic(information, config.covariance_floor),
      config.covariance_floor);
  result.belief.timestamp = self.timestamp;
  for (const auto& peer : beliefs) {
    result.belief.timestamp = std::max(result.belief.timestamp, peer.timestamp);
  }
  result.belief.sequence = self.sequence;
  result.belief.valid = true;
  result.success = true;
  result.status = "ok";
  return result;
}

}  // namespace hercules_localization
