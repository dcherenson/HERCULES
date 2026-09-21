#include "hercules_localization_ros/localization_adapter.hpp"

#include <hercules_localization/localization.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include "hercules_localization/global_ci.hpp"
#include "hercules_localization/gs_ci.hpp"
#include "hercules_localization/math.hpp"
#include "hercules_localization/recursive_localization.hpp"

namespace hercules_localization_ros {
namespace {

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
  return value;
}

template <typename Array>
bool finiteArray(const Array &values) {
  for (const auto value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

template <typename Array>
double diagonalSum(const Array &covariance, std::size_t dimension) {
  double result = 0.0;
  for (std::size_t index = 0; index < dimension; ++index) {
    result += covariance[index * dimension + index];
  }
  return result;
}

bool validGlobalWire(const GlobalCiBelief &wire) {
  if (!wire.valid || wire.ordered_agent_ids.empty() ||
      (wire.sender_id.empty() && wire.agent_id.empty()) || !std::isfinite(wire.timestamp)) {
    return false;
  }
  const std::size_t dimension = 2U * wire.ordered_agent_ids.size() + 1U;
  return wire.global_mean.size() == dimension &&
         wire.global_covariance.size() == dimension * dimension &&
         finiteArray(wire.global_mean) && finiteArray(wire.global_covariance);
}

hercules_localization::GlobalCiBelief coreGlobalBelief(const GlobalCiBelief &wire) {
  hercules_localization::GlobalCiBelief result;
  result.sender_id = wire.sender_id.empty() ? wire.agent_id : wire.sender_id;
  result.sequence = wire.sequence;
  result.ordered_agent_ids = wire.ordered_agent_ids;
  const Eigen::Index dimension = static_cast<Eigen::Index>(wire.global_mean.size());
  result.mean = Eigen::Map<const Eigen::VectorXd>(wire.global_mean.data(), dimension);
  result.covariance = Eigen::Map<const Eigen::MatrixXd>(
      wire.global_covariance.data(), dimension, dimension);
  result.timestamp = wire.timestamp;
  result.valid = wire.valid;
  return result;
}

void stampMetadata(LocalizationEstimate &estimate, const LocalizationMeasurement &measurement,
                   const std::string &agent_id, const std::string &algorithm,
                   std::uint64_t sequence) {
  if (estimate.header.stamp.sec == 0 && estimate.header.stamp.nanosec == 0) {
    estimate.header = measurement.header;
  }
  if (estimate.agent_id.empty()) {
    estimate.agent_id = agent_id;
  }
  if (estimate.frame_id.empty()) {
    estimate.frame_id = measurement.frame_id;
  }
  if (estimate.algorithm.empty()) {
    estimate.algorithm = algorithm;
  }
  estimate.sequence = sequence;
}

void stampMetadata(LocalizationDiagnostics &diagnostics, const LocalizationMeasurement &measurement,
                   const std::string &agent_id, const std::string &algorithm,
                   std::uint64_t updates, std::uint64_t measurements_received,
                   std::uint64_t peer_estimates_received, std::uint64_t measurements_rejected,
                   std::uint64_t peer_estimates_rejected) {
  if (diagnostics.header.stamp.sec == 0 && diagnostics.header.stamp.nanosec == 0) {
    diagnostics.header = measurement.header;
  }
  if (diagnostics.agent_id.empty()) {
    diagnostics.agent_id = agent_id;
  }
  if (diagnostics.algorithm.empty()) {
    diagnostics.algorithm = algorithm;
  }
  diagnostics.updates = updates;
  diagnostics.measurements_received = measurements_received;
  diagnostics.peer_estimates_received = peer_estimates_received;
  diagnostics.measurements_rejected = measurements_rejected;
  diagnostics.peer_estimates_rejected = peer_estimates_rejected;
}

}  // namespace

LocalizationAdapter::LocalizationAdapter(std::string agent_id, std::string algorithm)
    : agent_id_(std::move(agent_id)) {
  if (agent_id_.empty()) {
    throw std::invalid_argument("localization agent_id must not be empty");
  }
  if (!setAlgorithm(algorithm)) {
    throw std::invalid_argument("unsupported localization algorithm: " + algorithm);
  }
}

std::string LocalizationAdapter::canonicalAlgorithm(const std::string &algorithm) {
  const auto value = lower(algorithm);
  if (value == "recursive_decentralized" || value == "recursive-decentralized" ||
      value == "recursive" || value == "rde") {
    return kRecursiveDecentralized;
  }
  if (value == "gs_ci" || value == "gs-ci" || value == "gsci" || value == "ci") {
    return kGsCi;
  }
  return {};
}

bool LocalizationAdapter::isSupportedAlgorithm(const std::string &algorithm) {
  return !canonicalAlgorithm(algorithm).empty();
}

bool LocalizationAdapter::setAlgorithm(const std::string &algorithm) {
  const auto canonical = canonicalAlgorithm(algorithm);
  if (canonical.empty()) {
    return false;
  }
  algorithm_ = canonical;
  return true;
}

bool LocalizationAdapter::setGlobalAgentIds(
    std::vector<std::string> ordered_agent_ids) {
  const std::set<std::string> unique_ids(
      ordered_agent_ids.begin(), ordered_agent_ids.end());
  if (ordered_agent_ids.empty() || unique_ids.size() != ordered_agent_ids.size() ||
      unique_ids.count(agent_id_) != 1U || unique_ids.count("") != 0U ||
      unique_ids.count("Target1") != 0U) {
    return false;
  }
  if (got_gps_ && global_belief_.has_value() &&
      global_belief_->ordered_agent_ids != ordered_agent_ids) {
    return false;
  }
  global_agent_ids_ = std::move(ordered_agent_ids);
  return true;
}

bool LocalizationAdapter::registerAlgorithm(const std::string &algorithm, UpdateCallback callback) {
  const auto canonical = canonicalAlgorithm(algorithm);
  if (canonical.empty() || !callback) {
    return false;
  }
  callbacks_[canonical] = std::move(callback);
  return true;
}

void LocalizationAdapter::setConfig(
    const hercules_localization::LocalizationConfig &config) {
  config_ = config;
  if (recursive_state_.has_value()) recursive_state_->setConfig(config_);
}

bool LocalizationAdapter::syncEstimateFromGlobalBelief() {
  if (!global_belief_.has_value() ||
      !hercules_localization::isValidGlobalCiBelief(
          *global_belief_, config_.covariance_floor)) {
    return false;
  }
  const auto own = std::find(global_belief_->ordered_agent_ids.begin(),
                             global_belief_->ordered_agent_ids.end(), agent_id_);
  if (own == global_belief_->ordered_agent_ids.end()) return false;
  const Eigen::Index position_offset = static_cast<Eigen::Index>(
      2U * static_cast<std::size_t>(
               std::distance(global_belief_->ordered_agent_ids.begin(), own)));
  const Eigen::Index yaw_offset = global_belief_->mean.size() - 1;
  estimate_.mean.position = global_belief_->mean.segment<2>(position_offset);
  estimate_.mean.yaw = hercules_localization::wrapAngle(
      global_belief_->mean[yaw_offset]);
  estimate_.covariance.topLeftCorner<2, 2>() =
      global_belief_->covariance.block<2, 2>(position_offset, position_offset);
  estimate_.covariance.topRightCorner<2, 1>() =
      global_belief_->covariance.block<2, 1>(position_offset, yaw_offset);
  estimate_.covariance.bottomLeftCorner<1, 2>() =
      global_belief_->covariance.block<1, 2>(yaw_offset, position_offset);
  estimate_.covariance(2, 2) = global_belief_->covariance(yaw_offset, yaw_offset);
  estimate_.covariance = hercules_localization::regularizeCovariance(
      estimate_.covariance, config_.covariance_floor);
  estimate_.timestamp = global_belief_->timestamp;
  estimate_.valid = true;
  return true;
}

hercules_localization::RecursiveTransactionResult
LocalizationAdapter::installRecursivePairPosterior(
    const std::string &peer_id,
    const hercules_localization::PoseEstimate &posterior,
    const Eigen::Matrix3d &peer_factor,
    const std::string &event_id, std::uint64_t sequence) {
  hercules_localization::RecursiveTransactionResult result;
  if (algorithm_ != kRecursiveDecentralized || !recursive_state_.has_value()) {
    result.rejected = true;
    result.status = "recursive_state_unavailable";
    result.result.status = result.status;
    return result;
  }
  recursive_state_->setConfig(config_);
  result = recursive_state_->installPairPosterior(
      peer_id, posterior, peer_factor, event_id, sequence);
  if (result.accepted && !result.duplicate) {
    estimate_ = recursive_state_->estimate();
    ++accepted_relative_updates_;
  }
  return result;
}

bool LocalizationAdapter::finiteMeasurement(const LocalizationMeasurement &measurement) {
  if (!finiteArray(measurement.position) || !finiteArray(measurement.velocity) ||
      !finiteArray(measurement.orientation) || !std::isfinite(measurement.yaw) ||
      !std::isfinite(measurement.yaw_rate) || !finiteArray(measurement.covariance) ||
      !std::isfinite(measurement.robustness_margin)) {
    return false;
  }
  const Eigen::Map<const Eigen::Matrix<double, 3, 3>> covariance(measurement.covariance.data());
  if ((covariance - covariance.transpose()).norm() >
      1e-8 * (1.0 + covariance.norm())) return false;
  const Eigen::LLT<Eigen::Matrix3d> factor(covariance);
  return factor.info() == Eigen::Success;
}

bool LocalizationAdapter::finitePeerEstimate(const LocalizationPeerEstimate &estimate) {
  if (!finiteArray(estimate.position) || !finiteArray(estimate.velocity) ||
      !finiteArray(estimate.orientation) || !std::isfinite(estimate.yaw) ||
      !std::isfinite(estimate.yaw_rate) || !finiteArray(estimate.covariance) ||
      !std::isfinite(estimate.robustness_margin)) {
    return false;
  }
  const Eigen::Map<const Eigen::Matrix<double, 3, 3>> covariance(estimate.covariance.data());
  if ((covariance - covariance.transpose()).norm() >
      1e-8 * (1.0 + covariance.norm())) return false;
  const Eigen::LLT<Eigen::Matrix3d> factor(covariance);
  return factor.info() == Eigen::Success;
}

AdapterOutput LocalizationAdapter::passThrough(const LocalizationMeasurement &measurement,
                                               std::size_t valid_peer_count,
                                               std::size_t rejected_peer_count) {
  AdapterOutput output;
  const bool safe_measurement = measurement.valid && finiteMeasurement(measurement);
  output.estimate.header = measurement.header;
  output.estimate.agent_id = agent_id_;
  output.estimate.frame_id = measurement.frame_id;
  output.estimate.position = safe_measurement ? measurement.position
                                              : std::array<double, 3>{0.0, 0.0, 0.0};
  output.estimate.velocity = safe_measurement ? measurement.velocity
                                              : std::array<double, 3>{0.0, 0.0, 0.0};
  output.estimate.orientation = safe_measurement ? measurement.orientation
                                                 : std::array<double, 4>{1.0, 0.0, 0.0, 0.0};
  output.estimate.yaw = safe_measurement ? measurement.yaw : 0.0;
  output.estimate.yaw_rate = safe_measurement ? measurement.yaw_rate : 0.0;
  output.estimate.covariance = safe_measurement
      ? measurement.covariance
      : std::array<double, 9>{1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0};
  output.estimate.robustness_margin = safe_measurement
      ? measurement.robustness_margin : 0.0;
  output.estimate.algorithm = algorithm_;
  output.estimate.sequence = sequence_;
  output.estimate.initialized = safe_measurement;
  output.estimate.valid = safe_measurement;

  output.diagnostics.header = measurement.header;
  output.diagnostics.agent_id = agent_id_;
  output.diagnostics.algorithm = algorithm_;
  output.diagnostics.initialized = safe_measurement;
  output.diagnostics.valid = safe_measurement;
  output.diagnostics.peer_estimates_received = valid_peer_count;
  output.diagnostics.peer_estimates_rejected = rejected_peer_count;
  output.diagnostics.robustness_margin = output.estimate.robustness_margin;
  output.diagnostics.covariance_trace = diagonalSum(output.estimate.covariance, 3);
  output.diagnostics.status = safe_measurement ? "pass_through" : "invalid_measurement";
  output.diagnostics.communication_state = safe_measurement ? "local_only" : "rejected";
  output.diagnostics.detail =
      "No numerical callback registered; local observation forwarded unchanged";
  return output;
}

AdapterOutput LocalizationAdapter::numericalUpdate(
    const LocalizationMeasurement &measurement,
    const std::vector<LocalizationPeerEstimate> &peers,
    const std::vector<PlanarRelativeMeasurement> &relative,
    const std::vector<GlobalCiBelief> &global_beliefs) {
  AdapterOutput output;
  // The conformal/robustness stage supplies this per-update margin on the
  // wire; keep the static config for all other numerical settings.
  config_.robustness_margin = measurement.robustness_margin;
  if (recursive_state_.has_value()) recursive_state_->setConfig(config_);
  const auto source = measurement.source_id;
  const bool is_gps = source == "gps" || source == "global_gps";
  const bool is_odom = source == "odometry" || source == "odom_local";
  const bool was_initialized = got_gps_;
  if (is_odom) {
    latest_odom_yaw_ = measurement.yaw;
    have_latest_odom_yaw_ = true;
  }
  if (is_odom && !got_gps_ && allow_odom_seed_) {
    estimate_.mean = hercules_localization::Pose2D(
        measurement.position[0], measurement.position[1], measurement.yaw);
    estimate_.covariance = Eigen::Map<const Eigen::Matrix<double, 3, 3>>(
        measurement.covariance.data());
    estimate_.valid = true;
    got_gps_ = true;
    allow_odom_seed_ = false;
    if (algorithm_ == kRecursiveDecentralized) {
      recursive_state_.emplace(agent_id_, estimate_, config_);
    }
  }
  // GPS initializes the common pose once. Subsequent GPS samples are not
  // allowed to teleport a running odometry/relative-fusion estimate.
  if (is_gps && !got_gps_) {
    const double initial_yaw = have_latest_odom_yaw_ ? latest_odom_yaw_ : measurement.yaw;
    estimate_.mean = hercules_localization::Pose2D(
        measurement.position[0], measurement.position[1], initial_yaw);
    estimate_.covariance = Eigen::Map<const Eigen::Matrix<double, 3, 3>>(
        measurement.covariance.data());
    estimate_.timestamp = static_cast<double>(measurement.header.stamp.sec) +
        1e-9 * static_cast<double>(measurement.header.stamp.nanosec);
    estimate_.valid = true;
    latest_velocity_ = Eigen::Vector2d(measurement.velocity[0], measurement.velocity[1]);
    latest_yaw_rate_ = measurement.yaw_rate;
    got_gps_ = true;
    initial_yaw_from_odometry_ = have_latest_odom_yaw_;
    if (algorithm_ == kRecursiveDecentralized) {
      recursive_state_.emplace(agent_id_, estimate_, config_);
    } else if (algorithm_ == kGsCi) {
      const auto ordered_ids = global_agent_ids_.empty()
          ? std::vector<std::string>{agent_id_}
          : global_agent_ids_;
      auto belief = hercules_localization::initializeGlobalCiBelief(
          agent_id_, ordered_ids, estimate_, 1.0e6, config_.covariance_floor);
      if (belief.valid) global_belief_ = std::move(belief);
    }
  }
  // Drone1 is the configured absolute anchor.  Keep consuming its later GPS
  // fixes as private EKF observations; all other agents intentionally use
  // only their first fix plus odometry increments and cooperative contacts.
  if (is_gps && was_initialized && agent_id_ == anchor_agent_id_ && estimate_.valid) {
    if (algorithm_ == kRecursiveDecentralized && recursive_state_.has_value()) {
      const Eigen::Matrix3d gps_covariance =
          Eigen::Map<const Eigen::Matrix<double, 3, 3>>(
              measurement.covariance.data());
      recursive_state_->applyPrivatePositionMeasurement(
          Eigen::Vector2d(measurement.position[0], measurement.position[1]),
          gps_covariance.topLeftCorner<2, 2>(),
          static_cast<double>(measurement.header.stamp.sec) +
              1e-9 * static_cast<double>(measurement.header.stamp.nanosec));
      estimate_ = recursive_state_->estimate();
    } else if (algorithm_ == kGsCi && global_belief_.has_value()) {
      const Eigen::Matrix3d gps_covariance =
          Eigen::Map<const Eigen::Matrix<double, 3, 3>>(
              measurement.covariance.data());
      const double timestamp = static_cast<double>(measurement.header.stamp.sec) +
          1e-9 * static_cast<double>(measurement.header.stamp.nanosec);
      const auto updated = hercules_localization::globalCiPrivatePositionUpdate(
          *global_belief_, agent_id_,
          Eigen::Vector2d(measurement.position[0], measurement.position[1]),
          gps_covariance.topLeftCorner<2, 2>(), timestamp, config_);
      if (updated.success) {
        global_belief_ = updated.belief;
        syncEstimateFromGlobalBelief();
      }
    } else {
    const Eigen::Matrix3d prior_covariance =
        hercules_localization::regularizeCovariance(
            estimate_.covariance, config_.covariance_floor);
    const Eigen::Matrix3d raw_gps_covariance =
        Eigen::Map<const Eigen::Matrix<double, 3, 3>>(
            measurement.covariance.data());
    const Eigen::Matrix3d gps_covariance =
        hercules_localization::regularizeCovariance(
            raw_gps_covariance, config_.covariance_floor);
    Eigen::Vector2d innovation;
    innovation << measurement.position[0] - estimate_.mean.position.x(),
        measurement.position[1] - estimate_.mean.position.y();
    const Eigen::Matrix<double, 2, 3> position_selector =
        (Eigen::Matrix<double, 2, 3>() << 1.0, 0.0, 0.0,
                                           0.0, 1.0, 0.0).finished();
    const Eigen::Matrix2d gps_position_covariance =
        gps_covariance.topLeftCorner<2, 2>();
    const Eigen::Matrix2d raw_innovation_covariance =
        position_selector * prior_covariance * position_selector.transpose() +
        gps_position_covariance;
    const Eigen::Matrix2d innovation_covariance =
        hercules_localization::regularizeCovariance(
            raw_innovation_covariance, config_.covariance_floor);
    const Eigen::Matrix<double, 3, 2> gain = prior_covariance *
        position_selector.transpose() * hercules_localization::safeInverse(
            innovation_covariance, config_.covariance_floor);
    estimate_.mean = hercules_localization::Pose2D::fromVector(
        estimate_.vector() + gain * innovation);
    estimate_.mean.yaw = hercules_localization::wrapAngle(estimate_.mean.yaw);
    const Eigen::Matrix3d residual = Eigen::Matrix3d::Identity() - gain * position_selector;
    const Eigen::Matrix3d raw_posterior_covariance =
        residual * prior_covariance * residual.transpose() +
        gain * gps_position_covariance * gain.transpose();
    estimate_.covariance = hercules_localization::regularizeCovariance(
        raw_posterior_covariance, config_.covariance_floor);
    }
  }
  if (is_odom && got_gps_) {
    const Eigen::Vector2d position(measurement.position[0], measurement.position[1]);
    const double yaw = measurement.yaw;
    if (!have_odom_) {
      // GPS and odometry are independent ROS streams.  If the first GPS fix
      // won the race, replace its placeholder heading with the first actual
      // odometry yaw before any odometry delta or cooperative observation is
      // applied.
      if (!initial_yaw_from_odometry_) {
        estimate_.mean.yaw = hercules_localization::wrapAngle(yaw);
        estimate_.covariance(0, 2) = estimate_.covariance(2, 0) = 0.0;
        estimate_.covariance(1, 2) = estimate_.covariance(2, 1) = 0.0;
        estimate_.covariance(2, 2) = measurement.covariance[8];
        estimate_.covariance = hercules_localization::regularizeCovariance(
            estimate_.covariance, config_.covariance_floor);
        if (algorithm_ == kRecursiveDecentralized) {
          if (!recursive_state_.has_value()) {
            recursive_state_.emplace(agent_id_, estimate_, config_);
          } else if (recursive_state_->replaceInitialYaw(
                         estimate_.mean.yaw, estimate_.covariance(2, 2))) {
            estimate_ = recursive_state_->estimate();
          }
        } else if (algorithm_ == kGsCi && global_belief_.has_value()) {
          const Eigen::Index yaw_offset = global_belief_->mean.size() - 1;
          global_belief_->mean[yaw_offset] = estimate_.mean.yaw;
          global_belief_->covariance.col(yaw_offset).setZero();
          global_belief_->covariance.row(yaw_offset).setZero();
          global_belief_->covariance(yaw_offset, yaw_offset) = estimate_.covariance(2, 2);
          global_belief_->covariance = hercules_localization::regularizeCovariance(
              Eigen::Ref<const Eigen::MatrixXd>(global_belief_->covariance),
              config_.covariance_floor);
          global_belief_->valid = true;
          syncEstimateFromGlobalBelief();
        }
        initial_yaw_from_odometry_ = true;
      }
      previous_odom_position_ = position;
      previous_odom_yaw_ = yaw;
      have_odom_ = true;
    } else {
      const Eigen::Vector2d position_delta = position - previous_odom_position_;
      const double yaw_delta = hercules_localization::angleDifference(yaw, previous_odom_yaw_);
      const double measurement_timestamp =
          static_cast<double>(measurement.header.stamp.sec) +
          1e-9 * static_cast<double>(measurement.header.stamp.nanosec);
      const double dt = std::max(0.0, measurement_timestamp - estimate_.timestamp);
      if (algorithm_ == kGsCi && global_belief_.has_value()) {
        const Eigen::Matrix3d process_covariance =
            hercules_localization::regularizeCovariance(
                Eigen::Matrix3d(Eigen::Map<const Eigen::Matrix<double, 3, 3>>(
                    measurement.covariance.data())),
                config_.process_covariance_floor);
        hercules_localization::PoseVector world_delta;
        world_delta << position_delta.x(), position_delta.y(), yaw_delta;
        auto propagated = hercules_localization::propagateGlobalCiBelief(
            *global_belief_, agent_id_, world_delta, process_covariance, dt,
            config_.unknown_motion_variance, config_.covariance_floor);
        if (propagated.valid) {
          global_belief_ = std::move(propagated);
          syncEstimateFromGlobalBelief();
        }
      }
      if (algorithm_ == kRecursiveDecentralized) {
        if (!recursive_state_.has_value()) {
          recursive_state_.emplace(agent_id_, estimate_, config_);
        }
        Eigen::Matrix3d process_covariance =
            Eigen::Map<const Eigen::Matrix<double, 3, 3>>(
                measurement.covariance.data());
        process_covariance = hercules_localization::regularizeCovariance(
            process_covariance, config_.process_covariance_floor);
        hercules_localization::PoseVector world_delta;
        world_delta << position_delta.x(), position_delta.y(), yaw_delta;
        recursive_state_->propagateWorldDelta(
            world_delta, process_covariance, measurement_timestamp);
        estimate_ = recursive_state_->estimate();
      } else if (!global_belief_.has_value()) {
        estimate_.mean.position += position_delta;
        estimate_.mean.yaw = hercules_localization::wrapAngle(
            estimate_.mean.yaw + yaw_delta);
        estimate_.covariance += Eigen::Matrix3d::Identity() * 0.0025;
        estimate_.covariance = hercules_localization::regularizeCovariance(
            estimate_.covariance, config_.covariance_floor);
      }
      previous_odom_position_ = position;
      previous_odom_yaw_ = yaw;
    }
    latest_velocity_ = Eigen::Vector2d(measurement.velocity[0], measurement.velocity[1]);
    latest_yaw_rate_ = measurement.yaw_rate;
    estimate_.timestamp = static_cast<double>(measurement.header.stamp.sec) +
        1e-9 * static_cast<double>(measurement.header.stamp.nanosec);
  }
  if (!got_gps_ || !estimate_.valid) {
    output = passThrough(measurement, peers.size(), 0);
    output.estimate.initialized = false;
    output.estimate.valid = false;
    output.diagnostics.initialized = false;
    output.diagnostics.valid = false;
    output.diagnostics.status = "waiting_for_first_gps";
    return output;
  }

  // GS-CI exchanges an ordered global position state plus the sender's yaw.
  // Build the local belief from the current planar estimate, retain remote
  // blocks from the previous CI result when available, and fuse only fresh,
  // dimensionally valid peer beliefs.  The local marginal is then fed back
  // into the adapter's pose state for the common control/logging path.
  if (algorithm_ == kGsCi && global_belief_.has_value() && !global_beliefs.empty()) {
    global_belief_->sender_id = agent_id_;
    global_belief_->sequence = sequence_;
    std::vector<hercules_localization::GlobalCiBelief> peers_for_ci;
    peers_for_ci.reserve(global_beliefs.size());
    for (const auto &wire : global_beliefs) {
      if (validGlobalWire(wire) &&
          wire.ordered_agent_ids == global_belief_->ordered_agent_ids) {
        peers_for_ci.push_back(coreGlobalBelief(wire));
      }
    }
    const auto ci = hercules_localization::globalCiUpdate(
        *global_belief_, peers_for_ci, config_);
    if (ci.success) {
      global_belief_ = ci.belief;
      syncEstimateFromGlobalBelief();
    }
  }

  std::vector<hercules_localization::RelativeObservation> observations;
  std::vector<std::pair<std::string, std::uint64_t>> observation_sequences;
  std::size_t gs_used_measurements = 0U;
  std::size_t gs_rejected_measurements = 0U;
  std::vector<std::pair<std::string, std::uint64_t>> applied_sequences;
  for (const auto &item : relative) {
    const std::string observer = !item.observer_id.empty() ? item.observer_id : item.source_id;
    const std::string observed = !item.observed_id.empty() ? item.observed_id : item.target_id;
    if (!item.valid || observer != agent_id_ || observed.empty()) continue;
    const std::string pair_key = observer + "->" + observed;
    if (item.sequence != 0) {
      const auto applied = applied_relative_sequences_.find(pair_key);
      if (applied != applied_relative_sequences_.end() && item.sequence <= applied->second) {
        continue;
      }
    }
    const Eigen::Map<const Eigen::Matrix<double, 2, 2>> wire_covariance(item.covariance.data());
    const Eigen::LLT<Eigen::Matrix2d> covariance_factor(wire_covariance);
    if (!wire_covariance.allFinite() || covariance_factor.info() != Eigen::Success) {
      continue;
    }
    hercules_localization::RangeBearingMeasurement core_measurement;
    core_measurement.neighbor_id = observed;
    core_measurement.range = item.range;
    core_measurement.bearing = item.bearing;
    if (!std::isfinite(core_measurement.range) ||
        !std::isfinite(core_measurement.bearing) ||
        core_measurement.range <= config_.range_epsilon) {
      const double x = item.relative_position[0];
      const double y = item.relative_position[1];
      core_measurement.range = std::hypot(x, y);
      core_measurement.bearing = std::atan2(y, x);
    }
    core_measurement.covariance << item.covariance[0], item.covariance[1],
        item.covariance[2], item.covariance[3];
    core_measurement.timestamp = item.timestamp;
    core_measurement.valid = item.valid;
    if (algorithm_ == kGsCi) {
      if (!global_belief_.has_value()) {
        ++gs_rejected_measurements;
        continue;
      }
      const auto updated = hercules_localization::globalCiRangeBearingUpdate(
          *global_belief_, observer, observed, core_measurement, config_);
      if (!updated.success) {
        ++gs_rejected_measurements;
        continue;
      }
      global_belief_ = updated.belief;
      syncEstimateFromGlobalBelief();
      ++gs_used_measurements;
      applied_sequences.emplace_back(pair_key, item.sequence);
      continue;
    }
    const auto peer = std::find_if(peers.begin(), peers.end(), [&](const auto &candidate) {
      return candidate.sender_id == observed && candidate.valid;
    });
    if (peer == peers.end()) continue;
    hercules_localization::RelativeObservation observation;
    observation.measurement = core_measurement;
    observation.neighbor.agent_id = observed;
    observation.neighbor.mean = hercules_localization::Pose2D(
        peer->position[0], peer->position[1], peer->yaw);
    observation.neighbor.covariance = Eigen::Map<const Eigen::Matrix<double, 3, 3>>(
        peer->covariance.data());
    observation.neighbor.timestamp = static_cast<double>(peer->header.stamp.sec) +
        1e-9 * static_cast<double>(peer->header.stamp.nanosec);
    observation.neighbor.valid = peer->valid;
    observations.push_back(observation);
    observation_sequences.emplace_back(pair_key, item.sequence);
  }
  hercules_localization::LocalizationResult result;
  if (algorithm_ == kGsCi) {
    result.estimate = estimate_;
    result.success = global_belief_.has_value() && global_belief_->valid && estimate_.valid;
    result.status = result.success
        ? (gs_used_measurements > 0U ? "ok" : "no_measurements")
        : "invalid_global_belief";
    result.used_measurements = gs_used_measurements;
    result.rejected_measurements = gs_rejected_measurements;
  } else {
    if (!recursive_state_.has_value()) {
      recursive_state_.emplace(agent_id_, estimate_, config_);
    }
    result.estimate = recursive_state_->estimate();
    result.success = true;
    result.status = observations.empty() ? "no_measurements" : "no_valid_measurements";
    for (std::size_t index = 0; index < observations.size(); ++index) {
      const auto &observation = observations[index];
      const std::string &peer_id = observation.measurement.neighbor_id;
      if (recursive_state_->correlationFactors().count(peer_id) == 0U) {
        recursive_state_->initializePeer(peer_id);
      }
      hercules_localization::RecursivePairTransaction transaction;
      transaction.observer_id = agent_id_;
      transaction.observed_id = peer_id;
      transaction.measurement = observation.measurement;
      transaction.observed = observation.neighbor;
      transaction.reciprocal_factor = Eigen::Matrix3d::Identity();
      if (index < observation_sequences.size()) {
        transaction.sequence = observation_sequences[index].second;
        if (transaction.sequence != 0U) {
          transaction.event_id = observation_sequences[index].first + ":" +
              std::to_string(transaction.sequence);
        }
      }
      const auto transaction_result =
          recursive_state_->applyPairTransaction(transaction);
      if (transaction_result.accepted) {
        if (index < observation_sequences.size()) {
          applied_sequences.push_back(observation_sequences[index]);
        }
        if (!transaction_result.duplicate) {
          ++result.used_measurements;
          result.maximum_innovation = std::max(
              result.maximum_innovation,
              transaction_result.result.maximum_innovation);
          result.maximum_mahalanobis_squared = std::max(
              result.maximum_mahalanobis_squared,
              transaction_result.result.maximum_mahalanobis_squared);
        }
      } else if (transaction_result.rejected) {
        ++result.rejected_measurements;
      }
    }
    estimate_ = recursive_state_->estimate();
    result.estimate = estimate_;
    result.status = result.used_measurements > 0 ? "ok" : result.status;
  }
  accepted_relative_updates_ += static_cast<std::uint64_t>(std::max(0, result.used_measurements));
  for (const auto &[pair_key, sequence] : applied_sequences) {
    if (sequence != 0) applied_relative_sequences_[pair_key] = sequence;
  }
  if (result.success) estimate_ = result.estimate;
  output.estimate.header = measurement.header;
  output.estimate.agent_id = agent_id_;
  output.estimate.frame_id = measurement.frame_id.empty() ? "airsim_world_ned" : measurement.frame_id;
  output.estimate.position = {estimate_.mean.position.x(), estimate_.mean.position.y(), measurement.position[2]};
  output.estimate.velocity = {latest_velocity_.x(), latest_velocity_.y(), measurement.velocity[2]};
  output.estimate.orientation = measurement.orientation;
  output.estimate.yaw = estimate_.mean.yaw;
  output.estimate.yaw_rate = latest_yaw_rate_;
  output.estimate.covariance = {estimate_.covariance(0, 0), estimate_.covariance(0, 1), estimate_.covariance(0, 2),
      estimate_.covariance(1, 0), estimate_.covariance(1, 1), estimate_.covariance(1, 2),
      estimate_.covariance(2, 0), estimate_.covariance(2, 1), estimate_.covariance(2, 2)};
  output.estimate.robustness_margin = config_.robustness_margin;
  output.estimate.algorithm = algorithm_;
  output.estimate.initialized = got_gps_;
  output.estimate.valid = result.success && estimate_.valid;
  output.estimate.stale = false;
  output.diagnostics.header = measurement.header;
  output.diagnostics.agent_id = agent_id_;
  output.diagnostics.algorithm = algorithm_;
  output.diagnostics.initialized = got_gps_;
  output.diagnostics.valid = output.estimate.valid;
  output.diagnostics.stale = false;
  output.diagnostics.measurements_received = measurements_received_;
  output.diagnostics.measurements_rejected = measurements_rejected_ + result.rejected_measurements;
  output.diagnostics.peer_estimates_received = peer_estimates_received_;
  output.diagnostics.peer_estimates_rejected = peer_estimates_rejected_;
  output.diagnostics.accepted_relative_updates = accepted_relative_updates_;
  const Eigen::LLT<Eigen::Matrix3d> covariance_factor(estimate_.covariance);
  output.diagnostics.covariance_spd = estimate_.covariance.allFinite() &&
                                      covariance_factor.info() == Eigen::Success;
  output.diagnostics.covariance_trace = estimate_.covariance.trace();
  output.diagnostics.robustness_margin = config_.robustness_margin;
  output.diagnostics.status = result.status;
  output.diagnostics.detail = result.message;
  output.diagnostics.communication_state = result.success ? "accepted" : "rejected";
  output.diagnostics.innovation_norm = result.maximum_innovation;
  return output;
}

AdapterOutput LocalizationAdapter::update(
    const LocalizationMeasurement &measurement,
    const std::vector<LocalizationPeerEstimate> &peer_estimates) {
  ++updates_;
  ++measurements_received_;
  ++sequence_;

  std::vector<LocalizationPeerEstimate> valid_peers;
  valid_peers.reserve(peer_estimates.size());
  for (const auto &peer : peer_estimates) {
    if (peer.valid && finitePeerEstimate(peer)) {
      valid_peers.push_back(peer);
      ++peer_estimates_received_;
    } else {
      ++peer_estimates_rejected_;
    }
  }

  const bool valid_measurement = measurement.valid && finiteMeasurement(measurement);
  if (!valid_measurement) {
    ++measurements_rejected_;
  }

  AdapterOutput output;
  const auto callback = callbacks_.find(algorithm_);
  if (valid_measurement && callback != callbacks_.end()) {
    output = callback->second(measurement, valid_peers, algorithm_);
  } else {
    output = passThrough(measurement, valid_peers.size(), peer_estimates.size() - valid_peers.size());
  }

  stampMetadata(output.estimate, measurement, agent_id_, algorithm_, sequence_);
  stampMetadata(output.diagnostics, measurement, agent_id_, algorithm_, updates_,
                measurements_received_, peer_estimates_received_, measurements_rejected_,
                peer_estimates_rejected_);
  if (output.diagnostics.status.empty()) {
    output.diagnostics.status = valid_measurement ? "ok" : "invalid_measurement";
  }
  if (!std::isfinite(output.diagnostics.robustness_margin)) {
    output.diagnostics.robustness_margin = measurement.robustness_margin;
  }
  return output;
}

AdapterOutput LocalizationAdapter::update(
    const LocalizationMeasurement &measurement,
    const std::vector<LocalizationPeerEstimate> &peer_estimates,
    const std::vector<PlanarRelativeMeasurement> &relative_measurements,
    const std::vector<GlobalCiBelief> &global_beliefs) {
  ++updates_;
  ++measurements_received_;
  ++sequence_;
  std::vector<LocalizationPeerEstimate> valid_peers;
  for (const auto &peer : peer_estimates) {
    if (peer.valid && finitePeerEstimate(peer)) { valid_peers.push_back(peer); ++peer_estimates_received_; }
    else ++peer_estimates_rejected_;
  }
  const bool valid_measurement = measurement.valid && finiteMeasurement(measurement);
  if (!valid_measurement) ++measurements_rejected_;
  const auto callback = callbacks_.find(algorithm_);
  auto output = !valid_measurement
                    ? passThrough(measurement, valid_peers.size(),
                                  peer_estimates.size() - valid_peers.size())
                    : callback != callbacks_.end()
                        ? callback->second(measurement, valid_peers, algorithm_)
                        : numericalUpdate(measurement, valid_peers, relative_measurements,
                                          global_beliefs);
  stampMetadata(output.estimate, measurement, agent_id_, algorithm_, sequence_);
  stampMetadata(output.diagnostics, measurement, agent_id_, algorithm_, updates_, measurements_received_,
                peer_estimates_received_, measurements_rejected_, peer_estimates_rejected_);
  if (output.diagnostics.status.empty()) {
    output.diagnostics.status = valid_measurement ? "ok" : "invalid_measurement";
  }
  if (!std::isfinite(output.diagnostics.robustness_margin)) {
    output.diagnostics.robustness_margin = measurement.robustness_margin;
  }
  return output;
}

AdapterOutput LocalizationAdapter::update(
    const LocalizationMeasurement &measurement,
    const std::vector<LocalizationPeerEstimate> &peer_estimates,
    const std::vector<PlanarRelativeMeasurement> &relative_measurements) {
  ++updates_;
  ++measurements_received_;
  ++sequence_;
  std::vector<LocalizationPeerEstimate> valid_peers;
  for (const auto &peer : peer_estimates) {
    if (peer.valid && finitePeerEstimate(peer)) { valid_peers.push_back(peer); ++peer_estimates_received_; }
    else ++peer_estimates_rejected_;
  }
  const bool valid_measurement = measurement.valid && finiteMeasurement(measurement);
  if (!valid_measurement) ++measurements_rejected_;
  const auto callback = callbacks_.find(algorithm_);
  auto output = !valid_measurement
                    ? passThrough(measurement, valid_peers.size(),
                                  peer_estimates.size() - valid_peers.size())
                    : callback != callbacks_.end()
                        ? callback->second(measurement, valid_peers, algorithm_)
                        : numericalUpdate(measurement, valid_peers, relative_measurements, {});
  stampMetadata(output.estimate, measurement, agent_id_, algorithm_, sequence_);
  stampMetadata(output.diagnostics, measurement, agent_id_, algorithm_, updates_, measurements_received_,
                peer_estimates_received_, measurements_rejected_, peer_estimates_rejected_);
  if (output.diagnostics.status.empty()) {
    output.diagnostics.status = valid_measurement ? "ok" : "invalid_measurement";
  }
  if (!std::isfinite(output.diagnostics.robustness_margin)) {
    output.diagnostics.robustness_margin = measurement.robustness_margin;
  }
  return output;
}

void LocalizationAdapter::reset() {
  sequence_ = 0;
  updates_ = 0;
  measurements_received_ = 0;
  peer_estimates_received_ = 0;
  measurements_rejected_ = 0;
  peer_estimates_rejected_ = 0;
  accepted_relative_updates_ = 0;
  applied_relative_sequences_.clear();
  estimate_ = {};
  global_belief_.reset();
  recursive_state_.reset();
  got_gps_ = false;
  have_odom_ = false;
  have_latest_odom_yaw_ = false;
  initial_yaw_from_odometry_ = false;
  latest_odom_yaw_ = 0.0;
  allow_odom_seed_ = false;
}

LocalizationPeerEstimate LocalizationAdapter::toPeerEstimate(
    const LocalizationEstimate &estimate, const std::string &sender_id) {
  LocalizationPeerEstimate output;
  output.header = estimate.header;
  output.sender_id = sender_id;
  output.frame_id = estimate.frame_id;
  output.position = estimate.position;
  output.velocity = estimate.velocity;
  output.orientation = estimate.orientation;
  output.yaw = estimate.yaw;
  output.yaw_rate = estimate.yaw_rate;
  output.covariance = estimate.covariance;
  output.robustness_margin = estimate.robustness_margin;
  output.algorithm = estimate.algorithm;
  output.sequence = estimate.sequence;
  output.valid = estimate.valid;
  return output;
}

}  // namespace hercules_localization_ros
