#pragma once

#include <map>
#include <set>
#include <vector>

#include "hercules_localization/math.hpp"

namespace hercules_localization {

/** Perform one robust EKF-style update for a known neighbor estimate. */
ObservationUpdate updateFromRangeBearing(
    const PoseEstimate& prior, const RelativeObservation& observation,
    const LocalizationConfig& config = {});

/**
 * Sequential recursive decentralized localization.
 *
 * Measurements are consumed in the supplied order.  Each neighbor covariance
 * is projected into the range/bearing innovation covariance, so correlated or
 * uncertain neighbors cannot accidentally make the local estimate overconfident.
 */
LocalizationResult recursiveDecentralizedUpdate(
    const PoseEstimate& prior,
    const std::vector<RelativeObservation>& observations,
    const LocalizationConfig& config = {});

// Descriptive aliases used by adapters and callers that prefer the long name.
LocalizationResult recursiveDecentralizedLocalization(
    const PoseEstimate& prior,
    const std::vector<RelativeObservation>& observations,
    const LocalizationConfig& config = {});

/** Apply a sequence of explicitly identified pair transactions in order. */
LocalizationResult recursiveDecentralizedUpdate(
    const PoseEstimate& prior,
    const std::vector<RecursivePairTransaction>& transactions,
    const LocalizationConfig& config = {});

class RecursiveDecentralizedLocalizer {
 public:
  explicit RecursiveDecentralizedLocalizer(
      PoseEstimate initial = {}, LocalizationConfig config = {});

  void reset(const PoseEstimate& estimate);
  LocalizationResult update(
      const std::vector<RelativeObservation>& observations);
  LocalizationResult update(const PoseEstimate& prior,
                            const std::vector<RelativeObservation>& observations);

  const PoseEstimate& estimate() const { return estimate_; }
  const LocalizationConfig& config() const { return config_; }
  void setConfig(const LocalizationConfig& config) { config_ = config; }

 private:
  PoseEstimate estimate_;
  LocalizationConfig config_;
};

/**
 * Stateful recursive estimator boundary used by asynchronous ROS pair
 * transport.  It keeps one cross-correlation factor per peer and remembers
 * event IDs/sequence numbers so retries cannot apply an update twice.
 */
class RecursiveDecentralizedState {
 public:
  explicit RecursiveDecentralizedState(std::string agent_id = {},
                                       PoseEstimate initial = {},
                                       LocalizationConfig config = {});

  void reset(const PoseEstimate& estimate = {});
  void initializePeer(const std::string& peer_id,
                      const Eigen::Matrix3d& factor = Eigen::Matrix3d::Zero());
  void propagate(const Twist2D& twist, double dt,
                 const Eigen::Matrix3d& process_covariance = Eigen::Matrix3d::Zero());
  void propagateWorldDelta(
      const PoseVector& world_delta,
      const Eigen::Matrix3d& process_covariance = Eigen::Matrix3d::Zero(),
      double timestamp = 0.0);
  bool applyPrivatePositionMeasurement(
      const Eigen::Vector2d& position, const Eigen::Matrix2d& covariance,
      double timestamp = 0.0);
  /** Replace the placeholder startup yaw without discarding peer factors. */
  bool replaceInitialYaw(double yaw, double variance);
  RecursiveTransactionResult applyPairTransaction(
      const RecursivePairTransaction& transaction);
  RecursiveTransactionResult installPairPosterior(
      const std::string& peer_id, const PoseEstimate& posterior,
      const Eigen::Matrix3d& peer_factor, const std::string& event_id = {},
      std::uint64_t sequence = 0);

  const std::string& agentId() const { return agent_id_; }
  const PoseEstimate& estimate() const { return estimate_; }
  const CorrelationFactors& correlationFactors() const { return factors_; }
  const std::set<std::string>& processedEvents() const { return processed_events_; }
  void setConfig(const LocalizationConfig& config) { config_ = config; }

 private:
  std::string agent_id_;
  PoseEstimate estimate_;
  LocalizationConfig config_;
  CorrelationFactors factors_;
  std::set<std::string> processed_events_;
  std::map<std::string, std::uint64_t> latest_sequences_;
};

}  // namespace hercules_localization
