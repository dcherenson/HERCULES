#pragma once

#include <hercules_interfaces/msg/localization_diagnostics.hpp>
#include <hercules_interfaces/msg/localization_estimate.hpp>
#include <hercules_interfaces/msg/localization_measurement.hpp>
#include <hercules_interfaces/msg/localization_peer_estimate.hpp>
#include <hercules_interfaces/msg/planar_relative_measurement.hpp>
#include <hercules_interfaces/msg/global_ci_belief.hpp>
#include <hercules_localization/recursive_localization.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace hercules_localization_ros {

using LocalizationMeasurement = hercules_interfaces::msg::LocalizationMeasurement;
using LocalizationEstimate = hercules_interfaces::msg::LocalizationEstimate;
using LocalizationPeerEstimate = hercules_interfaces::msg::LocalizationPeerEstimate;
using LocalizationDiagnostics = hercules_interfaces::msg::LocalizationDiagnostics;
using PlanarRelativeMeasurement = hercules_interfaces::msg::PlanarRelativeMeasurement;
using GlobalCiBelief = hercules_interfaces::msg::GlobalCiBelief;

/** Names of the numerical methods exposed by the ROS boundary. */
constexpr char kRecursiveDecentralized[] = "recursive_decentralized";
constexpr char kGsCi[] = "gs_ci";

/**
 * Output from one adapter update.
 *
 * The adapter owns transport metadata and algorithm selection, while the
 * numerical implementation is dispatched to the pure core or, when
 * registered, supplied as a callback.
 */
struct AdapterOutput {
  LocalizationEstimate estimate;
  LocalizationDiagnostics diagnostics;
};

/**
 * Numerical-core hook used by the adapter.
 *
 * The callback receives the local observation, the peer messages retained by
 * the transport, and the canonical selected algorithm name. It may fill an
 * estimate and diagnostics; empty metadata is filled by the adapter.
 */
using UpdateCallback = std::function<AdapterOutput(
    const LocalizationMeasurement &, const std::vector<LocalizationPeerEstimate> &,
    const std::string &algorithm)>;

/**
 * ROS-independent conversion and lifecycle boundary for cooperative
 * localization. The numerical implementation is called only through the
 * separately built hercules_localization library.
 */
class LocalizationAdapter {
public:
  explicit LocalizationAdapter(std::string agent_id,
                               std::string algorithm = kRecursiveDecentralized);

  /** Return true and canonicalize an algorithm name, or leave it unchanged. */
  bool setAlgorithm(const std::string &algorithm);

  /** Register/rebind the numerical callback for one supported algorithm. */
  bool registerAlgorithm(const std::string &algorithm, UpdateCallback callback);

  const std::string &agentId() const { return agent_id_; }
  void setAnchorAgent(std::string anchor_agent) { anchor_agent_id_ = std::move(anchor_agent); }
  /** Configure the canonical ordered position blocks used by GS-CI. */
  bool setGlobalAgentIds(std::vector<std::string> ordered_agent_ids);
  const std::string &algorithm() const { return algorithm_; }
  std::uint64_t sequence() const { return sequence_; }
  void setConfig(const hercules_localization::LocalizationConfig &config);
  const std::optional<hercules_localization::GlobalCiBelief> &globalBelief() const {
    return global_belief_;
  }
  const std::optional<hercules_localization::RecursiveDecentralizedState> &recursiveState() const {
    return recursive_state_;
  }
  hercules_localization::RecursiveTransactionResult installRecursivePairPosterior(
      const std::string &peer_id,
      const hercules_localization::PoseEstimate &posterior,
      const Eigen::Matrix3d &peer_factor,
      const std::string &event_id = {}, std::uint64_t sequence = 0);

  /**
   * Run one transport update. This overload has no pairwise observations and
   * therefore still exercises GPS/odometry lifecycle handling but cannot fuse
   * a neighbor.
   */
  AdapterOutput update(
      const LocalizationMeasurement &measurement,
      const std::vector<LocalizationPeerEstimate> &peer_estimates);

  /**
   * Update with typed pairwise observations. This is the node's main path and
   * dispatches recursive_decentralized or gs_ci after converting wire values
   * to pure-core types.
   */
  AdapterOutput update(
      const LocalizationMeasurement &measurement,
      const std::vector<LocalizationPeerEstimate> &peer_estimates,
      const std::vector<PlanarRelativeMeasurement> &relative_measurements);

  /** Update with full ordered GS-CI beliefs in addition to pairwise inputs. */
  AdapterOutput update(
      const LocalizationMeasurement &measurement,
      const std::vector<LocalizationPeerEstimate> &peer_estimates,
      const std::vector<PlanarRelativeMeasurement> &relative_measurements,
      const std::vector<GlobalCiBelief> &global_beliefs);

  /** Reset sequence and counters held by the transport adapter. */
  void reset();

  /** Canonicalize accepted names (hyphenated spellings are accepted too). */
  static std::string canonicalAlgorithm(const std::string &algorithm);
  static bool isSupportedAlgorithm(const std::string &algorithm);

  /** Convert a published estimate into a broadcast peer message. */
  static LocalizationPeerEstimate toPeerEstimate(
      const LocalizationEstimate &estimate, const std::string &sender_id);

private:
  static bool finiteMeasurement(const LocalizationMeasurement &measurement);
  static bool finitePeerEstimate(const LocalizationPeerEstimate &estimate);
  AdapterOutput passThrough(const LocalizationMeasurement &measurement,
                            std::size_t valid_peer_count,
                            std::size_t rejected_peer_count);
  AdapterOutput numericalUpdate(const LocalizationMeasurement &measurement,
                                const std::vector<LocalizationPeerEstimate> &peers,
                                const std::vector<PlanarRelativeMeasurement> &relative,
                                const std::vector<GlobalCiBelief> &global_beliefs);
  bool syncEstimateFromGlobalBelief();

  std::string agent_id_;
  std::string anchor_agent_id_{"Drone1"};
  std::vector<std::string> global_agent_ids_;
  std::string algorithm_;
  std::map<std::string, UpdateCallback> callbacks_;
  std::uint64_t sequence_{0};
  std::uint64_t updates_{0};
  std::uint64_t measurements_received_{0};
  std::uint64_t peer_estimates_received_{0};
  std::uint64_t measurements_rejected_{0};
  std::uint64_t peer_estimates_rejected_{0};
  std::uint64_t accepted_relative_updates_{0};
  std::map<std::string, std::uint64_t> applied_relative_sequences_;
  hercules_localization::PoseEstimate estimate_{};
  hercules_localization::LocalizationConfig config_{};
  bool got_gps_{false};
  bool allow_odom_seed_{false};
  bool have_odom_{false};
  bool have_latest_odom_yaw_{false};
  bool initial_yaw_from_odometry_{false};
  Eigen::Vector2d previous_odom_position_{Eigen::Vector2d::Zero()};
  double previous_odom_yaw_{0.0};
  double latest_odom_yaw_{0.0};
  Eigen::Vector2d latest_velocity_{Eigen::Vector2d::Zero()};
  double latest_yaw_rate_{0.0};
  std::optional<hercules_localization::GlobalCiBelief> global_belief_;
  std::optional<hercules_localization::RecursiveDecentralizedState> recursive_state_;
};

}  // namespace hercules_localization_ros
