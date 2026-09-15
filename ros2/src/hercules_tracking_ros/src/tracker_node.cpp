#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hercules_interfaces/msg/target_estimate.hpp>
#include <hercules_interfaces/msg/target_measurement.hpp>
#include <hercules_interfaces/msg/tracking_consensus.hpp>
#include <hercules_interfaces/msg/tracking_diagnostics.hpp>
#include <hercules_interfaces/msg/tracking_epoch.hpp>
#include <hercules_interfaces/msg/tracking_handoff.hpp>
#include <hercules_interfaces/msg/tracking_round_status.hpp>
#include <rclcpp/rclcpp.hpp>

#include "hercules_tracking/target_tracker.hpp"
#include "hercules_tracking_ros/neighbor_graph.hpp"
#include "hercules_tracking_ros/tracking_adapter.hpp"
#include "hercules_tracking_ros/tracking_protocol.hpp"

namespace hercules_tracking_ros {
namespace {
using SteadyClock = std::chrono::steady_clock;

enum class Stage { kIdle, kSeed, kRound, kStatus };

hercules_tracking::TrackConfig trackingConfig(rclcpp::Node& node) {
  hercules_tracking::TrackConfig value;
  value.window_seconds = node.declare_parameter<double>("tracking_window", 5.0);
  value.rho = node.declare_parameter<double>("tracking_admm_rho", 1.0);
  value.max_iterations = node.declare_parameter<int>("tracking_admm_max_iterations", 20);
  value.tolerance = node.declare_parameter<double>("tracking_admm_tolerance", 0.001);
  value.process_noise_spectral_density =
      node.declare_parameter<double>("tracking_process_noise", 0.20);
  value.measurement_std = node.declare_parameter<double>("tracking_measurement_std", 0.25);
  return value;
}
}  // namespace

class TrackerNode : public rclcpp::Node {
 public:
  TrackerNode()
      : Node("target_tracker"),
        agent_id_(declare_parameter<std::string>("agent_id", "")),
        target_id_(declare_parameter<std::string>("target_id", "Target1")),
        config_(trackingConfig(*this)), tracker_(agent_id_, config_),
        protocol_(agent_id_, target_id_) {
    if (agent_id_.empty()) throw std::invalid_argument("agent_id parameter is required");
    round_timeout_ = declare_parameter<double>("round_timeout_sec", 0.008);
    seed_timeout_ = declare_parameter<double>("seed_timeout_sec", 0.020);
    measurement_wait_ = declare_parameter<double>("measurement_wait_sec", 0.010);
    if (round_timeout_ <= 0.0 || seed_timeout_ <= 0.0) {
      throw std::invalid_argument("tracking timeouts must be positive");
    }
    const auto reliable = rclcpp::QoS(100).reliable();
    measurement_subscription_ = create_subscription<hercules_interfaces::msg::TargetMeasurement>(
        "/hercules_tracking/" + agent_id_ + "/" + target_id_ + "/measurement", reliable,
        [this](hercules_interfaces::msg::TargetMeasurement::ConstSharedPtr message) {
          if (message->target_id == target_id_ && message->source_id == agent_id_) {
            latest_measurement_ = *message;
          }
        });
    epoch_subscription_ = create_subscription<hercules_interfaces::msg::TrackingEpoch>(
        "/hercules_tracking/epoch", reliable,
        [this](hercules_interfaces::msg::TrackingEpoch::ConstSharedPtr message) { queueEpoch(*message); });
    consensus_subscription_ = create_subscription<hercules_interfaces::msg::TrackingConsensus>(
        "/hercules_tracking/consensus", reliable,
        [this](hercules_interfaces::msg::TrackingConsensus::ConstSharedPtr message) {
          receiveConsensus(*message);
        });
    status_subscription_ = create_subscription<hercules_interfaces::msg::TrackingRoundStatus>(
        "/hercules_tracking/round_status", reliable,
        [this](hercules_interfaces::msg::TrackingRoundStatus::ConstSharedPtr message) {
          receiveStatus(*message);
        });
    handoff_subscription_ = create_subscription<hercules_interfaces::msg::TrackingHandoff>(
        "/hercules_tracking/handoff", reliable,
        [this](hercules_interfaces::msg::TrackingHandoff::ConstSharedPtr message) {
          receiveHandoff(*message);
        });
    consensus_publisher_ = create_publisher<hercules_interfaces::msg::TrackingConsensus>(
        "/hercules_tracking/consensus", reliable);
    status_publisher_ = create_publisher<hercules_interfaces::msg::TrackingRoundStatus>(
        "/hercules_tracking/round_status", reliable);
    handoff_publisher_ = create_publisher<hercules_interfaces::msg::TrackingHandoff>(
        "/hercules_tracking/handoff", reliable);
    estimate_publisher_ = create_publisher<hercules_interfaces::msg::TargetEstimate>(
        "/hercules_tracking/" + agent_id_ + "/" + target_id_ + "/estimate", reliable);
    diagnostics_publisher_ = create_publisher<hercules_interfaces::msg::TrackingDiagnostics>(
        "/hercules_tracking/" + agent_id_ + "/" + target_id_ + "/diagnostics", reliable);
    timer_ = create_wall_timer(std::chrono::milliseconds(1), [this]() { advance(); });
    RCLCPP_INFO(get_logger(), "local tracker ready for %s/%s", agent_id_.c_str(), target_id_.c_str());
  }

 private:
  void queueEpoch(const hercules_interfaces::msg::TrackingEpoch& epoch) {
    if (epoch.target_id != target_id_ || epoch.epoch_id <= last_epoch_id_) return;
    if (stage_ != Stage::kIdle) finishEpoch(true);
    pending_epoch_ = epoch;
    last_epoch_id_ = epoch.epoch_id;
    deadline_ = SteadyClock::now() + std::chrono::duration_cast<SteadyClock::duration>(
                                          std::chrono::duration<double>(measurement_wait_));
  }

  void beginEpoch(const hercules_interfaces::msg::TrackingEpoch& epoch) {
    std::vector<std::string> neighbors;
    try {
      neighbors = neighborsFromFlattened(agent_id_, epoch.agent_ids, epoch.adjacency);
    } catch (const std::exception& error) {
      RCLCPP_ERROR(get_logger(), "rejected epoch graph: %s", error.what());
      return;
    }
    epoch_ = epoch;
    future_consensus_.clear();
    future_status_.clear();
    protocol_.begin(epoch, std::move(neighbors));
    direct_observation_ = false;
    epoch_timed_out_ = false;
    missing_neighbor_messages_ = 0;
    hercules_tracking::MeasurementMap measurements;
    std::optional<hercules_tracking::TargetMeasurement> measurement;
    if (latest_measurement_) {
      const bool fresh_capture = latest_measurement_->capture_id.empty() ||
                                 latest_measurement_->capture_id != consumed_capture_id_;
      if (fresh_capture) {
        consumed_capture_id_ = latest_measurement_->capture_id;
        if (latest_measurement_->valid) {
          try {
            measurement = measurementFromRos(*latest_measurement_);
            direct_observation_ = true;
          } catch (const std::exception& error) {
            RCLCPP_WARN(get_logger(), "invalid measurement: %s", error.what());
          }
        }
      }
    }
    measurements[target_id_] = measurement;
    tracker_.beginEpoch(timeSeconds(epoch.stamp), measurements,
                        std::max(1, static_cast<int>(epoch.agent_ids.size())));
    stage_ = Stage::kSeed;
    protocol_.setRound(0);
    deadline_ = SteadyClock::now() + std::chrono::duration_cast<SteadyClock::duration>(
                                          std::chrono::duration<double>(seed_timeout_));
    publishConsensus(0);
  }

  void receiveConsensus(const hercules_interfaces::msg::TrackingConsensus& message) {
    if (message.sender_id == agent_id_) return;
    if (message.epoch_id == epoch_.epoch_id && message.target_id == target_id_ &&
        message.sender_id != agent_id_ &&
        std::find(protocol_.neighbors().begin(), protocol_.neighbors().end(), message.sender_id) !=
            protocol_.neighbors().end() &&
        message.round_id == protocol_.roundId() + 1 &&
        (stage_ == Stage::kSeed || stage_ == Stage::kStatus)) {
      future_consensus_[message.round_id][message.sender_id] = message;
      return;
    }
    if (stage_ != Stage::kSeed && stage_ != Stage::kRound) {
      if (stage_ != Stage::kIdle) ++rejected_messages_;
      return;
    }
    const auto disposition = protocol_.accept(message);
    if (disposition != MessageDisposition::kAccepted) {
      ++rejected_messages_;
      return;
    }
    if (stage_ == Stage::kSeed && message.active) {
      const bool was_active = tracker_.messages().count(target_id_) != 0;
      try {
        tracker_.seedFromMessages({consensusFromRos(message)}, timeSeconds(epoch_.stamp));
      } catch (const std::exception& error) {
        ++rejected_messages_;
        RCLCPP_WARN(get_logger(), "rejected seed message: %s", error.what());
        return;
      }
      if (!was_active && tracker_.messages().count(target_id_)) publishConsensus(0);
    }
  }

  void receiveStatus(const hercules_interfaces::msg::TrackingRoundStatus& message) {
    if (stage_ == Stage::kStatus) {
      if (message.epoch_id == epoch_.epoch_id && message.target_id == target_id_ &&
          message.round_id == protocol_.roundId() + 1) {
        future_status_[message.round_id][message.sender_id] = message;
      } else if (!protocol_.acceptStatus(message)) {
        ++rejected_messages_;
      }
      return;
    }
    if ((stage_ == Stage::kSeed || stage_ == Stage::kRound) &&
        message.epoch_id == epoch_.epoch_id && message.target_id == target_id_ &&
        (message.round_id == protocol_.roundId() ||
         message.round_id == protocol_.roundId() + 1)) {
      future_status_[message.round_id][message.sender_id] = message;
    }
  }

  void receiveHandoff(const hercules_interfaces::msg::TrackingHandoff& message) {
    if (message.receiver_id != agent_id_ || message.target_id != target_id_ ||
        message.sender_id == agent_id_ || message.epoch_id != epoch_.epoch_id) return;
    if (std::find(protocol_.neighbors().begin(), protocol_.neighbors().end(), message.sender_id) ==
        protocol_.neighbors().end()) {
      ++rejected_messages_;
      return;
    }
    try {
      if (tracker_.acceptHandoff(handoffFromRos(message))) ++handoffs_accepted_;
    } catch (const std::exception&) {
      ++rejected_messages_;
    }
  }

  void publishConsensus(uint32_t round) {
    hercules_tracking::TrackMessage value;
    value.target_id = target_id_;
    const auto messages = tracker_.messages();
    const auto found = messages.find(target_id_);
    if (found != messages.end()) value = found->second;
    consensus_publisher_->publish(
        consensusToRos(agent_id_, epoch_.epoch_id, round, value, timeSeconds(epoch_.stamp)));
    ++consensus_messages_published_;
  }

  void startRound(uint32_t round) {
    protocol_.setRound(round);
    const auto future = future_consensus_.find(round);
    if (future != future_consensus_.end()) {
      for (const auto& [sender, message] : future->second) {
        (void)sender;
        protocol_.accept(message);
      }
      future_consensus_.erase(future);
    }
    stage_ = Stage::kRound;
    deadline_ = SteadyClock::now() + std::chrono::duration_cast<SteadyClock::duration>(
                                          std::chrono::duration<double>(round_timeout_));
    publishConsensus(round);
  }

  void applyRound(bool deadline_expired) {
    std::map<std::string, std::vector<hercules_tracking::TrackMessage>> inbound;
    for (const auto& [sender, message] : protocol_.inbound()) {
      (void)sender;
      try {
        inbound[target_id_].push_back(consensusFromRos(message));
      } catch (const std::exception&) {
        ++rejected_messages_;
      }
    }
    const auto missing = protocol_.missingNeighbors();
    if (deadline_expired && !missing.empty()) {
      epoch_timed_out_ = true;
      missing_neighbor_messages_ += missing.size();
    }
    const double residual = tracker_.consensusRound(inbound);
    hercules_interfaces::msg::TrackingRoundStatus status;
    status.sender_id = agent_id_;
    status.target_id = target_id_;
    status.epoch_id = epoch_.epoch_id;
    status.round_id = protocol_.roundId();
    status.residual = residual;
    status.round_timed_out = deadline_expired && !missing.empty();
    protocol_.acceptStatus(status);
    status_publisher_->publish(status);
    stage_ = Stage::kStatus;
    const auto future = future_status_.find(protocol_.roundId());
    if (future != future_status_.end()) {
      for (const auto& [sender, value] : future->second) {
        (void)sender;
        protocol_.acceptStatus(value);
      }
      future_status_.erase(future);
    }
    deadline_ = SteadyClock::now() + std::chrono::duration_cast<SteadyClock::duration>(
                                          std::chrono::duration<double>(round_timeout_));
  }

  void evaluateStatus(bool deadline_expired) {
    if (deadline_expired && !protocol_.allStatusesReceived()) epoch_timed_out_ = true;
    double maximum = 0.0;
    for (const auto& [sender, status] : protocol_.statuses()) {
      (void)sender;
      maximum = std::max(maximum, status.residual);
      epoch_timed_out_ = epoch_timed_out_ || status.round_timed_out;
    }
    const bool converged = protocol_.allStatusesReceived() && maximum <= config_.tolerance;
    if (converged || protocol_.roundId() >= static_cast<uint32_t>(config_.max_iterations) ||
        (deadline_expired && !protocol_.allStatusesReceived())) {
      finishEpoch(epoch_timed_out_);
    } else {
      startRound(protocol_.roundId() + 1);
    }
  }

  void finishEpoch(bool timed_out) {
    if (stage_ == Stage::kIdle) return;
    if (!direct_observation_) {
      const hercules_interfaces::msg::TrackingConsensus* best = nullptr;
      double best_trace = std::numeric_limits<double>::infinity();
      for (const auto& [sender, message] : protocol_.inbound()) {
        (void)sender;
        if (!message.active) continue;
        const double trace = message.state_covariance[0] + message.state_covariance[5] +
                             message.state_covariance[10] + message.state_covariance[15];
        if (trace < best_trace) { best_trace = trace; best = &message; }
      }
      if (best) {
        try {
          const auto support = consensusFromRos(*best);
          auto found = tracker_.tracks().find(target_id_);
          if (found != tracker_.tracks().end() && found->second.active()) {
            found->second.setConsensusSupport(timeSeconds(epoch_.stamp), support.state_covariance);
          }
        } catch (const std::exception&) {
          ++rejected_messages_;
        }
      }
    }
    auto estimates = tracker_.finishEpoch();
    hercules_tracking::TargetEstimate estimate;
    estimate.target_id = target_id_;
    estimate.timestamp = timeSeconds(epoch_.stamp);
    const auto found = estimates.find(target_id_);
    if (found != estimates.end()) estimate = found->second;
    estimate_publisher_->publish(estimateToRos(estimate));

    auto handoffs = tracker_.handoffMessages(timeSeconds(epoch_.stamp));
    auto handoff = handoffs.find(target_id_);
    if (handoff != handoffs.end()) {
      std::vector<std::string> candidates;
      for (const auto& [sender, message] : protocol_.inbound()) {
        if (message.active) candidates.push_back(sender);
      }
      if (!candidates.empty()) {
        std::sort(candidates.begin(), candidates.end());
        handoff->second.receiver_id = candidates.front();
        handoff_publisher_->publish(handoffToRos(epoch_.epoch_id, handoff->second));
        ++handoffs_sent_;
      }
    }

    hercules_interfaces::msg::TrackingDiagnostics diagnostics;
    diagnostics.sender_id = agent_id_;
    diagnostics.target_id = target_id_;
    diagnostics.epoch_id = epoch_.epoch_id;
    diagnostics.stamp = epoch_.stamp;
    diagnostics.epoch_complete = true;
    diagnostics.epoch_timed_out = timed_out;
    diagnostics.active = estimate.active;
    diagnostics.direct_observation = direct_observation_;
    diagnostics.consensus_iterations = static_cast<uint32_t>(std::max(0, estimate.admm_iterations));
    diagnostics.consensus_residual = estimate.consensus_residual;
    diagnostics.expected_neighbor_count = static_cast<uint32_t>(protocol_.neighbors().size());
    diagnostics.missing_neighbor_messages = static_cast<uint32_t>(missing_neighbor_messages_);
    diagnostics.consensus_messages_published = consensus_messages_published_;
    diagnostics.rejected_messages = rejected_messages_;
    diagnostics.handoffs_sent = handoffs_sent_;
    diagnostics.handoffs_accepted = handoffs_accepted_;
    diagnostics_publisher_->publish(diagnostics);
    stage_ = Stage::kIdle;
  }

  void advance() {
    if (stage_ == Stage::kIdle) {
      if (pending_epoch_ && SteadyClock::now() >= deadline_) {
        const auto epoch = *pending_epoch_;
        pending_epoch_.reset();
        beginEpoch(epoch);
      }
      return;
    }
    const bool expired = SteadyClock::now() >= deadline_;
    if (stage_ == Stage::kSeed) {
      if (expired) startRound(1);
    } else if (stage_ == Stage::kRound) {
      if (protocol_.allNeighborMessagesReceived() || expired) applyRound(expired);
    } else if (stage_ == Stage::kStatus) {
      if (protocol_.allStatusesReceived() || expired) evaluateStatus(expired);
    }
  }

  std::string agent_id_;
  std::string target_id_;
  hercules_tracking::TrackConfig config_;
  hercules_tracking::TargetTracker tracker_;
  TrackingProtocol protocol_;
  Stage stage_{Stage::kIdle};
  double round_timeout_{0.008};
  double seed_timeout_{0.020};
  double measurement_wait_{0.010};
  uint64_t last_epoch_id_{0};
  bool direct_observation_{false};
  bool epoch_timed_out_{false};
  std::size_t missing_neighbor_messages_{0};
  uint64_t consensus_messages_published_{0};
  uint64_t rejected_messages_{0};
  uint64_t handoffs_sent_{0};
  uint64_t handoffs_accepted_{0};
  std::string consumed_capture_id_;
  std::optional<hercules_interfaces::msg::TargetMeasurement> latest_measurement_;
  std::optional<hercules_interfaces::msg::TrackingEpoch> pending_epoch_;
  std::map<uint32_t, std::map<std::string, hercules_interfaces::msg::TrackingConsensus>>
      future_consensus_;
  std::map<uint32_t, std::map<std::string, hercules_interfaces::msg::TrackingRoundStatus>>
      future_status_;
  hercules_interfaces::msg::TrackingEpoch epoch_;
  SteadyClock::time_point deadline_;
  rclcpp::Subscription<hercules_interfaces::msg::TargetMeasurement>::SharedPtr measurement_subscription_;
  rclcpp::Subscription<hercules_interfaces::msg::TrackingEpoch>::SharedPtr epoch_subscription_;
  rclcpp::Subscription<hercules_interfaces::msg::TrackingConsensus>::SharedPtr consensus_subscription_;
  rclcpp::Subscription<hercules_interfaces::msg::TrackingRoundStatus>::SharedPtr status_subscription_;
  rclcpp::Subscription<hercules_interfaces::msg::TrackingHandoff>::SharedPtr handoff_subscription_;
  rclcpp::Publisher<hercules_interfaces::msg::TrackingConsensus>::SharedPtr consensus_publisher_;
  rclcpp::Publisher<hercules_interfaces::msg::TrackingRoundStatus>::SharedPtr status_publisher_;
  rclcpp::Publisher<hercules_interfaces::msg::TrackingHandoff>::SharedPtr handoff_publisher_;
  rclcpp::Publisher<hercules_interfaces::msg::TargetEstimate>::SharedPtr estimate_publisher_;
  rclcpp::Publisher<hercules_interfaces::msg::TrackingDiagnostics>::SharedPtr diagnostics_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace hercules_tracking_ros

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<hercules_tracking_ros::TrackerNode>());
  rclcpp::shutdown();
  return 0;
}
