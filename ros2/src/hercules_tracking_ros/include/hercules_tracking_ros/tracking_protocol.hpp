#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <hercules_interfaces/msg/tracking_consensus.hpp>
#include <hercules_interfaces/msg/tracking_epoch.hpp>
#include <hercules_interfaces/msg/tracking_round_status.hpp>

namespace hercules_tracking_ros {

enum class MessageDisposition { kAccepted, kWrongEpoch, kWrongRound,
                                kWrongTarget, kWrongSender, kDisconnected };

class TrackingProtocol {
 public:
  TrackingProtocol(std::string agent_id, std::string target_id);
  void begin(const hercules_interfaces::msg::TrackingEpoch& epoch,
             std::vector<std::string> neighbors);
  void setRound(uint32_t round_id);
  MessageDisposition accept(
      const hercules_interfaces::msg::TrackingConsensus& message);
  bool acceptStatus(
      const hercules_interfaces::msg::TrackingRoundStatus& message);
  bool allNeighborMessagesReceived() const;
  bool allStatusesReceived() const;
  std::vector<std::string> missingNeighbors() const;
  const std::map<std::string, hercules_interfaces::msg::TrackingConsensus>& inbound() const {
    return inbound_;
  }
  const std::map<std::string, hercules_interfaces::msg::TrackingRoundStatus>& statuses() const {
    return statuses_;
  }
  const std::vector<std::string>& neighbors() const { return neighbors_; }
  const std::vector<std::string>& agents() const { return agents_; }
  uint64_t epochId() const { return epoch_id_; }
  uint32_t roundId() const { return round_id_; }

 private:
  std::string agent_id_;
  std::string target_id_;
  uint64_t epoch_id_{0};
  uint32_t round_id_{0};
  std::vector<std::string> neighbors_;
  std::vector<std::string> agents_;
  std::set<std::string> neighbor_set_;
  std::set<std::string> agent_set_;
  std::map<std::string, hercules_interfaces::msg::TrackingConsensus> inbound_;
  std::map<std::string, hercules_interfaces::msg::TrackingRoundStatus> statuses_;
};

}  // namespace hercules_tracking_ros
