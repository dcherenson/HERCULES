#include "hercules_tracking_ros/tracking_protocol.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace hercules_tracking_ros {

TrackingProtocol::TrackingProtocol(std::string agent_id, std::string target_id)
    : agent_id_(std::move(agent_id)), target_id_(std::move(target_id)) {
  if (agent_id_.empty() || target_id_.empty()) throw std::invalid_argument("identities are required");
}

void TrackingProtocol::begin(const hercules_interfaces::msg::TrackingEpoch& epoch,
                             std::vector<std::string> neighbors) {
  epoch_id_ = epoch.epoch_id;
  round_id_ = 0;
  agents_ = epoch.agent_ids;
  neighbors_ = std::move(neighbors);
  neighbor_set_ = {neighbors_.begin(), neighbors_.end()};
  agent_set_ = {agents_.begin(), agents_.end()};
  inbound_.clear();
  statuses_.clear();
}

void TrackingProtocol::setRound(uint32_t round_id) {
  round_id_ = round_id;
  inbound_.clear();
  statuses_.clear();
}

MessageDisposition TrackingProtocol::accept(
    const hercules_interfaces::msg::TrackingConsensus& message) {
  if (message.epoch_id != epoch_id_) return MessageDisposition::kWrongEpoch;
  if (message.round_id != round_id_) return MessageDisposition::kWrongRound;
  if (message.target_id != target_id_) return MessageDisposition::kWrongTarget;
  if (!agent_set_.count(message.sender_id) || message.sender_id == agent_id_) {
    return MessageDisposition::kWrongSender;
  }
  if (!neighbor_set_.count(message.sender_id)) return MessageDisposition::kDisconnected;
  inbound_[message.sender_id] = message;
  return MessageDisposition::kAccepted;
}

bool TrackingProtocol::acceptStatus(
    const hercules_interfaces::msg::TrackingRoundStatus& message) {
  if (message.epoch_id != epoch_id_ || message.round_id != round_id_ ||
      message.target_id != target_id_ || !agent_set_.count(message.sender_id)) return false;
  statuses_[message.sender_id] = message;
  return true;
}

bool TrackingProtocol::allNeighborMessagesReceived() const {
  return inbound_.size() == neighbor_set_.size();
}

bool TrackingProtocol::allStatusesReceived() const {
  return statuses_.size() == agent_set_.size();
}

std::vector<std::string> TrackingProtocol::missingNeighbors() const {
  std::vector<std::string> output;
  for (const auto& neighbor : neighbors_) {
    if (!inbound_.count(neighbor)) output.push_back(neighbor);
  }
  return output;
}

}  // namespace hercules_tracking_ros
