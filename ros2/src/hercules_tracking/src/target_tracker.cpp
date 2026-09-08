#include "hercules_tracking/target_tracker.hpp"

#include <set>
#include <stdexcept>
#include <utility>

namespace hercules_tracking {

TargetTracker::TargetTracker(std::string agent_id, TrackConfig config)
    : agent_id_(std::move(agent_id)), config_(config) {
  if (agent_id_.empty()) throw std::invalid_argument("agent_id must not be empty");
}

TargetTrack& TargetTracker::track(const std::string& target_id) {
  auto found = tracks_.find(target_id);
  if (found == tracks_.end()) {
    found = tracks_.try_emplace(target_id, target_id, config_).first;
  }
  return found->second;
}

void TargetTracker::beginEpoch(double timestamp, const MeasurementMap& measurements,
                               int active_tracker_count) {
  std::set<std::string> target_ids;
  for (const auto& [target_id, unused] : tracks_) {
    (void)unused;
    target_ids.insert(target_id);
  }
  for (const auto& [target_id, unused] : measurements) {
    (void)unused;
    target_ids.insert(target_id);
  }
  for (const auto& target_id : target_ids) {
    const auto found = measurements.find(target_id);
    track(target_id).begin(timestamp,
        found == measurements.end() ? std::optional<TargetMeasurement>{} : found->second,
        active_tracker_count);
  }
}

void TargetTracker::seedFromMessages(const std::vector<TrackMessage>& messages, double timestamp) {
  for (const auto& message : messages) {
    if (message.target_id.empty() || message.trajectory.size() == 0) continue;
    track(message.target_id).seed(message, timestamp);
  }
}

MessageMap TargetTracker::messages() const {
  MessageMap output;
  for (const auto& [target_id, value] : tracks_) {
    if (value.active()) output.emplace(target_id, value.message());
  }
  return output;
}

double TargetTracker::consensusRound(
    const std::map<std::string, std::vector<TrackMessage>>& inbound_messages) {
  double maximum = 0.0;
  for (auto& [target_id, value] : tracks_) {
    const auto found = inbound_messages.find(target_id);
    maximum = std::max(maximum, value.consensusStep(
        found == inbound_messages.end() ? std::vector<TrackMessage>{} : found->second));
  }
  return maximum;
}

EstimateMap TargetTracker::finishEpoch() const {
  EstimateMap output;
  for (const auto& [target_id, value] : tracks_) {
    if (value.active()) output.emplace(target_id, value.finalize());
  }
  return output;
}

EstimateMap TargetTracker::predictedEstimates(double timestamp) const {
  EstimateMap output;
  for (const auto& [target_id, value] : tracks_) {
    if (value.active()) output.emplace(target_id, value.predicted(timestamp));
  }
  return output;
}

std::map<std::string, HandoffMessage> TargetTracker::handoffMessages(double timestamp) {
  std::map<std::string, HandoffMessage> output;
  for (auto& [target_id, value] : tracks_) {
    auto message = value.handoffMessage(timestamp);
    if (message) {
      message->source_id = agent_id_;
      output.emplace(target_id, std::move(*message));
    }
  }
  return output;
}

bool TargetTracker::acceptHandoff(const HandoffMessage& message) {
  if (message.target_id.empty()) return false;
  return track(message.target_id).acceptHandoff(message);
}

}  // namespace hercules_tracking
