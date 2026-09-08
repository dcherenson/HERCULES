#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "hercules_tracking/target_track.hpp"

namespace hercules_tracking {

using MeasurementMap = std::map<std::string, std::optional<TargetMeasurement>>;
using MessageMap = std::map<std::string, TrackMessage>;
using EstimateMap = std::map<std::string, TargetEstimate>;

class TargetTracker {
 public:
  explicit TargetTracker(std::string agent_id, TrackConfig config = {});

  void beginEpoch(double timestamp, const MeasurementMap& measurements,
                  int active_tracker_count = 1);
  void seedFromMessages(const std::vector<TrackMessage>& messages, double timestamp);
  MessageMap messages() const;
  double consensusRound(const std::map<std::string, std::vector<TrackMessage>>& inbound_messages);
  EstimateMap finishEpoch() const;
  EstimateMap predictedEstimates(double timestamp) const;
  std::map<std::string, HandoffMessage> handoffMessages(double timestamp);
  bool acceptHandoff(const HandoffMessage& message);

  const std::string& agentId() const { return agent_id_; }
  std::map<std::string, TargetTrack>& tracks() { return tracks_; }
  const std::map<std::string, TargetTrack>& tracks() const { return tracks_; }

 private:
  TargetTrack& track(const std::string& target_id);
  std::string agent_id_;
  TrackConfig config_;
  std::map<std::string, TargetTrack> tracks_;
};

}  // namespace hercules_tracking
