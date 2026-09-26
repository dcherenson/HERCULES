#include "hercules_tracking/synchronous_network.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace hercules_tracking {

SynchronousTrackingNetwork::SynchronousTrackingNetwork(
    const std::vector<std::string>& agent_ids, TrackConfig config,
    int max_iterations, double tolerance)
    : max_iterations_(config.maicp_enabled ? kPaperAdmmIterations : max_iterations),
      tolerance_(tolerance), paper_mode_(config.maicp_enabled) {
  for (const auto& agent_id : agent_ids) modules_.try_emplace(agent_id, agent_id, config);
  if (modules_.empty()) throw std::invalid_argument("network requires at least one agent");
}

NetworkResult SynchronousTrackingNetwork::update(double timestamp,
                                                  const AgentMeasurements& measurements,
                                                  const Adjacency& adjacency) {
  std::set<std::string> target_ids;
  for (const auto& [agent, values] : measurements) {
    (void)agent;
    for (const auto& [target_id, unused] : values) {
      (void)unused;
      target_ids.insert(target_id);
    }
  }
  for (const auto& [agent, module] : modules_) {
    (void)agent;
    for (const auto& [target_id, unused] : module.tracks()) {
      (void)unused;
      target_ids.insert(target_id);
    }
  }
  const int active_count = std::max(1, static_cast<int>(modules_.size()));
  for (auto& [agent_id, module] : modules_) {
    const auto found = measurements.find(agent_id);
    module.beginEpoch(timestamp,
        found == measurements.end() ? MeasurementMap{} : found->second, active_count);
  }

  for (std::size_t propagation = 0; propagation < std::max<std::size_t>(1, modules_.size());
       ++propagation) {
    bool changed = false;
    std::map<std::string, MessageMap> announcements;
    for (const auto& [agent_id, module] : modules_) announcements.emplace(agent_id, module.messages());
    for (auto& [agent_id, module] : modules_) {
      bool has_all = true;
      for (const auto& target_id : target_ids) has_all &= module.tracks().count(target_id) != 0;
      if (has_all) continue;
      std::vector<TrackMessage> neighbors;
      const auto edges = adjacency.find(agent_id);
      if (edges != adjacency.end()) {
        for (const auto& neighbor : edges->second) {
          const auto source = announcements.find(neighbor);
          if (source == announcements.end()) continue;
          for (const auto& [target_id, message] : source->second) {
            (void)target_id;
            neighbors.push_back(message);
          }
        }
      }
      const auto before = module.tracks().size();
      module.seedFromMessages(neighbors, timestamp);
      changed |= module.tracks().size() != before;
    }
    if (!changed) break;
  }

  double residual = std::numeric_limits<double>::infinity();
  int rounds = 0;
  std::map<std::string, MessageMap> messages;
  std::vector<std::map<std::string, std::map<std::string, TrackRoundState>>> history;
  for (rounds = 1; rounds <= max_iterations_; ++rounds) {
    messages.clear();
    for (const auto& [agent_id, module] : modules_) messages.emplace(agent_id, module.messages());
    double round_residual = 0.0;
    for (auto& [agent_id, module] : modules_) {
      std::map<std::string, std::vector<TrackMessage>> inbound;
      const auto edges = adjacency.find(agent_id);
      if (edges != adjacency.end()) {
        for (const auto& neighbor : edges->second) {
          const auto source = messages.find(neighbor);
          if (source == messages.end()) continue;
          for (const auto& [target_id, message] : source->second) inbound[target_id].push_back(message);
        }
      }
      round_residual = std::max(round_residual, module.consensusRound(inbound));
    }
    std::map<std::string, std::map<std::string, TrackRoundState>> snapshot;
    for (const auto& [agent_id, module] : modules_) {
      for (const auto& [target_id, track] : module.tracks()) {
        if (track.active()) snapshot[agent_id][target_id] =
            {track.trajectory(), track.dual(), track.consensusResidual()};
      }
    }
    history.push_back(std::move(snapshot));
    residual = round_residual;
    if (!paper_mode_ && residual <= tolerance_) break;
  }
  if (rounds > max_iterations_) rounds = max_iterations_;

  std::map<std::string, std::map<std::string, TrackMessage>> support_messages;
  for (const auto& [agent_id, module] : modules_) {
    (void)module;
    const auto edges = adjacency.find(agent_id);
    if (edges == adjacency.end()) continue;
    for (const auto& neighbor : edges->second) {
      const auto source = messages.find(neighbor);
      if (source == messages.end()) continue;
      for (const auto& [target_id, message] : source->second) {
        if (!message.active) continue;
        auto& candidates = support_messages[agent_id];
        const auto previous = candidates.find(target_id);
        if (previous == candidates.end() || message.state_covariance.trace() <
                                             previous->second.state_covariance.trace()) {
          candidates[target_id] = message;
        }
      }
    }
  }

  for (auto& [agent_id, module] : modules_) {
    for (auto& [target_id, track] : module.tracks()) {
      if (!track.active()) continue;
      const auto agent_measurements = measurements.find(agent_id);
      bool has_measurement = false;
      if (agent_measurements != measurements.end()) {
        const auto measurement = agent_measurements->second.find(target_id);
        has_measurement = measurement != agent_measurements->second.end() &&
                          measurement->second && measurement->second->valid;
      }
      if (has_measurement) continue;
      const auto support_agent = support_messages.find(agent_id);
      if (support_agent == support_messages.end()) continue;
      const auto support = support_agent->second.find(target_id);
      if (support != support_agent->second.end()) {
        track.setConsensusSupport(timestamp, support->second.state_covariance);
      }
    }
  }

  NetworkResult result;
  for (const auto& [agent_id, module] : modules_) {
    result.estimates.emplace(agent_id, module.finishEpoch());
    result.messages.emplace(agent_id, module.messages());
  }
  result.iterations = rounds;
  result.residual = residual;
  result.active_targets.assign(target_ids.begin(), target_ids.end());
  result.round_history = std::move(history);
  last_result_ = result;
  return result;
}

std::map<std::string, EstimateMap> SynchronousTrackingNetwork::predictedEstimates(
    double timestamp) const {
  std::map<std::string, EstimateMap> output;
  for (const auto& [agent_id, module] : modules_) {
    output.emplace(agent_id, module.predictedEstimates(timestamp));
  }
  return output;
}

std::vector<HandoffMessage> SynchronousTrackingNetwork::performHandoffs(
    double timestamp, const Adjacency& adjacency) {
  std::vector<HandoffMessage> handoffs;
  for (auto& [source_id, module] : modules_) {
    auto source_messages = module.handoffMessages(timestamp);
    for (auto& [target_id, message] : source_messages) {
      std::vector<std::string> candidates;
      const auto edges = adjacency.find(source_id);
      if (edges != adjacency.end()) {
        for (const auto& neighbor : edges->second) {
          const auto found = modules_.find(neighbor);
          if (found != modules_.end() && found->second.tracks().count(target_id)) {
            candidates.push_back(neighbor);
          }
        }
      }
      if (candidates.empty()) continue;
      std::sort(candidates.begin(), candidates.end());
      message.source_id = source_id;
      message.receiver_id = candidates.front();
      modules_.at(message.receiver_id).acceptHandoff(message);
      handoffs.push_back(message);
    }
  }
  last_result_.handoffs.insert(last_result_.handoffs.end(), handoffs.begin(), handoffs.end());
  return handoffs;
}

}  // namespace hercules_tracking
