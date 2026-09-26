#pragma once

#include <map>
#include <string>
#include <vector>

#include "hercules_tracking/target_tracker.hpp"

namespace hercules_tracking {

using AgentMeasurements = std::map<std::string, MeasurementMap>;
using Adjacency = std::map<std::string, std::vector<std::string>>;

struct TrackRoundState {
  Eigen::VectorXd trajectory;
  Eigen::VectorXd dual;
  double residual{0.0};
};

struct NetworkResult {
  std::map<std::string, EstimateMap> estimates;
  std::map<std::string, MessageMap> messages;
  int iterations{0};
  double residual{0.0};
  std::vector<std::string> active_targets;
  std::vector<HandoffMessage> handoffs;
  std::vector<std::map<std::string, std::map<std::string, TrackRoundState>>> round_history;
};

class SynchronousTrackingNetwork {
 public:
  static constexpr int kPaperAdmmIterations = 50;

  SynchronousTrackingNetwork(const std::vector<std::string>& agent_ids,
                             TrackConfig config = {}, int max_iterations = 20,
                             double tolerance = 1e-3);

  NetworkResult update(double timestamp, const AgentMeasurements& measurements,
                       const Adjacency& adjacency);
  std::map<std::string, EstimateMap> predictedEstimates(double timestamp) const;
  std::vector<HandoffMessage> performHandoffs(double timestamp, const Adjacency& adjacency);

  TargetTracker& module(const std::string& agent_id) { return modules_.at(agent_id); }
  const TargetTracker& module(const std::string& agent_id) const { return modules_.at(agent_id); }
  const NetworkResult& lastResult() const { return last_result_; }

 private:
  std::map<std::string, TargetTracker> modules_;
  int max_iterations_;
  double tolerance_;
  bool paper_mode_{false};
  NetworkResult last_result_;
};

}  // namespace hercules_tracking
