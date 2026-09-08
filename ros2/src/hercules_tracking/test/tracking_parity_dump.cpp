#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "hercules_tracking/synchronous_network.hpp"

namespace ht = hercules_tracking;

namespace {

ht::TargetMeasurement measurement(const std::string& target, double x, double y,
                                  double timestamp, double variance = 0.25) {
  return {target, ht::Position(x, y), ht::PositionMatrix::Identity() * variance, timestamp};
}

template <typename Derived>
void writeMatrix(const Eigen::MatrixBase<Derived>& value) {
  std::cout << '[';
  bool first = true;
  for (Eigen::Index row = 0; row < value.rows(); ++row) {
    for (Eigen::Index column = 0; column < value.cols(); ++column) {
      if (!first) std::cout << ',';
      first = false;
      std::cout << value(row, column);
    }
  }
  std::cout << ']';
}

void writeTimes(const std::vector<double>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

void writeEstimate(const ht::TargetEstimate& estimate) {
  std::cout << "{\"position\":";
  writeMatrix(estimate.position);
  std::cout << ",\"velocity\":";
  writeMatrix(estimate.velocity);
  std::cout << ",\"covariance\":";
  writeMatrix(estimate.covariance);
  std::cout << ",\"state_covariance\":";
  writeMatrix(estimate.state_covariance);
  std::cout << ",\"timestamp\":" << estimate.timestamp
            << ",\"active\":" << (estimate.active ? "true" : "false") << '}';
}

ht::Adjacency lineGraph() {
  return {{"Drone1", {"Husky1"}},
          {"Husky1", {"Drone1", "Husky2"}},
          {"Husky2", {"Husky1"}}};
}

void writeModels() {
  std::cout << "\"models\":[";
  const std::vector<double> deltas{0.0, 0.5, 1.3, -0.7};
  for (std::size_t index = 0; index < deltas.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << "{\"dt\":" << deltas[index] << ",\"transition\":";
    writeMatrix(ht::constantVelocityTransition(deltas[index]));
    std::cout << ",\"process\":";
    writeMatrix(ht::constantAccelerationProcessNoise(deltas[index], 0.7));
    std::cout << '}';
  }
  std::cout << ']';
}

void writeLocal() {
  ht::TargetTrack track("Target1");
  track.begin(0.0, measurement("Target1", 1.0, -2.0, 0.0), 2);
  track.begin(0.7, std::nullopt, 2);
  track.begin(1.6, measurement("Target1", 2.1, -1.1, 1.4), 2);
  std::cout << "\"local\":{\"times\":";
  writeTimes(track.times());
  std::cout << ",\"information_shape\":[" << track.information().rows() << ','
            << track.information().cols() << "],\"information\":";
  writeMatrix(track.information());
  std::cout << ",\"information_vector\":";
  writeMatrix(track.informationVector());
  std::cout << ",\"trajectory\":";
  writeMatrix(track.trajectory());
  std::cout << ",\"full_covariance\":";
  writeMatrix(track.covariance());
  std::cout << ",\"estimate\":";
  writeEstimate(track.finalize());
  std::cout << '}';
}

void writeNetwork(const std::string& key, const std::vector<std::string>& agents,
                  const ht::AgentMeasurements& observations, const ht::Adjacency& graph,
                  int max_iterations = 8, double tolerance = 0.0) {
  ht::TrackConfig config;
  config.max_iterations = max_iterations;
  config.tolerance = tolerance;
  ht::SynchronousTrackingNetwork network(agents, config, max_iterations, tolerance);
  const auto result = network.update(0.0, observations, graph);
  std::cout << '\"' << key << "\":{\"iterations\":" << result.iterations
            << ",\"residual\":" << result.residual << ",\"rounds\":[";
  for (std::size_t round = 0; round < result.round_history.size(); ++round) {
    if (round) std::cout << ',';
    std::cout << '{';
    bool first_agent = true;
    for (const auto& [agent, targets] : result.round_history[round]) {
      if (!first_agent) std::cout << ',';
      first_agent = false;
      const auto& state = targets.at("Target1");
      std::cout << '\"' << agent << "\":{\"trajectory\":";
      writeMatrix(state.trajectory);
      std::cout << ",\"dual\":";
      writeMatrix(state.dual);
      std::cout << ",\"residual\":" << state.residual << '}';
    }
    std::cout << '}';
  }
  std::cout << "],\"estimates\":{";
  bool first = true;
  for (const auto& agent : agents) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << '\"' << agent << "\":";
    writeEstimate(result.estimates.at(agent).at("Target1"));
  }
  std::cout << "}}";
}

void writeAsync() {
  ht::TargetTrack track("Target1");
  track.begin(10.0, measurement("Target1", 4.0, -3.0, 9.25));
  track.begin(10.5, std::nullopt);
  std::cout << "\"async\":{\"times\":";
  writeTimes(track.times());
  std::cout << ",\"information\":";
  writeMatrix(track.information());
  std::cout << ",\"trajectory\":";
  writeMatrix(track.trajectory());
  std::cout << '}';
}

void writeHandoff() {
  ht::TrackConfig config;
  config.window_seconds = 1.0;
  ht::SynchronousTrackingNetwork network({"a", "b"}, config, 4, 0.0);
  const ht::Adjacency graph{{"a", {"b"}}, {"b", {"a"}}};
  network.update(0.0,
      {{"a", {{"Target1", measurement("Target1", 3.0, 4.0, 0.0)}}}, {"b", {}}}, graph);
  network.update(1.0, {{"a", {}}, {"b", {}}}, graph);
  const auto handoffs = network.performHandoffs(1.0, graph);
  const auto& receiver = network.module("b").tracks().at("Target1");
  std::cout << "\"handoff\":{\"count\":" << handoffs.size() << ",\"matrix\":";
  writeMatrix(handoffs.at(0).information_matrix);
  std::cout << ",\"vector\":";
  writeMatrix(handoffs.at(0).information_vector);
  std::cout << ",\"receiver_information\":";
  writeMatrix(receiver.information());
  std::cout << ",\"receiver_vector\":";
  writeMatrix(receiver.informationVector());
  std::cout << ",\"receiver_state\":";
  writeMatrix(receiver.trajectory());
  std::cout << ",\"receiver_active\":" << (receiver.active() ? "true" : "false") << '}';
}

}  // namespace

int main() {
  std::cout << std::setprecision(17) << '{';
  writeModels();
  std::cout << ',';
  writeLocal();
  std::cout << ',';
  writeNetwork("connected", {"Drone1", "Husky1", "Husky2"},
      {{"Drone1", {{"Target1", measurement("Target1", 0.0, 0.0, 0.0)}}},
       {"Husky1", {{"Target1", measurement("Target1", 1.0, 0.0, 0.0)}}},
       {"Husky2", {}}}, lineGraph());
  std::cout << ',';
  writeNetwork("disconnected", {"a", "b"},
      {{"a", {{"Target1", measurement("Target1", 0.0, 0.0, 0.0)}}},
       {"b", {{"Target1", measurement("Target1", 10.0, 0.0, 0.0)}}}},
      {{"a", {}}, {"b", {}}}, 4, 0.0);
  std::cout << ',';
  writeAsync();
  std::cout << ',';
  writeHandoff();
  std::cout << "}\n";
  return 0;
}
