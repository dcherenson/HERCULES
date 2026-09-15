#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "hercules_tracking/synchronous_network.hpp"

int main() {
  namespace ht = hercules_tracking;
  ht::TrackConfig config;
  config.window_seconds = 5.0;
  config.process_noise_spectral_density = 0.2;
  config.measurement_std = 0.25;
  config.rho = 1.0;
  config.max_iterations = 8;
  config.tolerance = 0.0;
  const std::vector<std::string> agents = {"Drone1", "Husky1", "Husky2"};
  ht::SynchronousTrackingNetwork network(agents, config, 8, 0.0);
  ht::AgentMeasurements measurements;
  const ht::PositionMatrix covariance = ht::PositionMatrix::Identity() * 0.0625;
  measurements["Drone1"]["Target1"] =
      ht::TargetMeasurement("Target1", ht::Position(0.0, 0.0), covariance, 0.0);
  measurements["Husky1"]["Target1"] =
      ht::TargetMeasurement("Target1", ht::Position(1.0, 0.0), covariance, 0.0);
  measurements["Husky2"] = {};
  ht::Adjacency graph = {{"Drone1", {"Husky1"}},
                         {"Husky1", {"Drone1", "Husky2"}},
                         {"Husky2", {"Husky1"}}};
  const auto result = network.update(0.0, measurements, graph);
  std::cout << std::setprecision(17) << '{';
  for (std::size_t index = 0; index < agents.size(); ++index) {
    if (index) std::cout << ',';
    const auto& estimate = result.estimates.at(agents[index]).at("Target1");
    std::cout << '\"' << agents[index] << "\":{\"position\":["
              << estimate.position.x() << ',' << estimate.position.y()
              << "],\"velocity\":[" << estimate.velocity.x() << ','
              << estimate.velocity.y() << "]}";
  }
  std::cout << "}\n";
  return 0;
}
