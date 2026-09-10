#include "hercules_tracking_ros/neighbor_graph.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace hercules_tracking_ros {

Adjacency buildNeighborGraph(const PositionMap& positions, double communication_range) {
  if (communication_range < 0.0) throw std::invalid_argument("communication_range must be nonnegative");
  Adjacency result;
  for (const auto& [id, unused] : positions) {
    (void)unused;
    result[id] = {};
  }
  for (auto first = positions.begin(); first != positions.end(); ++first) {
    for (auto second = std::next(first); second != positions.end(); ++second) {
      if ((first->second - second->second).norm() <= communication_range) {
        result[first->first].push_back(second->first);
        result[second->first].push_back(first->first);
      }
    }
  }
  return result;
}

std::vector<uint8_t> flattenAdjacency(const std::vector<std::string>& agent_ids,
                                      const Adjacency& adjacency) {
  std::vector<uint8_t> output(agent_ids.size() * agent_ids.size(), 0);
  for (std::size_t row = 0; row < agent_ids.size(); ++row) {
    const auto found = adjacency.find(agent_ids[row]);
    if (found == adjacency.end()) continue;
    for (const auto& neighbor : found->second) {
      const auto column = std::find(agent_ids.begin(), agent_ids.end(), neighbor);
      if (column != agent_ids.end()) {
        output[row * agent_ids.size() + static_cast<std::size_t>(column - agent_ids.begin())] = 1;
      }
    }
  }
  return output;
}

std::vector<std::string> neighborsFromFlattened(
    const std::string& agent_id, const std::vector<std::string>& agent_ids,
    const std::vector<uint8_t>& adjacency) {
  if (adjacency.size() != agent_ids.size() * agent_ids.size()) {
    throw std::invalid_argument("adjacency must contain N*N entries");
  }
  const auto row = std::find(agent_ids.begin(), agent_ids.end(), agent_id);
  if (row == agent_ids.end()) throw std::invalid_argument("agent is absent from epoch graph");
  const std::size_t index = static_cast<std::size_t>(row - agent_ids.begin());
  std::vector<std::string> output;
  for (std::size_t column = 0; column < agent_ids.size(); ++column) {
    if (column != index && adjacency[index * agent_ids.size() + column]) {
      output.push_back(agent_ids[column]);
    }
  }
  return output;
}

}  // namespace hercules_tracking_ros
