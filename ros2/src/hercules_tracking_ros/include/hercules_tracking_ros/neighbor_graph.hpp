#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace hercules_tracking_ros {

using PositionMap = std::map<std::string, Eigen::Vector3d>;
using Adjacency = std::map<std::string, std::vector<std::string>>;

Adjacency buildNeighborGraph(const PositionMap& positions, double communication_range);
std::vector<uint8_t> flattenAdjacency(const std::vector<std::string>& agent_ids,
                                      const Adjacency& adjacency);
std::vector<std::string> neighborsFromFlattened(
    const std::string& agent_id, const std::vector<std::string>& agent_ids,
    const std::vector<uint8_t>& adjacency);

}  // namespace hercules_tracking_ros
