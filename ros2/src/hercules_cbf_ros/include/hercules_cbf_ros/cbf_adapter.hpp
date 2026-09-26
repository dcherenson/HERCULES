#pragma once

#include <string>
#include <vector>

#include <hercules_interfaces/msg/cbf_diagnostics.hpp>
#include <hercules_interfaces/msg/ground_truth_state.hpp>
#include <hercules_interfaces/msg/obstacle_proxy_array.hpp>
#include <hercules_interfaces/msg/target_estimate.hpp>

#include "hercules_cbf/filter.hpp"

namespace hercules_cbf_ros {

struct AdapterResult {
  hercules_cbf::CBFResult result;
  hercules_interfaces::msg::CBFDiagnostics diagnostics;
};

hercules_cbf::AgentState stateFromRos(
    const hercules_interfaces::msg::GroundTruthState& message);
// Populate the frozen class model and margin that are consumed by Wang.  The
// model is affine in the planar position with coefficients
// [bias_x, x_x, y_x, bias_y, x_y, y_y].
void applyConfiguredWangModel(hercules_cbf::AgentState& state,
                              const hercules_cbf::CBFConfig& config);
hercules_cbf::ObstacleProxy obstacleFromRos(
    const hercules_interfaces::msg::ObstacleProxy& message);
void appendTargetProxy(const hercules_interfaces::msg::TargetEstimate& estimate,
                       const std::string& agent_id, double ugv_radius,
                       std::vector<hercules_cbf::ObstacleProxy>& obstacles,
                       double target_z = 0.0);
void appendTargetProxy(const hercules_interfaces::msg::TargetEstimate& estimate,
                       const std::string& agent_id, double ugv_radius,
                       std::vector<hercules_interfaces::msg::ObstacleProxy>& obstacles,
                       double target_z = 0.0);
AdapterResult filterRequest(
    const hercules_interfaces::msg::GroundTruthState& ego,
    const Eigen::VectorXd& nominal,
    const std::vector<hercules_interfaces::msg::GroundTruthState>& neighbors,
    const std::vector<hercules_interfaces::msg::ObstacleProxy>& obstacles,
    const hercules_cbf::CBFConfig& config, bool sensor_valid,
    const std::string& selected_method, const std::string& effective_method);

}  // namespace hercules_cbf_ros
