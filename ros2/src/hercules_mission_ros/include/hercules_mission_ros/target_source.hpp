#pragma once

#include "hercules_mission_core/formation_controller.hpp"

namespace hercules_mission_ros {

class TargetSource {
 public:
  virtual ~TargetSource() = default;
  virtual hercules_mission_core::TargetEstimate estimate(
      const hercules_mission_core::AgentState& target_truth) const = 0;
};

class TruthTargetSource final : public TargetSource {
 public:
  hercules_mission_core::TargetEstimate estimate(
      const hercules_mission_core::AgentState& target_truth) const override {
    hercules_mission_core::TargetEstimate result;
    result.position = target_truth.position.head<2>();
    result.velocity = target_truth.velocity.head<2>();
    result.active = true;
    return result;
  }
};

}  // namespace hercules_mission_ros
