#pragma once

#include <optional>

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

struct TimestampedTargetEstimate {
  hercules_mission_core::TargetEstimate estimate;
  double estimator_timestamp{0.0};
  double receipt_age{0.0};
};

class DistributedTargetSource {
 public:
  explicit DistributedTargetSource(double stale_after_seconds)
      : stale_after_seconds_(stale_after_seconds) {}

  hercules_mission_core::TargetEstimate estimate(
      const std::optional<TimestampedTargetEstimate>& local,
      double mission_timestamp) const {
    hercules_mission_core::TargetEstimate result;
    if (!local || !local->estimate.active || local->receipt_age > stale_after_seconds_ ||
        mission_timestamp < local->estimator_timestamp ||
        mission_timestamp - local->estimator_timestamp > stale_after_seconds_) {
      return result;
    }
    result = local->estimate;
    const double delta = mission_timestamp - local->estimator_timestamp;
    result.position += delta * result.velocity;
    return result;
  }

 private:
  double stale_after_seconds_;
};

}  // namespace hercules_mission_ros
