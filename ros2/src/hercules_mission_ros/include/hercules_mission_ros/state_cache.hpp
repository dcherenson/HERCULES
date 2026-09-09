#pragma once

#include <cstdint>
#include <optional>

#include "hercules_mission_ros/state_adapter.hpp"

namespace hercules_mission_ros {

enum class ReceiveStatus { kAdvanced, kRepeated, kRejected };

// Receipt time is caller-supplied steady-clock seconds. Simulator sample time
// remains the state stamp and is never replaced by receipt time.
class StateCache {
 public:
  explicit StateCache(double freshness_timeout_seconds = 0.5);

  ReceiveStatus receive(const CanonicalState& state,
                        double receipt_time_seconds);
  bool fresh(double now_seconds) const;

  const std::optional<CanonicalState>& state() const { return state_; }
  uint64_t messages() const { return messages_; }
  uint64_t distinctStamps() const { return distinct_stamps_; }
  uint64_t rejectedMessages() const { return rejected_messages_; }

 private:
  double freshness_timeout_seconds_{0.5};
  std::optional<CanonicalState> state_;
  double last_receipt_seconds_{0.0};
  double last_advanced_seconds_{0.0};
  uint64_t messages_{0};
  uint64_t distinct_stamps_{0};
  uint64_t rejected_messages_{0};
  bool invalid_since_last_advance_{false};
};

}  // namespace hercules_mission_ros
