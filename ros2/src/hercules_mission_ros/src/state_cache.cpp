#include "hercules_mission_ros/state_cache.hpp"

#include <cmath>
#include <stdexcept>

namespace hercules_mission_ros {

StateCache::StateCache(double freshness_timeout_seconds)
    : freshness_timeout_seconds_(freshness_timeout_seconds) {
  if (!std::isfinite(freshness_timeout_seconds_) ||
      freshness_timeout_seconds_ <= 0.0) {
    throw std::invalid_argument("freshness timeout must be finite and positive");
  }
}

ReceiveStatus StateCache::receive(const CanonicalState& state,
                                  double receipt_time_seconds) {
  ++messages_;
  if (!std::isfinite(receipt_time_seconds) || receipt_time_seconds < 0.0 ||
      !state.valid || state.stamp_ns <= 0 ||
      (state_ && state.stamp_ns < state_->stamp_ns)) {
    ++rejected_messages_;
    invalid_since_last_advance_ = true;
    return ReceiveStatus::kRejected;
  }

  last_receipt_seconds_ = receipt_time_seconds;
  if (state_ && state.stamp_ns == state_->stamp_ns) {
    state_ = state;
    return ReceiveStatus::kRepeated;
  }

  state_ = state;
  last_advanced_seconds_ = receipt_time_seconds;
  ++distinct_stamps_;
  invalid_since_last_advance_ = false;
  return ReceiveStatus::kAdvanced;
}

bool StateCache::fresh(double now_seconds) const {
  return state_ && !invalid_since_last_advance_ && distinct_stamps_ >= 2 &&
         std::isfinite(now_seconds) &&
         now_seconds >= last_receipt_seconds_ &&
         now_seconds >= last_advanced_seconds_ &&
         now_seconds - last_receipt_seconds_ < freshness_timeout_seconds_ &&
         now_seconds - last_advanced_seconds_ < freshness_timeout_seconds_;
}

}  // namespace hercules_mission_ros
