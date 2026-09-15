#include "hercules_cbf_ros/obstacle_cache.hpp"

#include <cmath>

namespace hercules_cbf_ros {

ObstacleCache::ObstacleCache(double stale_after_seconds)
    : stale_after_seconds_(stale_after_seconds) {}

bool ObstacleCache::receive(const hercules_interfaces::msg::ObstacleProxyArray& message,
                            std::chrono::steady_clock::time_point receipt) {
  if (message.agent_id.empty() || message.header.frame_id != "airsim_world_ned" ||
      !message.acquisition_success || !message.valid || message.capture_id.empty()) {
    return false;
  }
  if (!capture_id_.empty() && message.capture_id == capture_id_) return false;
  for (const auto& proxy : message.proxies) {
    for (double value : proxy.center) {
      if (!std::isfinite(value)) return false;
    }
    if (!std::isfinite(proxy.radius) || proxy.radius < 0.0) return false;
  }
  latest_ = ObstacleSnapshot{message, 0.0, true};
  receipt_ = receipt;
  capture_id_ = message.capture_id;
  return true;
}

std::optional<ObstacleSnapshot> ObstacleCache::snapshot(
    std::chrono::steady_clock::time_point now) const {
  if (!latest_) return std::nullopt;
  auto result = *latest_;
  result.age_seconds = std::chrono::duration<double>(now - receipt_).count();
  result.valid = result.age_seconds <= stale_after_seconds_;
  return result;
}

void ObstacleCache::reset() {
  latest_.reset();
  capture_id_.clear();
}

}  // namespace hercules_cbf_ros
