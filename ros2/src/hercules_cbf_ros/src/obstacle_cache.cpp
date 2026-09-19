#include "hercules_cbf_ros/obstacle_cache.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace hercules_cbf_ros {

namespace {

std::optional<double> captureStampSeconds(
    const hercules_interfaces::msg::ObstacleProxyArray& message) {
  const auto& stamp = message.capture_stamp;
  if (stamp.sec == 0 && stamp.nanosec == 0) return std::nullopt;
  if (stamp.sec < 0 || stamp.nanosec >= 1000000000U) return std::nullopt;
  const double seconds = static_cast<double>(stamp.sec) +
                         1e-9 * static_cast<double>(stamp.nanosec);
  if (!std::isfinite(seconds) || seconds <= 0.0) return std::nullopt;
  return seconds;
}

}  // namespace

ObstacleCache::ObstacleCache(double stale_after_seconds, std::string expected_agent_id,
                             std::string expected_clock_domain)
    : stale_after_seconds_(stale_after_seconds),
      expected_agent_id_(std::move(expected_agent_id)),
      expected_clock_domain_(std::move(expected_clock_domain)) {}

bool ObstacleCache::receive(const hercules_interfaces::msg::ObstacleProxyArray& message,
                            std::chrono::steady_clock::time_point receipt) {
  if (message.agent_id.empty() ||
      (!expected_agent_id_.empty() && message.agent_id != expected_agent_id_) ||
      message.header.frame_id != "airsim_world_ned" ||
      (!expected_clock_domain_.empty() && message.clock_domain != expected_clock_domain_) ||
      !message.acquisition_success || !message.valid || message.capture_id.empty()) {
    return false;
  }
  if (!capture_id_.empty() && message.capture_id == capture_id_) return false;
  if (!capture_clock_domain_.empty() && !message.clock_domain.empty() &&
      message.clock_domain != capture_clock_domain_) {
    return false;
  }
  const auto message_capture_stamp = captureStampSeconds(message);
  const bool capture_stamp_present = message.capture_stamp.sec != 0 ||
                                     message.capture_stamp.nanosec != 0;
  if (capture_stamp_present && !message_capture_stamp) return false;
  // Capture timestamps are comparable only inside one declared clock
  // domain.  Receipt age remains steady-clock based, so a producer that does
  // not provide capture metadata still gets safe freshness semantics.
  if (message.clock_domain == capture_clock_domain_ && message_capture_stamp &&
      has_capture_stamp_ && *message_capture_stamp <= capture_stamp_seconds_) {
    return false;
  }
  if (message.clock_domain == capture_clock_domain_ && has_capture_stamp_ &&
      !message_capture_stamp) {
    return false;
  }
  std::unordered_set<std::string> proxy_ids;
  for (const auto& proxy : message.proxies) {
    if (!proxy.proxy_id.empty() && !proxy_ids.insert(proxy.proxy_id).second) return false;
    for (double value : proxy.center) {
      if (!std::isfinite(value)) return false;
    }
    if (!std::isfinite(proxy.radius) || proxy.radius < 0.0) return false;
    if (proxy.has_velocity) {
      for (double value : proxy.velocity) {
        if (!std::isfinite(value)) return false;
      }
    }
  }
  latest_ = ObstacleSnapshot{message, 0.0, true};
  receipt_ = receipt;
  capture_id_ = message.capture_id;
  capture_clock_domain_ = message.clock_domain;
  if (message_capture_stamp) {
    capture_stamp_seconds_ = *message_capture_stamp;
    has_capture_stamp_ = true;
  } else {
    capture_stamp_seconds_ = 0.0;
    has_capture_stamp_ = false;
  }
  return true;
}

std::optional<ObstacleSnapshot> ObstacleCache::snapshot(
    std::chrono::steady_clock::time_point now) const {
  if (!latest_) return std::nullopt;
  auto result = *latest_;
  result.age_seconds = std::max(
      0.0, std::chrono::duration<double>(now - receipt_).count());
  result.valid = result.age_seconds <= stale_after_seconds_;
  return result;
}

void ObstacleCache::reset() {
  latest_.reset();
  capture_id_.clear();
  capture_clock_domain_.clear();
  capture_stamp_seconds_ = 0.0;
  has_capture_stamp_ = false;
}

}  // namespace hercules_cbf_ros
