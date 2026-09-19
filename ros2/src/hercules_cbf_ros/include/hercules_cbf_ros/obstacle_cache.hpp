#pragma once

#include <chrono>
#include <optional>
#include <unordered_set>
#include <string>

#include <hercules_interfaces/msg/obstacle_proxy_array.hpp>

namespace hercules_cbf_ros {

struct ObstacleSnapshot {
  hercules_interfaces::msg::ObstacleProxyArray message;
  double age_seconds{0.0};
  bool valid{false};
};

class ObstacleCache {
 public:
  explicit ObstacleCache(double stale_after_seconds = 1.3,
                         std::string expected_agent_id = {},
                         std::string expected_clock_domain = {});
  bool receive(const hercules_interfaces::msg::ObstacleProxyArray& message,
               std::chrono::steady_clock::time_point receipt = std::chrono::steady_clock::now());
  std::optional<ObstacleSnapshot> snapshot(
      std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()) const;
  void reset();
  double staleAfter() const { return stale_after_seconds_; }

 private:
  double stale_after_seconds_;
  std::optional<ObstacleSnapshot> latest_;
  std::chrono::steady_clock::time_point receipt_{};
  std::string capture_id_;
  std::string expected_agent_id_;
  std::string expected_clock_domain_;
  std::string capture_clock_domain_;
  double capture_stamp_seconds_{0.0};
  bool has_capture_stamp_{false};
};

}  // namespace hercules_cbf_ros
