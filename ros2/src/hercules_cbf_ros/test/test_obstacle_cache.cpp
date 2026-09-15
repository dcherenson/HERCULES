#include <gtest/gtest.h>
#include "hercules_cbf_ros/obstacle_cache.hpp"

namespace {
hercules_interfaces::msg::ObstacleProxyArray msg(const std::string& capture) {
  hercules_interfaces::msg::ObstacleProxyArray value;
  value.header.frame_id = "airsim_world_ned";
  value.agent_id = "Drone1";
  value.capture_id = capture;
  value.acquisition_success = true;
  value.valid = true;
  value.proxies.resize(1);
  value.proxies[0].radius = 1.0;
  return value;
}
}

TEST(ObstacleCache, RejectsRepeatedCaptureAndExpiresByReceiptAge) {
  hercules_cbf_ros::ObstacleCache cache(1.0);
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_TRUE(cache.receive(msg("1"), t0));
  EXPECT_FALSE(cache.receive(msg("1"), t0 + std::chrono::milliseconds(1)));
  auto fresh = cache.snapshot(t0 + std::chrono::milliseconds(500));
  ASSERT_TRUE(fresh);
  EXPECT_TRUE(fresh->valid);
  EXPECT_DOUBLE_EQ(fresh->age_seconds, 0.5);
  auto stale = cache.snapshot(t0 + std::chrono::milliseconds(1500));
  ASSERT_TRUE(stale);
  EXPECT_FALSE(stale->valid);
}

TEST(ObstacleCache, FailureDoesNotReplaceLastSuccessfulCapture) {
  hercules_cbf_ros::ObstacleCache cache;
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_TRUE(cache.receive(msg("1"), t0));
  auto failed = msg("2");
  failed.acquisition_success = false;
  EXPECT_FALSE(cache.receive(failed, t0 + std::chrono::seconds(1)));
  ASSERT_TRUE(cache.snapshot(t0 + std::chrono::milliseconds(10)));
  EXPECT_EQ(cache.snapshot(t0)->message.capture_id, "1");
}
