#include <gtest/gtest.h>

#include "hercules_mission_ros/target_source.hpp"

TEST(TruthTargetSource, CopiesTruthAndActivatesEstimate) {
  hercules_mission_core::AgentState truth;
  truth.agent_id = "Target1";
  truth.position = {1.0, 2.0, 3.0};
  truth.velocity = {4.0, 5.0, 6.0};
  const auto estimate = hercules_mission_ros::TruthTargetSource().estimate(truth);
  EXPECT_TRUE(estimate.active);
  EXPECT_TRUE(estimate.position.isApprox(Eigen::Vector2d(1.0, 2.0)));
  EXPECT_TRUE(estimate.velocity.isApprox(Eigen::Vector2d(4.0, 5.0)));
}
