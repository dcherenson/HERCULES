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

TEST(DistributedTargetSource, PredictsOnlyFreshActiveLocalEstimate) {
  hercules_mission_ros::DistributedTargetSource source(5.0);
  hercules_mission_ros::TimestampedTargetEstimate local;
  local.estimate.position = {2.0, -1.0};
  local.estimate.velocity = {0.5, 2.0};
  local.estimate.active = true;
  local.estimator_timestamp = 10.0;
  local.receipt_age = 0.1;
  const auto estimate = source.estimate(local, 12.0);
  EXPECT_TRUE(estimate.active);
  EXPECT_TRUE(estimate.position.isApprox(Eigen::Vector2d(3.0, 3.0)));
}

TEST(DistributedTargetSource, RejectsMissingInactiveStaleAndFutureEstimates) {
  hercules_mission_ros::DistributedTargetSource source(5.0);
  EXPECT_FALSE(source.estimate(std::nullopt, 10.0).active);

  hercules_mission_ros::TimestampedTargetEstimate local;
  local.estimate.active = false;
  local.estimator_timestamp = 10.0;
  EXPECT_FALSE(source.estimate(local, 10.0).active);
  local.estimate.active = true;
  local.receipt_age = 5.01;
  EXPECT_FALSE(source.estimate(local, 10.0).active);
  local.receipt_age = 0.0;
  EXPECT_FALSE(source.estimate(local, 9.99).active);
  EXPECT_FALSE(source.estimate(local, 15.01).active);
}
