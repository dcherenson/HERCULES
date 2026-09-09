#include <gtest/gtest.h>

#include <Eigen/Core>

#include "hercules_mission_ros/mission_actuation.hpp"

TEST(MissionActuation, UavIntegratesAndClipsComponentwise) {
  const auto command = hercules_mission_ros::uavVelocityCommand(
      {0.8, -0.5, 0.0}, {5.0, -10.0, 1.0}, 0.1, 1.0);
  EXPECT_TRUE(command.isApprox(Eigen::Vector3d(1.0, -1.0, 0.1)));
}

TEST(MissionActuation, UgvMatchesPythonSpeedAdapter) {
  const auto command = hercules_mission_ros::ugvCarCommand(1.0, 0.5, 0.0, 1.0);
  EXPECT_DOUBLE_EQ(command.throttle, 0.08);
  EXPECT_DOUBLE_EQ(command.brake, 0.0);
  EXPECT_DOUBLE_EQ(command.steering, 0.5);
  EXPECT_FALSE(command.handbrake);
  const auto stopped = hercules_mission_ros::ugvCarCommand(0.0, 0.0, 0.0, 1.0);
  EXPECT_DOUBLE_EQ(stopped.throttle, 0.0);
  EXPECT_DOUBLE_EQ(stopped.brake, 1.0);
  EXPECT_TRUE(stopped.handbrake);
}

TEST(MissionActuation, UgvBrakesOverspeedAndSaturatesSteering) {
  const auto command = hercules_mission_ros::ugvCarCommand(0.5, -4.0, 1.0, 1.0);
  EXPECT_DOUBLE_EQ(command.throttle, 0.0);
  EXPECT_NEAR(command.brake, 0.8, 1e-12);
  EXPECT_DOUBLE_EQ(command.steering, -1.0);
}
