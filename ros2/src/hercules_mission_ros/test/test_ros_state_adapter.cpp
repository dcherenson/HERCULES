#include <gtest/gtest.h>

#include <nav_msgs/msg/odometry.hpp>

#include "hercules_mission_ros/ros_state_adapter.hpp"

namespace hmr = hercules_mission_ros;
namespace hmc = hercules_mission_core;

TEST(RosStateAdapter, ExtractsTransportFieldsWithoutReinterpretation) {
  nav_msgs::msg::Odometry message;
  message.header.stamp.sec = 12;
  message.header.stamp.nanosec = 345;
  message.pose.pose.position.x = 1.0;
  message.pose.pose.position.y = 2.0;
  message.pose.pose.position.z = 3.0;
  message.pose.pose.orientation.w = 0.5;
  message.pose.pose.orientation.x = 0.5;
  message.pose.pose.orientation.y = -0.5;
  message.pose.pose.orientation.z = -0.5;
  message.twist.twist.linear.x = 4.0;
  message.twist.twist.linear.y = 5.0;
  message.twist.twist.linear.z = 6.0;
  message.twist.twist.angular.x = 7.0;
  message.twist.twist.angular.y = 8.0;
  message.twist.twist.angular.z = 9.0;
  const auto data = hmr::wrapperDataFromMessage(message);
  EXPECT_EQ(data.stamp_ns, 12000000345LL);
  EXPECT_TRUE(data.position.isApprox(Eigen::Vector3d(1.0, 2.0, 3.0)));
  EXPECT_TRUE(data.velocity.isApprox(Eigen::Vector3d(4.0, 5.0, 6.0)));
  EXPECT_TRUE(data.angular_velocity.isApprox(Eigen::Vector3d(7.0, 8.0, 9.0)));
  EXPECT_DOUBLE_EQ(data.orientation.w(), 0.5);
  EXPECT_DOUBLE_EQ(data.orientation.x(), 0.5);
  EXPECT_DOUBLE_EQ(data.orientation.y(), -0.5);
  EXPECT_DOUBLE_EQ(data.orientation.z(), -0.5);
}

TEST(RosStateAdapter, PublishesTypedCanonicalContractAndQuaternionOrder) {
  hmr::CanonicalState state;
  state.stamp_ns = 3000000004LL;
  state.agent_id = "Husky1";
  state.vehicle_type = hmc::VehicleType::kUgv;
  state.position = {1.0, 2.0, 3.0};
  state.velocity = {4.0, 5.0, 6.0};
  state.orientation = Eigen::Quaterniond(0.5, 0.1, 0.2, 0.3);
  state.yaw = 0.7;
  state.yaw_rate = -0.4;
  state.valid = true;
  const auto message = hmr::groundTruthMessage(state);
  EXPECT_EQ(message.header.frame_id, hmr::kCanonicalFrame);
  EXPECT_EQ(message.header.stamp.sec, 3);
  EXPECT_EQ(message.header.stamp.nanosec, 4U);
  EXPECT_EQ(message.agent_id, "Husky1");
  EXPECT_EQ(message.vehicle_type, "ugv");
  EXPECT_EQ(message.position[1], 2.0);
  EXPECT_EQ(message.velocity[2], 6.0);
  EXPECT_EQ(message.orientation[0], 0.5);
  EXPECT_EQ(message.orientation[1], 0.1);
  EXPECT_EQ(message.orientation[2], 0.2);
  EXPECT_EQ(message.orientation[3], 0.3);
  EXPECT_TRUE(message.valid);
}
