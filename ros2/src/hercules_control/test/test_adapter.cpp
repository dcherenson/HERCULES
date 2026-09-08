#include "hercules_control/ros_adapter.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <cmath>
using namespace hercules_control;
TEST(Adapter, QuaternionOrderAndNoDuplicateAxisFlip) {
  nav_msgs::msg::Odometry m; m.header.stamp.sec = 2; m.header.stamp.nanosec = 3;
  m.pose.pose.position.y = -2; m.pose.pose.position.z = 4;
  m.pose.pose.orientation.w = std::sqrt(.5); m.pose.pose.orientation.x = .5;
  m.pose.pose.orientation.y = .5; m.pose.pose.orientation.z = 0;
  m.twist.twist.linear.y = 3;
  const auto s = from_odometry(m);
  EXPECT_EQ(s.stamp_ns,2000000003LL); EXPECT_EQ(s.position.y(),-2);
  EXPECT_EQ(s.position.z(),4); EXPECT_DOUBLE_EQ(s.orientation.w(),std::sqrt(.5));
  EXPECT_DOUBLE_EQ(s.orientation.x(),.5); EXPECT_DOUBLE_EQ(s.orientation.y(),.5);
  EXPECT_DOUBLE_EQ(s.orientation.z(),0); EXPECT_EQ(s.velocity.y(),3);
  EXPECT_TRUE(valid(s));
}
TEST(Adapter, NativeCommandSignsAndBounds) {
  VelocityCommand c; c.velocity = {1,.1,.15}; c.yaw_rate = .1;
  auto m=to_drone_message(c); EXPECT_DOUBLE_EQ(m.twist.linear.x,.2);
  EXPECT_DOUBLE_EQ(m.twist.linear.y,-.1); EXPECT_DOUBLE_EQ(m.twist.linear.z,-.15);
  EXPECT_DOUBLE_EQ(m.twist.angular.z,-.1);
  auto ground=to_ground_message({1,0,2}); EXPECT_FLOAT_EQ(ground.throttle,.15f);
  EXPECT_FLOAT_EQ(ground.brake,1); EXPECT_FALSE(ground.manual);
  c.velocity.x()=std::numeric_limits<double>::infinity(); EXPECT_THROW(to_drone_message(c),std::invalid_argument);
}
