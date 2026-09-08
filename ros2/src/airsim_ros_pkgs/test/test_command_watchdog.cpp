#include "command_watchdog.hpp"
#include <gtest/gtest.h>
TEST(CommandWatchdog, LatchedCommandExpiresOnlyOnceAndCanBeRenewed) {
  hercules::CommandWatchdog w;
  EXPECT_FALSE(w.expired(100,.5));
  w.received(100); EXPECT_FALSE(w.expired(100.4,.5));
  EXPECT_TRUE(w.expired(100.5,.5)); EXPECT_FALSE(w.expired(101,.5));
  w.received(102); EXPECT_FALSE(w.expired(200,0));
  EXPECT_TRUE(w.expired(200,.5));
}

TEST(CommandWatchdog, ExpiredLatchedActuationIsReplacedWithBraking) {
  struct Controls { float throttle = 0, steering = 0, brake = 0; };
  hercules::CommandWatchdog w;
  Controls command{.15f, 0, 0};
  w.received(1.0);
  EXPECT_FALSE(w.stop_if_expired(1.49,.5,command));
  EXPECT_FLOAT_EQ(command.throttle,.15f);
  EXPECT_TRUE(w.stop_if_expired(1.5,.5,command));
  EXPECT_FLOAT_EQ(command.throttle,0); EXPECT_FLOAT_EQ(command.brake,1);
  EXPECT_FALSE(w.stop_if_expired(1.6,.5,command));
}
