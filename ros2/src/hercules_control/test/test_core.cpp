#include "hercules_control/core.hpp"
#include <gtest/gtest.h>
#include <limits>
using namespace hercules_control;

static void sample(StateReceiver &r, double t, double x = 0, double z = 0, double speed = 0) {
  State s; s.stamp_ns = static_cast<int64_t>((t + 1) * 1e9);
  s.position = {x, 0, z}; s.velocity.x() = speed;
  ASSERT_TRUE(r.receive(s, t));
}
TEST(State, RejectInvalidAndRegressingData) {
  StateReceiver r; sample(r, 0); sample(r, .1);
  EXPECT_TRUE(r.fresh(.2)); EXPECT_FALSE(r.fresh(.7));
  auto s = *r.state(); s.position.x() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(r.receive(s, .2)); EXPECT_FALSE(r.fresh(.2));
  s = *r.state(); s.orientation.coeffs().setZero(); EXPECT_FALSE(r.receive(s, .3));
  s = *r.state(); s.stamp_ns = 0; EXPECT_FALSE(r.receive(s, .3));
}
TEST(State, RepeatedStampsDoNotStayFresh) {
  StateReceiver r; sample(r, 0); sample(r, .1); auto s = *r.state();
  EXPECT_TRUE(r.receive(s, .7)); EXPECT_FALSE(r.fresh(.7));
  EXPECT_EQ(r.distinct_stamps, 2u);
}
TEST(Sequence, GroundMotionAndVerifiedStop) {
  StateReceiver r; SmokeSequence seq(false, 0); sample(r, 0); sample(r, .1);
  auto step = seq.tick(.1, r, true); EXPECT_EQ(step.action, Action::Move);
  EXPECT_DOUBLE_EQ(step.ground.throttle, .15);
  sample(r, 1.11, .1); EXPECT_EQ(seq.tick(1.11, r, true).action, Action::Stop);
  sample(r, 2.2, .1); EXPECT_EQ(seq.tick(2.2, r, true).action, Action::Finished);
  EXPECT_TRUE(seq.success());
}
TEST(Sequence, NoMotionIsNotSuccess) {
  StateReceiver r; SmokeSequence seq(false, 0); sample(r, 0); sample(r, .1); seq.tick(.1,r,true);
  sample(r, 1.2); seq.tick(1.2,r,true); sample(r, 2.3); seq.tick(2.3,r,true);
  EXPECT_TRUE(seq.done()); EXPECT_FALSE(seq.success());
}
TEST(Sequence, StaleStateAndInterruptStop) {
  for (bool interrupt : {false, true}) {
    StateReceiver r; SmokeSequence seq(false,0); sample(r,0); sample(r,.1); seq.tick(.1,r,true);
    EXPECT_EQ(seq.tick(interrupt ? .2 : .7,r,true,interrupt).action,Action::Stop);
    EXPECT_FALSE(seq.error().empty());
  }
}
TEST(Sequence, DisplacementAndSpeedGuards) {
  StateReceiver r; SmokeSequence seq(false,0); sample(r,0); sample(r,.1); seq.tick(.1,r,true);
  sample(r,.2,.6); EXPECT_EQ(seq.tick(.2,r,true).action,Action::Stop);
  EXPECT_FALSE(seq.error().empty());
  SmokeSequence seq2(false,.2); sample(r,.3,.6); seq2.tick(.3,r,true);
  sample(r,.4,.7,0,.6); EXPECT_EQ(seq2.tick(.4,r,true).action,Action::Stop);
}
TEST(Sequence, DroneTakeoffPulseAndLandingWithoutVelocity) {
  StateReceiver r; SmokeSequence seq(true,0); sample(r,0); sample(r,.1);
  EXPECT_EQ(seq.tick(.1,r,true).action,Action::Takeoff); seq.service_result(true);
  sample(r,1,0,3); EXPECT_EQ(seq.tick(1,r,true).action,Action::None);
  sample(r,1.6,0,3); EXPECT_EQ(seq.tick(1.6,r,true).action,Action::Move);
  sample(r,2.7,.1,3); EXPECT_EQ(seq.tick(2.7,r,true).action,Action::Stop);
  sample(r,3.8,.1,3); EXPECT_EQ(seq.tick(3.8,r,true).action,Action::Land);
  sample(r,4,.1,2); EXPECT_EQ(seq.tick(4,r,true).action,Action::None);
  seq.service_result(true); sample(r,5,.1,0);
  EXPECT_EQ(seq.tick(5,r,true).action,Action::Finished); EXPECT_TRUE(seq.success());
}
TEST(Sequence, FailedTakeoffAndLandingTimeoutFail) {
  StateReceiver r; SmokeSequence seq(true,0); sample(r,0); sample(r,.1); seq.tick(.1,r,true);
  seq.service_result(false); sample(r,.2); EXPECT_EQ(seq.tick(.2,r,true).action,Action::Stop);
  sample(r,1.3); EXPECT_EQ(seq.tick(1.3,r,true).action,Action::Land);
  sample(r,62); EXPECT_EQ(seq.tick(62,r,true).action,Action::Finished); EXPECT_FALSE(seq.success());
}
TEST(Sequence, NoStateDeadline) {
  StateReceiver r; SmokeSequence seq(false,0);
  EXPECT_EQ(seq.tick(11,r,false).action,Action::Stop);
  EXPECT_EQ(seq.tick(17,r,false).action,Action::Finished); EXPECT_FALSE(seq.success());
}
TEST(Sequence, OvershootWhileBrakingCannotPass) {
  StateReceiver r; SmokeSequence seq(false,0); sample(r,0); sample(r,.1); seq.tick(.1,r,true);
  sample(r,1.2,.1); seq.tick(1.2,r,true);
  sample(r,2.3,.6); seq.tick(2.3,r,true);
  EXPECT_TRUE(seq.done()); EXPECT_FALSE(seq.success());
  EXPECT_NE(seq.error().find("displacement"), std::string::npos);
}

TEST(Sequence, TakeoffDeadlineAndAltitudeCeiling) {
  for (bool altitude : {false, true}) {
    StateReceiver r; SmokeSequence seq(true,0); sample(r,0); sample(r,.1); seq.tick(.1,r,true);
    const double now = altitude ? .2 : 20.2;
    sample(r,now,0,altitude ? 5.1 : 0);
    EXPECT_EQ(seq.tick(now,r,true).action,Action::Stop);
    EXPECT_NE(seq.error().find(altitude ? "altitude" : "takeoff"),std::string::npos);
  }
}
