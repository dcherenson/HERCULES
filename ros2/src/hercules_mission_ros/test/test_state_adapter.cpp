#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include <Eigen/Geometry>

#include "hercules_mission_core/target_motion.hpp"
#include "hercules_mission_ros/state_adapter.hpp"
#include "hercules_mission_ros/state_cache.hpp"

namespace hmr = hercules_mission_ros;
namespace hmc = hercules_mission_core;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

hmr::WrapperOdometryData wrapperFromNed(
    int64_t stamp_ns, const Eigen::Vector3d& local_position,
    const Eigen::Vector3d& velocity, const Eigen::Vector3d& angular_velocity,
    const Eigen::Quaterniond& orientation) {
  hmr::WrapperOdometryData wrapper;
  wrapper.stamp_ns = stamp_ns;
  wrapper.position = {
      local_position.x(), -local_position.y(), -local_position.z()};
  wrapper.velocity = {velocity.x(), -velocity.y(), -velocity.z()};
  wrapper.angular_velocity = {
      angular_velocity.x(), -angular_velocity.y(), -angular_velocity.z()};
  wrapper.orientation = Eigen::Quaterniond(
      orientation.w(), orientation.x(), -orientation.y(), -orientation.z());
  return wrapper;
}

hmr::CanonicalState validState(int64_t stamp_ns) {
  return hmr::fromWrapperOdometry(
      wrapperFromNed(stamp_ns, Eigen::Vector3d::Zero(),
                     Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                     Eigen::Quaterniond::Identity()),
      Eigen::Vector3d::Zero(), "Drone1", hmc::VehicleType::kDrone);
}

}  // namespace

TEST(StateAdapter, ReconstructsWorldPositionWithNonzeroOriginAndOffset) {
  const Eigen::Vector3d origin(11.0, -7.0, 2.5);
  const Eigen::Vector3d local(-3.0, 4.0, -1.5);
  const auto state = hmr::fromWrapperOdometry(
      wrapperFromNed(10, local, Eigen::Vector3d::Zero(),
                     Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity()),
      origin, "Drone1", hmc::VehicleType::kDrone);
  ASSERT_TRUE(state.valid);
  EXPECT_TRUE(state.position.isApprox(origin + local, 1e-12));
}

TEST(StateAdapter, InvertsFixedAxisVelocityAndAngularVelocity) {
  const Eigen::Vector3d velocity(1.25, -2.5, 3.75);
  const Eigen::Vector3d angular(-0.2, 0.4, -0.6);
  const auto state = hmr::fromWrapperOdometry(
      wrapperFromNed(20, Eigen::Vector3d::Zero(), velocity, angular,
                     Eigen::Quaterniond::Identity()),
      Eigen::Vector3d::Zero(), "Husky1", hmc::VehicleType::kUgv);
  ASSERT_TRUE(state.valid);
  EXPECT_TRUE(state.velocity.isApprox(velocity, 1e-12));
  EXPECT_TRUE(state.angular_velocity.isApprox(angular, 1e-12));
  EXPECT_DOUBLE_EQ(state.yaw_rate, angular.z());
}

TEST(StateAdapter, RecoversQuaternionAndYawAcrossWrapBoundary) {
  for (const double yaw : {0.0, kPi / 2.0, -kPi / 2.0,
                           kPi - 1e-7, -kPi + 1e-7}) {
    const Eigen::Quaterniond orientation(
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    const auto state = hmr::fromWrapperOdometry(
        wrapperFromNed(30, Eigen::Vector3d::Zero(),
                       Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                       orientation),
        Eigen::Vector3d::Zero(), "Drone1", hmc::VehicleType::kDrone);
    ASSERT_TRUE(state.valid);
    EXPECT_TRUE(state.orientation.isApprox(orientation, 1e-12));
    EXPECT_NEAR(hmc::wrapAngle(state.yaw - yaw), 0.0, 1e-12);
  }

  // Exercise all quaternion components, not only the pure-yaw Z component.
  const Eigen::Quaterniond orientation =
      Eigen::AngleAxisd(0.37, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(-0.21, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(0.13, Eigen::Vector3d::UnitX());
  const auto state = hmr::fromWrapperOdometry(
      wrapperFromNed(31, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                     Eigen::Vector3d::Zero(), orientation),
      Eigen::Vector3d::Zero(), "Drone1", hmc::VehicleType::kDrone);
  ASSERT_TRUE(state.valid);
  EXPECT_TRUE(state.orientation.isApprox(orientation, 1e-12));
  EXPECT_NEAR(state.yaw, 0.37, 1e-12);
}

TEST(StateAdapter, DifferentOriginsPreventFalseCollocation) {
  const auto wrapper = wrapperFromNed(
      40, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
  const auto drone = hmr::fromWrapperOdometry(
      wrapper, {0.0, -3.0, -0.2}, "Drone1", hmc::VehicleType::kDrone);
  const auto husky = hmr::fromWrapperOdometry(
      wrapper, {0.0, 3.0, -0.2}, "Husky1", hmc::VehicleType::kUgv);
  EXPECT_TRUE(drone.position.isApprox(Eigen::Vector3d(0.0, -3.0, -0.2)));
  EXPECT_TRUE(husky.position.isApprox(Eigen::Vector3d(0.0, 3.0, -0.2)));
  EXPECT_DOUBLE_EQ((drone.position - husky.position).norm(), 6.0);
}

TEST(StateAdapter, MapsDirectlyToMissionCoreAgentState) {
  const auto state = hmr::fromWrapperOdometry(
      wrapperFromNed(50, {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0},
                     Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity()),
      {10.0, 20.0, 30.0}, "Husky1", hmc::VehicleType::kUgv);
  const auto agent = state.agentState();
  EXPECT_EQ(agent.agent_id, "Husky1");
  EXPECT_EQ(agent.vehicle_type, hmc::VehicleType::kUgv);
  EXPECT_TRUE(agent.position.isApprox(state.position));
  EXPECT_TRUE(agent.velocity.isApprox(state.velocity));
  EXPECT_DOUBLE_EQ(agent.yaw, state.yaw);
}

TEST(StateAdapter, RejectsInvalidInputs) {
  auto wrapper = wrapperFromNed(
      60, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
  wrapper.position.x() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(hmr::fromWrapperOdometry(
      wrapper, Eigen::Vector3d::Zero(), "Drone1", hmc::VehicleType::kDrone).error,
      hmr::ConversionError::kNonfinitePosition);

  wrapper = wrapperFromNed(
      60, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
  wrapper.velocity.z() = std::numeric_limits<double>::infinity();
  EXPECT_EQ(hmr::fromWrapperOdometry(
      wrapper, Eigen::Vector3d::Zero(), "Drone1", hmc::VehicleType::kDrone).error,
      hmr::ConversionError::kNonfiniteVelocity);

  wrapper = wrapperFromNed(
      60, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
  wrapper.orientation = Eigen::Quaterniond(0.0, 0.0, 0.0, 0.0);
  EXPECT_EQ(hmr::fromWrapperOdometry(
      wrapper, Eigen::Vector3d::Zero(), "Drone1", hmc::VehicleType::kDrone).error,
      hmr::ConversionError::kInvalidQuaternion);
  wrapper.stamp_ns = 0;
  EXPECT_EQ(hmr::fromWrapperOdometry(
      wrapper, Eigen::Vector3d::Zero(), "Drone1", hmc::VehicleType::kDrone).error,
      hmr::ConversionError::kNonpositiveTimestamp);
}

TEST(StateCache, RepeatedAndRegressingStampsCannotRemainFresh) {
  hmr::StateCache cache(0.5);
  EXPECT_EQ(cache.receive(validState(100), 1.0), hmr::ReceiveStatus::kAdvanced);
  EXPECT_FALSE(cache.fresh(1.0));
  EXPECT_EQ(cache.receive(validState(200), 1.1), hmr::ReceiveStatus::kAdvanced);
  EXPECT_TRUE(cache.fresh(1.1));
  EXPECT_EQ(cache.receive(validState(200), 1.4), hmr::ReceiveStatus::kRepeated);
  EXPECT_TRUE(cache.fresh(1.4));
  EXPECT_FALSE(cache.fresh(1.61));
  EXPECT_EQ(cache.receive(validState(150), 1.62), hmr::ReceiveStatus::kRejected);
  EXPECT_FALSE(cache.fresh(1.62));
  EXPECT_EQ(cache.receive(validState(300), 1.7), hmr::ReceiveStatus::kAdvanced);
  EXPECT_TRUE(cache.fresh(1.7));
  EXPECT_EQ(cache.distinctStamps(), 3U);
  EXPECT_EQ(cache.rejectedMessages(), 1U);
}

TEST(StateCache, InvalidStateAndReceiptTimeAreRejected) {
  hmr::StateCache cache;
  auto invalid = validState(100);
  invalid.valid = false;
  EXPECT_EQ(cache.receive(invalid, 1.0), hmr::ReceiveStatus::kRejected);
  EXPECT_EQ(cache.receive(validState(100),
                          std::numeric_limits<double>::quiet_NaN()),
            hmr::ReceiveStatus::kRejected);
  EXPECT_FALSE(cache.fresh(1.0));
}
