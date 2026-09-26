#include <cmath>

#include <gtest/gtest.h>

#include "hercules_mission_core/periodic_mission.hpp"

namespace hm = hercules_mission_core;

TEST(PeriodicMission, StartsAndEndsAtHomeWithPositiveYTangent) {
  hm::PeriodicMissionConfig config;
  config.home = Eigen::Vector3d(4.0, -2.0, -5.0);
  config.radius = 2.0;
  config.duration = 10.0;
  const auto start = hm::periodicMissionReference(
      config.home, config.radius, config.duration, 0.0);
  const auto end = hm::periodicMissionReference(
      config.home, config.radius, config.duration, config.duration);
  const double speed = 2.0 * std::acos(-1.0) * config.radius / config.duration;

  EXPECT_TRUE(start.position.isApprox(config.home, 1e-12));
  EXPECT_TRUE(end.position.isApprox(config.home, 1e-12));
  EXPECT_NEAR(start.velocity.x(), 0.0, 1e-12);
  EXPECT_NEAR(start.velocity.y(), speed, 1e-12);
  EXPECT_TRUE(start.velocity.isApprox(end.velocity, 1e-12));
  EXPECT_TRUE(start.acceleration.isApprox(end.acceleration, 1e-12));
}

TEST(PeriodicMission, RadiusDurationSetConstantSpeedAndCorrectAcceleration) {
  hm::PeriodicMissionConfig config;
  config.radius = 2.5;
  config.duration = 10.0;
  const double pi = std::acos(-1.0);
  const double omega = 2.0 * pi / config.duration;
  const double expected_speed = config.radius * omega;
  const double expected_acceleration = config.radius * omega * omega;

  for (int index = 0; index <= 100; ++index) {
    const auto state = hm::periodicMissionReference(
        config, config.duration * static_cast<double>(index) / 100.0);
    EXPECT_NEAR(state.velocity.norm(), expected_speed, 1e-12);
    EXPECT_NEAR(state.acceleration.norm(), expected_acceleration, 1e-12);
    EXPECT_NEAR((state.position - config.home).head<2>().norm(),
                config.radius * std::sqrt(2.0 -
                    2.0 * std::cos(omega * config.duration * index / 100.0)),
                1e-10);
  }
}

TEST(PeriodicMission, SampledTravelLengthIsOneCircumference) {
  hm::PeriodicMissionConfig config;
  config.radius = 2.5;
  config.duration = 10.0;
  constexpr int sample_count = 2000;
  double length = 0.0;
  auto previous = hm::periodicMissionReference(config, 0.0).position;
  for (int index = 1; index <= sample_count; ++index) {
    const auto state = hm::periodicMissionReference(
        config, config.duration * static_cast<double>(index) / sample_count);
    length += (state.position - previous).norm();
    previous = state.position;
  }
  EXPECT_NEAR(length, 2.0 * std::acos(-1.0) * config.radius, 1e-4);
}

TEST(PeriodicMission, TimeIsClampedToTheCompletedLoop) {
  hm::PeriodicMissionConfig config;
  config.home = Eigen::Vector3d(-1.0, 3.0, 2.0);
  const auto start = hm::periodicMissionReference(config, -1.0);
  const auto end = hm::periodicMissionReference(config, config.duration + 1.0);
  EXPECT_TRUE(start.position.isApprox(config.home, 1e-12));
  EXPECT_TRUE(end.position.isApprox(config.home, 1e-12));
}

TEST(PeriodicMission, OutAndBackEndpointsAreExactlyHomeAndClamped) {
  const Eigen::Vector3d home(4.0, -2.0, -5.0);
  constexpr double length = 2.0;
  constexpr double duration = 10.0;
  const auto start = hm::outAndBackMissionReference(
      home, length, duration, 0.0);
  const auto end = hm::outAndBackMissionReference(
      home, length, duration, duration);
  const auto before = hm::outAndBackMissionReference(
      home, length, duration, -1.0);
  const auto after = hm::outAndBackMissionReference(
      home, length, duration, duration + 1.0);

  EXPECT_EQ(start.position, home);
  EXPECT_EQ(end.position, home);
  EXPECT_EQ(before.position, home);
  EXPECT_EQ(after.position, home);
  EXPECT_TRUE(start.velocity.isApprox(end.velocity, 1e-12));
  EXPECT_TRUE(start.acceleration.isApprox(end.acceleration, 1e-12));

}

TEST(PeriodicMission, OutAndBackMidpointReachesLengthAndReverses) {
  const Eigen::Vector3d home(-1.0, 3.0, 2.0);
  constexpr double length = 2.0;
  constexpr double duration = 10.0;
  const auto midpoint = hm::outAndBackMissionReference(
      home, length, duration, duration / 2.0);

  EXPECT_NEAR(midpoint.position.x(), home.x(), 1e-12);
  EXPECT_NEAR(midpoint.position.y(), home.y() + length, 1e-12);
  EXPECT_NEAR(midpoint.position.z(), home.z(), 1e-12);
  EXPECT_NEAR(midpoint.velocity.norm(), 0.0, 1e-12);
  EXPECT_NEAR(midpoint.acceleration.x(), 0.0, 1e-12);
  EXPECT_NEAR(midpoint.acceleration.y(),
              -2.0 * length * std::acos(-1.0) * std::acos(-1.0) /
                  (duration * duration),
              1e-12);
}

TEST(PeriodicMission, OutAndBackDerivativesMatchAnalyticReference) {
  const Eigen::Vector3d home(0.5, -0.25, 1.5);
  constexpr double length = 3.0;
  constexpr double duration = 8.0;
  constexpr double elapsed = 1.75;
  const double pi = std::acos(-1.0);
  const double theta = 2.0 * pi * elapsed / duration;
  const auto state = hm::outAndBackMissionReference(
      home, length, duration, elapsed);

  EXPECT_NEAR(state.position.y(),
              home.y() + 0.5 * length * (1.0 - std::cos(theta)), 1e-12);
  EXPECT_NEAR(state.velocity.y(),
              length * pi / duration * std::sin(theta), 1e-12);
  EXPECT_NEAR(state.acceleration.y(),
              2.0 * length * pi * pi / (duration * duration) * std::cos(theta),
              1e-12);
  EXPECT_NEAR(state.position.x(), home.x(), 1e-12);
  EXPECT_NEAR(state.position.z(), home.z(), 1e-12);
  EXPECT_NEAR(state.velocity.x(), 0.0, 1e-12);
  EXPECT_NEAR(state.velocity.z(), 0.0, 1e-12);
}

TEST(PeriodicMission, OutAndBackSampledTravelLengthIsTwiceLength) {
  const Eigen::Vector3d home(2.0, -1.0, 0.0);
  constexpr double length = 2.0;
  constexpr double duration = 10.0;
  constexpr int sample_count = 2000;
  double travel = 0.0;
  auto previous = hm::outAndBackMissionReference(
      home, length, duration, 0.0).position;
  for (int index = 1; index <= sample_count; ++index) {
    const auto state = hm::outAndBackMissionReference(
        home, length, duration,
        duration * static_cast<double>(index) / sample_count);
    travel += (state.position - previous).norm();
    previous = state.position;
  }

  EXPECT_NEAR(travel, 2.0 * length, 1e-5);
}
