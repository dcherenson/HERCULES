#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "hercules_mission_core/mission_config.hpp"

namespace hmc = hercules_mission_core;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

hmc::AgentState state(const std::string& id, const Eigen::Vector3d& position,
                      const Eigen::Vector3d& velocity, double yaw,
                      hmc::VehicleType type) {
  return {id, position, velocity, yaw, type};
}

}  // namespace

TEST(TargetMotion, GeronoGeometryIsRouteAligned) {
  hmc::FigureEightConfig config;
  config.route_heading = kPi / 2.0;
  hmc::FigureEightTargetController controller(config);
  const auto [forward, left] = hmc::routeBasis(config.route_heading);
  double maximum_forward = -std::numeric_limits<double>::infinity();
  double maximum_left = -std::numeric_limits<double>::infinity();
  for (const auto& point : controller.points()) {
    maximum_forward = std::max(maximum_forward, point.head<2>().dot(forward));
    maximum_left = std::max(maximum_left, point.head<2>().dot(left));
  }
  EXPECT_NEAR(maximum_forward, 5.0, 1e-12);
  EXPECT_NEAR(maximum_left, 4.0, 1e-12);
  EXPECT_EQ(controller.index(), 8);
  EXPECT_FALSE(controller.points()[static_cast<std::size_t>(controller.index())]
                   .head<2>().isZero(1e-12));
}

TEST(TargetMotion, StartPlacementAndReferencePreservePhase) {
  hmc::FigureEightTargetController controller({});
  const Eigen::Vector3d desired(12.0, -4.0, 0.7);
  const int index = controller.index();
  controller.placeStartAt(desired);
  EXPECT_TRUE(controller.points()[static_cast<std::size_t>(index)].isApprox(desired, 1e-12));
  EXPECT_EQ(controller.index(), index);
  EXPECT_TRUE(controller.reference(0).isApprox(desired, 1e-12));
}

TEST(TargetMotion, OvershootAdvancesOnlyForward) {
  hmc::FigureEightTargetController controller({});
  controller.setIndex(8);
  const auto points = controller.points();
  controller.update(points[11], 0.0, 0.1);
  EXPECT_GE(controller.index(), 11);
  EXPECT_LE(controller.index(), 16);
}

TEST(TargetMotion, DirectionAndCommandsMatchBoundedModelSemantics) {
  hmc::FigureEightConfig config;
  config.sample_count = 64;
  config.direction = -1;
  config.waypoint_radius = 0.0;
  config.speed = 0.1;
  hmc::FigureEightTargetController controller(config);
  controller.setIndex(16);
  const auto points = controller.points();
  EXPECT_TRUE(controller.reference(1).isApprox(points[15], 1e-12));
  const Eigen::Vector2d delta = points[15].head<2>() - points[16].head<2>();
  const auto command = controller.update(points[16], std::atan2(delta.y(), delta.x()), 12.0);
  EXPECT_EQ(controller.index(), 15);
  EXPECT_GT(command.x(), 0.0);
  EXPECT_LE(command.x(), config.speed);
  EXPECT_LE(std::abs(command.y()), config.max_yaw_rate);
}

TEST(TargetFormation, FiveUavSlotsAreExact) {
  EXPECT_TRUE(hmc::uavTargetSlot("Drone1").isApprox(Eigen::Vector3d(-2.0, -2.0, 0.0)));
  EXPECT_TRUE(hmc::uavTargetSlot("Drone2").isApprox(Eigen::Vector3d(2.0, -2.0, 0.0)));
  EXPECT_TRUE(hmc::uavTargetSlot("SimpleFlight").isZero());
  EXPECT_TRUE(hmc::uavTargetSlot("Drone4").isApprox(Eigen::Vector3d(-2.0, 2.0, 0.0)));
  EXPECT_TRUE(hmc::uavTargetSlot("Drone5").isApprox(Eigen::Vector3d(2.0, 2.0, 0.0)));
}

TEST(TargetFormation, HeadingThresholdIsStrictAndRotatesSlots) {
  const Eigen::Vector3d position(10.0, 20.0, -1.0);
  const auto threshold = hmc::targetCenteredSlot(
      "Drone1", hmc::VehicleType::kDrone, position, {2.5, 0.0, 0.0},
      kPi / 2.0, -5.0);
  EXPECT_NEAR(threshold.heading, kPi / 2.0, 1e-12);
  const auto rotating = hmc::targetCenteredSlot(
      "Drone1", hmc::VehicleType::kDrone, position, {0.0, 3.0, 0.0},
      0.0, -5.0);
  EXPECT_TRUE(rotating.position.isApprox(Eigen::Vector3d(12.0, 18.0, -5.0), 1e-12));
  EXPECT_TRUE(rotating.velocity.isApprox(Eigen::Vector3d(0.0, 3.0, 0.0), 1e-12));
}

TEST(TargetFormation, UgvsFormCenteredCircumradiusTriangle) {
  const Eigen::Vector3d target(4.0, 8.0, -1.0);
  const Eigen::Vector3d velocity(3.0, 0.0, 0.0);
  std::vector<Eigen::Vector3d> positions;
  for (const auto& name : {"Husky1", "Husky2", "Husky3"}) {
    positions.push_back(hmc::targetCenteredSlot(
        name, hmc::VehicleType::kUgv, target, velocity, kPi / 2.0,
        -5.0, -1.0, 5.0).position);
  }
  EXPECT_TRUE(((positions[0] + positions[1] + positions[2]) / 3.0)
                  .isApprox(target, 1e-12));
  for (const auto& position : positions) {
    EXPECT_NEAR((position - target).head<2>().norm(), 5.0, 1e-12);
  }
}

TEST(FormationControl, UavInactiveAndSpeedSaturation) {
  const auto mission = hmc::ruralTargetTrackingConfig();
  const hmc::FormationController controller(mission.formation);
  const hmc::TargetEstimate inactive{{10.0, 20.0}, {1.0, 0.0}, false};
  EXPECT_TRUE(controller.targetNominalControl(
      state("Drone1", Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones(), 0.0,
            hmc::VehicleType::kDrone), inactive, 0.0).isZero());
  const hmc::TargetEstimate active{{10.0, 20.0}, {0.0, 0.0}, true};
  const auto command = controller.targetNominalControl(
      state("Drone5", {-100.0, 100.0, -50.0}, Eigen::Vector3d::Zero(), 0.0,
            hmc::VehicleType::kDrone), active, 0.0, -1.0, 5.0);
  EXPECT_NEAR(command.norm(), mission.formation.velocity_gain * mission.formation.max_speed,
              1e-12);
}

TEST(FormationControl, UgvHoldAlignmentAndYawSaturation) {
  const auto mission = hmc::ruralTargetTrackingConfig();
  const hmc::FormationController controller(mission.formation);
  const hmc::TargetEstimate estimate{{10.0, 20.0}, {0.0, 0.0}, true};
  const auto slot = hmc::targetCenteredSlot(
      "Husky1", hmc::VehicleType::kUgv, {10.0, 20.0, -1.0},
      Eigen::Vector3d::Zero(), 0.0, -5.0, -1.0, 5.0);
  EXPECT_TRUE(controller.targetNominalUnicycleControl(
      state("Husky1", slot.position + Eigen::Vector3d(0.5, 0.0, 0.0),
            Eigen::Vector3d::Zero(), 0.0, hmc::VehicleType::kUgv),
      estimate, 0.0, -1.0, 5.0).isZero());
  const auto turning = controller.targetNominalUnicycleControl(
      state("Husky1", {0.0, 20.0, -1.0}, Eigen::Vector3d::Zero(), kPi,
            hmc::VehicleType::kUgv), estimate, 0.0, -1.0, 5.0);
  EXPECT_GE(turning.x(), 0.25 * mission.formation.leader_max_speed - 1e-12);
  EXPECT_NEAR(std::abs(turning.y()), mission.formation.ugv_max_yaw_rate, 1e-12);
}

TEST(MissionConfig, RuralFixtureFreezesNamesAndTestedValues) {
  const auto config = hmc::ruralTargetTrackingConfig();
  const std::vector<std::string> expected{
      "Drone1", "Drone2", "SimpleFlight",
      "Husky1", "Husky2", "Husky3"};
  ASSERT_EQ(config.agents.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(config.agents[index].id, expected[index]);
  }
  EXPECT_EQ(config.target.id, "Target1");
  EXPECT_DOUBLE_EQ(config.control_dt, 0.1);
  EXPECT_DOUBLE_EQ(config.tracking_rate, 4.0);
  EXPECT_DOUBLE_EQ(config.target_motion.speed, 0.10);
  EXPECT_DOUBLE_EQ(config.target_toward_robots_offset, 5.0);
  EXPECT_DOUBLE_EQ(config.initial_heading_offset_deg, 90.0);
  EXPECT_DOUBLE_EQ(config.target_ugv_circumradius, 5.0);
  EXPECT_DOUBLE_EQ(config.formation.position_gain, 0.5);
  EXPECT_DOUBLE_EQ(config.tracking.process_noise, 0.20);
  EXPECT_DOUBLE_EQ(config.tracking.measurement_std, 0.25);
  EXPECT_EQ(config.source_revision, "2ef27d8ddc7f019f4b1af0196f7fbb4d91ce595f");
  EXPECT_TRUE(hmc::ruralTargetStartAnchor(
      Eigen::Vector3d::Zero(), Eigen::Vector3d(20.0, 8.0, -1.0), 0.0, config)
                  .isApprox(Eigen::Vector3d(0.0, 7.0, -1.0), 1e-12));
}

TEST(MissionRegression, SixAgentNominalKinematicsRemainDeterministic) {
  const auto mission = hmc::ruralTargetTrackingConfig();
  hmc::FigureEightConfig motion = mission.target_motion;
  motion.center = Eigen::Vector3d::Zero();
  motion.route_heading = 0.0;
  hmc::FigureEightTargetController target_controller(motion);
  target_controller.setIndex(mission.target_start_sample_index);
  target_controller.placeStartAt(Eigen::Vector3d::Zero());
  Eigen::Vector3d target_position = Eigen::Vector3d::Zero();
  const Eigen::Vector2d first_delta =
      target_controller.reference(1).head<2>() - target_position.head<2>();
  double target_yaw = std::atan2(first_delta.y(), first_delta.x());
  Eigen::Vector3d target_velocity = Eigen::Vector3d::Zero();
  const hmc::FormationController formation(mission.formation);

  std::map<std::string, hmc::AgentState> agents;
  for (const auto& item : mission.agents) {
    const auto slot = hmc::targetCenteredSlot(
        item.id, item.type, target_position, target_velocity, 0.0,
        mission.uav_altitude, mission.target_ground_z,
        mission.target_ugv_circumradius);
    agents.emplace(item.id, state(item.id, slot.position, Eigen::Vector3d::Zero(),
                                  0.0, item.type));
  }

  double maximum_uav_error = 0.0;
  double maximum_ugv_error = 0.0;
  double minimum_ugv_radius = std::numeric_limits<double>::infinity();
  double maximum_ugv_radius = 0.0;
  const int initial_index = target_controller.index();
  for (int step = 0; step < 160; ++step) {
    const Eigen::Vector2d target_command = target_controller.update(
        target_position, target_yaw, mission.control_dt);
    hmc::TargetEstimate estimate{
        target_position.head<2>(), target_velocity.head<2>(), true};
    for (auto& [name, agent] : agents) {
      const auto reference = hmc::targetCenteredSlot(
          name, agent.vehicle_type, target_position, target_velocity, 0.0,
          mission.uav_altitude, mission.target_ground_z,
          mission.target_ugv_circumradius);
      const double error = agent.vehicle_type == hmc::VehicleType::kDrone
                               ? (agent.position - reference.position).norm()
                               : (agent.position - reference.position).head<2>().norm();
      if (agent.vehicle_type == hmc::VehicleType::kDrone) {
        maximum_uav_error = std::max(maximum_uav_error, error);
        const Eigen::Vector3d acceleration = formation.targetNominalControl(
            agent, estimate, 0.0, mission.target_ground_z,
            mission.target_ugv_circumradius);
        agent.velocity += acceleration * mission.control_dt;
        agent.position += agent.velocity * mission.control_dt;
      } else {
        maximum_ugv_error = std::max(maximum_ugv_error, error);
        const double radius = (agent.position - target_position).head<2>().norm();
        minimum_ugv_radius = std::min(minimum_ugv_radius, radius);
        maximum_ugv_radius = std::max(maximum_ugv_radius, radius);
        const Eigen::Vector2d command = formation.targetNominalUnicycleControl(
            agent, estimate, 0.0, mission.target_ground_z,
            mission.target_ugv_circumradius);
        agent.yaw = hmc::wrapAngle(agent.yaw + command.y() * mission.control_dt);
        agent.velocity = Eigen::Vector3d(
            command.x() * std::cos(agent.yaw), command.x() * std::sin(agent.yaw), 0.0);
        agent.position += agent.velocity * mission.control_dt;
      }
    }
    target_yaw = hmc::wrapAngle(target_yaw + target_command.y() * mission.control_dt);
    target_velocity = Eigen::Vector3d(
        target_command.x() * std::cos(target_yaw),
        target_command.x() * std::sin(target_yaw), 0.0);
    target_position += target_velocity * mission.control_dt;
  }
  const int progress =
      (target_controller.index() - initial_index + motion.sample_count) % motion.sample_count;
  std::cout << "MISSION_REGRESSION_METRICS {\"maximum_uav_slot_error\":"
            << maximum_uav_error << ",\"maximum_ugv_slot_error\":"
            << maximum_ugv_error << ",\"minimum_ugv_target_radius\":"
            << minimum_ugv_radius << ",\"maximum_ugv_target_radius\":"
            << maximum_ugv_radius << ",\"target_path_progress_samples\":"
            << progress << "}\n";
  EXPECT_LT(maximum_uav_error, 1.0);
  EXPECT_LT(maximum_ugv_error, 1.0);
  EXPECT_GT(minimum_ugv_radius, 4.0);
  EXPECT_LT(maximum_ugv_radius, 6.0);
  EXPECT_GT(progress, 0);
}
