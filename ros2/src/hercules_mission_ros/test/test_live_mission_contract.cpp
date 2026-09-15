#include <gtest/gtest.h>

#include <map>
#include <set>

#include "hercules_mission_core/mission_config.hpp"
#include "hercules_mission_core/target_formation.hpp"
#include "hercules_mission_ros/mission_actuation.hpp"
#include "hercules_mission_ros/mission_gate.hpp"

TEST(LiveMissionContract, HasEightFixedControlledAgentsAndSeparateTarget) {
  const auto config = hercules_mission_core::ruralTargetTrackingConfig();
  const std::vector<std::string> expected = {
      "Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5",
      "Husky1", "Husky2", "Husky3"};
  ASSERT_EQ(config.agents.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(config.agents[i].id, expected[i]);
    EXPECT_NE(config.agents[i].id, config.target.id);
  }
  EXPECT_EQ(config.target.id, "Target1");
  EXPECT_TRUE(hercules_mission_core::uavTargetSlot("Drone1").isApprox(
      Eigen::Vector3d(-2, -2, 0)));
  EXPECT_TRUE(hercules_mission_core::uavTargetSlot("Drone2").isApprox(
      Eigen::Vector3d(2, -2, 0)));
  EXPECT_TRUE(hercules_mission_core::uavTargetSlot("SimpleFlight").isZero());
  EXPECT_TRUE(hercules_mission_core::uavTargetSlot("Drone4").isApprox(
      Eigen::Vector3d(-2, 2, 0)));
  EXPECT_TRUE(hercules_mission_core::uavTargetSlot("Drone5").isApprox(
      Eigen::Vector3d(2, 2, 0)));
  EXPECT_DOUBLE_EQ(hercules_mission_core::ugvTargetSlotPhase("Husky1"), 0.0);
  EXPECT_GT(hercules_mission_core::ugvTargetSlotPhase("Husky2"), 0.0);
  EXPECT_LT(hercules_mission_core::ugvTargetSlotPhase("Husky3"), 0.0);
}

TEST(LiveMissionContract, RequiresEveryFreshStateAndOrigin) {
  const std::vector<std::string> required = {"Drone1", "Target1"};
  std::set<std::string> states = {"Drone1", "Target1"};
  std::set<std::string> origins = {"Drone1", "Target1"};
  std::map<std::string, double> ages = {{"Drone1", 0.1}, {"Target1", 0.1}};
  EXPECT_TRUE(hercules_mission_ros::missionReady(required, states, origins, ages, 0.5));
  origins.erase("Target1");
  EXPECT_FALSE(hercules_mission_ros::missionReady(required, states, origins, ages, 0.5));
  origins.insert("Target1");
  ages["Drone1"] = 0.6;
  EXPECT_FALSE(hercules_mission_ros::missionReady(required, states, origins, ages, 0.5));
  EXPECT_FALSE(hercules_mission_ros::actuationAllowed(true, true));
  EXPECT_TRUE(hercules_mission_ros::actuationAllowed(false, true));
  EXPECT_FALSE(hercules_mission_ros::actuationAllowed(false, false));
}

TEST(LiveMissionContract, StopCommandBrakesAndHandbrakesCars) {
  const auto stop = hercules_mission_ros::stoppedCarCommand();
  EXPECT_DOUBLE_EQ(stop.throttle, 0.0);
  EXPECT_DOUBLE_EQ(stop.brake, 1.0);
  EXPECT_TRUE(stop.handbrake);
}
