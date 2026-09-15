#include <gtest/gtest.h>

#include "hercules_tracking_ros/neighbor_graph.hpp"
#include "hercules_tracking_ros/tracking_protocol.hpp"

namespace {
hercules_interfaces::msg::TrackingEpoch epoch() {
  hercules_interfaces::msg::TrackingEpoch value;
  value.epoch_id = 7;
  value.target_id = "Target1";
  value.agent_ids = {"Drone1", "Husky1", "Husky2"};
  return value;
}

hercules_interfaces::msg::TrackingConsensus message(
    const std::string& sender, uint64_t epoch_id = 7, uint32_t round = 2) {
  hercules_interfaces::msg::TrackingConsensus value;
  value.sender_id = sender;
  value.target_id = "Target1";
  value.epoch_id = epoch_id;
  value.round_id = round;
  return value;
}
}  // namespace

TEST(NeighborGraph, MatchesInclusiveThreeDimensionalDistanceAndFlattening) {
  const hercules_tracking_ros::PositionMap positions = {
      {"Drone1", {0, 0, 0}}, {"Husky1", {3, 4, 0}}, {"Husky2", {20, 0, 0}}};
  const auto graph = hercules_tracking_ros::buildNeighborGraph(positions, 5.0);
  EXPECT_EQ(graph.at("Drone1"), std::vector<std::string>({"Husky1"}));
  const std::vector<std::string> ids = {"Drone1", "Husky1", "Husky2"};
  const auto flat = hercules_tracking_ros::flattenAdjacency(ids, graph);
  EXPECT_EQ(hercules_tracking_ros::neighborsFromFlattened("Husky1", ids, flat),
            std::vector<std::string>({"Drone1"}));
}

TEST(TrackingProtocol, FiltersEpochRoundSenderTargetAndGraph) {
  hercules_tracking_ros::TrackingProtocol protocol("Husky1", "Target1");
  protocol.begin(epoch(), {"Drone1"});
  protocol.setRound(2);
  EXPECT_EQ(protocol.accept(message("Drone1")),
            hercules_tracking_ros::MessageDisposition::kAccepted);
  EXPECT_EQ(protocol.accept(message("Drone1", 6)),
            hercules_tracking_ros::MessageDisposition::kWrongEpoch);
  EXPECT_EQ(protocol.accept(message("Drone1", 7, 3)),
            hercules_tracking_ros::MessageDisposition::kWrongRound);
  auto wrong_target = message("Drone1");
  wrong_target.target_id = "Other";
  EXPECT_EQ(protocol.accept(wrong_target),
            hercules_tracking_ros::MessageDisposition::kWrongTarget);
  EXPECT_EQ(protocol.accept(message("Husky2")),
            hercules_tracking_ros::MessageDisposition::kDisconnected);
  EXPECT_EQ(protocol.accept(message("Husky1")),
            hercules_tracking_ros::MessageDisposition::kWrongSender);
}

TEST(TrackingProtocol, ReportsMissingNeighborAndRequiresAllAgentStatuses) {
  hercules_tracking_ros::TrackingProtocol protocol("Husky1", "Target1");
  protocol.begin(epoch(), {"Drone1", "Husky2"});
  protocol.setRound(1);
  EXPECT_FALSE(protocol.allNeighborMessagesReceived());
  EXPECT_EQ(protocol.missingNeighbors().size(), 2U);
  EXPECT_EQ(protocol.accept(message("Drone1", 7, 1)),
            hercules_tracking_ros::MessageDisposition::kAccepted);
  EXPECT_EQ(protocol.missingNeighbors(), std::vector<std::string>({"Husky2"}));
  for (const auto& agent : epoch().agent_ids) {
    hercules_interfaces::msg::TrackingRoundStatus status;
    status.sender_id = agent;
    status.target_id = "Target1";
    status.epoch_id = 7;
    status.round_id = 1;
    EXPECT_TRUE(protocol.acceptStatus(status));
  }
  EXPECT_TRUE(protocol.allStatusesReceived());
}
