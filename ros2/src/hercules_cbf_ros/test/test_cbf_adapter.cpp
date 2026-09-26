#include <gtest/gtest.h>
#include "hercules_cbf_ros/cbf_adapter.hpp"

TEST(CbfAdapter, ConvertsStateAndPreservesInvalidSensorZeroFallback) {
  hercules_interfaces::msg::GroundTruthState ego;
  ego.header.frame_id = "airsim_world_ned";
  ego.agent_id = "Drone1";
  ego.vehicle_type = "drone";
  ego.position = {1.0, 2.0, -5.0};
  ego.velocity = {2.0, -3.0, 0.0};
  const auto converted = hercules_cbf_ros::stateFromRos(ego);
  EXPECT_EQ(converted.agent_id, "Drone1");
  EXPECT_TRUE(converted.position.isApprox(Eigen::Vector3d(1, 2, -5)));
  hercules_cbf::CBFConfig config;
  const auto output = hercules_cbf_ros::filterRequest(
      ego, Eigen::Vector3d::Zero(), {}, {}, config, false, "mestres", "mestres");
  EXPECT_FALSE(output.result.success);
  EXPECT_TRUE(output.result.safe_control.isApprox(Eigen::Vector3d::Zero()));
  EXPECT_EQ(output.diagnostics.solver_status, "invalid_or_stale_sensor");
}

TEST(CbfAdapter, TargetProxyUsesPositionCovarianceAndVelocity) {
  hercules_interfaces::msg::TargetEstimate estimate;
  estimate.active = true;
  estimate.position = {3.0, -2.0};
  estimate.velocity = {0.5, 1.0};
  estimate.state_covariance[0] = 4.0;
  estimate.state_covariance[5] = 1.0;
  std::vector<hercules_cbf::ObstacleProxy> obstacles;
  hercules_cbf_ros::appendTargetProxy(estimate, "Husky1", 1.25, obstacles);
  ASSERT_EQ(obstacles.size(), 1u);
  EXPECT_DOUBLE_EQ(obstacles[0].radius, 5.25);
  EXPECT_TRUE(obstacles[0].velocity->isApprox(Eigen::Vector3d(.5, 1, 0)));
}

TEST(CbfAdapter, AppliesFrozenClassAccelerationModelAndMargin) {
  hercules_interfaces::msg::GroundTruthState ego;
  ego.agent_id = "Drone1";
  ego.vehicle_type = "drone";
  ego.position = {2.0, -3.0, -5.0};
  ego.velocity = {0.0, 0.0, 0.0};
  hercules_cbf::CBFConfig config;
  config.method = hercules_cbf::Method::kWang;
  config.uav_margin = 0.4;
  config.uav_acceleration_model_coefficients = {1.0, 2.0, 3.0,
                                                -1.0, 4.0, 5.0};
  hercules_cbf::AgentState state = hercules_cbf_ros::stateFromRos(ego);
  hercules_cbf_ros::applyConfiguredWangModel(state, config);
  ASSERT_TRUE(state.learned_acceleration.has_value());
  EXPECT_TRUE(state.learned_acceleration->isApprox(Eigen::Vector3d(-4.0, -8.0, 0.0)));
  EXPECT_DOUBLE_EQ(state.margin, 0.4);
}
