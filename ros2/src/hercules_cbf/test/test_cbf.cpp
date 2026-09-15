#include <gtest/gtest.h>

#include "hercules_cbf/filter.hpp"

namespace hc = hercules_cbf;

hc::AgentState drone(const Eigen::Vector3d& p, const Eigen::Vector3d& v = Eigen::Vector3d::Zero()) {
  hc::AgentState state;
  state.agent_id = "Drone1";
  state.position = p;
  state.velocity = v;
  state.vehicle_type = hc::VehicleType::kDrone;
  return state;
}

TEST(CbfCore, SafeWangPassesNominalAndKeepsAltitudeRow) {
  hc::CBFRequest request;
  request.ego = drone({0, 0, -5});
  request.nominal_control = Eigen::Vector3d(0.1, 0, 0);
  hc::AgentState neighbor = drone({20, 0, -5});
  neighbor.agent_id = "Drone2";
  request.neighbors.push_back(neighbor);
  hc::CBFConfig config;
  config.method = hc::Method::kWang;
  auto result = hc::filter(request, config);
  ASSERT_TRUE(result.success);
  EXPECT_EQ(result.constraint_count, 2);
  EXPECT_TRUE(result.safe_control.isApprox(request.nominal_control, 1e-3));
}

TEST(CbfCore, WangSplitsOnlyNeighborRhs) {
  hc::CBFRequest request;
  request.ego = drone({0, 0, -5}, {1, 0, 0});
  request.nominal_control = Eigen::Vector3d::Zero();
  hc::AgentState neighbor = drone({3, 0, -5});
  neighbor.agent_id = "Drone2";
  request.neighbors.push_back(neighbor);
  hc::CBFConfig mestres, wang;
  mestres.method = hc::Method::kMestres;
  wang.method = hc::Method::kWang;
  auto a = hc::buildConstraints(request, mestres);
  auto b = hc::buildConstraints(request, wang);
  ASSERT_EQ(a.rows.size(), 2u);
  ASSERT_EQ(b.rows.size(), 2u);
  EXPECT_TRUE(a.rows[0].isApprox(b.rows[0], 1e-12));
  EXPECT_DOUBLE_EQ(b.rhs[0], 0.5 * a.rhs[0]);
  EXPECT_DOUBLE_EQ(b.rhs[1], a.rhs[1]);
}

TEST(CbfCore, UnicycleUsesLookaheadAndNonnegativeSpeed) {
  hc::CBFRequest request;
  request.ego.agent_id = "Husky1";
  request.ego.vehicle_type = hc::VehicleType::kUgv;
  request.ego.yaw = 0.4;
  request.nominal_control = Eigen::Vector2d(1, 0);
  hc::CBFConfig config;
  config.method = hc::Method::kMestres;
  auto result = hc::filter(request, config);
  EXPECT_TRUE(result.success);
  EXPECT_TRUE(result.safe_control.isApprox(Eigen::Vector2d(1, 0), 1e-3));
  EXPECT_EQ(result.safe_control.size(), 2);
  EXPECT_GE(result.safe_control[0], 0.0);
}

TEST(CbfCore, ProjectionClipsAndReportsRounds) {
  hc::ConstraintSet constraints;
  constraints.rows = {Eigen::Vector2d(1, 0)};
  constraints.rhs = {0.5};
  constraints.lower = Eigen::Vector2d(-1, -1);
  constraints.upper = Eigen::Vector2d(1, 1);
  hc::CBFConfig config;
  int rounds = 0;
  const auto value = hc::projectedCorrection(Eigen::Vector2d(0, 0), constraints, config, &rounds);
  EXPECT_TRUE(value.isApprox(Eigen::Vector2d(0.5, 0), 1e-9));
  EXPECT_EQ(rounds, 2);
}

TEST(CbfCore, InvalidSensorIsZeroAndInfeasibleUsesNegativeVelocityFallback) {
  hc::CBFRequest request;
  request.ego = drone({0, 0, -5}, {2, -3, 0});
  request.nominal_control = Eigen::Vector3d::Zero();
  request.sensor_valid = false;
  auto invalid = hc::filter(request, {});
  EXPECT_FALSE(invalid.success);
  EXPECT_EQ(invalid.status, "invalid_or_stale_sensor");
  EXPECT_TRUE(invalid.safe_control.isApprox(Eigen::Vector3d::Zero(), 1e-12));

  request.sensor_valid = true;
  hc::ObstacleProxy obstacle;
  obstacle.obstacle_id = "occupied";
  obstacle.center = request.ego.position;
  obstacle.radius = 2.0;
  request.obstacles.push_back(obstacle);
  auto infeasible = hc::filter(request, {});
  EXPECT_FALSE(infeasible.success);
  EXPECT_TRUE(infeasible.safe_control.isApprox(Eigen::Vector3d(-2, 3, 0), 1e-12));
}

TEST(CbfCore, PostClipCanRemainBarrierInfeasibleLikePython) {
  hc::CBFRequest request;
  request.ego = drone({0, 0, -5});
  request.nominal_control = Eigen::Vector3d::Zero();
  hc::ObstacleProxy obstacle;
  obstacle.obstacle_id = "impossible";
  obstacle.center = Eigen::Vector3d(-0.5, 0, -5);
  obstacle.radius = 2.0;
  request.obstacles.push_back(obstacle);
  hc::CBFConfig config;
  config.method = hc::Method::kWang;
  config.uav_acceleration_limit = 0.1;
  auto result = hc::filter(request, config);
  EXPECT_TRUE(result.success);
  EXPECT_GT(result.maximum_row_violation, 0.0);
}
