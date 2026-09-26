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

TEST(CbfCore, WangUsesPlanarBrakingDistanceAndAuthoritySplit) {
  hc::CBFRequest request;
  request.ego = drone({0, 0, -5}, {1, 0, 0});
  request.ego.acceleration_limit = 6.0;
  request.nominal_control = Eigen::Vector3d::Zero();
  hc::AgentState neighbor = drone({3, 0, -5});
  neighbor.agent_id = "Drone2";
  neighbor.acceleration_limit = 2.0;
  request.neighbors.push_back(neighbor);
  hc::CBFConfig wang;
  wang.method = hc::Method::kWang;
  wang.wang_gamma = 1.5;
  auto constraints = hc::buildConstraints(request, wang);
  ASSERT_TRUE(constraints.valid);
  ASSERT_EQ(constraints.rows.size(), 2u);  // pair plus the UAV altitude row
  EXPECT_TRUE(constraints.rows[0].isApprox(Eigen::Vector3d(-3, 0, 0), 1e-12));
  EXPECT_NEAR(hc::strategyBAuthorityWeight(6.0, 2.0), 0.75, 1e-12);
  EXPECT_NEAR(constraints.barriers[0],
              hc::wangBarrier(request.ego.position, neighbor.position,
                              request.ego.velocity, neighbor.velocity, 6.0, 2.0, 2.0),
              1e-12);
}

TEST(CbfCore, WangModelAndMarginEnterPairConstraint) {
  hc::CBFRequest base;
  base.ego = drone({0, 0, -5}, {1, 0, 0});
  base.nominal_control = Eigen::Vector3d::Zero();
  hc::AgentState neighbor = drone({8, 0, -5});
  neighbor.agent_id = "Drone2";
  base.neighbors.push_back(neighbor);
  hc::CBFConfig config;
  config.method = hc::Method::kWang;
  config.uav_margin = 0.25;
  const auto nominal = hc::buildConstraints(base, config);
  base.ego.learned_acceleration = Eigen::Vector3d(0.5, 0, 0);
  neighbor.learned_acceleration = Eigen::Vector3d(-0.25, 0, 0);
  base.neighbors[0] = neighbor;
  const auto modeled = hc::buildConstraints(base, config);
  ASSERT_TRUE(nominal.valid);
  ASSERT_TRUE(modeled.valid);
  ASSERT_EQ(modeled.rows.size(), 2u);
  const double distance = 8.0;
  EXPECT_DOUBLE_EQ(modeled.robust_terms[0], distance * (0.25 + 0.25));
  EXPECT_NEAR(modeled.rhs[0] - nominal.rhs[0],
              -0.5 * (-8.0 * (0.5 - (-0.25))), 1e-12);
}

TEST(CbfCore, WangStaticObstacleUsesOneSidedTighteningAndUgvAcceleration) {
  hc::CBFRequest request;
  request.ego.agent_id = "Husky1";
  request.ego.vehicle_type = hc::VehicleType::kUgv;
  request.ego.position = Eigen::Vector3d(0, 0, 0);
  request.ego.velocity = Eigen::Vector3d::Zero();
  request.nominal_control = Eigen::Vector3d::Zero();
  request.ego.learned_acceleration = Eigen::Vector3d(0.4, 0, 0);
  hc::ObstacleProxy obstacle;
  obstacle.obstacle_id = "wall";
  obstacle.center = Eigen::Vector3d(8, 0, 0);
  obstacle.radius = 1.0;
  request.obstacles.push_back(obstacle);
  hc::CBFConfig config;
  config.method = hc::Method::kWang;
  config.ugv_margin = 0.2;
  const auto constraints = hc::buildConstraints(request, config);
  ASSERT_TRUE(constraints.valid);
  ASSERT_EQ(constraints.rows.size(), 1u);
  EXPECT_EQ(constraints.rows[0].size(), 3);
  EXPECT_TRUE(constraints.rows[0].isApprox(Eigen::Vector3d(-8, 0, 0), 1e-12));
  EXPECT_DOUBLE_EQ(constraints.robust_terms[0], 8.0 * 0.2);
  EXPECT_FALSE(hc::isUnicycle(request, config));
}

TEST(CbfCore, WangOverlapIsTruthfulFallback) {
  hc::CBFRequest request;
  request.ego = drone({0, 0, -5}, {2, -3, 0});
  request.nominal_control = Eigen::Vector3d::Zero();
  hc::ObstacleProxy obstacle;
  obstacle.obstacle_id = "occupied";
  obstacle.center = request.ego.position;
  obstacle.radius = 2.0;
  request.obstacles.push_back(obstacle);
  hc::CBFConfig config;
  config.method = hc::Method::kWang;
  const auto result = hc::filter(request, config);
  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.fallback);
  EXPECT_EQ(result.status, "invalid_wang_obstacle_geometry:occupied");
  EXPECT_TRUE(result.safe_control.isApprox(Eigen::Vector3d(-2, 3, 0), 1e-12));
}

TEST(CbfCore, WangCorridorRowsUseInwardAcceleration) {
  hc::CBFRequest request;
  request.ego = drone({0, 0, -5});
  request.nominal_control = Eigen::Vector3d::Zero();
  hc::CBFConfig config;
  config.method = hc::Method::kWang;
  config.wang_corridor_y_min = -7.0;
  config.wang_corridor_y_max = 7.0;
  const auto constraints = hc::buildConstraints(request, config);
  ASSERT_TRUE(constraints.valid);
  ASSERT_EQ(constraints.rows.size(), 3u);  // two walls plus altitude
  EXPECT_TRUE(constraints.rows[0].isApprox(Eigen::Vector3d(0, 1, 0), 1e-12));
  EXPECT_TRUE(constraints.rows[1].isApprox(Eigen::Vector3d(0, -1, 0), 1e-12));
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
  config.method = hc::Method::kMestres;
  config.uav_acceleration_limit = 0.1;
  auto result = hc::filter(request, config);
  EXPECT_TRUE(result.success);
  EXPECT_GT(result.maximum_row_violation, 0.0);
}
