#include "hercules_localization_ros/localization_adapter.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace {

using hercules_localization_ros::LocalizationAdapter;
using hercules_localization_ros::LocalizationMeasurement;
using hercules_localization_ros::LocalizationPeerEstimate;
using hercules_localization_ros::AdapterOutput;

LocalizationMeasurement measurement() {
  LocalizationMeasurement value;
  value.agent_id = "Drone1";
  value.source_id = "gps";
  value.frame_id = "map";
  value.position = {1.0, 2.0, 3.0};
  value.velocity = {0.1, 0.2, 0.3};
  value.orientation = {1.0, 0.0, 0.0, 0.0};
  value.yaw = 0.4;
  value.yaw_rate = 0.05;
  value.covariance = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0};
  value.valid = true;
  return value;
}

LocalizationPeerEstimate peerEstimate(const std::string &sender_id) {
  LocalizationPeerEstimate value;
  value.sender_id = sender_id;
  value.frame_id = "map";
  value.position = {0.0, 0.0, 0.0};
  value.velocity = {0.0, 0.0, 0.0};
  value.orientation = {1.0, 0.0, 0.0, 0.0};
  value.covariance = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  value.valid = true;
  return value;
}

TEST(LocalizationAdapter, CanonicalizesSupportedAlgorithmNames) {
  EXPECT_EQ(LocalizationAdapter::canonicalAlgorithm("recursive-decentralized"),
            "recursive_decentralized");
  EXPECT_EQ(LocalizationAdapter::canonicalAlgorithm("GS-CI"), "gs_ci");
  EXPECT_TRUE(LocalizationAdapter::isSupportedAlgorithm("rde"));
  EXPECT_FALSE(LocalizationAdapter::isSupportedAlgorithm("unknown"));
}

TEST(LocalizationAdapter, RejectsUnknownAlgorithmWithoutChangingSelection) {
  LocalizationAdapter adapter("Drone1", "recursive_decentralized");
  EXPECT_FALSE(adapter.setAlgorithm("not-an-estimator"));
  EXPECT_EQ(adapter.algorithm(), "recursive_decentralized");
  EXPECT_TRUE(adapter.setAlgorithm("gs_ci"));
  EXPECT_EQ(adapter.algorithm(), "gs_ci");
}

TEST(LocalizationAdapter, DefaultTransportPassesThroughMeasurement) {
  LocalizationAdapter adapter("Drone1");
  const auto output = adapter.update(measurement(), {});

  EXPECT_EQ(output.estimate.agent_id, "Drone1");
  EXPECT_EQ(output.estimate.frame_id, "map");
  EXPECT_EQ(output.estimate.position[0], 1.0);
  EXPECT_EQ(output.estimate.velocity[2], 0.3);
  EXPECT_EQ(output.estimate.algorithm, "recursive_decentralized");
  EXPECT_EQ(output.estimate.sequence, 1U);
  EXPECT_TRUE(output.estimate.initialized);
  EXPECT_TRUE(output.estimate.valid);
  EXPECT_EQ(output.diagnostics.status, "pass_through");
  EXPECT_EQ(output.diagnostics.covariance_trace, 6.0);
}

TEST(LocalizationAdapter, RegisteredCallbackReceivesSelectedAlgorithmAndPeers) {
  LocalizationAdapter adapter("Drone1", "gs_ci");
  bool called = false;
  EXPECT_TRUE(adapter.registerAlgorithm(
      "gs_ci", [&](const LocalizationMeasurement &local,
                    const std::vector<LocalizationPeerEstimate> &peers,
                    const std::string &algorithm) {
        called = true;
        EXPECT_EQ(local.position[0], 1.0);
        EXPECT_EQ(peers.size(), 1U);
        EXPECT_EQ(algorithm, "gs_ci");
        AdapterOutput result;
        result.estimate.valid = true;
        result.diagnostics.status = "core_stub";
        return result;
      }));

  LocalizationPeerEstimate peer;
  peer.sender_id = "Husky1";
  peer.position = {4.0, 5.0, 6.0};
  peer.velocity = {0.0, 0.0, 0.0};
  peer.orientation = {1.0, 0.0, 0.0, 0.0};
  peer.covariance = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  peer.valid = true;

  const auto output = adapter.update(measurement(), {peer});
  EXPECT_TRUE(called);
  EXPECT_EQ(output.estimate.agent_id, "Drone1");
  EXPECT_EQ(output.estimate.algorithm, "gs_ci");
  EXPECT_EQ(output.diagnostics.status, "core_stub");
  EXPECT_EQ(output.diagnostics.peer_estimates_received, 1U);
}

TEST(LocalizationAdapter, InvalidMeasurementIsMarkedInvalidAndCounted) {
  LocalizationAdapter adapter("Drone1");
  auto value = measurement();
  value.valid = false;
  value.position[1] = std::numeric_limits<double>::quiet_NaN();

  const auto output = adapter.update(value, {});
  EXPECT_FALSE(output.estimate.valid);
  EXPECT_FALSE(output.diagnostics.valid);
  EXPECT_EQ(output.diagnostics.measurements_rejected, 1U);
  EXPECT_EQ(output.diagnostics.status, "invalid_measurement");
  for (const double component : output.estimate.position) EXPECT_TRUE(std::isfinite(component));
  for (const double component : output.estimate.covariance) EXPECT_TRUE(std::isfinite(component));
}

TEST(LocalizationAdapter, PeerConversionPreservesEstimateAndSender) {
  LocalizationAdapter adapter("Drone1");
  const auto estimate = adapter.update(measurement(), {}).estimate;
  const auto peer = LocalizationAdapter::toPeerEstimate(estimate, "Drone1");
  EXPECT_EQ(peer.sender_id, "Drone1");
  EXPECT_EQ(peer.sequence, estimate.sequence);
  EXPECT_EQ(peer.position[1], estimate.position[1]);
  EXPECT_EQ(peer.algorithm, estimate.algorithm);
}

TEST(LocalizationAdapter, NumericalCoreConsumesTypedPairAndReportsSelectedMethod) {
  LocalizationAdapter adapter("Drone1", "recursive_decentralized");
  auto local = measurement();
  local.position = {0.0, 0.0, 0.0};
  local.yaw = 0.0;

  LocalizationPeerEstimate peer;
  peer.sender_id = "Drone2";
  peer.position = {10.0, 0.0, 0.0};
  peer.velocity = {0.0, 0.0, 0.0};
  peer.orientation = {1.0, 0.0, 0.0, 0.0};
  peer.yaw = 0.0;
  peer.covariance = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  peer.valid = true;

  hercules_localization_ros::PlanarRelativeMeasurement relative;
  relative.source_id = "Drone1";
  relative.target_id = "Drone2";
  relative.observer_id = "Drone1";
  relative.observed_id = "Drone2";
  relative.range = 10.0;
  relative.bearing = 0.0;
  relative.covariance = {0.25, 0.0, 0.0, 0.01};
  relative.valid = true;

  const auto output = adapter.update(local, {peer}, {relative});
  EXPECT_EQ(output.estimate.algorithm, "recursive_decentralized");
  EXPECT_TRUE(output.estimate.initialized);
  EXPECT_TRUE(output.estimate.valid);
  EXPECT_EQ(output.diagnostics.algorithm, "recursive_decentralized");
  EXPECT_EQ(output.diagnostics.status, "ok");

  adapter.reset();
  ASSERT_TRUE(adapter.setAlgorithm("gs_ci"));
  ASSERT_TRUE(adapter.setGlobalAgentIds({"Drone1", "Drone2"}));
  const auto gs_output = adapter.update(local, {peer}, {relative});
  EXPECT_EQ(gs_output.estimate.algorithm, "gs_ci");
  EXPECT_TRUE(gs_output.estimate.valid);
}

TEST(LocalizationAdapter, UsesOdometryIncrementsAndKeepsNonAnchorGpsFixed) {
  LocalizationAdapter adapter("Drone2", "recursive_decentralized");
  auto first_fix = measurement();
  first_fix.position = {10.0, 20.0, 0.0};
  first_fix.yaw = 0.1;
  auto output = adapter.update(first_fix, {}, {});
  ASSERT_TRUE(output.estimate.initialized);
  EXPECT_DOUBLE_EQ(output.estimate.position[0], 10.0);

  auto second_fix = first_fix;
  second_fix.position[0] = 999.0;
  output = adapter.update(second_fix, {}, {});
  EXPECT_DOUBLE_EQ(output.estimate.position[0], 10.0);

  auto odom = first_fix;
  odom.source_id = "odom_local";
  odom.position = {100.0, 200.0, -5.0};
  output = adapter.update(odom, {}, {});
  ASSERT_TRUE(output.estimate.valid);
  EXPECT_DOUBLE_EQ(output.estimate.position[0], 10.0);
  odom.position[0] = 101.5;
  odom.position[1] = 198.0;
  output = adapter.update(odom, {}, {});
  EXPECT_NEAR(output.estimate.position[0], 11.5, 1e-12);
  EXPECT_NEAR(output.estimate.position[1], 18.0, 1e-12);
}

TEST(LocalizationAdapter, Drone1ConsumesSubsequentPrivateGpsFixes) {
  LocalizationAdapter adapter("Drone1", "recursive_decentralized");
  auto first_fix = measurement();
  first_fix.position = {0.0, 0.0, 0.0};
  first_fix.covariance = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  ASSERT_TRUE(adapter.update(first_fix, {}, {}).estimate.initialized);
  auto second_fix = first_fix;
  second_fix.position[0] = 2.0;
  const auto output = adapter.update(second_fix, {}, {});
  EXPECT_GT(output.estimate.position[0], 0.0);
  EXPECT_LT(output.estimate.position[0], 2.0);
  EXPECT_DOUBLE_EQ(output.estimate.yaw, first_fix.yaw);
}

TEST(LocalizationAdapter, RecursiveStatePersistsFactorsAcrossGpsAndOdometry) {
  LocalizationAdapter adapter("Drone1", "recursive_decentralized");
  auto gps = measurement();
  gps.position = {0.0, 0.0, 0.0};
  gps.covariance = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  ASSERT_TRUE(adapter.update(gps, {}, {}).estimate.initialized);

  auto peer = peerEstimate("Drone2");
  peer.position = {5.0, 0.0, 0.0};
  peer.covariance = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  hercules_localization_ros::PlanarRelativeMeasurement relative;
  relative.source_id = "Drone1";
  relative.target_id = "Drone2";
  relative.observer_id = "Drone1";
  relative.observed_id = "Drone2";
  relative.sequence = 1;
  relative.range = 4.5;
  relative.bearing = 0.0;
  relative.covariance = {0.25, 0.0, 0.0, 0.01};
  relative.valid = true;
  ASSERT_TRUE(adapter.update(gps, {peer}, {relative}).estimate.valid);
  ASSERT_TRUE(adapter.recursiveState().has_value());
  ASSERT_EQ(adapter.recursiveState()->correlationFactors().count("Drone2"), 1U);
  const Eigen::Matrix3d factor_after_pair =
      adapter.recursiveState()->correlationFactors().at("Drone2");
  EXPECT_GT(factor_after_pair.norm(), 0.0);

  auto private_gps = gps;
  private_gps.position[0] = 0.2;
  ASSERT_TRUE(adapter.update(private_gps, {peer}, {}).estimate.valid);
  const Eigen::Matrix3d factor_after_private =
      adapter.recursiveState()->correlationFactors().at("Drone2");
  EXPECT_FALSE(factor_after_private.isApprox(factor_after_pair, 1e-12));

  auto odom = gps;
  odom.source_id = "odom_local";
  odom.header.stamp.sec = 1;
  ASSERT_TRUE(adapter.update(odom, {}, {}).estimate.valid);
  const double x_before_delta = adapter.recursiveState()->estimate().mean.position.x();
  odom.position[0] = 1.0;
  odom.header.stamp.sec = 2;
  ASSERT_TRUE(adapter.update(odom, {}, {}).estimate.valid);
  EXPECT_NEAR(adapter.recursiveState()->estimate().mean.position.x(),
              x_before_delta + 1.0, 1e-12);
  EXPECT_TRUE(adapter.recursiveState()->correlationFactors().at("Drone2").isApprox(
      factor_after_private, 1e-12));
}

TEST(LocalizationAdapter, RejectedPairSequenceCanBeRetried) {
  LocalizationAdapter adapter("Drone1", "recursive_decentralized");
  auto gps = measurement();
  gps.position = {0.0, 0.0, 0.0};
  gps.covariance = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  ASSERT_TRUE(adapter.update(gps, {}, {}).estimate.initialized);

  auto peer = peerEstimate("Drone2");
  peer.position = {5.0, 0.0, 0.0};
  hercules_localization_ros::PlanarRelativeMeasurement relative;
  relative.source_id = "Drone1";
  relative.target_id = "Drone2";
  relative.observer_id = "Drone1";
  relative.observed_id = "Drone2";
  relative.sequence = 9;
  relative.range = -1.0;
  relative.bearing = 0.0;
  relative.relative_position = {0.0, 0.0};
  relative.covariance = {0.25, 0.0, 0.0, 0.01};
  relative.valid = true;
  const auto rejected = adapter.update(gps, {peer}, {relative});
  EXPECT_EQ(rejected.diagnostics.accepted_relative_updates, 0U);

  relative.range = 4.5;
  const auto retried = adapter.update(gps, {peer}, {relative});
  EXPECT_EQ(retried.diagnostics.accepted_relative_updates, 1U);
}

TEST(LocalizationAdapter, InstallsReturnedPairPosteriorExactlyOnce) {
  LocalizationAdapter adapter("Drone1", "recursive_decentralized");
  auto gps = measurement();
  gps.position = {0.0, 0.0, 0.0};
  gps.covariance = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  ASSERT_TRUE(adapter.update(gps, {}, {}).estimate.initialized);

  hercules_localization::PoseEstimate posterior;
  posterior.mean = hercules_localization::Pose2D(0.4, -0.1, 0.05);
  posterior.covariance = Eigen::Matrix3d::Identity() * 0.5;
  posterior.timestamp = 2.0;
  posterior.valid = true;
  const Eigen::Matrix3d factor = Eigen::Matrix3d::Identity() * 0.15;
  const auto installed = adapter.installRecursivePairPosterior(
      "Drone2", posterior, factor, "returned-pair", 11);
  ASSERT_TRUE(installed.accepted);
  EXPECT_FALSE(installed.duplicate);
  ASSERT_TRUE(adapter.recursiveState().has_value());
  EXPECT_TRUE(adapter.recursiveState()->estimate().vector().isApprox(
      posterior.vector(), 1e-12));
  EXPECT_TRUE(adapter.recursiveState()->correlationFactors().at("Drone2").isApprox(
      factor, 1e-12));

  const auto duplicate = adapter.installRecursivePairPosterior(
      "Drone2", posterior, factor, "returned-pair", 11);
  EXPECT_TRUE(duplicate.accepted);
  EXPECT_TRUE(duplicate.duplicate);
}

TEST(LocalizationAdapter, FirstGpsUsesLatestOdometryYawWhenAvailable) {
  LocalizationAdapter adapter("Drone2", "recursive_decentralized");
  auto odom = measurement();
  odom.source_id = "odom_local";
  odom.position = {100.0, 200.0, 0.0};
  odom.yaw = 1.25;
  ASSERT_FALSE(adapter.update(odom, {}, {}).estimate.initialized);

  auto gps = measurement();
  gps.source_id = "gps";
  gps.position = {3.0, 4.0, 0.0};
  gps.yaw = -0.75;
  const auto output = adapter.update(gps, {}, {});
  ASSERT_TRUE(output.estimate.initialized);
  EXPECT_DOUBLE_EQ(output.estimate.yaw, 1.25);
}

TEST(LocalizationAdapter, FirstOdometryCorrectsYawWhenGpsArrivesFirst) {
  LocalizationAdapter adapter("Drone2", "recursive_decentralized");
  auto gps = measurement();
  gps.source_id = "gps";
  gps.yaw = 0.0;
  ASSERT_TRUE(adapter.update(gps, {}, {}).estimate.initialized);

  auto odom = measurement();
  odom.source_id = "odom_local";
  odom.position = {100.0, 200.0, 0.0};
  odom.yaw = -1.2;
  const auto output = adapter.update(odom, {}, {});
  ASSERT_TRUE(output.estimate.valid);
  EXPECT_DOUBLE_EQ(output.estimate.yaw, -1.2);
  ASSERT_TRUE(adapter.recursiveState().has_value());
  EXPECT_DOUBLE_EQ(adapter.recursiveState()->estimate().mean.yaw, -1.2);
}

TEST(LocalizationAdapter, ResetStillRequiresFirstGpsFix) {
  LocalizationAdapter adapter("Drone2", "recursive_decentralized");
  ASSERT_TRUE(adapter.update(measurement(), {}, {}).estimate.initialized);
  adapter.reset();
  auto odom = measurement();
  odom.source_id = "odom_local";
  odom.position = {7.0, 8.0, 0.0};
  const auto waiting = adapter.update(odom, {}, {});
  EXPECT_FALSE(waiting.estimate.initialized);
  EXPECT_FALSE(waiting.estimate.valid);
  EXPECT_EQ(waiting.diagnostics.status, "waiting_for_first_gps");
}

TEST(LocalizationAdapter, GsCiConsumesOrderedGlobalBeliefPackets) {
  LocalizationAdapter adapter("Drone1", "gs_ci");
  ASSERT_TRUE(adapter.setGlobalAgentIds({"Drone1", "Drone2"}));
  auto local = measurement();
  local.position = {0.0, 0.0, 0.0};
  local.covariance = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  ASSERT_TRUE(adapter.update(local, {}, {}).estimate.initialized);

  hercules_localization_ros::GlobalCiBelief peer;
  peer.agent_id = "Drone2";
  peer.sender_id = "Drone2";
  peer.ordered_agent_ids = {"Drone1", "Drone2"};
  peer.global_mean = {2.0, 0.0, 10.0, 0.0, 0.0};
  peer.global_covariance.assign(25U, 0.0);
  for (std::size_t index = 0; index < 5U; ++index) peer.global_covariance[index * 5U + index] = 1.0;
  peer.timestamp = 1.0;
  peer.sequence = 1U;
  peer.valid = true;
  const auto output = adapter.update(local, {}, {}, {peer});
  EXPECT_TRUE(output.estimate.valid);
  EXPECT_GT(output.estimate.position[0], 0.0);
  EXPECT_EQ(output.estimate.algorithm, "gs_ci");
}

TEST(LocalizationAdapter, GsCiOdometryInflatesRemotePositionBlocks) {
  LocalizationAdapter adapter("Drone1", "gs_ci");
  ASSERT_TRUE(adapter.setGlobalAgentIds({"Drone1", "Drone2"}));
  hercules_localization::LocalizationConfig config;
  config.unknown_motion_variance = 2.0;
  adapter.setConfig(config);

  auto local = measurement();
  local.position = {0.0, 0.0, 0.0};
  local.covariance = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  ASSERT_TRUE(adapter.update(local, {}, {}).estimate.initialized);

  hercules_localization_ros::GlobalCiBelief peer;
  peer.agent_id = "Drone2";
  peer.sender_id = "Drone2";
  peer.ordered_agent_ids = {"Drone1", "Drone2"};
  peer.global_mean = {0.0, 0.0, 10.0, 0.0, 0.0};
  peer.global_covariance.assign(25U, 0.0);
  for (std::size_t index = 0; index < 5U; ++index) {
    peer.global_covariance[index * 5U + index] = 1.0;
  }
  peer.timestamp = 1.0;
  peer.sequence = 1U;
  peer.valid = true;
  ASSERT_TRUE(adapter.update(local, {}, {}, {peer}).estimate.valid);
  ASSERT_TRUE(adapter.globalBelief().has_value());
  const double remote_variance_before = adapter.globalBelief()->covariance(2, 2);

  auto odom = local;
  odom.source_id = "odom_local";
  odom.header.stamp.sec = 2;
  ASSERT_TRUE(adapter.update(odom, {}, {}).estimate.valid);
  odom.position[0] = 1.0;
  odom.header.stamp.sec = 3;
  ASSERT_TRUE(adapter.update(odom, {}, {}).estimate.valid);

  ASSERT_TRUE(adapter.globalBelief().has_value());
  EXPECT_GT(adapter.globalBelief()->covariance(2, 2), remote_variance_before);
  EXPECT_GT(adapter.globalBelief()->covariance(3, 3), remote_variance_before);
}

TEST(LocalizationAdapter, GsCiAppliesRelativeMeasurementToFullGlobalStateWithoutPeerPose) {
  LocalizationAdapter adapter("Drone1", "gs_ci");
  ASSERT_TRUE(adapter.setGlobalAgentIds({"Drone1", "Drone2"}));
  auto local = measurement();
  local.position = {0.0, 0.0, 0.0};
  local.covariance = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  ASSERT_TRUE(adapter.update(local, {}, {}).estimate.initialized);

  hercules_localization_ros::GlobalCiBelief peer;
  peer.agent_id = "Drone2";
  peer.sender_id = "Drone2";
  peer.ordered_agent_ids = {"Drone1", "Drone2"};
  peer.global_mean = {0.0, 0.0, 5.0, 0.0, 0.0};
  peer.global_covariance.assign(25U, 0.0);
  for (std::size_t index = 0; index < 5U; ++index) {
    peer.global_covariance[index * 5U + index] = 1.0;
  }
  peer.timestamp = 1.0;
  peer.sequence = 1U;
  peer.valid = true;
  ASSERT_TRUE(adapter.update(local, {}, {}, {peer}).estimate.valid);
  ASSERT_TRUE(adapter.globalBelief().has_value());
  const Eigen::VectorXd before = adapter.globalBelief()->mean;

  hercules_localization_ros::PlanarRelativeMeasurement relative;
  relative.source_id = "Drone1";
  relative.target_id = "Drone2";
  relative.observer_id = "Drone1";
  relative.observed_id = "Drone2";
  relative.sequence = 7U;
  relative.range = 4.0;
  relative.bearing = 0.0;
  relative.covariance = {0.25, 0.0, 0.0, 0.01};
  relative.valid = true;
  const auto output = adapter.update(local, {}, {relative}, {});
  ASSERT_TRUE(output.estimate.valid);
  EXPECT_EQ(output.diagnostics.accepted_relative_updates, 1U);
  ASSERT_TRUE(adapter.globalBelief().has_value());
  EXPECT_FALSE(adapter.globalBelief()->mean.isApprox(before, 1e-12));
}

}  // namespace
