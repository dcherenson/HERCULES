#include <gtest/gtest.h>

#include "hercules_tracking_ros/tracking_adapter.hpp"

TEST(TrackingAdapter, MeasurementPreservesRowMajorCovarianceAndDelayedTimestamp) {
  hercules_interfaces::msg::TargetMeasurement message;
  message.target_id = "Target1";
  message.source_id = "Drone1";
  message.capture_id = "capture-7";
  message.sensor_id = "target_bottom";
  message.stamp = hercules_tracking_ros::timeMessage(9.25);
  message.position = {3.0, -4.0};
  message.covariance = {1.0, 0.2, 0.2, 2.0};
  message.valid = true;
  message.visible = true;
  const auto value = hercules_tracking_ros::measurementFromRos(message);
  EXPECT_DOUBLE_EQ(value.timestamp, 9.25);
  EXPECT_NEAR(value.covariance(0, 1), 0.2, 1e-12);
  EXPECT_NEAR(value.covariance(1, 0), 0.2, 1e-12);
  EXPECT_EQ(value.capture_id, "capture-7");
}

TEST(TrackingAdapter, ConsensusTrajectoryAndCovarianceRoundTrip) {
  hercules_tracking::TrackMessage value;
  value.target_id = "Target1";
  value.times = {1.0, 2.0};
  value.trajectory.resize(8);
  value.trajectory << 1, 2, 3, 4, 5, 6, 7, 8;
  value.state_covariance << 1, 2, 3, 4, 5, 6, 7, 8,
                            9, 10, 11, 12, 13, 14, 15, 16;
  value.active = true;
  const auto message = hercules_tracking_ros::consensusToRos("Drone1", 4, 3, value, 2.0);
  EXPECT_EQ(message.epoch_id, 4U);
  EXPECT_EQ(message.round_id, 3U);
  EXPECT_EQ(message.trajectory_state[4], 5.0);
  EXPECT_EQ(message.state_covariance[6], 7.0);
  const auto restored = hercules_tracking_ros::consensusFromRos(message);
  EXPECT_TRUE(restored.trajectory.isApprox(value.trajectory, 0.0));
  EXPECT_TRUE(restored.state_covariance.isApprox(value.state_covariance, 0.0));
  EXPECT_TRUE(restored.state.isApprox((hercules_tracking::State() << 5, 6, 7, 8).finished()));
}

TEST(TrackingAdapter, RejectsMalformedTrajectory) {
  hercules_interfaces::msg::TrackingConsensus message;
  message.trajectory_state = {1.0, 2.0, 3.0};
  EXPECT_THROW(hercules_tracking_ros::consensusFromRos(message), std::invalid_argument);
}

TEST(TrackingAdapter, HandoffAndEstimateRoundTrip) {
  hercules_tracking::HandoffMessage value;
  value.source_id = "Drone1";
  value.receiver_id = "Husky1";
  value.target_id = "Target1";
  value.timestamp = 2.5;
  value.information_matrix.setIdentity();
  value.information_vector << 1, 2, 3, 4;
  value.state_covariance = hercules_tracking::StateMatrix::Identity() * 2.0;
  const auto message = hercules_tracking_ros::handoffToRos(9, value);
  const auto restored = hercules_tracking_ros::handoffFromRos(message);
  EXPECT_EQ(message.epoch_id, 9U);
  EXPECT_TRUE(restored.information_matrix.isApprox(value.information_matrix));
  EXPECT_TRUE(restored.information_vector.isApprox(value.information_vector));
  EXPECT_TRUE(restored.state_covariance->isApprox(*value.state_covariance));

  hercules_tracking::TargetEstimate estimate;
  estimate.target_id = "Target1";
  estimate.timestamp = 4.0;
  estimate.position = {1.0, 2.0};
  estimate.velocity = {3.0, 4.0};
  estimate.state_covariance = value.state_covariance.value();
  estimate.active = true;
  estimate.admm_iterations = 6;
  estimate.consensus_residual = 0.02;
  estimate.measurement_residual = hercules_tracking::Position(0.1, -0.2);
  const auto estimate_message = hercules_tracking_ros::estimateToRos(estimate);
  EXPECT_EQ(estimate_message.state_covariance[10], 2.0);
  EXPECT_EQ(estimate_message.consensus_iterations, 6U);
  EXPECT_TRUE(estimate_message.has_measurement_residual);
}
