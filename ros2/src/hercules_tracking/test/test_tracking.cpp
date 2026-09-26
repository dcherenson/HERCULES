#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Eigenvalues>

#include "hercules_tracking/linear_algebra.hpp"
#include "hercules_tracking/synchronous_network.hpp"

namespace ht = hercules_tracking;

namespace {

ht::TargetMeasurement measurement(const std::string& target_id, double x, double y,
                                  double timestamp, double variance = 0.25) {
  return {target_id, ht::Position(x, y), ht::PositionMatrix::Identity() * variance, timestamp};
}

ht::Adjacency lineGraph() {
  return {{"Drone1", {"Husky1"}},
          {"Husky1", {"Drone1", "Husky2"}},
          {"Husky2", {"Husky1"}}};
}

void expectVectorNear(const Eigen::VectorXd& left, const Eigen::VectorXd& right, double tolerance) {
  ASSERT_EQ(left.size(), right.size());
  EXPECT_LE((left - right).cwiseAbs().maxCoeff(), tolerance);
}

}  // namespace

TEST(Models, ConstantVelocityTransitionAndNegativeClamp) {
  ht::StateMatrix expected = ht::StateMatrix::Identity();
  expected(0, 2) = expected(1, 3) = 0.5;
  EXPECT_TRUE(ht::constantVelocityTransition(0.5).isApprox(expected, 1e-12));
  EXPECT_TRUE(ht::constantVelocityTransition(-2.0).isApprox(ht::StateMatrix::Identity(), 1e-12));
}

TEST(Models, WhiteAccelerationProcessNoise) {
  ht::StateMatrix expected = ht::StateMatrix::Zero();
  expected << 8.0 / 3.0, 0.0, 2.0, 0.0,
              0.0, 8.0 / 3.0, 0.0, 2.0,
              2.0, 0.0, 2.0, 0.0,
              0.0, 2.0, 0.0, 2.0;
  EXPECT_TRUE(ht::constantAccelerationProcessNoise(2.0).isApprox(expected, 1e-12));
  EXPECT_TRUE(ht::constantAccelerationProcessNoise(-1.0, -3.0).isZero(1e-12));
}

TEST(Models, PositionMeasurementMatrix) {
  Eigen::Matrix<double, 2, 4> expected = Eigen::Matrix<double, 2, 4>::Zero();
  expected(0, 0) = expected(1, 1) = 1.0;
  EXPECT_TRUE(ht::positionMeasurementMatrix().isApprox(expected, 0.0));
}

TEST(LinearAlgebra, BlockCholeskyMatchesIndependentDenseSolve) {
  std::vector<double> times{0.0, 0.25, 0.5};
  std::vector<std::optional<ht::TargetMeasurement>> measurements{
      measurement("Target1", 1.0, -2.0, 0.0), std::nullopt,
      measurement("Target1", 1.5, -1.5, 0.5)};
  auto [information, vector] = ht::assembleWindowInformation(
      times, measurements, ht::State::Zero(), ht::StateMatrix::Identity() * 4.0, 0.5);
  expectVectorNear(ht::solveBlockTridiagonal(information, vector),
                   ht::denseInformationSolution(information, vector), 1e-8);
}

TEST(LinearAlgebra, PaperMarginInflatesOnlyDirectMeasurementCovariance) {
  const std::vector<double> times{0.0, 1.0};
  const std::vector<std::optional<ht::TargetMeasurement>> measurements{
      std::nullopt, measurement("Target1", 1.0, -2.0, 1.0, 0.4)};
  const ht::State prior_mean = ht::State::Zero();
  const ht::StateMatrix prior_covariance = ht::StateMatrix::Identity() * 3.0;
  const auto nominal = ht::assembleWindowInformation(
      times, measurements, prior_mean, prior_covariance, 0.5, 1,
      std::nullopt, std::nullopt, false);
  const auto paper = ht::assembleWindowInformation(
      times, measurements, prior_mean, prior_covariance, 0.5, 1,
      std::nullopt, std::nullopt, true, 2.0, 0.30,
      ht::MaicpClass::kUgv);

  // The paper policy is R_nom + beta*r*I = R_nom + 0.6 I.  Only the final
  // position measurement factor changes; the prior and process blocks do not.
  const Eigen::MatrixXd prior_difference =
      paper.first.topLeftCorner(4, 4) - nominal.first.topLeftCorner(4, 4);
  EXPECT_TRUE(prior_difference.isZero(1e-12));
  const Eigen::MatrixXd measurement_difference =
      (paper.first - nominal.first).block(4, 4, 2, 2);
  EXPECT_TRUE(measurement_difference.isApprox(
      (Eigen::Matrix2d::Identity() * (1.0 / 1.0 - 1.0 / 0.4)), 1e-12));
  EXPECT_TRUE((paper.second - nominal.second).head<4>().isZero(1e-12));
}

TEST(Network, PaperModeRunsFixedFiftyRoundsWithoutToleranceExit) {
  ht::TrackConfig config;
  config.maicp_enabled = true;
  config.maicp_margin = 0.5;
  config.maicp_covariance_gain = 0.30;
  ht::SynchronousTrackingNetwork network({"a", "b"}, config, 2, 1e9);
  const ht::Adjacency graph{{"a", {"b"}}, {"b", {"a"}}};
  const auto result = network.update(
      0.0,
      {{"a", {{"Target1", measurement("Target1", 0.0, 0.0, 0.0)}}},
       {"b", {{"Target1", measurement("Target1", 1.0, 0.0, 0.0)}}}},
      graph);
  EXPECT_EQ(result.iterations, ht::SynchronousTrackingNetwork::kPaperAdmmIterations);
}

TEST(Network, ConnectedThreeAgentConsensus) {
  ht::TrackConfig config;
  config.max_iterations = 20;
  config.tolerance = 1e-3;
  ht::SynchronousTrackingNetwork network({"Drone1", "Husky1", "Husky2"}, config, 20, 1e-3);
  ht::AgentMeasurements observations{
      {"Drone1", {{"Target1", measurement("Target1", 0.0, 0.0, 0.0)}}},
      {"Husky1", {{"Target1", measurement("Target1", 1.0, 0.0, 0.0)}}},
      {"Husky2", {}}};
  const auto result = network.update(0.0, observations, lineGraph());
  const auto estimates = network.predictedEstimates(0.0);
  std::vector<double> x;
  for (const auto& agent : {"Drone1", "Husky1", "Husky2"}) {
    x.push_back(estimates.at(agent).at("Target1").position.x());
  }
  EXPECT_EQ(result.active_targets, std::vector<std::string>{"Target1"});
  EXPECT_LT(*std::max_element(x.begin(), x.end()) - *std::min_element(x.begin(), x.end()), 0.02);
  const double mean = (x[0] + x[1] + x[2]) / 3.0;
  EXPECT_GT(mean, 0.2);
  EXPECT_LT(mean, 0.8);
}

TEST(Track, MissingSamplesRemainInRollingWindow) {
  ht::TargetTracker tracker("Drone1");
  tracker.beginEpoch(0.0, {{"Target1", measurement("Target1", 0.0, 0.0, 0.0)}}, 3);
  tracker.beginEpoch(0.25, {}, 3);
  const auto& track = tracker.tracks().at("Target1");
  ASSERT_EQ(track.times().size(), 2u);
  EXPECT_FALSE(track.measurements()[1].has_value());
}

TEST(Network, DisconnectedComponentsDoNotExchangeTargets) {
  ht::SynchronousTrackingNetwork network({"a", "b"}, {}, 20, 1e-3);
  const auto result = network.update(
      0.0,
      {{"a", {{"Target1", measurement("Target1", 0.0, 0.0, 0.0)}}},
       {"b", {{"Target1", measurement("Target1", 10.0, 0.0, 0.0)}}}},
      {{"a", {}}, {"b", {}}});
  EXPECT_GT(std::abs(result.estimates.at("a").at("Target1").position.x() -
                     result.estimates.at("b").at("Target1").position.x()), 1.0);
  EXPECT_EQ(result.iterations, 1);
}

TEST(Track, MultipleTargetsRemainIndependent) {
  ht::TargetTracker tracker("Drone1");
  tracker.beginEpoch(0.0,
      {{"Target1", measurement("Target1", 1.0, 0.0, 0.0)},
       {"Target2", measurement("Target2", 20.0, 0.0, 0.0)}});
  const auto estimates = tracker.predictedEstimates(0.0);
  EXPECT_EQ(estimates.size(), 2u);
  EXPECT_LT(estimates.at("Target1").position.x(), estimates.at("Target2").position.x());
}

TEST(Handoff, InformationSustainsReceiver) {
  ht::TrackConfig config;
  config.window_seconds = 5.0;
  config.max_iterations = 10;
  ht::SynchronousTrackingNetwork network({"a", "b"}, config, 10, 1e-3);
  const ht::Adjacency graph{{"a", {"b"}}, {"b", {"a"}}};
  network.update(0.0, {{"a", {{"Target1", measurement("Target1", 3.0, 4.0, 0.0)}}},
                       {"b", {}}}, graph);
  network.update(5.0, {{"a", {}}, {"b", {}}}, graph);
  const auto handoffs = network.performHandoffs(5.0, graph);
  ASSERT_FALSE(handoffs.empty());
  EXPECT_EQ(handoffs[0].target_id, "Target1");
  EXPECT_EQ(handoffs[0].receiver_id, "b");
  EXPECT_TRUE(network.module("b").tracks().at("Target1").active());
}

TEST(Network, ConnectedSilentReceiverRetainsBoundedCovariance) {
  ht::TrackConfig config;
  config.window_seconds = 2.0;
  config.max_iterations = 10;
  ht::SynchronousTrackingNetwork network({"Drone1", "Husky1"}, config, 10, 1e-3);
  const ht::Adjacency graph{{"Drone1", {"Husky1"}}, {"Husky1", {"Drone1"}}};
  for (int epoch = 0; epoch < 7; ++epoch) {
    const double timestamp = epoch * 0.5;
    network.update(timestamp,
        {{"Drone1", {{"Target1", measurement("Target1", timestamp, 0.0, timestamp)}}},
         {"Husky1", {}}}, graph);
  }
  const auto estimate = network.predictedEstimates(3.0).at("Husky1").at("Target1");
  EXPECT_TRUE(estimate.active);
  EXPECT_LT(Eigen::SelfAdjointEigenSolver<ht::StateMatrix>(estimate.state_covariance)
                .eigenvalues().maxCoeff(), 10.0);
}

TEST(Track, AsynchronousCaptureUsesMeasurementTimestamp) {
  ht::TargetTracker tracker("a");
  tracker.beginEpoch(10.0, {{"Target1", measurement("Target1", 1.0, 2.0, 9.25)}});
  const auto& times = tracker.tracks().at("Target1").times();
  ASSERT_EQ(times.size(), 1u);
  EXPECT_DOUBLE_EQ(times[0], 9.25);
  tracker.beginEpoch(10.5, {});
  ASSERT_EQ(tracker.tracks().at("Target1").times().size(), 2u);
  EXPECT_DOUBLE_EQ(tracker.tracks().at("Target1").times()[1], 10.5);
}

TEST(Track, NearDuplicateCaptureTimeUsesPythonIsCloseRule) {
  ht::TargetTracker tracker("a");
  tracker.beginEpoch(10.00005,
                     {{"Target1", measurement("Target1", 1.0, 2.0, 10.00005)}});
  tracker.beginEpoch(10.5,
                     {{"Target1", measurement("Target1", 3.0, 4.0, 10.0)}});
  const auto& track = tracker.tracks().at("Target1");
  ASSERT_EQ(track.times().size(), 1u);
  EXPECT_DOUBLE_EQ(track.times()[0], 10.00005);
  ASSERT_TRUE(track.measurements()[0].has_value());
  EXPECT_DOUBLE_EQ(track.measurements()[0]->position.x(), 3.0);
}

TEST(Consensus, DifferentTrajectoryGridsInterpolate) {
  ht::TargetTrack interpolated("Target1");
  ht::TargetTrack directly_aligned("Target1");
  for (double time : {0.0, 1.0, 2.0}) {
    const auto value = time == 0.0 ? std::optional<ht::TargetMeasurement>(measurement("Target1", 0, 0, 0))
                                   : std::nullopt;
    interpolated.begin(time, value);
    directly_aligned.begin(time, value);
  }
  ht::TrackMessage sparse;
  sparse.target_id = "Target1";
  sparse.active = true;
  sparse.times = {0.0, 2.0};
  sparse.trajectory.resize(8);
  sparse.trajectory << 0, 0, 0, 0, 2, 4, 6, 8;
  ht::TrackMessage aligned = sparse;
  aligned.times.clear();
  aligned.trajectory.resize(12);
  aligned.trajectory << 0, 0, 0, 0, 1, 2, 3, 4, 2, 4, 6, 8;
  interpolated.consensusStep({sparse});
  directly_aligned.consensusStep({aligned});
  expectVectorNear(interpolated.trajectory(), directly_aligned.trajectory(), 1e-12);
  expectVectorNear(interpolated.dual(), directly_aligned.dual(), 1e-12);
}

TEST(Consensus, UnequalLengthUsesLatestStateFallback) {
  ht::TargetTrack fallback("Target1");
  ht::TargetTrack explicitly_tiled("Target1");
  for (double time : {0.0, 1.0, 2.0}) {
    const auto value = time == 0.0 ? std::optional<ht::TargetMeasurement>(measurement("Target1", 0, 0, 0))
                                   : std::nullopt;
    fallback.begin(time, value);
    explicitly_tiled.begin(time, value);
  }
  ht::TrackMessage short_message;
  short_message.target_id = "Target1";
  short_message.active = true;
  short_message.trajectory.resize(8);
  short_message.trajectory << -1, -1, -1, -1, 2, 3, 4, 5;
  ht::TrackMessage tiled = short_message;
  tiled.trajectory.resize(12);
  tiled.trajectory << 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5;
  fallback.consensusStep({short_message});
  explicitly_tiled.consensusStep({tiled});
  expectVectorNear(fallback.trajectory(), explicitly_tiled.trajectory(), 1e-12);
}

TEST(Track, StaleTrackExpires) {
  ht::TrackConfig config;
  config.window_seconds = 1.0;
  ht::TargetTrack track("Target1", config);
  ASSERT_TRUE(track.begin(0.0, measurement("Target1", 1, 2, 0)));
  EXPECT_FALSE(track.begin(1.01, std::nullopt));
  EXPECT_FALSE(track.active());
}

TEST(Track, FreshObservationAfterExpiryStartsFreshWindow) {
  ht::TrackConfig config;
  config.window_seconds = 1.0;
  ht::TargetTrack track("Target1", config);
  track.begin(0.0, measurement("Target1", 1, 2, 0));
  track.begin(0.5, std::nullopt);
  EXPECT_FALSE(track.begin(1.1, std::nullopt));
  ASSERT_TRUE(track.begin(1.2, measurement("Target1", 8, 9, 1.2)));
  ASSERT_EQ(track.times().size(), 1u);
  EXPECT_DOUBLE_EQ(track.times()[0], 1.2);
  EXPECT_NEAR(track.trajectory().tail(4)(0), 8.0, 1e-6);
}

TEST(Validation, MalformedOrNonfiniteInputsFail) {
  Eigen::MatrixXd nonsquare = Eigen::MatrixXd::Zero(2, 3);
  EXPECT_THROW(ht::positiveDefinite(nonsquare), std::invalid_argument);
  EXPECT_THROW(ht::solveBlockTridiagonal(Eigen::MatrixXd::Identity(5, 5),
                                         Eigen::VectorXd::Zero(5)), std::invalid_argument);
  ht::PositionMatrix invalid = ht::PositionMatrix::Identity();
  invalid(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(ht::TargetMeasurement("Target1", ht::Position::Zero(), invalid, 0.0),
               std::invalid_argument);
}

TEST(LinearAlgebra, CovariancesRemainSymmetricPositiveDefinite) {
  Eigen::MatrixXd indefinite(2, 2);
  indefinite << 1.0, 3.0, -2.0, -4.0;
  const Eigen::MatrixXd regularized = ht::positiveDefinite(indefinite);
  EXPECT_TRUE(regularized.isApprox(regularized.transpose(), 1e-14));
  EXPECT_GE(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(regularized)
                .eigenvalues().minCoeff(), ht::kEigenvalueFloor - 1e-12);
  const auto m = measurement("Target1", 1, 2, 0, 0.0);
  EXPECT_GE(Eigen::SelfAdjointEigenSolver<ht::PositionMatrix>(m.covariance)
                .eigenvalues().minCoeff(), ht::kEigenvalueFloor - 1e-12);
}

TEST(Handoff, PendingInformationIsConsumedOnce) {
  ht::TargetTrack receiver("Target1");
  ht::HandoffMessage message;
  message.target_id = "Target1";
  message.source_id = "source";
  message.timestamp = 0.0;
  message.information_matrix = ht::StateMatrix::Identity() * 2.0;
  message.information_vector << 2.0, 4.0, 0.0, 0.0;
  ASSERT_TRUE(receiver.acceptHandoff(message));
  ASSERT_TRUE(receiver.hasPendingHandoff());
  ASSERT_TRUE(receiver.begin(0.0, std::nullopt));
  EXPECT_FALSE(receiver.hasPendingHandoff());
  ASSERT_TRUE(receiver.begin(0.5, std::nullopt));
  EXPECT_FALSE(receiver.hasPendingHandoff());
}

TEST(Consensus, SilentSeedPreservesPythonDefaultCovarianceUntilFinalize) {
  ht::TargetTrack receiver("Target1");
  ht::TrackMessage announcement;
  announcement.target_id = "Target1";
  announcement.state << 1.0, 2.0, 3.0, 4.0;
  announcement.active = true;
  receiver.seed(announcement, 0.0);
  EXPECT_TRUE(receiver.message().state_covariance.isApprox(
      ht::StateMatrix::Identity() * 25.0, 1e-12));
}

TEST(Prediction, PreservesPythonPositionOnlyCovarianceRule) {
  ht::TargetTrack track("Target1");
  track.begin(0.0, measurement("Target1", 1, 2, 0));
  const auto initial = track.finalize();
  const auto predicted = track.predicted(2.0);
  const auto expected = initial.covariance +
      ht::constantAccelerationProcessNoise(2.0).topLeftCorner<2, 2>();
  EXPECT_TRUE(predicted.covariance.isApprox(expected, 1e-12));
  EXPECT_TRUE(predicted.state_covariance.isApprox(initial.state_covariance, 1e-12));
}
