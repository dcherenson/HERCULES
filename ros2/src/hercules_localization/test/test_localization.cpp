#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Eigenvalues>

#include "hercules_localization/localization.hpp"
#include "hercules_localization/global_ci.hpp"

namespace hl = hercules_localization;

namespace {

hl::PoseEstimate estimate(double x, double y, double yaw,
                          double variance = 1.0) {
  hl::PoseEstimate result;
  result.mean = hl::Pose2D(x, y, yaw);
  result.covariance = hl::PoseCovariance::Identity() * variance;
  result.valid = true;
  return result;
}

hl::RelativeObservation observation(const hl::PoseEstimate& neighbor,
                                    double range, double bearing,
                                    double variance = 0.01) {
  hl::RangeBearingMeasurement measurement;
  measurement.neighbor_id = neighbor.agent_id;
  measurement.range = range;
  measurement.bearing = bearing;
  measurement.covariance = hl::MeasurementCovariance::Identity() * variance;
  return {measurement, neighbor};
}

hl::GlobalCiBelief globalBelief(const std::string& sender,
                               const std::vector<std::string>& ids,
                               const Eigen::VectorXd& mean,
                               const Eigen::MatrixXd& covariance,
                               double timestamp = 1.0,
                               std::uint64_t sequence = 1) {
  hl::GlobalCiBelief result;
  result.sender_id = sender;
  result.ordered_agent_ids = ids;
  result.mean = mean;
  result.covariance = covariance;
  result.timestamp = timestamp;
  result.sequence = sequence;
  result.valid = true;
  return result;
}

Eigen::MatrixXd globalSpd(int dimension) {
  Eigen::MatrixXd factor = Eigen::MatrixXd::Identity(dimension, dimension);
  for (int row = 0; row < dimension; ++row) {
    for (int col = 0; col < row; ++col) {
      factor(row, col) = 0.05 * static_cast<double>(row + col + 1);
    }
  }
  return factor * factor.transpose() + Eigen::MatrixXd::Identity(dimension, dimension);
}

}  // namespace

TEST(Math, AnglesAndRelativePoseUseWrappedYaw) {
  constexpr double pi = 3.14159265358979323846;
  EXPECT_NEAR(hl::wrapAngle(3.0 * pi), pi, 1e-12);
  EXPECT_NEAR(hl::angleDifference(-pi + 0.1, pi - 0.1), 0.2, 1e-12);

  const hl::Pose2D reference(1.0, 2.0, pi / 2.0);
  const hl::Pose2D target(1.0, 4.0, -pi + 0.2);
  const hl::Pose2D relative = hl::relativePose(reference, target);
  EXPECT_NEAR(relative.position.x(), 2.0, 1e-12);
  EXPECT_NEAR(relative.position.y(), 0.0, 1e-12);
  EXPECT_NEAR(relative.yaw, hl::wrapAngle(-pi + 0.2 - pi / 2.0), 1e-12);
  EXPECT_TRUE(hl::compose(reference, relative).position.isApprox(target.position, 1e-12));
}

TEST(Math, RangeBearingJacobianMatchesFiniteDifference) {
  const hl::Pose2D observer(1.2, -0.8, 0.4);
  const hl::Pose2D target(4.0, 1.7, -0.2);
  const auto analytic = hl::rangeBearingJacobianObserver(observer, target);
  const double eps = 1e-7;
  for (int column = 0; column < 3; ++column) {
    hl::Pose2D plus = observer;
    hl::Pose2D minus = observer;
    if (column < 2) {
      plus.position[column] += eps;
      minus.position[column] -= eps;
    } else {
      plus.yaw += eps;
      minus.yaw -= eps;
    }
    const auto z_plus = hl::predictRangeBearing(plus, target);
    const auto z_minus = hl::predictRangeBearing(minus, target);
    EXPECT_NEAR(analytic(0, column), (z_plus.range - z_minus.range) / (2.0 * eps),
                1e-6);
    EXPECT_NEAR(analytic(1, column),
                hl::angleDifference(z_plus.bearing, z_minus.bearing) / (2.0 * eps),
                1e-6);
  }
}

TEST(Covariance, IndefiniteAndNonfiniteInputsBecomeFinitePositive) {
  Eigen::Matrix3d input;
  input << 1.0, 4.0, std::numeric_limits<double>::quiet_NaN(),
      -2.0, -0.5, 0.0,
      0.0, 0.0, -3.0;
  const Eigen::Matrix3d fixed = hl::regularizeCovariance(input, 1e-6);
  EXPECT_TRUE(fixed.allFinite());
  EXPECT_TRUE(fixed.isApprox(fixed.transpose(), 1e-14));
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(fixed);
  ASSERT_EQ(solver.info(), Eigen::Success);
  EXPECT_GE(solver.eigenvalues().minCoeff(), 1e-6 - 1e-12);

  const Eigen::Matrix3d inverse = hl::safeInverse(input, 1e-6);
  EXPECT_TRUE(inverse.allFinite());
  EXPECT_TRUE(hl::mahalanobisSquared(Eigen::Vector3d(1.0, 0.0, 0.0), input, 1e-6) >=
              0.0);
}

TEST(Recursive, SequentialRangeBearingUpdateUsesNeighborCovariance) {
  const hl::PoseEstimate prior = estimate(0.0, 0.0, 0.0, 1.0);
  const hl::PoseEstimate neighbor = estimate(5.0, 0.0, 0.0, 0.25);
  const auto result = hl::recursiveDecentralizedUpdate(
      prior, {observation(neighbor, 4.8, 0.0)}, hl::LocalizationConfig{});
  ASSERT_TRUE(result.success);
  ASSERT_EQ(result.used_measurements, 1);
  EXPECT_GT(result.estimate.mean.position.x(), 0.0);
  EXPECT_LT(result.estimate.mean.position.x(), 0.3);
  EXPECT_LT(result.estimate.covariance(0, 0), prior.covariance(0, 0));
  EXPECT_TRUE(result.estimate.covariance.allFinite());
}

TEST(Recursive, PaperMarginUsesLinearMeasurementOnlyInflation) {
  hl::LocalizationConfig config;
  config.maicp_enabled = true;
  config.maicp_margin = 2.0;
  config.maicp_class = hl::MaicpClass::kUgv;
  config.maicp_covariance_gain = 0.0;
  const Eigen::Matrix2d nominal = Eigen::Matrix2d::Identity() * 0.4;
  const Eigen::Matrix2d effective = hl::measurementCovarianceForConfig(
      nominal, config, 1e-12);
  EXPECT_TRUE(effective.isApprox(Eigen::Matrix2d::Identity(), 1e-12));

  const hl::PoseEstimate prior = estimate(0.0, 0.0, 0.0, 1.0);
  const hl::PoseEstimate neighbor = estimate(5.0, 0.0, 0.0, 0.25);
  const auto relative = observation(neighbor, 4.8, 0.0, 0.4);
  const auto paper = hl::updateFromRangeBearing(prior, relative, config);
  const auto nominal_result = hl::updateFromRangeBearing(
      prior, relative, hl::LocalizationConfig{});
  ASSERT_TRUE(paper.accepted);
  ASSERT_TRUE(nominal_result.accepted);
  // With the same prior and neighbor, increasing R must reduce the update
  // magnitude and leave the prior/process model unchanged.
  EXPECT_LT(paper.estimate.mean.position.x(), nominal_result.estimate.mean.position.x());
  EXPECT_GT(paper.estimate.covariance(0, 0), nominal_result.estimate.covariance(0, 0));

  hl::RecursivePairTransaction transaction;
  transaction.event_id = "paper-pair";
  transaction.observer_id = "A";
  transaction.observed_id = "B";
  transaction.observed = neighbor;
  transaction.reciprocal_factor = Eigen::Matrix3d::Identity();
  transaction.measurement = relative.measurement;
  transaction.measurement.neighbor_id = "B";
  hl::RecursiveDecentralizedState transactional_state("A", prior, config);
  transactional_state.initializePeer("B");
  const auto transaction_result =
      transactional_state.applyPairTransaction(transaction);
  ASSERT_TRUE(transaction_result.accepted);
  EXPECT_GT(transaction_result.result.estimate.covariance(0, 0),
            nominal_result.estimate.covariance(0, 0));
}

TEST(Recursive, InvalidMeasurementIsDiagnosedWithoutCorruptingPrior) {
  const hl::PoseEstimate prior = estimate(1.0, 2.0, 0.2);
  auto invalid = observation(estimate(5.0, 1.0, 0.0), 2.0, 0.0);
  invalid.measurement.range = std::numeric_limits<double>::quiet_NaN();
  const auto result = hl::recursiveDecentralizedUpdate(prior, {invalid});
  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.used_measurements, 0);
  EXPECT_EQ(result.rejected_measurements, 1);
  EXPECT_EQ(result.status, "no_valid_measurements");
  EXPECT_TRUE(result.estimate.mean.vector().isApprox(prior.mean.vector(), 1e-12));
}

TEST(Recursive, RejectsNonSpdObservationCovariance) {
  const hl::PoseEstimate prior = estimate(0.0, 0.0, 0.0);
  const hl::PoseEstimate neighbor = estimate(5.0, 0.0, 0.0);
  auto invalid = observation(neighbor, 5.0, 0.0);
  invalid.measurement.covariance << 1.0, 2.0, 2.0, 1.0;
  const auto result = hl::updateFromRangeBearing(prior, invalid);
  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.status, "invalid_measurement");
  EXPECT_TRUE(result.estimate.covariance.allFinite());
}

TEST(CovarianceIntersection, EqualCovariancesUseSymmetricHalfWeight) {
  const hl::PoseEstimate first = estimate(0.0, 0.0, 3.1, 2.0);
  const hl::PoseEstimate second = estimate(2.0, 0.0, -3.1, 2.0);
  hl::LocalizationConfig config;
  config.ci_grid_points = 51;
  const auto result = hl::covarianceIntersection(first, second, config);
  ASSERT_TRUE(result.success);
  EXPECT_NEAR(result.weight, 0.5, 1e-12);
  EXPECT_NEAR(result.estimate.mean.position.x(), 1.0, 1e-12);
  EXPECT_NEAR(std::abs(result.estimate.mean.yaw), 3.14159265358979323846, 0.02);
  EXPECT_TRUE(result.estimate.covariance.isApprox(first.covariance, 1e-9));
}

TEST(GsCi, UpdatesInPlaceAndReportsWeights) {
  const hl::PoseEstimate prior = estimate(0.0, 0.0, 0.0, 1.0);
  const hl::PoseEstimate neighbor = estimate(5.0, 0.0, 0.0, 0.1);
  hl::LocalizationConfig config;
  config.max_iterations = 5;
  config.convergence_tolerance = 1e-8;
  const auto result = hl::gsCiUpdate(prior, {observation(neighbor, 4.5, 0.0)}, config);
  ASSERT_TRUE(result.success);
  EXPECT_GT(result.used_measurements, 0);
  EXPECT_FALSE(result.ci_weights.empty());
  for (const double weight : result.ci_weights) {
    EXPECT_GE(weight, 0.0);
    EXPECT_LE(weight, 1.0);
  }
  EXPECT_GT(result.estimate.mean.position.x(), -1e-9);
  EXPECT_TRUE(result.estimate.covariance.allFinite());
}

TEST(Dispatch, SelectsRequestedAlgorithm) {
  const hl::PoseEstimate prior = estimate(0.0, 0.0, 0.0);
  const hl::PoseEstimate neighbor = estimate(4.0, 0.0, 0.0);
  const auto recursive = hl::update(
      prior, {observation(neighbor, 3.8, 0.0)},
      hl::LocalizationAlgorithm::kRecursiveDecentralized);
  const auto gs_ci = hl::update(prior, {observation(neighbor, 3.8, 0.0)},
                                hl::LocalizationAlgorithm::kGsCi);
  EXPECT_TRUE(recursive.success);
  EXPECT_TRUE(gs_ci.success);
  EXPECT_TRUE(recursive.estimate.covariance.allFinite());
  EXPECT_TRUE(gs_ci.estimate.covariance.allFinite());
}

TEST(RecursiveState, InitializesFactorsAndMakesPairTransactionsIdempotent) {
  hl::LocalizationConfig config;
  config.lambda = 0.5;
  hl::RecursiveDecentralizedState state("A", estimate(0.0, 0.0, 0.0), config);
  state.initializePeer("B");
  ASSERT_EQ(state.correlationFactors().count("B"), 1U);
  EXPECT_TRUE(state.correlationFactors().at("B").isZero(1e-12));

  hl::RecursivePairTransaction transaction;
  transaction.event_id = "event-1";
  transaction.observer_id = "A";
  transaction.observed_id = "B";
  transaction.sequence = 7;
  transaction.measurement = observation(estimate(5.0, 0.0, 0.0), 4.8, 0.0).measurement;
  transaction.observed = estimate(5.0, 0.0, 0.0, 0.25);
  const auto first = state.applyPairTransaction(transaction);
  ASSERT_TRUE(first.accepted);
  EXPECT_FALSE(first.duplicate);
  const auto duplicate = state.applyPairTransaction(transaction);
  EXPECT_TRUE(duplicate.accepted);
  EXPECT_TRUE(duplicate.duplicate);
  EXPECT_TRUE(state.estimate().covariance.allFinite());
}

TEST(RecursiveState, RejectsOutOfRangeLambda) {
  hl::LocalizationConfig config;
  config.lambda = 1.1;
  hl::RecursiveDecentralizedState state("A", estimate(0.0, 0.0, 0.0), config);
  hl::RecursivePairTransaction transaction;
  transaction.observer_id = "A";
  transaction.observed_id = "B";
  const auto result = state.applyPairTransaction(transaction);
  EXPECT_TRUE(result.rejected);
  EXPECT_EQ(result.status, "invalid_lambda");
}

TEST(RecursiveState, MotionPropagationDoesNotApplyEquation26Lambda) {
  hl::LocalizationConfig config;
  config.lambda = 0.2;
  const hl::PoseEstimate initial = estimate(1.0, -2.0, 0.3);
  hl::RecursiveDecentralizedState state("A", initial, config);
  const Eigen::Matrix3d factor =
      (Eigen::Vector3d(0.4, 0.3, 0.2)).asDiagonal();
  state.initializePeer("B", factor);
  const hl::Twist2D twist(1.0, -0.25, 0.1);
  const double dt = 0.5;
  const Eigen::Matrix3d expected =
      hl::motionJacobian(initial.mean, twist, dt) * factor;

  state.propagate(twist, dt, Eigen::Matrix3d::Identity() * 0.01);

  EXPECT_TRUE(state.correlationFactors().at("B").isApprox(expected, 1e-12));
}

TEST(RecursiveState, PairUpdateMatchesCentralizedEkfAndEquation26) {
  hl::LocalizationConfig config;
  config.lambda = 0.7;
  config.covariance_floor = 1e-12;
  config.measurement_covariance_floor = 1e-12;
  hl::PoseEstimate observer = estimate(0.0, 0.0, 0.2);
  observer.covariance = (Eigen::Vector3d(1.0, 0.8, 0.4)).asDiagonal();
  hl::PoseEstimate observed = estimate(4.0, 1.0, -0.1);
  observed.covariance = (Eigen::Vector3d(0.7, 0.9, 0.3)).asDiagonal();
  const Eigen::Matrix3d sigma_ab = Eigen::Matrix3d::Identity() * 0.1;
  const Eigen::Matrix3d sigma_ac = Eigen::Matrix3d::Identity() * 0.05;
  hl::RecursiveDecentralizedState state("A", observer, config);
  state.initializePeer("B", sigma_ab);
  state.initializePeer("C", sigma_ac);

  hl::RecursivePairTransaction transaction;
  transaction.event_id = "centralized-oracle";
  transaction.observer_id = "A";
  transaction.observed_id = "B";
  transaction.sequence = 1;
  transaction.observed = observed;
  transaction.reciprocal_factor = Eigen::Matrix3d::Identity();
  transaction.measurement = observation(observed, 4.0, 0.05, 0.02).measurement;

  Eigen::Matrix<double, 6, 6> prior;
  prior << observer.covariance, sigma_ab,
      sigma_ab.transpose(), observed.covariance;
  Eigen::Matrix<double, 2, 6> h;
  h << hl::rangeBearingJacobianObserver(observer.mean, observed.mean),
      hl::rangeBearingJacobianTarget(observer.mean, observed.mean);
  const auto predicted = hl::predictRangeBearing(observer.mean, observed.mean);
  Eigen::Vector2d innovation;
  innovation << transaction.measurement.range - predicted.range,
      hl::angleDifference(transaction.measurement.bearing, predicted.bearing);
  const Eigen::Matrix2d noise = transaction.measurement.covariance;
  const Eigen::Matrix2d innovation_covariance = h * prior * h.transpose() + noise;
  const Eigen::Matrix<double, 6, 2> gain =
      prior * h.transpose() * innovation_covariance.inverse();
  Eigen::Matrix<double, 6, 1> prior_mean;
  prior_mean << observer.vector(), observed.vector();
  const Eigen::Matrix<double, 6, 1> posterior_mean = prior_mean + gain * innovation;
  const Eigen::Matrix<double, 6, 6> residual =
      Eigen::Matrix<double, 6, 6>::Identity() - gain * h;
  const Eigen::Matrix<double, 6, 6> posterior_covariance =
      residual * prior * residual.transpose() + gain * noise * gain.transpose();

  const auto result = state.applyPairTransaction(transaction);
  ASSERT_TRUE(result.accepted);
  EXPECT_TRUE(result.result.estimate.vector().isApprox(posterior_mean.head<3>(), 1e-9));
  EXPECT_TRUE(result.observed_estimate.vector().isApprox(posterior_mean.tail<3>(), 1e-9));
  EXPECT_TRUE(result.result.estimate.covariance.isApprox(
      posterior_covariance.topLeftCorner<3, 3>(), 1e-9));
  EXPECT_TRUE(result.observed_estimate.covariance.isApprox(
      posterior_covariance.bottomRightCorner<3, 3>(), 1e-9));
  EXPECT_TRUE(result.correlation_factor.isApprox(
      posterior_covariance.topRightCorner<3, 3>(), 1e-9));
  EXPECT_TRUE(result.reciprocal_factor.isApprox(Eigen::Matrix3d::Identity(), 1e-12));

  const Eigen::Matrix3d equation_26 = config.lambda *
      posterior_covariance.topLeftCorner<3, 3>() *
      observer.covariance.inverse() * sigma_ac;
  EXPECT_TRUE(state.correlationFactors().at("C").isApprox(equation_26, 1e-9));
}

TEST(GlobalCi, PropagatesOrderingAndInflatesRemoteBlocks) {
  hl::GlobalCiBelief belief;
  belief.sender_id = "A";
  belief.ordered_agent_ids = {"A", "B", "C"};
  belief.mean = Eigen::VectorXd::Zero(7);
  belief.covariance = Eigen::MatrixXd::Identity(7, 7);
  const auto propagated = hl::propagateGlobalCiBelief(
      belief, "A", Eigen::Vector3d(1.0, -2.0, 0.2),
      Eigen::Matrix3d::Identity() * 0.1, 2.0, 0.5);
  ASSERT_TRUE(propagated.valid);
  EXPECT_DOUBLE_EQ(propagated.mean[0], 1.0);
  EXPECT_DOUBLE_EQ(propagated.mean[1], -2.0);
  EXPECT_NEAR(propagated.mean[6], 0.2, 1e-12);
  EXPECT_GT(propagated.covariance(2, 2), belief.covariance(2, 2));
  EXPECT_GT(propagated.covariance(4, 4), belief.covariance(4, 4));
}

TEST(GlobalCi, RemovesSenderYawAndInsertsReceiverYawDuringBatchCi) {
  hl::GlobalCiBelief self;
  self.sender_id = "A";
  self.ordered_agent_ids = {"A", "B"};
  self.mean = Eigen::VectorXd::Zero(5);
  self.mean[4] = 3.13;
  self.covariance = Eigen::MatrixXd::Identity(5, 5);
  hl::GlobalCiBelief peer = self;
  peer.sender_id = "B";
  peer.mean[0] = 4.0;
  peer.mean[4] = -3.13;
  hl::LocalizationConfig config;
  config.ci_self_weight = 0.8;
  const auto result = hl::globalCiUpdate(self, {peer}, config);
  ASSERT_TRUE(result.success);
  ASSERT_EQ(result.weights.size(), 2U);
  EXPECT_NEAR(result.weights[0], 0.8, 1e-12);
  EXPECT_NEAR(result.weights[1], 0.2, 1e-12);
  EXPECT_TRUE(result.belief.covariance.allFinite());
  EXPECT_NEAR(hl::angleDifference(result.belief.mean[4], self.mean[4]), 0.0, 1e-12);
}

TEST(GlobalCi, RejectsNonSpdBeliefPackets) {
  hl::GlobalCiBelief belief;
  belief.sender_id = "A";
  belief.ordered_agent_ids = {"A", "B"};
  belief.mean = Eigen::VectorXd::Zero(5);
  belief.covariance = Eigen::MatrixXd::Identity(5, 5);
  belief.covariance(0, 0) = 0.0;
  belief.covariance(1, 1) = -1.0;
  EXPECT_FALSE(hl::isValidGlobalCiBelief(belief));
}

TEST(GlobalCi, InitializesOrderedStateWithLargeRemoteCovariance) {
  hl::PoseEstimate local = estimate(3.0, -2.0, 3.4, 0.5);
  local.agent_id = "A";
  local.timestamp = 7.0;
  local.covariance << 0.4, 0.02, 0.05,
      0.02, 0.6, -0.03,
      0.05, -0.03, 0.3;

  const auto belief = hl::initializeGlobalCiBelief(
      "A", {"A", "B", "C"}, local, 1.0e5);
  ASSERT_TRUE(belief.valid);
  ASSERT_TRUE(hl::isValidGlobalCiBelief(belief));
  ASSERT_EQ(belief.mean.size(), 7);
  EXPECT_DOUBLE_EQ(belief.mean[0], 3.0);
  EXPECT_DOUBLE_EQ(belief.mean[1], -2.0);
  EXPECT_NEAR(belief.mean[6], hl::wrapAngle(3.4), 1e-12);
  EXPECT_DOUBLE_EQ(belief.covariance(2, 2), 1.0e5);
  EXPECT_DOUBLE_EQ(belief.covariance(4, 4), 1.0e5);
  EXPECT_NEAR(belief.covariance(0, 6), local.covariance(0, 2), 1e-12);
  EXPECT_NEAR(belief.covariance(6, 1), local.covariance(2, 1), 1e-12);
  EXPECT_DOUBLE_EQ(belief.timestamp, 7.0);
}

TEST(GlobalCi, PrivatePositionUpdateTransformsCompleteCovarianceWithJosephForm) {
  constexpr int dimension = 7;
  const Eigen::VectorXd prior_mean =
      (Eigen::VectorXd(dimension) << 1.0, 2.0, 8.0, -1.0, -3.0, 4.0, 3.2).finished();
  const Eigen::MatrixXd prior_covariance = globalSpd(dimension);
  const auto prior = globalBelief("A", {"A", "B", "C"}, prior_mean,
                                 prior_covariance, 1.0);
  const Eigen::Vector2d measurement(1.6, 1.5);
  const Eigen::Matrix2d noise = (Eigen::Vector2d(0.2, 0.3)).asDiagonal();

  const auto result = hl::globalCiPrivatePositionUpdate(
      prior, "A", measurement, noise, 2.0);
  ASSERT_TRUE(result.success);
  ASSERT_TRUE(result.belief.valid);

  Eigen::MatrixXd selector = Eigen::MatrixXd::Zero(2, dimension);
  selector(0, 0) = 1.0;
  selector(1, 1) = 1.0;
  const Eigen::Matrix2d innovation_covariance =
      selector * prior_covariance * selector.transpose() + noise;
  const Eigen::MatrixXd gain = prior_covariance * selector.transpose() *
      innovation_covariance.inverse();
  const Eigen::VectorXd expected_mean = prior_mean + gain * (measurement - prior_mean.head<2>());
  const Eigen::MatrixXd residual = Eigen::MatrixXd::Identity(dimension, dimension) -
      gain * selector;
  const Eigen::MatrixXd expected_covariance = residual * prior_covariance *
      residual.transpose() + gain * noise * gain.transpose();

  EXPECT_TRUE(result.belief.mean.head<6>().isApprox(expected_mean.head<6>(), 1e-9));
  EXPECT_NEAR(result.belief.mean[6], hl::wrapAngle(expected_mean[6]), 1e-12);
  EXPECT_TRUE(result.belief.covariance.isApprox(expected_covariance, 1e-9));
  EXPECT_LT(result.belief.covariance(2, 2), prior_covariance(2, 2));
  EXPECT_NE(result.belief.covariance(2, 6), prior_covariance(2, 6));
  EXPECT_DOUBLE_EQ(result.belief.timestamp, 2.0);
}

TEST(GlobalCi, RangeBearingUpdateUsesFullOrderedStateAndObserverYaw) {
  constexpr int dimension = 7;
  const Eigen::VectorXd prior_mean =
      (Eigen::VectorXd(dimension) << 0.0, 0.0, 5.0, 1.0, -3.0, 2.0, 0.35).finished();
  const Eigen::MatrixXd prior_covariance = globalSpd(dimension);
  const auto prior = globalBelief("A", {"A", "B", "C"}, prior_mean,
                                 prior_covariance, 1.0);
  hl::RangeBearingMeasurement measurement;
  measurement.neighbor_id = "B";
  measurement.range = 4.7;
  measurement.bearing = 0.12;
  measurement.covariance << 0.15, 0.01, 0.01, 0.02;
  measurement.timestamp = 3.0;

  hl::LocalizationConfig config;
  config.covariance_floor = 1e-12;
  config.measurement_covariance_floor = 1e-12;
  const auto result = hl::globalCiRangeBearingUpdate(
      prior, "A", "B", measurement, config);
  ASSERT_TRUE(result.success);
  ASSERT_TRUE(result.belief.valid);

  const hl::Pose2D observer(prior_mean.segment<2>(0), prior_mean[6]);
  const hl::Pose2D observed(prior_mean.segment<2>(2), 0.0);
  const auto observer_jacobian =
      hl::rangeBearingJacobianObserver(observer, observed, config.range_epsilon);
  const auto observed_jacobian =
      hl::rangeBearingJacobianTarget(observer, observed, config.range_epsilon);
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(2, dimension);
  jacobian.block<2, 2>(0, 0) = observer_jacobian.leftCols<2>();
  jacobian.block<2, 2>(0, 2) = observed_jacobian.leftCols<2>();
  jacobian.col(6) = observer_jacobian.col(2);
  const auto predicted = hl::predictRangeBearing(observer, observed);
  Eigen::Vector2d innovation;
  innovation << measurement.range - predicted.range,
      hl::angleDifference(measurement.bearing, predicted.bearing);
  const Eigen::Matrix2d innovation_covariance =
      jacobian * prior_covariance * jacobian.transpose() + measurement.covariance;
  const Eigen::MatrixXd gain = prior_covariance * jacobian.transpose() *
      innovation_covariance.inverse();
  const Eigen::VectorXd expected_mean = prior_mean + gain * innovation;
  const Eigen::MatrixXd residual = Eigen::MatrixXd::Identity(dimension, dimension) -
      gain * jacobian;
  const Eigen::MatrixXd expected_covariance = residual * prior_covariance *
      residual.transpose() + gain * measurement.covariance * gain.transpose();

  EXPECT_TRUE(result.belief.mean.head<6>().isApprox(expected_mean.head<6>(), 1e-8));
  EXPECT_NEAR(result.belief.mean[6], hl::wrapAngle(expected_mean[6]), 1e-10);
  EXPECT_TRUE(result.belief.covariance.isApprox(expected_covariance, 1e-8));
  EXPECT_NE(result.belief.covariance(4, 4), prior_covariance(4, 4));
  EXPECT_DOUBLE_EQ(result.belief.timestamp, 3.0);
}

TEST(GlobalCi, PeerBatchIsPermutationInvariantAndRejectsDuplicateSender) {
  constexpr int dimension = 5;
  const auto self = globalBelief(
      "A", {"A", "B"}, Eigen::VectorXd::Zero(dimension),
      Eigen::MatrixXd::Identity(dimension, dimension), 5.0);
  auto peer_b = self;
  peer_b.sender_id = "B";
  peer_b.timestamp = 5.0;
  peer_b.sequence = 2;
  peer_b.mean[0] = 4.0;
  auto peer_b_duplicate = peer_b;
  peer_b_duplicate.sequence = 1;
  peer_b_duplicate.mean[0] = 9.0;
  auto peer_c = self;
  peer_c.sender_id = "C";
  // C is intentionally not in the configured ordered list and must be
  // rejected before it can influence the result.
  peer_c.ordered_agent_ids = {"A", "B", "C"};

  const auto first = hl::globalCiUpdate(self, {peer_c, peer_b_duplicate, peer_b});
  const auto second = hl::globalCiUpdate(self, {peer_b, peer_b_duplicate, peer_c});
  ASSERT_TRUE(first.success);
  ASSERT_TRUE(second.success);
  EXPECT_EQ(first.accepted_peers, 1U);
  EXPECT_EQ(first.rejected_peers, 2U);
  EXPECT_EQ(second.accepted_peers, first.accepted_peers);
  EXPECT_EQ(second.rejected_peers, first.rejected_peers);
  EXPECT_TRUE(first.belief.mean.isApprox(second.belief.mean, 1e-12));
  EXPECT_TRUE(first.belief.covariance.isApprox(second.belief.covariance, 1e-12));
  EXPECT_TRUE(first.weights == second.weights);
}

TEST(GlobalCi, MissingCommunicationLeavesSelfBeliefUnchangedAndNormalized) {
  const auto self = globalBelief(
      "A", {"A", "B"}, Eigen::VectorXd::LinSpaced(5, -1.0, 1.0),
      globalSpd(5), 5.0);
  hl::LocalizationConfig config;
  config.ci_self_weight = 0.8;
  const auto result = hl::globalCiUpdate(self, {}, config);
  ASSERT_TRUE(result.success);
  ASSERT_EQ(result.weights.size(), 1U);
  EXPECT_DOUBLE_EQ(result.weights[0], 1.0);
  EXPECT_TRUE(result.belief.mean.isApprox(self.mean, 1e-12));
  EXPECT_TRUE(result.belief.covariance.isApprox(self.covariance, 1e-10));
}

TEST(GlobalCi, RejectsIndefiniteLocalProcessCovariance) {
  hl::GlobalCiBelief prior;
  prior.sender_id = "A";
  prior.ordered_agent_ids = {"A", "B"};
  prior.mean = Eigen::VectorXd::Zero(5);
  prior.covariance = Eigen::MatrixXd::Identity(5, 5);
  Eigen::Matrix3d process = Eigen::Matrix3d::Identity();
  process(1, 1) = -0.1;
  const auto result = hl::propagateGlobalCiBelief(
      prior, "A", Eigen::Vector3d::Zero(), process, 1.0, 0.2);
  EXPECT_FALSE(result.valid);
}

TEST(GlobalCi, YawMappingPreservesReceiverCrossCovariance) {
  hl::GlobalCiBelief self;
  self.sender_id = "A";
  self.ordered_agent_ids = {"A", "B"};
  self.mean = Eigen::VectorXd::Zero(5);
  self.mean[4] = 3.12;
  self.covariance = Eigen::MatrixXd::Identity(5, 5);
  self.covariance(0, 4) = self.covariance(4, 0) = 0.25;
  self.covariance(1, 4) = self.covariance(4, 1) = -0.1;
  self.covariance(4, 4) = 1.4;
  self.timestamp = 2.0;
  auto peer = self;
  peer.sender_id = "B";
  peer.mean[0] = 5.0;
  peer.mean[4] = -3.12;
  peer.covariance(4, 4) = 30.0;
  peer.covariance(0, 4) = peer.covariance(4, 0) = 0.8;
  const auto result = hl::globalCiUpdate(self, {peer});
  ASSERT_TRUE(result.success);
  EXPECT_NEAR(hl::angleDifference(result.belief.mean[4], self.mean[4]), 0.0, 0.05);
  EXPECT_TRUE(result.belief.covariance.allFinite());
  EXPECT_NE(result.belief.covariance(0, 4), 0.0);
  EXPECT_LT(result.belief.covariance(4, 4), peer.covariance(4, 4));
}
