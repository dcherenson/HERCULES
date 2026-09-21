#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <map>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace hercules_localization {

inline constexpr int kPoseDimension = 3;
inline constexpr int kPositionDimension = 2;
inline constexpr int kMeasurementDimension = 2;
inline constexpr double kDefaultCovarianceFloor = 1e-9;
inline constexpr double kDefaultRangeEpsilon = 1e-9;

using Position2D = Eigen::Vector2d;
using PoseVector = Eigen::Vector3d;
using PoseCovariance = Eigen::Matrix3d;
using MeasurementVector = Eigen::Vector2d;
using MeasurementCovariance = Eigen::Matrix2d;

/** A planar pose in a right-handed x/y frame with yaw in radians. */
struct Pose2D {
  Position2D position{Position2D::Zero()};
  double yaw{0.0};

  Pose2D() = default;
  Pose2D(double x, double y, double yaw_in) : position(x, y), yaw(yaw_in) {}
  Pose2D(const Position2D& position_in, double yaw_in)
      : position(position_in), yaw(yaw_in) {}

  PoseVector vector() const {
    PoseVector value;
    value << position.x(), position.y(), yaw;
    return value;
  }

  static Pose2D fromVector(const PoseVector& value) {
    return Pose2D(value.x(), value.y(), value.z());
  }
};

using Pose2d = Pose2D;
using PlanarPose = Pose2D;

/** Body-frame planar twist used by the common motion helpers. */
struct Twist2D {
  double linear_x{0.0};
  double linear_y{0.0};
  double angular_z{0.0};

  Twist2D() = default;
  Twist2D(double x, double y, double yaw_rate)
      : linear_x(x), linear_y(y), angular_z(yaw_rate) {}
  Eigen::Vector3d vector() const {
    return Eigen::Vector3d(linear_x, linear_y, angular_z);
  }
};

using Twist2d = Twist2D;

/** Polar observation of a neighbor, expressed in the observing robot frame. */
struct RangeBearing {
  double range{0.0};
  double bearing{0.0};

  RangeBearing() = default;
  RangeBearing(double range_in, double bearing_in)
      : range(range_in), bearing(bearing_in) {}
  MeasurementVector vector() const {
    return MeasurementVector(range, bearing);
  }
};

using RangeBearingMeasurementValue = RangeBearing;

/**
 * A range/bearing measurement associated with a neighbor id.
 *
 * The covariance is in [range, bearing] order.  Bearing is always interpreted
 * in radians.  `valid` is a transport/data-quality flag; numerical routines
 * still validate all values before accepting a measurement.
 */
struct RangeBearingMeasurement {
  std::string neighbor_id;
  double range{0.0};
  double bearing{0.0};
  MeasurementCovariance covariance{MeasurementCovariance::Identity()};
  double timestamp{0.0};
  bool valid{true};
  std::string source_id;

  RangeBearingMeasurement() = default;
  RangeBearingMeasurement(std::string id, double range_in, double bearing_in,
                          const MeasurementCovariance& covariance_in =
                              MeasurementCovariance::Identity(),
                          double timestamp_in = 0.0)
      : neighbor_id(std::move(id)), range(range_in), bearing(bearing_in),
        covariance(covariance_in), timestamp(timestamp_in) {}

  RangeBearing value() const { return RangeBearing(range, bearing); }
};

using RangeBearingObservation = RangeBearingMeasurement;

/** A pose estimate with covariance and optional transport metadata. */
struct PoseEstimate {
  Pose2D mean{};
  PoseCovariance covariance{PoseCovariance::Identity()};
  double timestamp{0.0};
  bool valid{true};
  std::string agent_id;

  PoseEstimate() = default;
  PoseEstimate(const Pose2D& mean_in, const PoseCovariance& covariance_in,
               double timestamp_in = 0.0, bool valid_in = true)
      : mean(mean_in), covariance(covariance_in), timestamp(timestamp_in),
        valid(valid_in) {}

  PoseVector vector() const { return mean.vector(); }
};

using LocalizationEstimate = PoseEstimate;
using StateEstimate = PoseEstimate;

/** A measurement and the neighbor estimate used to interpret it. */
struct RelativeObservation {
  RangeBearingMeasurement measurement;
  PoseEstimate neighbor;

  RelativeObservation() = default;
  RelativeObservation(const RangeBearingMeasurement& measurement_in,
                      const PoseEstimate& neighbor_in)
      : measurement(measurement_in), neighbor(neighbor_in) {}
};

using LocalizationObservation = RelativeObservation;

enum class LocalizationAlgorithm {
  kRecursiveDecentralized,
  kGsCi,
};

using Algorithm = LocalizationAlgorithm;

enum class CovarianceIntersectionCost {
  kLogDet,
  kTrace,
};

/** Numerical and robustification settings shared by both algorithms. */
struct LocalizationConfig {
  double covariance_floor{kDefaultCovarianceFloor};
  double process_covariance_floor{kDefaultCovarianceFloor};
  double measurement_covariance_floor{kDefaultCovarianceFloor};
  double range_epsilon{kDefaultRangeEpsilon};
  double robustness_margin{0.0};
  double covariance_inflation{0.0};
  double innovation_gate{std::numeric_limits<double>::infinity()};
  int max_iterations{10};
  double convergence_tolerance{1e-6};
  CovarianceIntersectionCost ci_cost{CovarianceIntersectionCost::kLogDet};
  int ci_grid_points{101};
  bool joseph_form{true};
  // Luft et al. equation-26 nonparticipant correlation damping.  A value of
  // one preserves the propagated factor; zero deliberately drops it.
  double lambda{1.0};
  double ci_self_weight{0.8};
  double unknown_motion_variance{0.25};
  double communication_timeout_sec{0.5};
};

using CorrelationFactors = std::map<std::string, Eigen::Matrix3d>;

struct RecursivePairTransaction {
  std::string event_id;
  std::string observer_id;
  std::string observed_id;
  std::uint64_t sequence{0};
  RangeBearingMeasurement measurement;
  PoseEstimate observer;
  PoseEstimate observed;
  Eigen::Matrix3d correlation_factor{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d reciprocal_factor{Eigen::Matrix3d::Zero()};
};

struct ObservationUpdate {
  PoseEstimate estimate{};
  MeasurementVector innovation{MeasurementVector::Zero()};
  MeasurementCovariance innovation_covariance{
      MeasurementCovariance::Identity()};
  double mahalanobis_squared{std::numeric_limits<double>::infinity()};
  bool accepted{false};
  bool valid{false};
  std::string status;
};

struct CovarianceIntersectionResult {
  PoseEstimate estimate{};
  double weight{0.5};
  double objective{std::numeric_limits<double>::infinity()};
  bool success{false};
  std::string status;
};

struct LocalizationResult {
  PoseEstimate estimate{};
  bool success{false};
  std::string status;
  std::string message;
  int used_measurements{0};
  int rejected_measurements{0};
  int iterations{0};
  double maximum_innovation{0.0};
  double maximum_mahalanobis_squared{0.0};
  std::vector<double> ci_weights;
};

/** Result metadata for an idempotent recursive pair transaction. */
struct RecursiveTransactionResult {
  LocalizationResult result{};
  PoseEstimate observed_estimate{};
  Eigen::Matrix3d correlation_factor{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d reciprocal_factor{Eigen::Matrix3d::Zero()};
  bool accepted{false};
  bool duplicate{false};
  bool rejected{false};
  std::string status;
};

/** Full GS-CI state: one [x,y] block per ordered agent and sender yaw. */
struct GlobalCiBelief {
  std::string sender_id;
  std::vector<std::string> ordered_agent_ids;
  Eigen::VectorXd mean;
  Eigen::MatrixXd covariance;
  double timestamp{0.0};
  std::uint64_t sequence{0};
  bool valid{true};
};

struct GlobalCiResult {
  GlobalCiBelief belief;
  std::vector<double> weights;
  std::size_t accepted_peers{0};
  std::size_t rejected_peers{0};
  bool success{false};
  std::string status;
};

}  // namespace hercules_localization
