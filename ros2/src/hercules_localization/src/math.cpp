#include "hercules_localization/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "hercules_localization/covariance.hpp"

namespace hercules_localization {
namespace {

double safeRange(double value, double epsilon) {
  const double e = std::isfinite(epsilon) && epsilon > 0.0 ? epsilon : 1e-9;
  return std::max(std::abs(value), e);
}

Eigen::Matrix2d rotationMatrix(double yaw) {
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  Eigen::Matrix2d result;
  result << c, -s, s, c;
  return result;
}

}  // namespace

double wrapAngle(double angle) {
  if (!std::isfinite(angle)) return 0.0;
  constexpr double kPi = 3.141592653589793238462643383279502884;
  constexpr double kTwoPi = 2.0 * kPi;
  double wrapped = std::fmod(angle + kPi, kTwoPi);
  if (wrapped < 0.0) wrapped += kTwoPi;
  wrapped -= kPi;
  // Use a closed interval at +pi.  This keeps the representation stable at
  // the branch cut and matches the convention used by the paper equations
  // and the public angle-wrapping tests.
  return wrapped <= -kPi ? kPi : wrapped;
}

double angleDifference(double first, double second) {
  return wrapAngle(first - second);
}

Pose2D normalizePose(const Pose2D& pose) {
  Pose2D result = pose;
  if (!result.position.allFinite()) result.position.setZero();
  result.yaw = wrapAngle(result.yaw);
  return result;
}

Position2D rotate(const Position2D& point, double yaw) {
  return rotationMatrix(yaw) * point;
}

Pose2D compose(const Pose2D& first, const Pose2D& second) {
  return normalizePose(Pose2D(first.position + rotate(second.position, first.yaw),
                              first.yaw + second.yaw));
}

Pose2D inverse(const Pose2D& pose) {
  const double inverse_yaw = -pose.yaw;
  return normalizePose(Pose2D(rotate(-pose.position, inverse_yaw), inverse_yaw));
}

Pose2D relativePose(const Pose2D& reference, const Pose2D& target) {
  return compose(inverse(reference), target);
}

Position2D transformPoint(const Pose2D& pose, const Position2D& local_point) {
  return pose.position + rotate(local_point, pose.yaw);
}

Position2D inverseTransformPoint(const Pose2D& pose, const Position2D& world_point) {
  return rotate(world_point - pose.position, -pose.yaw);
}

Pose2D integrate(const Pose2D& pose, const Twist2D& twist, double dt) {
  if (!std::isfinite(dt) || dt <= 0.0 || !std::isfinite(twist.linear_x) ||
      !std::isfinite(twist.linear_y) || !std::isfinite(twist.angular_z)) {
    return normalizePose(pose);
  }
  const double w = twist.angular_z;
  const double angle = w * dt;
  Position2D body_delta;
  if (std::abs(w) <= 1e-9) {
    body_delta << twist.linear_x * dt, twist.linear_y * dt;
  } else {
    const double s = std::sin(angle);
    const double c = std::cos(angle);
    body_delta << (s * twist.linear_x - (1.0 - c) * twist.linear_y) / w,
        ((1.0 - c) * twist.linear_x + s * twist.linear_y) / w;
  }
  return normalizePose(Pose2D(pose.position + rotate(body_delta, pose.yaw),
                              pose.yaw + angle));
}

Eigen::Matrix3d motionJacobian(const Pose2D& pose, const Twist2D& twist, double dt) {
  Eigen::Matrix3d result = Eigen::Matrix3d::Identity();
  if (!std::isfinite(dt) || dt <= 0.0 || !std::isfinite(twist.linear_x) ||
      !std::isfinite(twist.linear_y) || !std::isfinite(twist.angular_z)) {
    return result;
  }
  const double w = twist.angular_z;
  const double angle = w * dt;
  Position2D body_delta;
  if (std::abs(w) <= 1e-9) {
    body_delta << twist.linear_x * dt, twist.linear_y * dt;
  } else {
    const double s = std::sin(angle);
    const double c = std::cos(angle);
    body_delta << (s * twist.linear_x - (1.0 - c) * twist.linear_y) / w,
        ((1.0 - c) * twist.linear_x + s * twist.linear_y) / w;
  }
  const Position2D rotated_delta = rotate(body_delta, pose.yaw);
  result(0, 2) = -rotated_delta.y();
  result(1, 2) = rotated_delta.x();
  return result;
}

PoseEstimate propagate(const PoseEstimate& estimate, const Twist2D& twist, double dt,
                       const Eigen::Matrix3d& process_covariance,
                       double covariance_floor) {
  PoseEstimate result = estimate;
  result.mean = integrate(estimate.mean, twist, dt);
  const Eigen::Matrix3d f = motionJacobian(estimate.mean, twist, dt);
  const Eigen::Matrix3d q = regularizeCovariance(process_covariance, covariance_floor);
  const Eigen::Matrix3d propagated_covariance =
      f * regularizeCovariance(estimate.covariance, covariance_floor) * f.transpose() + q;
  result.covariance = regularizeCovariance(propagated_covariance, covariance_floor);
  if (std::isfinite(dt) && dt > 0.0) result.timestamp += dt;
  result.valid = estimate.valid;
  return result;
}

RangeBearing rangeBearing(const Position2D& relative_position, double range_epsilon) {
  (void)range_epsilon;
  const double radius = relative_position.norm();
  if (!relative_position.allFinite() || !std::isfinite(radius)) return {};
  return RangeBearing(radius, std::atan2(relative_position.y(), relative_position.x()));
}

RangeBearing rangeBearing(const Pose2D& observer, const Pose2D& target,
                          double range_epsilon) {
  return rangeBearing(inverseTransformPoint(observer, target.position), range_epsilon);
}

Position2D polarToCartesian(const RangeBearing& measurement) {
  return polarToCartesian(measurement.range, measurement.bearing);
}

Position2D polarToCartesian(double range, double bearing) {
  if (!std::isfinite(range) || !std::isfinite(bearing)) return Position2D::Zero();
  return Position2D(range * std::cos(bearing), range * std::sin(bearing));
}

Eigen::Matrix2d polarJacobian(const RangeBearing& measurement) {
  Eigen::Matrix2d result;
  const double c = std::cos(measurement.bearing);
  const double s = std::sin(measurement.bearing);
  result << c, -measurement.range * s, s, measurement.range * c;
  return result;
}

Eigen::Matrix<double, 2, 3> rangeBearingJacobianObserver(
    const Pose2D& observer, const Pose2D& target, double range_epsilon) {
  const Position2D delta = target.position - observer.position;
  const double r = safeRange(delta.norm(), range_epsilon);
  const double r2 = r * r;
  Eigen::Matrix<double, 2, 3> result;
  // atan2(target_y-observer_y, target_x-observer_x)-observer_yaw
  result << -delta.x() / r, -delta.y() / r, 0.0,
      delta.y() / r2, -delta.x() / r2, -1.0;
  return result;
}

Eigen::Matrix<double, 2, 3> rangeBearingJacobianTarget(
    const Pose2D& observer, const Pose2D& target, double range_epsilon) {
  const Position2D delta = target.position - observer.position;
  const double r = safeRange(delta.norm(), range_epsilon);
  const double r2 = r * r;
  Eigen::Matrix<double, 2, 3> result;
  result << delta.x() / r, delta.y() / r, 0.0,
      -delta.y() / r2, delta.x() / r2, 0.0;
  return result;
}

Eigen::Matrix2d polarCovarianceToCartesian(const RangeBearing& measurement,
                                           const Eigen::Matrix2d& covariance,
                                           double floor) {
  const Eigen::Matrix2d cartesian_covariance =
      polarJacobian(measurement) * covariance * polarJacobian(measurement).transpose();
  return regularizeCovariance(cartesian_covariance, floor);
}

RangeBearing predictRangeBearing(const Pose2D& observer, const Pose2D& target,
                                 double range_epsilon) {
  const Position2D local = inverseTransformPoint(observer, target.position);
  RangeBearing result = rangeBearing(local, range_epsilon);
  result.bearing = wrapAngle(result.bearing);
  return result;
}

}  // namespace hercules_localization
