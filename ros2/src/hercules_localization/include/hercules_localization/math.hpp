#pragma once

#include <Eigen/Core>

#include "hercules_localization/types.hpp"

namespace hercules_localization {

double wrapAngle(double angle);
double angleDifference(double first, double second);

Pose2D normalizePose(const Pose2D& pose);
Pose2D compose(const Pose2D& first, const Pose2D& second);
Pose2D inverse(const Pose2D& pose);
Pose2D relativePose(const Pose2D& reference, const Pose2D& target);
Position2D rotate(const Position2D& point, double yaw);
Position2D transformPoint(const Pose2D& pose, const Position2D& local_point);
Position2D inverseTransformPoint(const Pose2D& pose,
                                  const Position2D& world_point);

/** Integrate a constant body-frame twist for a nonnegative duration. */
Pose2D integrate(const Pose2D& pose, const Twist2D& twist, double dt);

/** Jacobian of integrate() with respect to [x,y,yaw]. */
Eigen::Matrix3d motionJacobian(const Pose2D& pose, const Twist2D& twist,
                               double dt);

/** Constant process-noise propagation with an additive covariance. */
PoseEstimate propagate(const PoseEstimate& estimate, const Twist2D& twist,
                        double dt, const Eigen::Matrix3d& process_covariance,
                        double covariance_floor = kDefaultCovarianceFloor);

RangeBearing rangeBearing(const Position2D& relative_position,
                          double range_epsilon = kDefaultRangeEpsilon);
RangeBearing rangeBearing(const Pose2D& observer, const Pose2D& target,
                          double range_epsilon = kDefaultRangeEpsilon);
Position2D polarToCartesian(const RangeBearing& measurement);
Position2D polarToCartesian(double range, double bearing);
Eigen::Matrix2d polarJacobian(const RangeBearing& measurement);

/**
 * Jacobian of [range,bearing] with respect to observer pose [x,y,yaw].
 * The target pose is treated as constant.
 */
Eigen::Matrix<double, 2, 3> rangeBearingJacobianObserver(
    const Pose2D& observer, const Pose2D& target,
    double range_epsilon = kDefaultRangeEpsilon);

/** Jacobian of [range,bearing] with respect to target pose [x,y,yaw]. */
Eigen::Matrix<double, 2, 3> rangeBearingJacobianTarget(
    const Pose2D& observer, const Pose2D& target,
    double range_epsilon = kDefaultRangeEpsilon);

/** Convert a range/bearing covariance into local Cartesian covariance. */
Eigen::Matrix2d polarCovarianceToCartesian(
    const RangeBearing& measurement, const Eigen::Matrix2d& covariance,
    double floor = kDefaultCovarianceFloor);

/** Expected range/bearing and observer-pose Jacobian for one observation. */
RangeBearing predictRangeBearing(const Pose2D& observer, const Pose2D& target,
                                 double range_epsilon = kDefaultRangeEpsilon);

}  // namespace hercules_localization
