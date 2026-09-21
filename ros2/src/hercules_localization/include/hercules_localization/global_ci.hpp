#pragma once

#include <limits>
#include <string>
#include <vector>

#include "hercules_localization/types.hpp"

namespace hercules_localization {

/**
 * Build the ordered GS-CI state from one local pose.
 *
 * The state layout is `[x_1,y_1,...,x_n,y_n,yaw_sender]`.  The sender's
 * position/yaw marginal is copied from `local_pose`; every remote position
 * block starts at a deliberately large, finite variance so that a first
 * local observation does not claim information about agents it has not seen.
 * The sender id must occur exactly once in `ordered_agent_ids`.
 */
GlobalCiBelief initializeGlobalCiBelief(
    const std::string& sender_id,
    const std::vector<std::string>& ordered_agent_ids,
    const PoseEstimate& local_pose,
    double remote_position_variance = 1.0e6,
    double covariance_floor = kDefaultCovarianceFloor);

/** Convenience overload which uses `local_pose.agent_id` as the sender id. */
GlobalCiBelief initializeGlobalCiBelief(
    const PoseEstimate& local_pose,
    const std::vector<std::string>& ordered_agent_ids,
    double remote_position_variance = 1.0e6,
    double covariance_floor = kDefaultCovarianceFloor);

/** Descriptive aliases for initializeGlobalCiBelief(). */
GlobalCiBelief createGlobalCiBelief(
    const std::string& sender_id,
    const std::vector<std::string>& ordered_agent_ids,
    const PoseEstimate& local_pose,
    double remote_position_variance = 1.0e6,
    double covariance_floor = kDefaultCovarianceFloor);

GlobalCiBelief createGlobalCiBelief(
    const PoseEstimate& local_pose,
    const std::vector<std::string>& ordered_agent_ids,
    double remote_position_variance = 1.0e6,
    double covariance_floor = kDefaultCovarianceFloor);

/** Validate ordering, dimensions, finite values, and SPD covariance. */
bool isValidGlobalCiBelief(const GlobalCiBelief& belief,
                           double covariance_floor = kDefaultCovarianceFloor);

/** Propagate the local position/yaw block and inflate remote blocks. */
GlobalCiBelief propagateGlobalCiBelief(
    const GlobalCiBelief& prior, const std::string& receiver_id,
    const PoseVector& world_delta, const PoseCovariance& local_process_covariance,
    double dt, double unknown_motion_variance,
    double covariance_floor = kDefaultCovarianceFloor);

/**
 * Apply a private [x,y] observation to the complete ordered GS-CI state.
 *
 * The position selector is applied to the full covariance, so correlations
 * with remote blocks and the sender yaw are retained.  The covariance update
 * uses Joseph form and is symmetrized/floored before being returned.  On
 * invalid input the returned result has `success == false` and preserves the
 * input belief in `belief`.
 */
GlobalCiResult globalCiPrivatePositionUpdate(
    const GlobalCiBelief& prior, const std::string& receiver_id,
    const Position2D& measured_position,
    const MeasurementCovariance& measurement_covariance,
    double timestamp = std::numeric_limits<double>::quiet_NaN(),
    double covariance_floor = kDefaultCovarianceFloor);

/** Apply a private position update using the shared localization settings. */
GlobalCiResult globalCiPrivatePositionUpdate(
    const GlobalCiBelief& prior, const std::string& receiver_id,
    const Position2D& measured_position,
    const MeasurementCovariance& measurement_covariance, double timestamp,
    const LocalizationConfig& config);

/**
 * Apply one range/bearing observation to the complete ordered state.
 *
 * The observer must be the belief sender because the state contains only the
 * sender yaw.  Both observer and observed positions are represented in the
 * Jacobian; the observer yaw column is mapped to the final state element.
 */
GlobalCiResult globalCiRangeBearingUpdate(
    const GlobalCiBelief& prior, const std::string& observer_id,
    const std::string& observed_id,
    const RangeBearingMeasurement& measurement,
    const LocalizationConfig& config = {});

/** Short aliases used by callers that name the operations as EKF updates. */
GlobalCiResult globalCiPrivatePositionEkfUpdate(
    const GlobalCiBelief& prior, const std::string& receiver_id,
    const Position2D& measured_position,
    const MeasurementCovariance& measurement_covariance,
    double timestamp = std::numeric_limits<double>::quiet_NaN(),
    double covariance_floor = kDefaultCovarianceFloor);

GlobalCiResult globalCiRangeBearingEkfUpdate(
    const GlobalCiBelief& prior, const std::string& observer_id,
    const std::string& observed_id,
    const RangeBearingMeasurement& measurement,
    const LocalizationConfig& config = {});

/** Batch covariance intersection with deterministic self/peer coefficients. */
GlobalCiResult globalCiUpdate(
    const GlobalCiBelief& self, const std::vector<GlobalCiBelief>& peers,
    const LocalizationConfig& config = {});

}  // namespace hercules_localization
