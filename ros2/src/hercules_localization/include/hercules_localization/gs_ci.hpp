#pragma once

#include <vector>

#include "hercules_localization/recursive_localization.hpp"

namespace hercules_localization {

/** Fuse two possibly correlated pose estimates conservatively. */
CovarianceIntersectionResult covarianceIntersection(
    const PoseEstimate& first, const PoseEstimate& second,
    const LocalizationConfig& config = {});

/** Long descriptive alias for covarianceIntersection(). */
CovarianceIntersectionResult covarianceIntersectionFuse(
    const PoseEstimate& first, const PoseEstimate& second,
    const LocalizationConfig& config = {});

/**
 * In-place Gauss--Seidel covariance-intersection localization.
 *
 * A neighbor-derived pose candidate is formed from each range/bearing
 * observation.  The current estimate is immediately replaced by the CI result
 * before the next observation, and rounds repeat until the configured pose and
 * covariance change tolerance is reached.
 */
LocalizationResult gsCiUpdate(
    const PoseEstimate& prior,
    const std::vector<RelativeObservation>& observations,
    const LocalizationConfig& config = {});

LocalizationResult gaussSeidelCovarianceIntersection(
    const PoseEstimate& prior,
    const std::vector<RelativeObservation>& observations,
    const LocalizationConfig& config = {});

class GsCiLocalizer {
 public:
  explicit GsCiLocalizer(PoseEstimate initial = {}, LocalizationConfig config = {});

  void reset(const PoseEstimate& estimate);
  LocalizationResult update(
      const std::vector<RelativeObservation>& observations);
  LocalizationResult update(const PoseEstimate& prior,
                            const std::vector<RelativeObservation>& observations);

  const PoseEstimate& estimate() const { return estimate_; }
  const LocalizationConfig& config() const { return config_; }
  void setConfig(const LocalizationConfig& config) { config_ = config; }

 private:
  PoseEstimate estimate_;
  LocalizationConfig config_;
};

using GSCILocalizer = GsCiLocalizer;

}  // namespace hercules_localization
