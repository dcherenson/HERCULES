#pragma once

#include "hercules_localization/covariance.hpp"
#include "hercules_localization/gs_ci.hpp"
#include "hercules_localization/global_ci.hpp"
#include "hercules_localization/math.hpp"
#include "hercules_localization/recursive_localization.hpp"
#include "hercules_localization/types.hpp"

namespace hercules_localization {

/** Dispatch to the selected pure numerical localization algorithm. */
LocalizationResult update(const PoseEstimate& prior,
                          const std::vector<RelativeObservation>& observations,
                          LocalizationAlgorithm algorithm,
                          const LocalizationConfig& config = {});

}  // namespace hercules_localization
