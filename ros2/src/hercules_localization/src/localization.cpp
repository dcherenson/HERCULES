#include "hercules_localization/localization.hpp"

namespace hercules_localization {

LocalizationResult update(const PoseEstimate& prior,
                          const std::vector<RelativeObservation>& observations,
                          LocalizationAlgorithm algorithm,
                          const LocalizationConfig& config) {
  if (algorithm == LocalizationAlgorithm::kGsCi) {
    return gsCiUpdate(prior, observations, config);
  }
  return recursiveDecentralizedUpdate(prior, observations, config);
}

}  // namespace hercules_localization
