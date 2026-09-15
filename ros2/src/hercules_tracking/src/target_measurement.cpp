#include "hercules_tracking/target_measurement.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "hercules_tracking/linear_algebra.hpp"

namespace hercules_tracking {

TargetMeasurement::TargetMeasurement(
    std::string target_id_in, const Position& position_in,
    const PositionMatrix& covariance_in, double timestamp_in, bool valid_in,
    std::string source_in, std::optional<std::string> capture_id_in,
    std::optional<std::string> sensor_in, bool visible_in)
    : target_id(std::move(target_id_in)),
      position(position_in),
      covariance(positiveDefinite(covariance_in)),
      timestamp(timestamp_in),
      valid(valid_in),
      source(std::move(source_in)),
      capture_id(std::move(capture_id_in)),
      sensor(std::move(sensor_in)),
      visible(visible_in) {
  if (target_id.empty()) throw std::invalid_argument("target_id must not be empty");
  if (!position.allFinite() || !covariance_in.allFinite() || !std::isfinite(timestamp)) {
    throw std::invalid_argument("measurement values must be finite");
  }
}

}  // namespace hercules_tracking
