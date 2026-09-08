#pragma once

#include <optional>
#include <string>

#include "hercules_tracking/models.hpp"

namespace hercules_tracking {

struct TargetMeasurement {
  std::string target_id;
  Position position;
  PositionMatrix covariance;
  double timestamp;
  bool valid{true};
  std::string source{"unknown"};
  std::optional<std::string> capture_id;
  std::optional<std::string> sensor;
  bool visible{true};

  TargetMeasurement(std::string target_id_in, const Position& position_in,
                    const PositionMatrix& covariance_in, double timestamp_in,
                    bool valid_in = true, std::string source_in = "unknown",
                    std::optional<std::string> capture_id_in = std::nullopt,
                    std::optional<std::string> sensor_in = std::nullopt,
                    bool visible_in = true);
};

}  // namespace hercules_tracking
