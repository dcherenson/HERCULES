#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <Eigen/Core>

namespace hercules_mission_ros {

// Estimates the fixed translation between wrapper-local NED and AirSim world
// NED. Rotation is handled by the existing wrapper-to-NED state adapter.
class OriginCalibrator {
 public:
  OriginCalibrator(std::size_t required_samples = 10,
                   double stability_tolerance = 0.05);

  bool add(const Eigen::Vector3d& wrapper_local_ned,
           const Eigen::Vector3d& direct_world_ned);
  bool calibrated() const { return origin_.has_value(); }
  const Eigen::Vector3d& origin() const { return origin_.value(); }
  std::size_t samples() const { return candidates_.size(); }

 private:
  std::size_t required_samples_;
  double stability_tolerance_;
  std::vector<Eigen::Vector3d> candidates_;
  std::optional<Eigen::Vector3d> origin_;
};

}  // namespace hercules_mission_ros
