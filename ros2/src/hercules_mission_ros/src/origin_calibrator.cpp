#include "hercules_mission_ros/origin_calibrator.hpp"

#include <algorithm>
#include <stdexcept>

namespace hercules_mission_ros {

OriginCalibrator::OriginCalibrator(std::size_t required_samples,
                                   double stability_tolerance)
    : required_samples_(required_samples),
      stability_tolerance_(stability_tolerance) {
  if (required_samples_ == 0 || stability_tolerance_ <= 0.0) {
    throw std::invalid_argument("origin calibration parameters must be positive");
  }
}

bool OriginCalibrator::add(const Eigen::Vector3d& wrapper_local_ned,
                           const Eigen::Vector3d& direct_world_ned) {
  if (origin_) return true;
  const Eigen::Vector3d candidate = direct_world_ned - wrapper_local_ned;
  if (!candidate.allFinite()) return false;

  if (!candidates_.empty() &&
      (candidate - candidates_.back()).norm() > stability_tolerance_) {
    candidates_.clear();
  }
  candidates_.push_back(candidate);
  if (candidates_.size() < required_samples_) return false;

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const auto& value : candidates_) mean += value;
  mean /= static_cast<double>(candidates_.size());
  const bool stable = std::all_of(
      candidates_.begin(), candidates_.end(), [&](const Eigen::Vector3d& value) {
        return (value - mean).norm() <= stability_tolerance_;
      });
  if (stable) origin_ = mean;
  return calibrated();
}

}  // namespace hercules_mission_ros
