#include "hercules_mission_core/target_motion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace hercules_mission_core {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

int pythonRoundedEighth(int sample_count) {
  const double value = static_cast<double>(sample_count) * 0.125;
  const double lower = std::floor(value);
  const double fraction = value - lower;
  if (fraction < 0.5) return static_cast<int>(lower);
  if (fraction > 0.5) return static_cast<int>(lower + 1.0);
  const int integer = static_cast<int>(lower);
  return integer % 2 == 0 ? integer : integer + 1;
}

}  // namespace

double wrapAngle(double angle) {
  double wrapped = std::fmod(angle + kPi, 2.0 * kPi);
  if (wrapped < 0.0) wrapped += 2.0 * kPi;
  return wrapped - kPi;
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> routeBasis(double heading) {
  const Eigen::Vector2d forward(std::cos(heading), std::sin(heading));
  return {forward, Eigen::Vector2d(-forward.y(), forward.x())};
}

Eigen::Vector3d targetCenterBeforeGoal(const Eigen::Vector3d& start,
                                       const Eigen::Vector3d& goal,
                                       double fraction) {
  const double value = std::clamp(fraction, 0.0, 1.0);
  return start + value * (goal - start);
}

FigureEightTargetController::FigureEightTargetController(FigureEightConfig config)
    : config_(std::move(config)) {
  config_.sample_count = std::max(8, config_.sample_count);
  config_.direction = config_.direction >= 0 ? 1 : -1;
  buildPoints();
  index_ = wrappedIndex(pythonRoundedEighth(config_.sample_count));
}

void FigureEightTargetController::buildPoints() {
  const auto [forward, left] = routeBasis(config_.route_heading);
  points_.clear();
  points_.reserve(static_cast<std::size_t>(config_.sample_count));
  for (int index = 0; index < config_.sample_count; ++index) {
    const double theta = 2.0 * kPi * static_cast<double>(index) /
                         static_cast<double>(config_.sample_count);
    Eigen::Vector3d point = config_.center;
    point.head<2>() += 0.5 * config_.longitudinal_span * std::sin(theta) * forward +
                       0.5 * config_.lateral_span * std::sin(2.0 * theta) * left;
    points_.push_back(point);
  }
}

int FigureEightTargetController::wrappedIndex(int index) const {
  const int remainder = index % config_.sample_count;
  return remainder < 0 ? remainder + config_.sample_count : remainder;
}

void FigureEightTargetController::setIndex(int index) {
  index_ = wrappedIndex(index);
}

double FigureEightTargetController::phase() const {
  return 2.0 * kPi * static_cast<double>(index_) /
         static_cast<double>(config_.sample_count);
}

Eigen::Vector3d FigureEightTargetController::reference(int lookahead) const {
  return points_[static_cast<std::size_t>(
      wrappedIndex(index_ + config_.direction * lookahead))];
}

void FigureEightTargetController::placeStartAt(const Eigen::Vector3d& position) {
  config_.center += position - points_[static_cast<std::size_t>(index_)];
  buildPoints();
}

Eigen::Vector2d FigureEightTargetController::update(
    const Eigen::Vector3d& position, double yaw, double dt) {
  (void)dt;  // The executable Python controller accepts but does not use dt.
  const int starting_index = index_;
  while ((position.head<2>() - points_[static_cast<std::size_t>(index_)].head<2>()).norm() <=
         config_.waypoint_radius) {
    index_ = wrappedIndex(index_ + config_.direction);
    if (index_ == starting_index) break;
  }

  const int forward_count = std::max(
      2, std::min(config_.sample_count - 1, config_.sample_count / 8));
  const double current_distance =
      (position.head<2>() - points_[static_cast<std::size_t>(index_)].head<2>()).norm();
  int nearest_index = index_;
  double nearest_distance = std::numeric_limits<double>::infinity();
  for (int offset = 1; offset <= forward_count; ++offset) {
    const int candidate = wrappedIndex(index_ + config_.direction * offset);
    const double distance =
        (position.head<2>() - points_[static_cast<std::size_t>(candidate)].head<2>()).norm();
    if (distance < nearest_distance) {
      nearest_distance = distance;
      nearest_index = candidate;
    }
  }
  if (nearest_distance + 0.05 < current_distance) index_ = nearest_index;

  const Eigen::Vector2d delta = reference().head<2>() - position.head<2>();
  const double distance = delta.norm();
  if (distance <= 1e-9) return Eigen::Vector2d::Zero();
  const double desired_heading = std::atan2(delta.y(), delta.x());
  const double error = wrapAngle(desired_heading - yaw);
  double speed = std::min(config_.speed, distance);
  speed *= std::max(config_.minimum_alignment, std::cos(error));
  const double yaw_rate = std::clamp(
      config_.heading_gain * error, -config_.max_yaw_rate, config_.max_yaw_rate);
  return Eigen::Vector2d(speed, yaw_rate);
}

}  // namespace hercules_mission_core
