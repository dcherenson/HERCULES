#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace hercules_mission_core {

double wrapAngle(double angle);
std::pair<Eigen::Vector2d, Eigen::Vector2d> routeBasis(double heading);
Eigen::Vector3d targetCenterBeforeGoal(const Eigen::Vector3d& start,
                                       const Eigen::Vector3d& goal,
                                       double fraction = 0.7);

struct FigureEightConfig {
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  double route_heading{0.0};
  double longitudinal_span{10.0};
  double lateral_span{8.0};
  double speed{1.5};
  int sample_count{64};
  double waypoint_radius{1.0};
  double heading_gain{2.0};
  double max_yaw_rate{1.5};
  double minimum_alignment{0.75};
  int direction{1};
};

class FigureEightTargetController {
 public:
  explicit FigureEightTargetController(FigureEightConfig config);

  Eigen::Vector2d update(const Eigen::Vector3d& position, double yaw, double dt);
  Eigen::Vector3d reference(int lookahead = 2) const;
  void placeStartAt(const Eigen::Vector3d& position);

  const FigureEightConfig& config() const { return config_; }
  const std::vector<Eigen::Vector3d>& points() const { return points_; }
  int index() const { return index_; }
  void setIndex(int index);
  double phase() const;

 private:
  void buildPoints();
  int wrappedIndex(int index) const;

  FigureEightConfig config_;
  std::vector<Eigen::Vector3d> points_;
  int index_{0};
};

}  // namespace hercules_mission_core
