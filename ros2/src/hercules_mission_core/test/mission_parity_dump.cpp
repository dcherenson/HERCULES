#include <iomanip>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>

#include "hercules_mission_core/mission_config.hpp"

namespace hmc = hercules_mission_core;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

template <typename Derived>
void writeVector(const Eigen::MatrixBase<Derived>& value) {
  std::cout << '[';
  for (Eigen::Index index = 0; index < value.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << value(index);
  }
  std::cout << ']';
}

void writePoints(const std::vector<Eigen::Vector3d>& points) {
  std::cout << '[';
  bool first = true;
  for (const auto& point : points) {
    for (Eigen::Index component = 0; component < point.size(); ++component) {
      if (!first) std::cout << ',';
      first = false;
      std::cout << point(component);
    }
  }
  std::cout << ']';
}

hmc::FigureEightTargetController baselineController() {
  const hmc::RuralTargetTrackingConfig mission = hmc::ruralTargetTrackingConfig();
  const Eigen::Vector3d start = Eigen::Vector3d::Zero();
  const Eigen::Vector3d goal(20.0, 8.0, -1.0);
  const double heading = std::atan2(goal.y() - start.y(), goal.x() - start.x()) + kPi / 2.0;
  hmc::FigureEightConfig motion = mission.target_motion;
  motion.route_heading = heading;
  motion.center = hmc::ruralTargetStartAnchor(start, goal, heading, mission);
  hmc::FigureEightTargetController controller(motion);
  controller.setIndex(mission.target_start_sample_index);
  controller.placeStartAt(motion.center);
  return controller;
}

void writeGeometry() {
  const auto controller = baselineController();
  std::cout << "\"figure_geometry\":{\"center\":";
  writeVector(controller.config().center);
  std::cout << ",\"points\":";
  writePoints(controller.points());
  std::cout << ",\"index\":" << controller.index()
            << ",\"phase\":" << controller.phase() << ",\"references\":[";
  const std::vector<int> lookaheads{-3, 0, 1, 2, 7};
  for (std::size_t index = 0; index < lookaheads.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << "{\"lookahead\":" << lookaheads[index] << ",\"value\":";
    writeVector(controller.reference(lookaheads[index]));
    std::cout << '}';
  }
  std::cout << "]}";
}

void writeDynamics() {
  auto controller = baselineController();
  const auto points = controller.points();
  const std::vector<int> position_indices{5, 6, 12, 14, 2, 3};
  const std::vector<double> yaws{0.0, 0.5, -2.8, 2.5, 0.1, -1.1};
  const std::vector<double> deltas{0.1, 0.25, 0.05, 0.1, 0.2, 0.1};
  std::cout << "\"figure_dynamics\":[";
  for (std::size_t step = 0; step < position_indices.size(); ++step) {
    if (step) std::cout << ',';
    const Eigen::Vector2d command = controller.update(
        points[static_cast<std::size_t>(position_indices[step])], yaws[step], deltas[step]);
    std::cout << "{\"index\":" << controller.index()
              << ",\"phase\":" << controller.phase() << ",\"reference\":";
    writeVector(controller.reference());
    std::cout << ",\"command\":";
    writeVector(command);
    std::cout << '}';
  }
  std::cout << ']';
}

void writeSlots() {
  const auto mission = hmc::ruralTargetTrackingConfig();
  const double fallback = 0.6;
  const Eigen::Vector3d target_position(10.0, 20.0, -1.0);
  const std::vector<Eigen::Vector3d> velocities{
      Eigen::Vector3d::Zero(), Eigen::Vector3d(1.0, 0.0, 0.0),
      Eigen::Vector3d(0.0, 3.0, 0.0)};
  const std::vector<std::string> agents{
      "Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5",
      "Husky1", "Husky2", "Husky3"};
  std::cout << "\"slots\":[";
  bool first = true;
  for (std::size_t scenario = 0; scenario < velocities.size(); ++scenario) {
    for (const auto& agent : agents) {
      if (!first) std::cout << ',';
      first = false;
      const bool drone = agent.rfind("Husky", 0) != 0;
      const auto slot = hmc::targetCenteredSlot(
          agent, drone ? hmc::VehicleType::kDrone : hmc::VehicleType::kUgv,
          target_position, velocities[scenario], fallback, mission.uav_altitude,
          mission.target_ground_z, mission.target_ugv_circumradius);
      std::cout << "{\"case\":" << scenario << ",\"agent\":\"" << agent
                << "\",\"position\":";
      writeVector(slot.position);
      std::cout << ",\"velocity\":";
      writeVector(slot.velocity);
      std::cout << ",\"heading\":" << slot.heading << '}';
    }
  }
  std::cout << ']';
}

hmc::AgentState agent(const std::string& id, const Eigen::Vector3d& position,
                      const Eigen::Vector3d& velocity, double yaw,
                      hmc::VehicleType type) {
  return {id, position, velocity, yaw, type};
}

void writeControls() {
  const auto mission = hmc::ruralTargetTrackingConfig();
  const hmc::FormationController controller(mission.formation);
  const hmc::TargetEstimate active{{10.0, 20.0}, {0.5, 0.25}, true};
  const hmc::TargetEstimate inactive{{10.0, 20.0}, {0.5, 0.25}, false};
  const double fallback = 0.4;
  std::cout << "\"controls\":[";

  const std::vector<std::pair<std::string, Eigen::Vector3d>> uav_cases{
      {"uav_ordinary", controller.targetNominalControl(
          agent("Drone1", {8.0, 17.0, -4.0}, {0.2, -0.1, 0.3}, 0.0,
                hmc::VehicleType::kDrone), active, fallback, -1.0, 5.0)},
      {"uav_speed_saturation", controller.targetNominalControl(
          agent("Drone5", {-20.0, 40.0, -20.0}, {-1.0, 0.5, 0.0}, 0.0,
                hmc::VehicleType::kDrone), active, fallback, -1.0, 5.0)},
      {"uav_inactive", controller.targetNominalControl(
          agent("SimpleFlight", {2.0, 3.0, -5.0}, {1.0, 1.0, 0.0}, 0.0,
                hmc::VehicleType::kDrone), inactive, fallback, -1.0, 5.0)},
  };
  bool first = true;
  for (const auto& [name, command] : uav_cases) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"case\":\"" << name << "\",\"command\":";
    writeVector(command);
    std::cout << '}';
  }

  const auto hold_slot = hmc::targetCenteredSlot(
      "Husky3", hmc::VehicleType::kUgv, {10.0, 20.0, -1.0},
      {0.5, 0.25, 0.0}, fallback, -5.0, -1.0, 5.0);
  const std::vector<std::pair<std::string, Eigen::Vector2d>> ugv_cases{
      {"ugv_ordinary", controller.targetNominalUnicycleControl(
          agent("Husky1", {12.0, 18.0, -1.0}, {0.0, 0.0, 0.0}, 0.2,
                hmc::VehicleType::kUgv), active, fallback, -1.0, 5.0)},
      {"ugv_yaw_saturation", controller.targetNominalUnicycleControl(
          agent("Husky2", {-5.0, 5.0, -1.0}, {0.0, 0.0, 0.0}, 3.0,
                hmc::VehicleType::kUgv), active, fallback, -1.0, 5.0)},
      {"ugv_hold", controller.targetNominalUnicycleControl(
          agent("Husky3", hold_slot.position + Eigen::Vector3d(0.3, 0.0, 0.0),
                {0.0, 0.0, 0.0}, 0.0, hmc::VehicleType::kUgv),
          active, fallback, -1.0, 5.0)},
      {"ugv_inactive", controller.targetNominalUnicycleControl(
          agent("Husky1", {0.0, 0.0, -1.0}, {0.0, 0.0, 0.0}, 0.0,
                hmc::VehicleType::kUgv), inactive, fallback, -1.0, 5.0)},
  };
  for (const auto& [name, command] : ugv_cases) {
    std::cout << ",{\"case\":\"" << name << "\",\"command\":";
    writeVector(command);
    std::cout << '}';
  }
  std::cout << ']';
}

}  // namespace

int main() {
  std::cout << std::setprecision(17) << '{';
  writeGeometry();
  std::cout << ',';
  writeDynamics();
  std::cout << ',';
  writeSlots();
  std::cout << ',';
  writeControls();
  std::cout << "}\n";
  return 0;
}
