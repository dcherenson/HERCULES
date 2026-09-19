#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <Eigen/Core>
#include <airsim_interfaces/msg/car_controls.hpp>
#include <airsim_interfaces/msg/vel_cmd.hpp>
#include <airsim_interfaces/srv/land.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <hercules_interfaces/msg/ground_truth_state.hpp>
#include <hercules_interfaces/msg/target_estimate.hpp>
#include <hercules_interfaces/msg/target_measurement.hpp>
#include <hercules_interfaces/msg/target_observation_diagnostics.hpp>
#include <hercules_interfaces/msg/cbf_diagnostics_array.hpp>
#include <hercules_interfaces/msg/mission_collision.hpp>
#include <hercules_interfaces/msg/obstacle_proxy_array.hpp>
#include <hercules_interfaces/msg/tracking_diagnostics.hpp>
#include <hercules_interfaces/msg/tracking_epoch.hpp>
#include <hercules_mission_core/formation_controller.hpp>
#include <hercules_mission_core/mission_config.hpp>
#include <hercules_mission_core/target_formation.hpp>
#include <hercules_mission_core/target_motion.hpp>
#include <hercules_tracking/models.hpp>
#include <rclcpp/rclcpp.hpp>

#include "hercules_mission_ros/mission_actuation.hpp"
#include "hercules_mission_ros/mission_gate.hpp"
#include "hercules_mission_ros/target_source.hpp"
#include "hercules_cbf_ros/cbf_adapter.hpp"
#include "hercules_cbf_ros/obstacle_cache.hpp"
#include "hercules_tracking_ros/neighbor_graph.hpp"
#include "hercules_tracking_ros/tracking_adapter.hpp"

namespace hercules_mission_ros {
namespace {
using Clock = std::chrono::steady_clock;
constexpr const char* kDrones[] = {
    "Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5"};
constexpr const char* kUgvs[] = {"Husky1", "Husky2", "Husky3"};
constexpr const char* kControlled[] = {"Drone1", "Drone2", "SimpleFlight", "Drone4",
                                       "Drone5", "Husky1", "Husky2", "Husky3"};
constexpr const char* kAll[] = {"Drone1", "Drone2", "SimpleFlight", "Drone4",
                                "Drone5", "Husky1", "Husky2", "Husky3",
                                "Target1"};

double seconds(Clock::time_point then) {
  return std::chrono::duration<double>(Clock::now() - then).count();
}

struct ControlTargetEstimate {
  hercules_mission_core::TargetEstimate estimate;
  Eigen::Matrix2d covariance{Eigen::Matrix2d::Zero()};
  double timestamp{0.0};
  bool from_distributed{false};
};

const char* expectedVehicleType(const std::string& id) {
  if (id == "Target1") return "target_ugv";
  if (id.rfind("Drone", 0) == 0 || id == "SimpleFlight") return "drone";
  return "ugv";
}

bool validCanonicalState(const hercules_interfaces::msg::GroundTruthState& message,
                         const std::string& expected_id) {
  if (!message.valid || message.agent_id != expected_id ||
      message.header.frame_id != "airsim_world_ned" ||
      message.vehicle_type != expectedVehicleType(expected_id)) return false;
  for (const auto value : message.position) if (!std::isfinite(value)) return false;
  for (const auto value : message.velocity) if (!std::isfinite(value)) return false;
  for (const auto value : message.orientation) if (!std::isfinite(value)) return false;
  return std::isfinite(message.yaw) && std::isfinite(message.yaw_rate);
}

hercules_mission_core::AgentState agentState(
    const hercules_interfaces::msg::GroundTruthState& message) {
  hercules_mission_core::AgentState state;
  state.agent_id = message.agent_id;
  state.position = {message.position[0], message.position[1], message.position[2]};
  state.velocity = {message.velocity[0], message.velocity[1], message.velocity[2]};
  state.yaw = message.yaw;
  state.vehicle_type = message.vehicle_type == "drone"
                           ? hercules_mission_core::VehicleType::kDrone
                           : (message.vehicle_type == "target_ugv"
                                  ? hercules_mission_core::VehicleType::kTargetUgv
                                  : hercules_mission_core::VehicleType::kUgv);
  return state;
}

airsim_interfaces::msg::CarControls carMessage(const CarCommand& command) {
  airsim_interfaces::msg::CarControls message;
  message.throttle = static_cast<float>(command.throttle);
  message.brake = static_cast<float>(command.brake);
  message.steering = static_cast<float>(command.steering);
  message.handbrake = command.handbrake;
  message.manual = true;
  message.manual_gear = static_cast<int8_t>(command.manual_gear);
  message.gear_immediate = true;
  return message;
}
}  // namespace

class RuralNominalMissionNode : public rclcpp::Node {
 public:
  RuralNominalMissionNode()
      : Node("rural_nominal_mission"),
        config_(hercules_mission_core::ruralTargetTrackingConfig()),
        controller_(config_.formation), started_at_(Clock::now()) {
    dry_run_ = declare_parameter<bool>("dry_run", true);
    enable_target_ = declare_parameter<bool>("enable_target", true);
    enable_formation_ = declare_parameter<bool>("enable_formation", true);
    target_source_name_ = declare_parameter<std::string>("target_source", "truth");
    if (target_source_name_ != "truth" && target_source_name_ != "distributed_tracking") {
      throw std::invalid_argument("target_source must be truth or distributed_tracking");
    }
    target_observation_source_ =
        declare_parameter<std::string>("target_observation_source", "truth");
    duration_ = declare_parameter<double>("duration_sec", 30.0);
    startup_timeout_ = declare_parameter<double>("startup_timeout_sec", 30.0);
    freshness_timeout_ = declare_parameter<double>("freshness_timeout_sec", 0.5);
    route_heading_ = declare_parameter<double>("route_heading_rad", 0.06241881);
    target_speed_ = declare_parameter<double>("target_speed", config_.target_motion.speed);
    target_pattern_length_ = declare_parameter<double>(
        "target_pattern_length", config_.target_motion.longitudinal_span);
    target_pattern_width_ = declare_parameter<double>(
        "target_pattern_width", config_.target_motion.lateral_span);
    target_sample_count_ = declare_parameter<int>(
        "target_sample_count", config_.target_motion.sample_count);
    target_start_sample_index_ = declare_parameter<int>(
        "target_start_sample_index", config_.target_start_sample_index);
    target_direction_ = declare_parameter<int>(
        "target_direction", config_.target_motion.direction);
    target_waypoint_radius_ = declare_parameter<double>(
        "target_waypoint_radius", config_.target_motion.waypoint_radius);
    target_heading_gain_ = declare_parameter<double>(
        "target_heading_gain", config_.target_motion.heading_gain);
    target_max_yaw_rate_ = declare_parameter<double>(
        "target_max_yaw_rate", config_.target_motion.max_yaw_rate);
    target_minimum_alignment_ = declare_parameter<double>(
        "target_minimum_alignment", config_.target_motion.minimum_alignment);
    target_center_x_ = declare_parameter<double>(
        "target_center_x", std::numeric_limits<double>::quiet_NaN());
    target_center_y_ = declare_parameter<double>(
        "target_center_y", std::numeric_limits<double>::quiet_NaN());
    target_center_z_ = declare_parameter<double>(
        "target_center_z", std::numeric_limits<double>::quiet_NaN());
    uav_velocity_limit_ = declare_parameter<double>("uav_velocity_limit", 3.0);
    cbf_enabled_ = declare_parameter<bool>("cbf_enabled", false);
    cbf_method_name_ = declare_parameter<std::string>("cbf_method", "mestres");
    cbf_obstacle_source_ = declare_parameter<std::string>("cbf_obstacle_source", "none");
    truth_obstacle_fixture_ = declare_parameter<bool>("truth_obstacle_fixture", false);
    cbf_uncertainty_radius_ = declare_parameter<double>("uncertainty_radius", 0.0);
    cbf_config_.k1 = declare_parameter<double>("cbf_k1", cbf_config_.k1);
    cbf_config_.k2 = declare_parameter<double>("cbf_k2", cbf_config_.k2);
    cbf_config_.alpha = declare_parameter<double>("cbf_alpha", cbf_config_.alpha);
    cbf_config_.uav_radius = declare_parameter<double>("cbf_uav_radius", cbf_config_.uav_radius);
    cbf_config_.ugv_radius = declare_parameter<double>("cbf_ugv_radius", cbf_config_.ugv_radius);
    cbf_config_.obstacle_margin = declare_parameter<double>("cbf_obstacle_margin", cbf_config_.obstacle_margin);
    cbf_config_.uav_acceleration_limit = declare_parameter<double>("cbf_uav_acceleration_limit", cbf_config_.uav_acceleration_limit);
    cbf_config_.ugv_acceleration_limit = declare_parameter<double>("cbf_ugv_acceleration_limit", cbf_config_.ugv_acceleration_limit);
    cbf_config_.uav_velocity_limit = declare_parameter<double>("cbf_uav_velocity_limit", uav_velocity_limit_);
    cbf_config_.ugv_speed_limit = declare_parameter<double>("cbf_ugv_speed_limit", cbf_config_.ugv_speed_limit);
    cbf_config_.ugv_yaw_rate_limit = declare_parameter<double>("cbf_ugv_yaw_rate_limit", cbf_config_.ugv_yaw_rate_limit);
    cbf_config_.lookahead_distance = declare_parameter<double>("cbf_lookahead_distance", 0.1);
    cbf_config_.uav_altitude_floor = declare_parameter<double>(
        "cbf_uav_altitude_floor", config_.target_ground_z + 2.0);
    uav_altitude_ceiling_ = declare_parameter<double>(
        "uav_altitude_ceiling", config_.target_ground_z - 7.0);
    cbf_config_.solver_eps_abs = declare_parameter<double>("cbf_solver_eps_abs", cbf_config_.solver_eps_abs);
    cbf_config_.solver_eps_rel = declare_parameter<double>("cbf_solver_eps_rel", cbf_config_.solver_eps_rel);
    cbf_config_.solver_max_iter = declare_parameter<int>("cbf_solver_max_iter", cbf_config_.solver_max_iter);
    cbf_config_.distributed_rounds = declare_parameter<int>("cbf_projection_rounds", cbf_config_.distributed_rounds);
    cbf_config_.distributed_tolerance = declare_parameter<double>("cbf_projection_tolerance", cbf_config_.distributed_tolerance);
    cbf_obstacle_stale_after_ = declare_parameter<double>("cbf_obstacle_stale_after", 1.3);
    cbf_agent_deadline_ms_ = declare_parameter<double>("cbf_agent_deadline_ms", 5.0);
    actuation_profile_ = declare_parameter<std::string>("actuation_profile", "current_ros");
    if (actuation_profile_ != "current_ros" && actuation_profile_ != "python_cbf")
      throw std::invalid_argument("actuation_profile must be current_ros or python_cbf");
    if (cbf_method_name_ != "mestres" && cbf_method_name_ != "wang")
      throw std::invalid_argument("cbf_method must be mestres or wang");
    if (cbf_obstacle_source_ != "none" && cbf_obstacle_source_ != "truth" &&
        cbf_obstacle_source_ != "perception")
      throw std::invalid_argument("cbf_obstacle_source must be none, truth, or perception");
    if (cbf_obstacle_source_ == "truth" && !truth_obstacle_fixture_)
      throw std::invalid_argument(
          "cbf_obstacle_source=truth requires truth_obstacle_fixture=true; "
          "RuralAustralia parity uses none");
    if (!std::isfinite(cbf_uncertainty_radius_) || !std::isfinite(route_heading_) ||
        !std::isfinite(target_speed_) || !std::isfinite(target_pattern_length_) ||
        !std::isfinite(target_pattern_width_) || !std::isfinite(target_waypoint_radius_) ||
        !std::isfinite(target_heading_gain_) || !std::isfinite(target_max_yaw_rate_) ||
        !std::isfinite(target_minimum_alignment_) ||
        !std::isfinite(uav_velocity_limit_) ||
        !std::isfinite(cbf_config_.k1) || !std::isfinite(cbf_config_.k2) ||
        !std::isfinite(cbf_config_.alpha) || !std::isfinite(cbf_config_.uav_radius) ||
        !std::isfinite(cbf_config_.ugv_radius) || !std::isfinite(cbf_config_.obstacle_margin) ||
        !std::isfinite(cbf_config_.uav_acceleration_limit) ||
        !std::isfinite(cbf_config_.ugv_acceleration_limit) ||
        !std::isfinite(cbf_config_.uav_velocity_limit) ||
        !std::isfinite(cbf_config_.ugv_speed_limit) ||
        !std::isfinite(cbf_config_.ugv_yaw_rate_limit) ||
        !std::isfinite(cbf_config_.lookahead_distance) ||
        !std::isfinite(cbf_config_.uav_altitude_floor) || !std::isfinite(uav_altitude_ceiling_) ||
        !std::isfinite(cbf_config_.solver_eps_abs) || !std::isfinite(cbf_config_.solver_eps_rel) ||
        !std::isfinite(cbf_config_.distributed_tolerance) ||
        !std::isfinite(cbf_obstacle_stale_after_) || !std::isfinite(cbf_agent_deadline_ms_) ||
        cbf_config_.k1 < 0.0 || cbf_config_.k2 < 0.0 || cbf_config_.alpha < 0.0 ||
        cbf_config_.uav_radius < 0.0 || cbf_config_.ugv_radius < 0.0 ||
        cbf_config_.uav_acceleration_limit <= 0.0 || cbf_config_.ugv_acceleration_limit <= 0.0 ||
        cbf_config_.uav_velocity_limit <= 0.0 || cbf_config_.ugv_speed_limit <= 0.0 ||
        cbf_config_.ugv_yaw_rate_limit <= 0.0 || cbf_config_.lookahead_distance <= 0.0 ||
        cbf_config_.solver_eps_abs <= 0.0 || cbf_config_.solver_eps_rel <= 0.0 ||
        cbf_config_.solver_max_iter <= 0 || cbf_config_.distributed_rounds < 0 ||
        cbf_config_.distributed_tolerance < 0.0 || cbf_obstacle_stale_after_ <= 0.0 ||
        cbf_agent_deadline_ms_ < 0.0 || target_speed_ <= 0.0 ||
        target_pattern_length_ <= 0.0 || target_pattern_width_ <= 0.0 ||
        target_sample_count_ < 8 || target_waypoint_radius_ < 0.0 ||
        target_heading_gain_ < 0.0 || target_max_yaw_rate_ <= 0.0 ||
        target_minimum_alignment_ < 0.0 || target_minimum_alignment_ > 1.0 ||
        (target_direction_ != 1 && target_direction_ != -1))
      throw std::invalid_argument("invalid CBF limit, gain, or projection parameter");
    const bool any_target_center = std::isfinite(target_center_x_) ||
                                   std::isfinite(target_center_y_) ||
                                   std::isfinite(target_center_z_);
    const bool complete_target_center = std::isfinite(target_center_x_) &&
                                        std::isfinite(target_center_y_) &&
                                        std::isfinite(target_center_z_);
    if (any_target_center && !complete_target_center)
      throw std::invalid_argument("target_center_x/y/z must be all finite or all omitted");
    cbf_config_.method = cbf_method_name_ == "wang" ? hercules_cbf::Method::kWang
                                                     : hercules_cbf::Method::kMestres;
    cbf_config_.uncertainty_radius = cbf_uncertainty_radius_;
    cbf_diagnostics_publisher_ = create_publisher<hercules_interfaces::msg::CBFDiagnosticsArray>(
        "/hercules_mission/cbf_diagnostics", 10);
    log_path_ = declare_parameter<std::string>(
        "log_path", "ros2/validation/rural_nominal/artifacts/mission.jsonl");
    distributed_target_source_ = std::make_unique<DistributedTargetSource>(config_.tracking.window_seconds);

    for (const char* id : kAll) {
      state_subscriptions_.push_back(
          create_subscription<hercules_interfaces::msg::GroundTruthState>(
              std::string("/hercules_mission/ground_truth/") + id, 10,
              [this, id](hercules_interfaces::msg::GroundTruthState::ConstSharedPtr msg) {
                if (msg && validCanonicalState(*msg, id)) {
                  states_[id] = *msg;
                  receipts_[id] = Clock::now();
                } else {
                  // An invalid/nonfinite sample must invalidate the cached
                  // state immediately; otherwise an old valid sample could
                  // keep the mission gate open until its age timeout.
                  states_.erase(id);
                  receipts_.erase(id);
                }
              }));
      origin_subscriptions_.push_back(
          create_subscription<geometry_msgs::msg::PointStamped>(
              std::string("/hercules_mission/calibrated_origin/") + id,
              rclcpp::QoS(1).transient_local().reliable(),
              [this, id](geometry_msgs::msg::PointStamped::ConstSharedPtr msg) {
                const Eigen::Vector3d origin(msg->point.x, msg->point.y, msg->point.z);
                if (origin.allFinite()) {
                  origins_[id] = origin;
                } else {
                  origins_.erase(id);
                }
              }));
    }
    for (const char* id : kControlled) {
      obstacle_caches_[id] = std::make_unique<hercules_cbf_ros::ObstacleCache>(
          cbf_obstacle_stale_after_, id, "wall");
      obstacle_subscriptions_.push_back(create_subscription<hercules_interfaces::msg::ObstacleProxyArray>(
          std::string("/hercules_mission/obstacles/") + id, 10,
          [this, id](hercules_interfaces::msg::ObstacleProxyArray::ConstSharedPtr message) {
            obstacle_caches_.at(id)->receive(*message);
          }));
    }
    collision_subscription_ = create_subscription<hercules_interfaces::msg::MissionCollision>(
        "/hercules_mission/collisions", 20,
        [this](hercules_interfaces::msg::MissionCollision::ConstSharedPtr message) {
          collisions_[message->vehicle_name] = *message;
        });
    if (target_source_name_ == "distributed_tracking") {
      epoch_publisher_ = create_publisher<hercules_interfaces::msg::TrackingEpoch>(
          "/hercules_tracking/epoch", rclcpp::QoS(20).reliable());
      for (const char* id : kControlled) {
        estimate_subscriptions_.push_back(
            create_subscription<hercules_interfaces::msg::TargetEstimate>(
                std::string("/hercules_tracking/") + id + "/Target1/estimate", 20,
                [this, id](hercules_interfaces::msg::TargetEstimate::ConstSharedPtr message) {
                  local_estimates_[id] = *message;
                  estimate_receipts_[id] = Clock::now();
                }));
        tracking_diagnostic_subscriptions_.push_back(
            create_subscription<hercules_interfaces::msg::TrackingDiagnostics>(
                std::string("/hercules_tracking/") + id + "/Target1/diagnostics", 20,
                [this, id](hercules_interfaces::msg::TrackingDiagnostics::ConstSharedPtr message) {
                  tracking_diagnostics_[id] = *message;
                }));
        measurement_subscriptions_.push_back(
            create_subscription<hercules_interfaces::msg::TargetMeasurement>(
                std::string("/hercules_tracking/") + id + "/Target1/measurement", 20,
                [this, id](hercules_interfaces::msg::TargetMeasurement::ConstSharedPtr message) {
                  tracking_measurements_[id] = *message;
                }));
      }
      observation_diagnostics_subscription_ =
          create_subscription<hercules_interfaces::msg::TargetObservationDiagnostics>(
              "/hercules_tracking/observation_diagnostics", 10,
              [this](hercules_interfaces::msg::TargetObservationDiagnostics::ConstSharedPtr message) {
                observation_diagnostics_ = *message;
              });
    }
    for (const char* id : kDrones) {
      uav_publishers_[id] = create_publisher<airsim_interfaces::msg::VelCmd>(
          std::string("/hercules_drone/") + id + "/vel_cmd_world_frame", 1);
      land_clients_[id] = create_client<airsim_interfaces::srv::Land>(
          std::string("/hercules_drone/") + id + "/land");
    }
    for (const char* id : kUgvs) {
      car_publishers_[id] = create_publisher<airsim_interfaces::msg::CarControls>(
          std::string("/hercules_ugv/") + id + "/car_cmd", 1);
    }
    car_publishers_["Target1"] =
        create_publisher<airsim_interfaces::msg::CarControls>(
            "/hercules_ugv/Target1/car_cmd", 1);

    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(config_.control_dt)),
        [this]() { tick(); });
    RCLCPP_INFO(get_logger(),
                "Rural nominal mission waiting for nine calibrated states (%s)",
                dry_run_ ? "dry-run: actuation disabled" : "live actuation");
  }

 private:
  bool ready() const {
    std::vector<std::string> required;
    std::set<std::string> valid_states;
    std::set<std::string> calibrated_origins;
    std::map<std::string, double> ages;
    for (const char* id : kAll) {
      required.emplace_back(id);
      if (states_.count(id)) valid_states.insert(id);
      if (origins_.count(id)) calibrated_origins.insert(id);
      const auto receipt = receipts_.find(id);
      if (receipt != receipts_.end()) ages[id] = seconds(receipt->second);
    }
    return missionReady(required, valid_states, calibrated_origins, ages,
                        freshness_timeout_);
  }

  void tick() {
    if (finished_) return;
    if (!running_) {
      if (!ready()) {
        if (seconds(started_at_) > startup_timeout_) {
          fail("startup timed out before all nine calibrated states were fresh");
        }
        return;
      }
      startMission();
      return;
    }
    if (!ready()) {
      fail("state became stale or unavailable; actuation suppressed");
      return;
    }
    if (seconds(mission_started_at_) >= duration_) {
      finishMission();
      return;
    }
    step();
  }

  void startMission() {
    const auto target = agentState(states_.at("Target1"));
    auto motion = config_.target_motion;
    const bool has_target_center = std::isfinite(target_center_x_) &&
                                   std::isfinite(target_center_y_) &&
                                   std::isfinite(target_center_z_);
    if (has_target_center) {
      motion.center = Eigen::Vector3d(target_center_x_, target_center_y_, target_center_z_);
    } else {
      motion.center = target.position;
      motion.center.z() = config_.target_ground_z;
    }
    motion.route_heading = route_heading_;
    motion.speed = target_speed_;
    motion.longitudinal_span = target_pattern_length_;
    motion.lateral_span = target_pattern_width_;
    motion.sample_count = target_sample_count_;
    motion.waypoint_radius = target_waypoint_radius_;
    motion.heading_gain = target_heading_gain_;
    motion.max_yaw_rate = target_max_yaw_rate_;
    motion.minimum_alignment = target_minimum_alignment_;
    motion.direction = target_direction_;
    target_controller_ =
        std::make_unique<hercules_mission_core::FigureEightTargetController>(motion);
    target_controller_->setIndex(target_start_sample_index_);
    if (!has_target_center) target_controller_->placeStartAt(target.position);
    std::filesystem::create_directories(
        std::filesystem::path(log_path_).parent_path());
    log_.open(log_path_);
    if (!log_) {
      fail("could not open JSONL log: " + log_path_);
      return;
    }
    running_ = true;
    mission_started_at_ = Clock::now();
    RCLCPP_INFO(get_logger(), "preflight passed; mission started");
  }

  void publishTrackingEpoch(double mission_time) {
    hercules_tracking_ros::PositionMap positions;
    std::vector<std::string> ids;
    for (const char* id : kControlled) {
      ids.emplace_back(id);
      positions[id] = agentState(states_.at(id)).position;
    }
    current_adjacency_ = hercules_tracking_ros::buildNeighborGraph(
        positions, config_.communication_range);
    hercules_interfaces::msg::TrackingEpoch message;
    message.epoch_id = ++tracking_epoch_;
    message.stamp = hercules_tracking_ros::timeMessage(mission_time);
    message.target_id = "Target1";
    message.agent_ids = ids;
    message.adjacency = hercules_tracking_ros::flattenAdjacency(ids, current_adjacency_);
    epoch_publisher_->publish(message);
  }

  ControlTargetEstimate controlEstimate(
      const std::string& id, const hercules_mission_core::AgentState& target,
      double mission_time) const {
    ControlTargetEstimate result;
    result.timestamp = mission_time;
    if (target_source_name_ == "truth") {
      result.estimate = target_source_.estimate(target);
      return result;
    }
    const auto found = local_estimates_.find(id);
    const auto receipt = estimate_receipts_.find(id);
    if (found == local_estimates_.end() || receipt == estimate_receipts_.end()) {
      result.estimate = distributed_target_source_->estimate(std::nullopt, mission_time);
      return result;
    }
    TimestampedTargetEstimate input;
    input.estimate.position = {found->second.position[0], found->second.position[1]};
    input.estimate.velocity = {found->second.velocity[0], found->second.velocity[1]};
    input.estimate.active = found->second.active;
    input.estimator_timestamp = hercules_tracking_ros::timeSeconds(found->second.stamp);
    input.receipt_age = seconds(receipt->second);
    result.estimate = distributed_target_source_->estimate(input, mission_time);
    result.from_distributed = true;
    if (result.estimate.active) {
      // TargetTrack::predicted() reports the XY covariance with only the
      // position block of Q(dt) added.  It intentionally leaves the wire
      // state covariance untouched, so reconstruct that reported covariance
      // here for the target CBF proxy without changing tracker mathematics.
      const double delta = std::max(0.0, mission_time - input.estimator_timestamp);
      const auto process = hercules_tracking::constantAccelerationProcessNoise(
          delta, config_.tracking.process_noise).topLeftCorner<2, 2>();
      const auto& covariance = found->second.state_covariance;
      const double c00 = covariance[0];
      const double c01 = 0.5 * (covariance[1] + covariance[4]);
      const double c11 = covariance[5];
      if (std::isfinite(c00) && std::isfinite(c01) && std::isfinite(c11)) {
        result.covariance << c00, c01, c01, c11;
        result.covariance += process;
        if (!result.covariance.allFinite()) result.covariance.setZero();
      }
    }
    return result;
  }

  void step() {
    const auto tick_time = Clock::now();
    if (last_step_time_) {
      update_gaps_.push_back(std::chrono::duration<double>(tick_time - *last_step_time_).count());
    }
    last_step_time_ = tick_time;
    const auto target = agentState(states_.at("Target1"));
    const double mission_time = static_cast<double>(step_) * config_.control_dt;
    if (target_source_name_ == "distributed_tracking" &&
        mission_time + 1e-9 >= next_tracking_time_) {
      publishTrackingEpoch(mission_time);
      next_tracking_time_ += 1.0 / config_.tracking_rate;
    }
    const Eigen::Vector2d target_command =
        target_controller_->update(target.position, target.yaw, config_.control_dt);
    const double measured_target_speed = target.velocity.head<2>().norm();
    const CarCommand target_car = ugvCarCommand(
        target_command.x(), target_command.y(), measured_target_speed,
        config_.target_motion.max_yaw_rate, true);

    std::map<std::string, Eigen::Vector3d> commands;
    std::map<std::string, Eigen::Vector3d> desired_slots;
    std::map<std::string, double> slot_errors;
    std::map<std::string, hercules_mission_core::TargetEstimate> control_estimates;
    double step_max_uav = 0.0;
    double step_max_ugv = 0.0;
    double step_min_radius = 1e9;
    double step_max_radius = 0.0;
    hercules_interfaces::msg::CBFDiagnosticsArray cbf_diagnostics;
    cbf_diagnostics.header.frame_id = "airsim_world_ned";
    cbf_diagnostics.header.stamp = now();
    cbf_diagnostics.step_id = step_;

    for (const char* id : kDrones) {
      const auto agent = agentState(states_.at(id));
      const auto target_context = controlEstimate(id, target, mission_time);
      const auto& target_estimate = target_context.estimate;
      control_estimates[id] = target_estimate;
      const auto acceleration = controller_.targetNominalControl(
          agent, target_estimate, route_heading_, config_.target_ground_z,
          config_.target_ugv_circumradius);
      const auto nominal_velocity = target_estimate.active
          ? (actuation_profile_ == "python_cbf"
                 ? pythonCbfUavVelocityCommand(agent.velocity, acceleration, config_.control_dt,
                                               uav_velocity_limit_, uav_altitude_ceiling_,
                                               agent.position.z())
                 : uavVelocityCommand(agent.velocity, acceleration, config_.control_dt,
                                      uav_velocity_limit_))
          : Eigen::Vector3d::Zero();
      Eigen::Vector3d velocity = nominal_velocity;
      if (cbf_enabled_) {
        const auto filtered = runCbf(id, agent, acceleration, target_context);
        velocity = actuation_profile_ == "python_cbf"
            ? pythonCbfUavVelocityCommand(agent.velocity, filtered.result.safe_control,
                                          config_.control_dt, uav_velocity_limit_,
                                          uav_altitude_ceiling_, agent.position.z())
            : uavVelocityCommand(agent.velocity, filtered.result.safe_control,
                                 config_.control_dt, uav_velocity_limit_);
        cbf_diagnostics.entries.push_back(filtered.diagnostics);
      } else {
        cbf_diagnostics.entries.push_back(disabledDiagnostics(id, agent, acceleration));
      }
      commands[id] = velocity;
      const Eigen::Vector3d estimate_position(target_estimate.position.x(),
                                              target_estimate.position.y(),
                                              config_.target_ground_z);
      const Eigen::Vector3d estimate_velocity(target_estimate.velocity.x(),
                                              target_estimate.velocity.y(), 0.0);
      const auto slot = hercules_mission_core::targetCenteredSlot(
          id, hercules_mission_core::VehicleType::kDrone, estimate_position,
          estimate_velocity, route_heading_, config_.uav_altitude,
          config_.target_ground_z, config_.target_ugv_circumradius);
      desired_slots[id] = slot.position;
      slot_errors[id] = (agent.position - slot.position).norm();
      step_max_uav = std::max(step_max_uav, slot_errors[id]);
      if (!dry_run_ && enable_formation_) publishUav(id, velocity);
    }
    for (const char* id : kUgvs) {
      const auto agent = agentState(states_.at(id));
      const auto target_context = controlEstimate(id, target, mission_time);
      const auto& target_estimate = target_context.estimate;
      control_estimates[id] = target_estimate;
      const auto command = controller_.targetNominalUnicycleControl(
          agent, target_estimate, route_heading_, config_.target_ground_z,
          config_.target_ugv_circumradius);
      Eigen::Vector2d model_command(command.x(), command.y());
      if (cbf_enabled_) {
        const auto filtered = runCbf(id, agent, model_command, target_context);
        model_command = filtered.result.safe_control.head<2>();
        cbf_diagnostics.entries.push_back(filtered.diagnostics);
      } else {
        cbf_diagnostics.entries.push_back(disabledDiagnostics(id, agent, model_command));
      }
      commands[id] = {model_command.x(), model_command.y(), 0.0};
      const Eigen::Vector3d estimate_position(target_estimate.position.x(),
                                              target_estimate.position.y(),
                                              config_.target_ground_z);
      const Eigen::Vector3d estimate_velocity(target_estimate.velocity.x(),
                                              target_estimate.velocity.y(), 0.0);
      const auto slot = hercules_mission_core::targetCenteredSlot(
          id, hercules_mission_core::VehicleType::kUgv, estimate_position,
          estimate_velocity, route_heading_, config_.uav_altitude,
          config_.target_ground_z, config_.target_ugv_circumradius);
      desired_slots[id] = slot.position;
      slot_errors[id] =
          (agent.position.head<2>() - slot.position.head<2>()).norm();
      step_max_ugv = std::max(step_max_ugv, slot_errors[id]);
      const double radius = (agent.position.head<2>() - target.position.head<2>()).norm();
      step_min_radius = std::min(step_min_radius, radius);
      step_max_radius = std::max(step_max_radius, radius);
      if (!dry_run_ && enable_formation_ && (target_estimate.active || cbf_enabled_)) {
        publishCar(id, actuation_profile_ == "python_cbf"
            ? pythonCbfUgvCarCommand(model_command.x(), model_command.y(),
                                     agent.velocity.head<2>().norm(), cbf_config_.ugv_yaw_rate_limit,
                                     false, cbf_config_.ugv_speed_limit)
            : ugvCarCommand(model_command.x(), model_command.y(),
                            agent.velocity.head<2>().norm(), config_.formation.ugv_max_yaw_rate));
      } else if (!dry_run_ && enable_formation_) {
        publishCar(id, stoppedCarCommand());
      }
    }
    commands["Target1"] = {target_command.x(), target_command.y(), 0.0};
    if (!dry_run_ && enable_target_) publishCar("Target1", target_car);

    max_uav_error_ = std::max(max_uav_error_, step_max_uav);
    max_ugv_error_ = std::max(max_ugv_error_, step_max_ugv);
    min_ugv_radius_ = std::min(min_ugv_radius_, step_min_radius);
    max_ugv_radius_ = std::max(max_ugv_radius_, step_max_radius);
    writeRecord(target, target_command, commands, desired_slots, slot_errors,
                control_estimates, step_max_uav, step_max_ugv, cbf_diagnostics);
    cbf_diagnostics_publisher_->publish(cbf_diagnostics);
    ++step_;
  }

  template <typename Control>
  hercules_cbf_ros::AdapterResult runCbf(
      const std::string& id, const hercules_mission_core::AgentState& agent,
      const Control& nominal, const ControlTargetEstimate& target_context) {
    const auto& target_estimate = target_context.estimate;
    std::vector<hercules_interfaces::msg::GroundTruthState> neighbors;
    const auto& ego_message = states_.at(id);
    for (const char* other : kControlled) {
      if (std::string(other) == id) continue;
      const auto found = states_.find(other);
      if (found == states_.end()) continue;
      const auto other_state = agentState(found->second);
      if (other_state.vehicle_type != agent.vehicle_type) continue;
      if ((other_state.position - agent.position).norm() <= config_.communication_range)
        neighbors.push_back(found->second);
    }
    std::vector<hercules_interfaces::msg::ObstacleProxy> obstacles;
    std::optional<hercules_cbf_ros::ObstacleSnapshot> static_snapshot;
    bool static_sensor_valid = cbf_obstacle_source_ != "perception";
    if (cbf_obstacle_source_ == "perception") {
      static_snapshot = obstacle_caches_.at(id)->snapshot();
      // Preserve the last successful proxy set even after it becomes stale;
      // its validity is carried separately and controls the sensor gate.
      if (static_snapshot) obstacles = static_snapshot->message.proxies;
      static_sensor_valid = static_snapshot && static_snapshot->valid;
      if (static_snapshot) {
        // The Python mission replays static perception with zero obstacle
        // velocity and adds a bounded age margin only while the capture is
        // still fresh.  Keep stale geometry for the UGV target-proxy path,
        // but never let an expired snapshot acquire extra authority.
        const double age = std::max(0.0, static_snapshot->age_seconds);
        const double speed_limit = agent.vehicle_type ==
                hercules_mission_core::VehicleType::kUgv
            ? cbf_config_.ugv_speed_limit : cbf_config_.uav_velocity_limit;
        const double acceleration_limit = agent.vehicle_type ==
                hercules_mission_core::VehicleType::kUgv
            ? cbf_config_.ugv_acceleration_limit
            : cbf_config_.uav_acceleration_limit;
        const double age_margin = static_snapshot->valid
            ? std::min(speed_limit * age +
                           0.5 * acceleration_limit * age * age,
                       0.25)  // RuralAustralia max_proxy_radius is 1 m.
            : 0.0;
        std::vector<hercules_interfaces::msg::ObstacleProxy> filtered;
        filtered.reserve(obstacles.size());
        for (auto& obstacle : obstacles) {
          obstacle.has_velocity = false;
          obstacle.velocity = {0.0, 0.0, 0.0};
          obstacle.radius += age_margin;
          const bool explicit_proxy = obstacle.source.rfind("truth", 0) == 0 ||
                                      obstacle.source == "target_tracking";
          bool body_proxy = false;
          if (!explicit_proxy) {
            const Eigen::Vector3d center(obstacle.center[0], obstacle.center[1], obstacle.center[2]);
            for (const char* other : kControlled) {
              const auto state = states_.find(other);
              if (state == states_.end()) continue;
              const double body_radius = expectedVehicleType(other) == std::string("ugv")
                  ? cbf_config_.ugv_radius : cbf_config_.uav_radius;
              const Eigen::Vector3d position(state->second.position[0], state->second.position[1],
                                             state->second.position[2]);
              if ((center - position).norm() <= body_radius + 0.5) {
                body_proxy = true;
                break;
              }
            }
          }
          if (!body_proxy) filtered.push_back(obstacle);
        }
        obstacles.swap(filtered);
      }
    }
    if (agent.vehicle_type == hercules_mission_core::VehicleType::kUgv && target_estimate.active) {
      hercules_interfaces::msg::TargetEstimate target_message;
      const auto estimate_message = local_estimates_.find(id);
      if (target_source_name_ == "distributed_tracking" && estimate_message != local_estimates_.end())
        target_message = estimate_message->second;
      target_message.active = true;
      target_message.position = {target_estimate.position.x(), target_estimate.position.y()};
      target_message.velocity = {target_estimate.velocity.x(), target_estimate.velocity.y()};
      target_message.stamp = hercules_tracking_ros::timeMessage(target_context.timestamp);
      target_message.state_covariance[0] = target_context.covariance(0, 0);
      target_message.state_covariance[1] = target_context.covariance(0, 1);
      target_message.state_covariance[4] = target_context.covariance(1, 0);
      target_message.state_covariance[5] = target_context.covariance(1, 1);
      double target_z = config_.target_ground_z;
      const auto target_state = states_.find("Target1");
      if (target_state != states_.end() && target_state->second.position.size() >= 3)
        target_z = target_state->second.position[2];
      hercules_cbf_ros::appendTargetProxy(target_message, id, cbf_config_.ugv_radius,
                                          obstacles, target_z);
    }
    const bool target_proxy_active =
        agent.vehicle_type == hercules_mission_core::VehicleType::kUgv &&
        target_estimate.active;
    const bool sensor_valid = static_sensor_valid || target_proxy_active;
    hercules_cbf::CBFConfig config = cbf_config_;
    if (agent.vehicle_type == hercules_mission_core::VehicleType::kUgv)
      config.method = hercules_cbf::Method::kMestres;
    Eigen::VectorXd input;
    if constexpr (std::is_same_v<Control, Eigen::Vector3d>) input = nominal;
    else input = nominal;
    hercules_interfaces::msg::GroundTruthState ego = ego_message;
    const auto selected = cbf_method_name_;
    const auto effective = config.method == hercules_cbf::Method::kWang ? "wang" : "mestres";
    auto output = hercules_cbf_ros::filterRequest(ego, input, neighbors, obstacles, config,
                                                  sensor_valid, selected, effective);
    output.diagnostics.target_active = target_estimate.active;
    output.diagnostics.target_timestamp = target_context.timestamp;
    output.diagnostics.deadline_miss = output.result.solve_time_ms > cbf_agent_deadline_ms_;
    if (cbf_obstacle_source_ == "perception") {
      output.diagnostics.static_obstacles_valid = static_snapshot && static_snapshot->valid;
      output.diagnostics.static_obstacles_age = static_snapshot ? static_snapshot->age_seconds : 0.0;
    } else {
      output.diagnostics.static_obstacles_valid = true;
      output.diagnostics.static_obstacles_age = 0.0;
    }
    const auto receipt = estimate_receipts_.find(id);
    output.diagnostics.target_age = receipt == estimate_receipts_.end() ? 0.0 : seconds(receipt->second);
    return output;
  }

  hercules_interfaces::msg::CBFDiagnostics disabledDiagnostics(
      const std::string& id, const hercules_mission_core::AgentState& agent,
      const Eigen::VectorXd& nominal) const {
    hercules_interfaces::msg::CBFDiagnostics diagnostic;
    diagnostic.header.frame_id = "airsim_world_ned";
    diagnostic.agent_id = id;
    diagnostic.enabled = false;
    diagnostic.selected_method = cbf_method_name_;
    diagnostic.effective_method = agent.vehicle_type == hercules_mission_core::VehicleType::kUgv
        ? "mestres" : cbf_method_name_;
    diagnostic.control_kind = agent.vehicle_type == hercules_mission_core::VehicleType::kUgv
        ? "unicycle" : "double_integrator";
    diagnostic.control_dimension = static_cast<uint32_t>(nominal.size());
    diagnostic.nominal_control.assign(nominal.data(), nominal.data() + nominal.size());
    diagnostic.safe_control = diagnostic.nominal_control;
    diagnostic.success = false;
    diagnostic.fallback = false;
    diagnostic.solver_status = "disabled";
    diagnostic.solver_iterations = 0;
    diagnostic.final_feasible = true;
    diagnostic.static_obstacles_valid = cbf_obstacle_source_ != "perception";
    diagnostic.deadline_miss = false;
    return diagnostic;
  }

  void publishUav(const std::string& id, const Eigen::Vector3d& velocity) {
    airsim_interfaces::msg::VelCmd message;
    message.twist.linear.x = velocity.x();
    message.twist.linear.y = velocity.y();
    message.twist.linear.z = velocity.z();
    uav_publishers_.at(id)->publish(message);
  }

  void publishCar(const std::string& id, const CarCommand& command) {
    car_publishers_.at(id)->publish(carMessage(command));
  }

  void safeStop() {
    if (dry_run_) return;
    if (enable_formation_) {
      for (const char* id : kDrones) publishUav(id, Eigen::Vector3d::Zero());
      for (const char* id : kUgvs) publishCar(id, stoppedCarCommand());
    }
    if (enable_target_) publishCar("Target1", stoppedCarCommand());
  }

  void requestLanding() {
    if (dry_run_ || !enable_formation_) return;
    for (const char* id : kDrones) {
      if (!land_clients_.at(id)->service_is_ready()) {
        RCLCPP_ERROR(get_logger(), "%s land service unavailable", id);
        continue;
      }
      auto request = std::make_shared<airsim_interfaces::srv::Land::Request>();
      request->wait_on_last_task = false;
      land_clients_.at(id)->async_send_request(request);
    }
  }

  void finishMission() {
    safeStop();
    requestLanding();
    finished_ = true;
    timer_->cancel();
    log_.close();
    double mean_hz = 0.0;
    double max_gap = 0.0;
    if (!update_gaps_.empty()) {
      double sum = 0.0;
      for (double gap : update_gaps_) { sum += gap; max_gap = std::max(max_gap, gap); }
      mean_hz = static_cast<double>(update_gaps_.size()) / sum;
    }
    RCLCPP_INFO(get_logger(),
                "mission complete: steps=%zu max_uav_slot_error=%.3f "
                "max_ugv_slot_error=%.3f ugv_radius=[%.3f, %.3f] "
                "mean_frequency=%.3fHz max_gap=%.4fs log=%s",
                step_, max_uav_error_, max_ugv_error_, min_ugv_radius_,
                max_ugv_radius_, mean_hz, max_gap, log_path_.c_str());
  }

  void fail(const std::string& reason) {
    safeStop();
    finished_ = true;
    if (timer_) timer_->cancel();
    if (log_) log_.close();
    RCLCPP_FATAL(get_logger(), "%s", reason.c_str());
  }

  void writeVector(std::ostream& out, const Eigen::Vector3d& value) {
    out << '[' << value.x() << ',' << value.y() << ',' << value.z() << ']';
  }

  void writeFinite(std::ostream& out, double value) {
    out << (std::isfinite(value) ? value : 0.0);
  }

  void writeNullable(std::ostream& out, double value) {
    if (std::isfinite(value)) out << value;
    else out << "null";
  }

  void writeRecord(const hercules_mission_core::AgentState& target,
                   const Eigen::Vector2d& target_command,
                   const std::map<std::string, Eigen::Vector3d>& commands,
                   const std::map<std::string, Eigen::Vector3d>& desired_slots,
                   const std::map<std::string, double>& slot_errors,
                   const std::map<std::string, hercules_mission_core::TargetEstimate>& control_estimates,
                   double uav_error, double ugv_error,
                   const hercules_interfaces::msg::CBFDiagnosticsArray& cbf_diagnostics) {
    log_ << std::setprecision(15) << "{\"step\":" << step_
         << ",\"dt\":" << config_.control_dt
         << ",\"timestamp\":" << seconds(mission_started_at_)
         << ",\"wall_timestamp\":" << now().seconds()
         << ",\"vehicle_types\":{";
    for (std::size_t i = 0; i < std::size(kAll); ++i) {
      if (i) log_ << ',';
      const std::string id = kAll[i];
      log_ << '\"' << id << "\":\""
           << (id.rfind("Drone", 0) == 0 || id == "SimpleFlight" ? "drone" : "ugv")
           << '\"';
    }
    log_ << "},\"states\":{";
    for (std::size_t i = 0; i < std::size(kAll); ++i) {
      if (i) log_ << ',';
      const auto state = agentState(states_.at(kAll[i]));
      log_ << '\"' << kAll[i] << "\":{\"position\":";
      writeVector(log_, state.position);
      log_ << ",\"velocity\":";
      writeVector(log_, state.velocity);
      log_ << ",\"yaw\":" << state.yaw << '}';
    }
    log_ << "},\"calibrated_origins\":{";
    for (std::size_t i = 0; i < std::size(kAll); ++i) {
      if (i) log_ << ',';
      log_ << '\"' << kAll[i] << "\":";
      writeVector(log_, origins_.at(kAll[i]));
    }
    log_ << "},\"target_truth\":{\"name\":\"Target1\",\"position\":";
    writeVector(log_, target.position);
    log_ << ",\"velocity\":";
    writeVector(log_, target.velocity);
    log_ << ",\"yaw\":" << target.yaw << ",\"reference\":";
    writeVector(log_, target_controller_->reference());
    log_ << ",\"command\":["
         << target_command.x() << ',' << target_command.y()
         << "],\"phase\":" << target_controller_->phase()
         << ",\"index\":" << target_controller_->index()
         << ",\"collision\":{\"relevant\":false},\"pattern\":{\"type\":\"gerono_figure_eight\",\"center\":";
    writeVector(log_, target_controller_->config().center);
    log_ << ",\"route_heading\":" << route_heading_
         << ",\"longitudinal_span\":" << target_pattern_length_
         << ",\"lateral_span\":" << target_pattern_width_
         << ",\"speed\":" << target_speed_
         << ",\"sample_count\":" << target_sample_count_
         << ",\"direction\":" << target_direction_
         << ",\"waypoint_radius\":" << target_waypoint_radius_
         << ",\"heading_gain\":" << target_heading_gain_
         << ",\"max_yaw_rate\":" << target_max_yaw_rate_
         << ",\"minimum_alignment\":" << target_minimum_alignment_ << "}},";
    log_ << "\"target\":{\"name\":\"Target1\",\"position\":";
    writeVector(log_, target.position);
    log_ << "},\"targets\":{\"Target1\":{\"position\":";
    writeVector(log_, target.position);
    log_ << "}},\"commands\":{";
    std::size_t command_index = 0;
    for (const auto& [id, command] : commands) {
      if (command_index++) log_ << ',';
      log_ << '\"' << id << "\":";
      writeVector(log_, command);
    }
    log_ << "},\"desired_slots\":{";
    std::size_t slot_index = 0;
    for (const auto& [id, slot] : desired_slots) {
      if (slot_index++) log_ << ',';
      log_ << '\"' << id << "\":";
      writeVector(log_, slot);
    }
    log_ << "},\"slot_errors\":{";
    slot_index = 0;
    for (const auto& [id, error] : slot_errors) {
      if (slot_index++) log_ << ',';
      log_ << '\"' << id << "\":" << error;
    }
    log_ << "},\"target_tracking\":{\"enabled\":"
         << (target_source_name_ == "distributed_tracking" ? "true" : "false")
         << ",\"observation_source\":\"" << target_observation_source_
         << "\",\"target_id\":\"Target1\",\"epoch_id\":" << tracking_epoch_
         << ",\"agents\":{";
    std::size_t tracking_index = 0;
    for (const char* id : kControlled) {
      if (tracking_index++) log_ << ',';
      log_ << '\"' << id << "\":{";
      const auto measurement = tracking_measurements_.find(id);
      log_ << "\"measurement\":{";
      if (measurement != tracking_measurements_.end()) {
        const auto& value = measurement->second;
        log_ << "\"target_id\":\"" << value.target_id << "\",\"valid\":"
             << (value.valid ? "true" : "false") << ",\"visible\":"
             << (value.visible ? "true" : "false") << ",\"source\":\""
             << target_observation_source_ << "\",\"source_id\":\"" << value.source_id
             << "\",\"capture_id\":\"" << value.capture_id << "\",\"sensor\":\""
             << value.sensor_id << "\",\"timestamp\":";
        writeFinite(log_, hercules_tracking_ros::timeSeconds(value.stamp));
        log_ << ",\"capture_wall_timestamp\":";
        writeFinite(log_, hercules_tracking_ros::timeSeconds(value.capture_stamp));
        log_ << ",\"ros_receipt_timestamp\":";
        writeFinite(log_, hercules_tracking_ros::timeSeconds(value.receipt_stamp));
        log_ << ",\"position\":[" << value.position[0] << ',' << value.position[1]
             << "],\"covariance\":[[" << value.covariance[0] << ',' << value.covariance[1]
             << "],[" << value.covariance[2] << ',' << value.covariance[3] << "]]";
      }
      log_ << "},\"estimate\":{";
      const auto raw_estimate = local_estimates_.find(id);
      if (raw_estimate != local_estimates_.end()) {
        const auto& value = raw_estimate->second;
        log_ << "\"target_id\":\"" << value.target_id << "\",\"position\":["
             << value.position[0] << ',' << value.position[1] << "],\"velocity\":["
             << value.velocity[0] << ',' << value.velocity[1] << "],\"active\":"
             << (value.active ? "true" : "false") << ",\"iterations\":"
             << value.consensus_iterations << ",\"consensus_residual\":";
        writeFinite(log_, value.consensus_residual);
        log_ << ",\"timestamp\":";
        writeFinite(log_, hercules_tracking_ros::timeSeconds(value.stamp));
        log_ << ",\"covariance\":[[" << value.state_covariance[0] << ','
             << value.state_covariance[1] << "],[" << value.state_covariance[4]
             << ',' << value.state_covariance[5] << "]]";
        log_ << ",\"state_covariance\":[";
        for (std::size_t covariance_index = 0; covariance_index < 16; ++covariance_index) {
          if (covariance_index) log_ << ',';
          log_ << value.state_covariance[covariance_index];
        }
        log_ << ']';
      }
      const auto control = control_estimates.find(id);
      log_ << "},\"active\":"
           << (control != control_estimates.end() && control->second.active ? "true" : "false");
      const auto diagnostic = tracking_diagnostics_.find(id);
      if (diagnostic != tracking_diagnostics_.end()) {
        log_ << ",\"direct_observation\":"
             << (diagnostic->second.direct_observation ? "true" : "false")
             << ",\"epoch_timed_out\":"
             << (diagnostic->second.epoch_timed_out ? "true" : "false");
      }
      log_ << '}';
    }
    uint64_t messages = 0, rejected = 0, handoffs_sent = 0, handoffs_accepted = 0;
    std::size_t active_tracks = 0, direct_observations = 0, timed_out_epochs = 0;
    for (const auto& [id, value] : tracking_diagnostics_) {
      (void)id;
      messages += value.consensus_messages_published;
      rejected += value.rejected_messages;
      handoffs_sent += value.handoffs_sent;
      handoffs_accepted += value.handoffs_accepted;
      active_tracks += value.active;
      direct_observations += value.direct_observation;
      timed_out_epochs += value.epoch_timed_out;
    }
    log_ << "},\"active_track_count\":" << active_tracks
         << ",\"direct_observation_count\":" << direct_observations
         << ",\"epochs_timed_out\":" << timed_out_epochs
         << ",\"consensus_messages_published\":" << messages
         << ",\"consensus_messages_rejected\":" << rejected
         << ",\"handoffs_sent\":" << handoffs_sent
         << ",\"handoffs_accepted\":" << handoffs_accepted << "},";
    log_ << "\"target_observation_diagnostics\":{";
    if (observation_diagnostics_) {
      const auto& value = *observation_diagnostics_;
      log_ << "\"camera_captures\":" << value.camera_captures
           << ",\"camera_visible_detections\":" << value.visible_detections
           << ",\"camera_invalid_detections\":" << value.invalid_detections
           << ",\"camera_rpc_errors\":" << value.rpc_errors
           << ",\"mean_capture_rate_hz\":" << value.mean_capture_rate_hz
           << ",\"capture_interval_p95_sec\":" << value.capture_interval_p95_sec
           << ",\"capture_interval_max_sec\":" << value.capture_interval_max_sec
           << ",\"mean_image_rpc_duration_sec\":" << value.mean_image_rpc_duration_sec
           << ",\"max_image_rpc_duration_sec\":" << value.max_image_rpc_duration_sec;
    }
    log_ << "},\"formation\":{\"formation_max_error\":"
         << std::max(uav_error, ugv_error)
         << ",\"formation_xy_max_error\":" << std::max(uav_error, ugv_error)
         << "},\"tracking_communication_links\":[";
    std::size_t link_index = 0;
    for (const auto& [first, neighbors] : current_adjacency_) {
      for (const auto& second : neighbors) {
        if (first >= second) continue;
        if (link_index++) log_ << ',';
        log_ << "[\"" << first << "\",\"" << second << "\"]";
      }
    }
    log_ << "],\"obstacles\":{";
    std::size_t obstacle_agent_index = 0;
    for (const char* id : kControlled) {
      if (obstacle_agent_index++) log_ << ',';
      log_ << '\"' << id << "\":{\"source\":\"" << cbf_obstacle_source_
           << "\",\"valid\":";
      const auto snapshot = obstacle_caches_.at(id)->snapshot();
      log_ << ((snapshot && snapshot->valid) ? "true" : "false")
           << ",\"age\":";
      writeNullable(log_, snapshot ? snapshot->age_seconds : std::numeric_limits<double>::quiet_NaN());
      log_ << ",\"proxies\":[";
      if (snapshot) {
        for (std::size_t proxy_index = 0; proxy_index < snapshot->message.proxies.size(); ++proxy_index) {
          if (proxy_index) log_ << ',';
          const auto& proxy = snapshot->message.proxies[proxy_index];
          log_ << "{\"id\":\"" << proxy.proxy_id << "\",\"source\":\""
               << proxy.source << "\",\"center\":[";
          for (std::size_t coordinate = 0; coordinate < proxy.center.size(); ++coordinate) {
            if (coordinate) log_ << ',';
            log_ << proxy.center[coordinate];
          }
          log_ << "],\"radius\":" << proxy.radius << '}';
        }
      }
      log_ << "]}";
    }
    log_ << "},\"nominal_controls\":{";
    for (std::size_t i = 0; i < cbf_diagnostics.entries.size(); ++i) {
      if (i) log_ << ',';
      const auto& value = cbf_diagnostics.entries[i];
      log_ << '\"' << value.agent_id << "\":[";
      for (std::size_t j = 0; j < value.nominal_control.size(); ++j) {
        if (j) log_ << ',';
        log_ << value.nominal_control[j];
      }
      log_ << ']';
    }
    log_ << "},\"safe_controls\":{";
    for (std::size_t i = 0; i < cbf_diagnostics.entries.size(); ++i) {
      if (i) log_ << ',';
      const auto& value = cbf_diagnostics.entries[i];
      log_ << '\"' << value.agent_id << "\":[";
      for (std::size_t j = 0; j < value.safe_control.size(); ++j) {
        if (j) log_ << ',';
        log_ << value.safe_control[j];
      }
      log_ << ']';
    }
    log_ << "},\"actuation\":{\"profile\":\"" << actuation_profile_
         << "\",\"commands\":{";
    std::size_t actuation_index = 0;
    for (const auto& [id, command] : commands) {
      if (actuation_index++) log_ << ',';
      log_ << '\"' << id << "\":[" << command.x() << ',' << command.y() << ','
           << command.z() << ']';
    }
    log_ << "}},\"cbf\":{\"schema_version\":1,\"enabled\":"
         << (cbf_enabled_ ? "true" : "false")
         << ",\"step_id\":" << cbf_diagnostics.step_id << ",\"entries\":[";
    for (std::size_t i = 0; i < cbf_diagnostics.entries.size(); ++i) {
      if (i) log_ << ',';
      const auto& value = cbf_diagnostics.entries[i];
      log_ << "{\"agent_id\":\"" << value.agent_id
           << "\",\"enabled\":" << (value.enabled ? "true" : "false")
           << ",\"selected_method\":\"" << value.selected_method
           << "\",\"effective_method\":\"" << value.effective_method
           << "\",\"control_kind\":\"" << value.control_kind
           << "\",\"nominal_control\":[";
      for (std::size_t j = 0; j < value.nominal_control.size(); ++j) {
        if (j) log_ << ',';
        log_ << value.nominal_control[j];
      }
      log_ << "],\"safe_control\":[";
      for (std::size_t j = 0; j < value.safe_control.size(); ++j) {
        if (j) log_ << ',';
        log_ << value.safe_control[j];
      }
      log_ << "],\"success\":" << (value.success ? "true" : "false")
           << ",\"fallback\":" << (value.fallback ? "true" : "false")
           << ",\"solver_status\":\"" << value.solver_status
           << "\",\"solver_iterations\":" << value.solver_iterations
           << ",\"primal_residual\":";
      writeNullable(log_, value.primal_residual);
      log_ << ",\"dual_residual\":";
      writeNullable(log_, value.dual_residual);
      log_ << ",\"minimum_barrier\":";
      writeNullable(log_, value.minimum_barrier);
      log_
           << ",\"constraint_count\":" << value.constraint_count
           << ",\"active_constraints\":" << value.active_constraints
           << ",\"distributed_rounds\":" << value.distributed_rounds
           << ",\"filter_time_ms\":" << value.filter_time_ms
           << ",\"solver_time_ms\":" << value.solver_time_ms
           << ",\"maximum_row_violation\":" << value.maximum_row_violation
           << ",\"maximum_bound_violation\":" << value.maximum_bound_violation
           << ",\"final_feasible\":" << (value.final_feasible ? "true" : "false")
           << ",\"deadline_miss\":" << (value.deadline_miss ? "true" : "false")
           << ",\"static_obstacles_valid\":"
           << (value.static_obstacles_valid ? "true" : "false")
           << ",\"static_obstacles_age\":";
      writeNullable(log_, value.static_obstacles_age);
      log_ << ",\"target_active\":" << (value.target_active ? "true" : "false")
           << ",\"target_age\":";
      writeNullable(log_, value.target_age);
      log_ << ",\"target_timestamp\":";
      writeNullable(log_, value.target_timestamp);
      log_ << ",\"row_labels\":[";
      for (std::size_t row = 0; row < value.row_labels.size(); ++row) {
        if (row) log_ << ',';
        log_ << '\"' << value.row_labels[row] << '\"';
      }
      log_ << "],\"intervention_norm\":" << value.intervention_norm << '}';
    }
    log_ << "]},\"safety_communication_links\":[],\"collisions\":{";
    std::size_t collision_index = 0;
    for (const auto& [id, value] : collisions_) {
      if (collision_index++) log_ << ',';
      log_ << "\"" << id << "\":{\"available\":" << (value.available ? "true" : "false")
           << ",\"has_collided\":" << (value.has_collided ? "true" : "false")
           << ",\"relevant\":" << (value.relevant ? "true" : "false")
           << ",\"object_name\":\"" << value.object_name
           << "\",\"object_id\":" << value.object_id
           << ",\"penetration_depth\":" << value.penetration_depth
           << ",\"simulator_timestamp\":" << value.simulator_timestamp
           << ",\"event_id\":\"" << value.event_id << "\"}";
    }
    log_ << "}}\n";
    log_.flush();
  }

  hercules_mission_core::RuralTargetTrackingConfig config_;
  hercules_mission_core::FormationController controller_;
  TruthTargetSource target_source_;
  std::unique_ptr<DistributedTargetSource> distributed_target_source_;
  std::unique_ptr<hercules_mission_core::FigureEightTargetController>
      target_controller_;
  bool dry_run_{true};
  bool enable_target_{true};
  bool enable_formation_{true};
  bool running_{false};
  bool finished_{false};
  std::string target_source_name_{"truth"};
  std::string target_observation_source_{"truth"};
  std::string actuation_profile_{"current_ros"};
  double duration_{30.0};
  double startup_timeout_{30.0};
  double freshness_timeout_{0.5};
  bool cbf_enabled_{false};
  std::string cbf_method_name_{"mestres"};
  std::string cbf_obstacle_source_{"none"};
  bool truth_obstacle_fixture_{false};
  double cbf_uncertainty_radius_{0.0};
  hercules_cbf::CBFConfig cbf_config_;
  double route_heading_{0.0};
  double target_speed_{0.10};
  double target_pattern_length_{10.0};
  double target_pattern_width_{8.0};
  int target_sample_count_{64};
  int target_start_sample_index_{5};
  int target_direction_{1};
  double target_waypoint_radius_{1.0};
  double target_heading_gain_{2.0};
  double target_max_yaw_rate_{1.5};
  double target_minimum_alignment_{0.75};
  double target_center_x_{std::numeric_limits<double>::quiet_NaN()};
  double target_center_y_{std::numeric_limits<double>::quiet_NaN()};
  double target_center_z_{std::numeric_limits<double>::quiet_NaN()};
  double uav_velocity_limit_{3.0};
  double uav_altitude_ceiling_{-8.0};
  double cbf_obstacle_stale_after_{1.3};
  double cbf_agent_deadline_ms_{5.0};
  std::string log_path_;
  std::ofstream log_;
  std::size_t step_{0};
  uint64_t tracking_epoch_{0};
  double next_tracking_time_{0.0};
  double max_uav_error_{0.0};
  double max_ugv_error_{0.0};
  double min_ugv_radius_{1e9};
  double max_ugv_radius_{0.0};
  Clock::time_point started_at_;
  Clock::time_point mission_started_at_;
  std::optional<Clock::time_point> last_step_time_;
  std::vector<double> update_gaps_;
  std::map<std::string, hercules_interfaces::msg::GroundTruthState> states_;
  std::map<std::string, Clock::time_point> receipts_;
  std::map<std::string, Eigen::Vector3d> origins_;
  hercules_tracking_ros::Adjacency current_adjacency_;
  std::map<std::string, hercules_interfaces::msg::TargetEstimate> local_estimates_;
  std::map<std::string, Clock::time_point> estimate_receipts_;
  std::map<std::string, hercules_interfaces::msg::TargetMeasurement> tracking_measurements_;
  std::map<std::string, hercules_interfaces::msg::TrackingDiagnostics> tracking_diagnostics_;
  std::optional<hercules_interfaces::msg::TargetObservationDiagnostics> observation_diagnostics_;
  std::map<std::string, std::unique_ptr<hercules_cbf_ros::ObstacleCache>> obstacle_caches_;
  std::vector<rclcpp::Subscription<hercules_interfaces::msg::GroundTruthState>::SharedPtr>
      state_subscriptions_;
  std::vector<rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr>
      origin_subscriptions_;
  std::vector<rclcpp::Subscription<hercules_interfaces::msg::TargetEstimate>::SharedPtr>
      estimate_subscriptions_;
  std::vector<rclcpp::Subscription<hercules_interfaces::msg::TargetMeasurement>::SharedPtr>
      measurement_subscriptions_;
  std::vector<rclcpp::Subscription<hercules_interfaces::msg::TrackingDiagnostics>::SharedPtr>
      tracking_diagnostic_subscriptions_;
  rclcpp::Subscription<hercules_interfaces::msg::TargetObservationDiagnostics>::SharedPtr
      observation_diagnostics_subscription_;
  std::vector<rclcpp::Subscription<hercules_interfaces::msg::ObstacleProxyArray>::SharedPtr>
      obstacle_subscriptions_;
  rclcpp::Subscription<hercules_interfaces::msg::MissionCollision>::SharedPtr
      collision_subscription_;
  std::map<std::string, hercules_interfaces::msg::MissionCollision> collisions_;
  rclcpp::Publisher<hercules_interfaces::msg::TrackingEpoch>::SharedPtr epoch_publisher_;
  rclcpp::Publisher<hercules_interfaces::msg::CBFDiagnosticsArray>::SharedPtr cbf_diagnostics_publisher_;
  std::map<std::string, rclcpp::Publisher<airsim_interfaces::msg::VelCmd>::SharedPtr>
      uav_publishers_;
  std::map<std::string,
           rclcpp::Publisher<airsim_interfaces::msg::CarControls>::SharedPtr>
      car_publishers_;
  std::map<std::string, rclcpp::Client<airsim_interfaces::srv::Land>::SharedPtr>
      land_clients_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace hercules_mission_ros

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<hercules_mission_ros::RuralNominalMissionNode>());
  rclcpp::shutdown();
  return 0;
}
