#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
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

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <airsim_interfaces/msg/car_controls.hpp>
#include <airsim_interfaces/msg/vel_cmd.hpp>
#include <airsim_interfaces/srv/land.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <hercules_interfaces/msg/ground_truth_state.hpp>
#include <hercules_interfaces/msg/localization_diagnostics.hpp>
#include <hercules_interfaces/msg/localization_estimate.hpp>
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
#include <hercules_mission_core/periodic_mission.hpp>
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
    "Drone1", "Drone2", "SimpleFlight"};
constexpr const char* kUgvs[] = {"Husky1", "Husky2", "Husky3"};
constexpr const char* kControlled[] = {"Drone1", "Drone2", "SimpleFlight", "Husky1", "Husky2", "Husky3"};
constexpr const char* kAll[] = {"Drone1", "Drone2", "SimpleFlight", "Husky1", "Husky2", "Husky3",
                                "Target1"};
constexpr const char* kLocalizationFrame = "airsim_world_ned";

double seconds(Clock::time_point then) {
  return std::chrono::duration<double>(Clock::now() - then).count();
}

struct ControlTargetEstimate {
  hercules_mission_core::TargetEstimate estimate;
  Eigen::Matrix2d covariance{Eigen::Matrix2d::Zero()};
  double timestamp{0.0};
  bool from_distributed{false};
};

struct LocalizationOrderKey {
  std::int32_t stamp_sec{0};
  std::uint32_t stamp_nanosec{0};
  std::uint64_t sequence{0};
};

bool timestampValid(const builtin_interfaces::msg::Time& stamp) {
  // ROS 2 timestamps use a signed seconds field and a nanosecond remainder.
  // Simulation time is non-negative; accepting zero keeps the first valid
  // sample usable before /clock has advanced.
  return stamp.sec >= 0 && stamp.nanosec < 1000000000U;
}

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

bool validLocalizationEstimate(const hercules_interfaces::msg::LocalizationEstimate& message,
                               const std::string& expected_id,
                               const std::string& expected_algorithm) {
  if (!message.valid || !message.initialized || message.stale ||
      message.agent_id != expected_id || message.algorithm != expected_algorithm ||
      message.header.frame_id != kLocalizationFrame ||
      message.frame_id != kLocalizationFrame || !timestampValid(message.header.stamp) ||
      !std::all_of(message.position.begin(), message.position.end(),
                   [](double value) { return std::isfinite(value); }) ||
      !std::all_of(message.velocity.begin(), message.velocity.end(),
                   [](double value) { return std::isfinite(value); }) ||
      !std::all_of(message.covariance.begin(), message.covariance.end(),
                   [](double value) { return std::isfinite(value); }) ||
      !std::isfinite(message.yaw) || !std::isfinite(message.yaw_rate)) {
    return false;
  }
  Eigen::Map<const Eigen::Matrix3d> covariance(message.covariance.data());
  if (!covariance.isApprox(covariance.transpose(),
                           1e-8 * (1.0 + covariance.norm()))) {
    return false;
  }
  const Eigen::LLT<Eigen::Matrix3d> factor(covariance);
  return factor.info() == Eigen::Success &&
         factor.matrixL().toDenseMatrix().diagonal().array().all() > 0.0;
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
    maicp_case_ = declare_parameter<std::string>("maicp_case", "");
    if (!maicp_case_.empty() && maicp_case_ != "collision" &&
        maicp_case_ != "tracking" && maicp_case_ != "localization")
      throw std::invalid_argument("maicp_case must be collision, tracking, or localization");
    if (maicp_case_ == "tracking") config_.tracking_rate = 1.0;
    maicp_steps_ = declare_parameter<int>("maicp_steps", 100);
    maicp_loop_radius_ = declare_parameter<double>("maicp_loop_radius", 2.5);
    if (maicp_steps_ < 2 || !std::isfinite(maicp_loop_radius_) || maicp_loop_radius_ <= 0.0)
      throw std::invalid_argument("MAICP steps and route length must be positive");
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
    if (!maicp_case_.empty() && (!std::isfinite(duration_) || duration_ <= 0.0 || duration_ > 10.0))
      throw std::invalid_argument("paper missions require 0 < duration_sec <= 10");
    startup_timeout_ = declare_parameter<double>("startup_timeout_sec", 30.0);
    freshness_timeout_ = declare_parameter<double>("freshness_timeout_sec", 0.5);
    localization_algorithm_ = declare_parameter<std::string>(
        "localization_algorithm", "recursive_decentralized");
    if (localization_algorithm_ != "recursive_decentralized" &&
        localization_algorithm_ != "gs_ci") {
      throw std::invalid_argument(
          "localization_algorithm must be recursive_decentralized or gs_ci");
    }
    const auto localization_source =
        declare_parameter<std::string>("localization_source", "estimate");
    const auto control_source = declare_parameter<std::string>("control_source", "");
    localization_source_ = control_source.empty() ? localization_source : control_source;
    if (localization_source_ != "estimate" && localization_source_ != "truth") {
      throw std::invalid_argument("localization_source must be estimate or truth");
    }
    localization_stale_after_ = declare_parameter<double>(
        "localization_stale_after_sec", 0.5);
    if (!std::isfinite(localization_stale_after_) || localization_stale_after_ <= 0.0) {
      throw std::invalid_argument("localization_stale_after_sec must be positive");
    }
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
    cbf_config_.ugv_margin = declare_parameter<double>("maicp_margin_ugv", 0.0);
    cbf_config_.uav_margin = declare_parameter<double>("maicp_margin_uav", 0.0);
    if (!std::isfinite(cbf_config_.ugv_margin) || cbf_config_.ugv_margin < 0.0 ||
        !std::isfinite(cbf_config_.uav_margin) || cbf_config_.uav_margin < 0.0)
      throw std::invalid_argument("MAICP class margins must be finite and nonnegative");
    const auto read_model = [this](const std::string& name, std::array<double, 6>& model) {
      std::istringstream values(declare_parameter<std::string>(name, "0 0 0 0 0 0"));
      for (double& value : model) {
        if (!(values >> value) || !std::isfinite(value))
          throw std::invalid_argument(name + " needs six finite affine coefficients");
      }
      std::string extra;
      if (values >> extra) throw std::invalid_argument(name + " has extra coefficients");
    };
    read_model("maicp_model_ugv", cbf_config_.ugv_acceleration_model_coefficients);
    read_model("maicp_model_uav", cbf_config_.uav_acceleration_model_coefficients);
    if (maicp_case_ == "collision") {
      cbf_config_.wang_corridor_y_min = -14.75;
      cbf_config_.wang_corridor_y_max = 14.75;
    }
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
      localization_subscriptions_.push_back(
          create_subscription<hercules_interfaces::msg::LocalizationEstimate>(
              std::string("/hercules_localization/") + id + "/estimate", 20,
              [this, id](hercules_interfaces::msg::LocalizationEstimate::ConstSharedPtr msg) {
                if (!msg || !validLocalizationEstimate(*msg, id, localization_algorithm_)) {
                  // An invalid or explicitly stale sample invalidates the
                  // active control estimate immediately.  Retain the last
                  // accepted planar pose separately so CBF geometry can stay
                  // conservative, but never keep steering from an old sample
                  // during the freshness grace period.
                  localization_estimates_.erase(id);
                  localization_receipts_.erase(id);
                  ++localization_rejections_[id];
                  return;
                }
                const auto previous = localization_order_.find(id);
                const bool timestamp_older =
                    previous != localization_order_.end() &&
                    (msg->header.stamp.sec < previous->second.stamp_sec ||
                     (msg->header.stamp.sec == previous->second.stamp_sec &&
                      msg->header.stamp.nanosec < previous->second.stamp_nanosec));
                if (previous != localization_order_.end() &&
                    (msg->sequence <= previous->second.sequence || timestamp_older)) {
                  // A late/replayed estimate must not move the control state
                  // backwards.  Keep the last accepted estimate and let its
                  // receipt age drive the normal stale handling path.
                  ++localization_rejections_[id];
                  return;
                }
                localization_estimates_[id] = *msg;
                localization_receipts_[id] = Clock::now();
                localization_order_[id] = {
                    msg->header.stamp.sec, msg->header.stamp.nanosec, msg->sequence};
                last_localized_states_[id] = states_.count(id) ? states_.at(id) :
                    hercules_interfaces::msg::GroundTruthState{};
                auto& cached = last_localized_states_.at(id);
                cached.position[0] = msg->position[0];
                cached.position[1] = msg->position[1];
                cached.velocity[0] = msg->velocity[0];
                cached.velocity[1] = msg->velocity[1];
                cached.yaw = msg->yaw;
                cached.yaw_rate = msg->yaw_rate;
                cached.valid = true;
              }));
      localization_diagnostic_subscriptions_.push_back(
          create_subscription<hercules_interfaces::msg::LocalizationDiagnostics>(
              std::string("/hercules_localization/") + id + "/diagnostics", 20,
              [this, id](hercules_interfaces::msg::LocalizationDiagnostics::ConstSharedPtr msg) {
                if (msg && msg->agent_id == id &&
                    (msg->algorithm.empty() || msg->algorithm == localization_algorithm_) &&
                    timestampValid(msg->header.stamp)) {
                  localization_diagnostics_[id] = *msg;
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
                "Rural nominal mission waiting for seven calibrated states and six localization estimates (%s; algorithm=%s)",
                dry_run_ ? "dry-run: actuation disabled" : "live actuation",
                localization_algorithm_.c_str());
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

  bool localizationFresh(const std::string& id) const {
    if (localization_source_ == "truth") return true;
    const auto estimate = localization_estimates_.find(id);
    const auto receipt = localization_receipts_.find(id);
    return estimate != localization_estimates_.end() && estimate->second.valid &&
           estimate->second.initialized && !estimate->second.stale &&
           receipt != localization_receipts_.end() &&
           seconds(receipt->second) <= localization_stale_after_;
  }

  bool localizationReady() const {
    if (localization_source_ == "truth") return true;
    for (const char* id : kControlled) {
      if (!localizationFresh(id)) return false;
    }
    return true;
  }

  hercules_interfaces::msg::GroundTruthState localizedMessage(
      const std::string& id, bool retain_last = true) const {
    const auto truth = states_.find(id);
    if (truth == states_.end()) {
      return hercules_interfaces::msg::GroundTruthState();
    }
    if (localization_source_ == "truth") {
      return truth->second;
    }
    if (!localizationFresh(id)) {
      if (retain_last) {
        const auto cached = last_localized_states_.find(id);
        if (cached != last_localized_states_.end()) {
          // Keep the last accepted planar estimate as a static geometry
          // obstacle, but retain the current simulator truth for the vertical
          // channel.  AirSim can continue publishing z/vz while a planar
          // localization stream is delayed or disconnected.
          auto stopped = cached->second;
          stopped.position[2] = truth->second.position[2];
          stopped.velocity[2] = truth->second.velocity[2];
          stopped.velocity[0] = 0.0;
          stopped.velocity[1] = 0.0;
          stopped.yaw_rate = 0.0;
          stopped.valid = truth->second.valid;
          stopped.header = truth->second.header;
          stopped.agent_id = id;
          stopped.vehicle_type = truth->second.vehicle_type;
          return stopped;
        }
      }
      // No truth fallback is permitted for control.  Preserve the vehicle
      // type/altitude channel needed by the controller, but make the planar
      // state explicitly stationary and invalid-looking rather than silently
      // steering from simulator truth.
      auto stopped = truth->second;
      stopped.position[0] = 0.0;
      stopped.position[1] = 0.0;
      stopped.velocity[0] = 0.0;
      stopped.velocity[1] = 0.0;
      stopped.velocity[2] = truth->second.velocity[2];
      stopped.yaw = 0.0;
      stopped.yaw_rate = 0.0;
      return stopped;
    }
    auto result = truth->second;
    const auto& estimate = localization_estimates_.at(id);
    result.position[0] = estimate.position[0];
    result.position[1] = estimate.position[1];
    result.velocity[0] = estimate.velocity[0];
    result.velocity[1] = estimate.velocity[1];
    result.yaw = estimate.yaw;
    result.yaw_rate = estimate.yaw_rate;
    return result;
  }

  hercules_mission_core::AgentState controlAgentState(const std::string& id) const {
    // A stale agent is stationary at its last valid estimate for geometry and
    // tracking; the actuation loops separately suppress its command.
    return agentState(localizedMessage(id, true));
  }

  void tick() {
    if (finished_) return;
    if (!running_) {
      if (!ready() || !localizationReady()) {
        if (seconds(started_at_) > startup_timeout_) {
          fail("startup timed out before all seven calibrated states and six localization estimates were fresh");
        }
        return;
      }
      startMission();
      if (running_) step();
      return;
    }
    if (!ready()) {
      fail("state became stale or unavailable; actuation suppressed");
      return;
    }
    if (seconds(mission_started_at_) >= duration_ ||
        (!maicp_case_.empty() && step_ >= static_cast<std::size_t>(maicp_steps_))) {
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
    for (const char* id : kControlled) {
      maicp_home_[id] = controlAgentState(id).position;
      maicp_truth_home_[id] = agentState(states_.at(id)).position;
      maicp_last_truth_[id] = maicp_truth_home_[id];
      maicp_travel_[id] = 0.0;
      maicp_velocity_[id] = controlAgentState(id).velocity;
      maicp_returning_[id] = false;
    }
    maicp_home_["Target1"] = target.position;
    maicp_velocity_["Target1"] = target.velocity;
    running_ = true;
    mission_started_at_ = Clock::now();
    RCLCPP_INFO(get_logger(), "preflight passed; mission started");
  }

  void publishTrackingEpoch(double mission_time) {
    hercules_tracking_ros::PositionMap positions;
    std::vector<std::string> ids;
    for (const char* id : kControlled) {
      ids.emplace_back(id);
      positions[id] = controlAgentState(id).position;
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

  bool periodicMission() const {
    // All paper cases share a short closed patrol; each estimator/controller
    // is evaluated independently along that patrol.
    return !maicp_case_.empty();
  }

  hercules_mission_core::PeriodicReference loopReference(const std::string& id) const {
    const double period = std::max(1.0, duration_ - 2.5);
    const double t = std::min(period, seconds(mission_started_at_) + config_.control_dt);
    const bool ground = id.rfind("Husky", 0) == 0 || id == "Target1";
    auto reference = ground
        ? hercules_mission_core::outAndBackMissionReference(maicp_home_.at(id), 2.0, period, t)
        : hercules_mission_core::periodicMissionReference(maicp_home_.at(id), maicp_loop_radius_, period, t);
    if (t >= period) {
      reference.velocity.setZero();
      reference.acceleration.setZero();
    }
    return reference;
  }

  Eigen::Vector3d missionGoal(const std::string& id,
                             const hercules_mission_core::AgentState&) {
    maicp_returning_[id] = seconds(mission_started_at_) >= (duration_ - 2.5) * 0.75;
    return loopReference(id).position;
  }

  Eigen::Vector3d goalAcceleration(const hercules_mission_core::AgentState& agent,
                                   const Eigen::Vector3d& goal) const {
    Eigen::Vector3d desired_velocity = config_.formation.position_gain * (goal - agent.position);
    Eigen::Vector3d feedforward = Eigen::Vector3d::Zero();
    if (!maicp_case_.empty()) {
      const auto reference = loopReference(agent.agent_id);
      desired_velocity += reference.velocity;
      feedforward = reference.acceleration;
    }
    const double speed_limit = maicp_case_.empty() ? config_.formation.max_speed : 5.0;
    const double speed = desired_velocity.norm();
    if (speed > speed_limit) desired_velocity *= speed_limit / speed;
    return feedforward + config_.formation.velocity_gain * (desired_velocity - agent.velocity);
  }

  Eigen::Vector3d paperVelocity(const std::string& id, const Eigen::Vector3d& acceleration) {
    // Integrate the commanded acceleration. Resetting this integrator to the
    // measured velocity every tick attenuates it through AirSim's inner loop.
    auto& velocity = maicp_velocity_.at(id);
    velocity += config_.control_dt * acceleration;
    const double speed = velocity.head<2>().norm();
    if (speed > 5.0) velocity.head<2>() *= 5.0 / speed;
    // Only XY is the paper's double integrator. A separate height loop
    // supplies AirSim's vertical velocity setpoint without integrating it.
    const double height_error = maicp_home_.at(id).z() - agentState(states_.at(id)).position.z();
    velocity.z() = std::clamp(1.5 * height_error, -2.0, 2.0);
    return velocity;
  }

  Eigen::Vector2d paperGroundCommand(const hercules_mission_core::AgentState& agent,
                                     const Eigen::Vector3d& acceleration) {
    const auto desired = paperVelocity(agent.agent_id, acceleration);
    return ugvAccelerationCommand(desired, agent.yaw, Eigen::Vector3d::Zero(),
                                  config_.control_dt, 5.0, cbf_config_.ugv_yaw_rate_limit);
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
    const Eigen::Vector2d target_command = maicp_case_ == "tracking"
        ? paperGroundCommand(target, goalAcceleration(target, loopReference("Target1").position))
        : target_controller_->update(target.position, target.yaw, config_.control_dt);
    const double measured_target_speed = target.velocity.head<2>().norm();
    const CarCommand target_car = maicp_case_ == "tracking"
        ? paperUgvCarCommand(target_command.x(), target_command.y(), measured_target_speed,
                             cbf_config_.ugv_yaw_rate_limit)
        : ugvCarCommand(target_command.x(), target_command.y(), measured_target_speed,
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
      if (!localizationFresh(id)) {
        commands[id] = Eigen::Vector3d::Zero();
        // Keep the JSON/control schema complete while this agent is stopped;
        // the cached localized pose remains the static CBF obstacle and is
        // never replaced by simulator truth.
        const auto cached = localizedMessage(id);
        desired_slots[id] = Eigen::Vector3d(
            cached.position[0], cached.position[1], cached.position[2]);
        slot_errors[id] = 0.0;
        control_estimates[id] = hercules_mission_core::TargetEstimate{};
        cbf_diagnostics.entries.push_back(
            stoppedDiagnostics(id, agentState(cached)));
        if (!dry_run_ && enable_formation_) publishUav(id, Eigen::Vector3d::Zero());
        continue;
      }
      const auto agent = controlAgentState(id);
      const auto target_context = controlEstimate(id, target, mission_time);
      const auto& target_estimate = target_context.estimate;
      control_estimates[id] = target_estimate;
      const auto acceleration = periodicMission() ? goalAcceleration(agent, missionGoal(id, agent)) : controller_.targetNominalControl(
          agent, target_estimate, route_heading_, config_.target_ground_z,
          config_.target_ugv_circumradius);
      const auto nominal_velocity = (periodicMission() || target_estimate.active)
          ? (actuation_profile_ == "python_cbf"
                 ? pythonCbfUavVelocityCommand(agent.velocity, acceleration, config_.control_dt,
                                               uav_velocity_limit_, uav_altitude_ceiling_,
                                               agent.position.z())
                 : uavVelocityCommand(agent.velocity, acceleration, config_.control_dt,
                                      uav_velocity_limit_))
          : Eigen::Vector3d::Zero();
      Eigen::Vector3d velocity = (periodicMission() && !cbf_enabled_) ? paperVelocity(id, acceleration) : nominal_velocity;
      if (cbf_enabled_) {
        const auto filtered = runCbf(id, agent, acceleration, target_context);
        velocity = periodicMission() ? paperVelocity(id, filtered.result.safe_control)
            : actuation_profile_ == "python_cbf"
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
      desired_slots[id] = periodicMission() ? missionGoal(id, agent) : slot.position;
      slot_errors[id] = (agent.position - desired_slots[id]).norm();
      step_max_uav = std::max(step_max_uav, slot_errors[id]);
      if (!dry_run_ && enable_formation_) publishUav(id, velocity);
    }
    for (const char* id : kUgvs) {
      if (!localizationFresh(id)) {
        commands[id] = Eigen::Vector3d::Zero();
        const auto cached = localizedMessage(id);
        desired_slots[id] = Eigen::Vector3d(
            cached.position[0], cached.position[1], cached.position[2]);
        slot_errors[id] = 0.0;
        control_estimates[id] = hercules_mission_core::TargetEstimate{};
        cbf_diagnostics.entries.push_back(
            stoppedDiagnostics(id, agentState(cached)));
        if (!dry_run_ && enable_formation_) publishCar(id, stoppedCarCommand());
        continue;
      }
      const auto agent = controlAgentState(id);
      const auto target_context = controlEstimate(id, target, mission_time);
      const auto& target_estimate = target_context.estimate;
      control_estimates[id] = target_estimate;
      const auto command = controller_.targetNominalUnicycleControl(
          agent, target_estimate, route_heading_, config_.target_ground_z,
          config_.target_ugv_circumradius);
      Eigen::Vector2d model_command(command.x(), command.y());
      if (cbf_enabled_ && cbf_method_name_ == "wang") {
        Eigen::Vector3d goal;
        if (periodicMission()) goal = missionGoal(id, agent);
        else {
          const auto slot = hercules_mission_core::targetCenteredSlot(
              id, hercules_mission_core::VehicleType::kUgv,
              Eigen::Vector3d(target_estimate.position.x(), target_estimate.position.y(), agent.position.z()),
              Eigen::Vector3d(target_estimate.velocity.x(), target_estimate.velocity.y(), 0.0),
              route_heading_, config_.uav_altitude, config_.target_ground_z, config_.target_ugv_circumradius);
          goal = slot.position;
        }
        Eigen::Vector3d acceleration = goalAcceleration(agent, goal);
        acceleration.z() = 0.0;
        const auto filtered = runCbf(id, agent, acceleration, target_context);
        model_command = periodicMission() ? paperGroundCommand(agent, filtered.result.safe_control.head<3>())
            : ugvAccelerationCommand(agent.velocity, agent.yaw,
                filtered.result.safe_control.head<3>(), config_.control_dt,
                cbf_config_.ugv_speed_limit, cbf_config_.ugv_yaw_rate_limit);
        cbf_diagnostics.entries.push_back(filtered.diagnostics);
      } else {
        if (periodicMission()) {
          const Eigen::Vector3d acceleration = goalAcceleration(agent, missionGoal(id, agent));
          model_command = paperGroundCommand(agent, acceleration);
        }
        if (cbf_enabled_) {
          const auto filtered = runCbf(id, agent, model_command, target_context);
          model_command = filtered.result.safe_control.head<2>();
          cbf_diagnostics.entries.push_back(filtered.diagnostics);
        } else {
          cbf_diagnostics.entries.push_back(disabledDiagnostics(id, agent, model_command));
        }
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
      desired_slots[id] = periodicMission() ? missionGoal(id, agent) : slot.position;
      slot_errors[id] =
          (agent.position.head<2>() - desired_slots[id].head<2>()).norm();
      step_max_ugv = std::max(step_max_ugv, slot_errors[id]);
      const double radius = (agent.position.head<2>() - target.position.head<2>()).norm();
      step_min_radius = std::min(step_min_radius, radius);
      step_max_radius = std::max(step_max_radius, radius);
      if (!dry_run_ && enable_formation_ && (periodicMission() || target_estimate.active || cbf_enabled_)) {
        publishCar(id, !maicp_case_.empty()
            ? paperUgvCarCommand(model_command.x(), model_command.y(),
                agent.velocity.head<2>().norm(), cbf_config_.ugv_yaw_rate_limit)
            : actuation_profile_ == "python_cbf"
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
    const auto ego_message = localizedMessage(id);
    for (const char* other : kControlled) {
      if (std::string(other) == id) continue;
      const auto found = states_.find(other);
      if (found == states_.end()) continue;
      if (localization_source_ != "truth" &&
          !localization_estimates_.count(other) &&
          !last_localized_states_.count(other)) continue;
      const auto localized = localizedMessage(other);
      const auto other_state = agentState(localized);
      if (other_state.vehicle_type != agent.vehicle_type) continue;
      if ((other_state.position - agent.position).norm() <= config_.communication_range)
        neighbors.push_back(localized);
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
              const auto localized_state = localizedMessage(other);
              const Eigen::Vector3d position(localized_state.position[0], localized_state.position[1],
                                             localized_state.position[2]);
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
    if (maicp_case_ == "collision") {
      for (const auto& center : {Eigen::Vector2d(2.5, -9.0), Eigen::Vector2d(2.5, 0.0),
                                 Eigen::Vector2d(2.5, 9.0)}) {
        hercules_interfaces::msg::ObstacleProxy obstacle;
        obstacle.proxy_id = "maicp_column_" + std::to_string(obstacles.size());
        obstacle.source = "truth_maicp_fixture";
        obstacle.center = {center.x(), center.y(), agent.position.z()};
        obstacle.radius = std::sqrt(0.5);
        obstacle.is_planar = true;
        obstacles.push_back(obstacle);
      }
    }
    if (!periodicMission() && agent.vehicle_type == hercules_mission_core::VehicleType::kUgv && target_estimate.active) {
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
    if (!maicp_case_.empty() && maicp_case_ != "collision") {
      config.ugv_margin = 0.0;
      config.uav_margin = 0.0;
    }
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

  hercules_interfaces::msg::CBFDiagnostics stoppedDiagnostics(
      const std::string& id, const hercules_mission_core::AgentState& agent) const {
    const Eigen::Index dimension =
        agent.vehicle_type == hercules_mission_core::VehicleType::kUgv ? 2 : 3;
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(dimension);
    auto diagnostic = disabledDiagnostics(id, agent, zero);
    diagnostic.solver_status = "stopped_stale_localization";
    diagnostic.fallback = true;
    diagnostic.success = false;
    diagnostic.final_feasible = true;
    diagnostic.static_obstacles_valid = true;
    diagnostic.target_active = false;
    diagnostic.target_age = 0.0;
    diagnostic.target_timestamp = 0.0;
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
    if (!maicp_case_.empty()) {
      std::ofstream status(log_path_ + ".done");
      status << "complete\n";
    }
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
    if (!maicp_case_.empty()) {
      std::ofstream status(log_path_ + ".failed");
      status << reason << '\n';
    }
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

  void writeJsonString(std::ostream& out, const std::string& value) {
    out << '"';
    for (const unsigned char character : value) {
      switch (character) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
          if (character < 0x20U) {
            out << "\\u00" << std::hex << std::setw(2)
                << std::setfill('0') << static_cast<unsigned int>(character)
                << std::dec << std::setfill(' ');
          } else {
            out << static_cast<char>(character);
          }
      }
    }
    out << '"';
  }

  template <std::size_t Size>
  void writeNullableArray(std::ostream& out, const std::array<double, Size>& value,
                          bool present) {
    out << '[';
    for (std::size_t index = 0; index < Size; ++index) {
      if (index) out << ',';
      if (present) writeNullable(out, value[index]);
      else out << "null";
    }
    out << ']';
  }

  void writeNullArray(std::ostream& out, std::size_t size) {
    out << '[';
    for (std::size_t index = 0; index < size; ++index) {
      if (index) out << ',';
      out << "null";
    }
    out << ']';
  }

  void writeLocalizationDiagnostics(
      std::ostream& out, const std::string& id,
      const hercules_interfaces::msg::LocalizationEstimate* estimate,
      const hercules_interfaces::msg::LocalizationDiagnostics* diagnostic,
      bool stale) {
    const bool has_diagnostic = diagnostic != nullptr;
    const auto algorithm = estimate != nullptr && !estimate->algorithm.empty()
        ? estimate->algorithm
        : (has_diagnostic && !diagnostic->algorithm.empty()
               ? diagnostic->algorithm : localization_algorithm_);
    const bool initialized = has_diagnostic ? diagnostic->initialized
                                            : (estimate != nullptr && estimate->initialized);
    const bool valid = has_diagnostic ? diagnostic->valid
                                      : (estimate != nullptr && estimate->valid);
    const bool diagnostic_stale = stale || (has_diagnostic && diagnostic->stale);
    const std::string status = has_diagnostic && !diagnostic->status.empty()
        ? diagnostic->status
        : (diagnostic_stale ? "stopped_stale_localization" : "missing");
    const auto counter = [diagnostic](std::uint64_t value) {
      return diagnostic == nullptr ? std::uint64_t{0} : value;
    };
    const auto real = [diagnostic](double value) {
      return diagnostic == nullptr ? std::numeric_limits<double>::quiet_NaN() : value;
    };
    const auto boolean = [diagnostic](bool value) {
      return diagnostic == nullptr ? false : value;
    };
    out << "{\"agent_id\":";
    writeJsonString(out, id);
    out << ",\"algorithm\":";
    writeJsonString(out, algorithm);
    out << ",\"initialized\":" << (initialized ? "true" : "false")
        << ",\"valid\":" << (valid ? "true" : "false")
        << ",\"stale\":" << (diagnostic_stale ? "true" : "false")
        << ",\"updates\":" << counter(has_diagnostic ? diagnostic->updates : 0)
        << ",\"measurements_received\":"
        << counter(has_diagnostic ? diagnostic->measurements_received : 0)
        << ",\"peer_estimates_received\":"
        << counter(has_diagnostic ? diagnostic->peer_estimates_received : 0)
        << ",\"measurements_rejected\":"
        << counter(has_diagnostic ? diagnostic->measurements_rejected : 0)
        << ",\"peer_estimates_rejected\":"
        << counter(has_diagnostic ? diagnostic->peer_estimates_rejected : 0)
        << ",\"accepted_relative_updates\":"
        << counter(has_diagnostic ? diagnostic->accepted_relative_updates : 0)
        << ",\"accepted_communication_updates\":"
        << counter(has_diagnostic ? diagnostic->accepted_communication_updates : 0)
        << ",\"rejected_relative_updates\":"
        << counter(has_diagnostic ? diagnostic->rejected_relative_updates : 0)
        << ",\"rejected_communication_updates\":"
        << counter(has_diagnostic ? diagnostic->rejected_communication_updates : 0)
        << ",\"duplicate_updates\":"
        << counter(has_diagnostic ? diagnostic->duplicate_updates : 0)
        << ",\"communication_rejections\":"
        << counter(has_diagnostic ? diagnostic->communication_rejections : 0)
        << ",\"innovation_norm\":";
    writeNullable(out, real(has_diagnostic ? diagnostic->innovation_norm :
                             std::numeric_limits<double>::quiet_NaN()));
    out << ",\"covariance_trace\":";
    writeNullable(out, real(has_diagnostic ? diagnostic->covariance_trace :
                             std::numeric_limits<double>::quiet_NaN()));
    out << ",\"sensor_age\":";
    writeNullable(out, real(has_diagnostic ? diagnostic->sensor_age :
                             std::numeric_limits<double>::quiet_NaN()));
    out << ",\"communication_age\":";
    writeNullable(out, real(has_diagnostic ? diagnostic->communication_age :
                             std::numeric_limits<double>::quiet_NaN()));
    out << ",\"covariance_spd\":"
        << (boolean(has_diagnostic && diagnostic->covariance_spd) ? "true" : "false")
        << ",\"robustness_margin\":";
    writeNullable(out, real(has_diagnostic ? diagnostic->robustness_margin :
                             std::numeric_limits<double>::quiet_NaN()));
    out << ",\"update_latency_ms\":";
    writeNullable(out, real(has_diagnostic ? diagnostic->update_latency_ms :
                             std::numeric_limits<double>::quiet_NaN()));
    out << ",\"status\":";
    writeJsonString(out, status);
    out << ",\"detail\":";
    writeJsonString(out, has_diagnostic ? diagnostic->detail : std::string{});
    out << ",\"failure_reason\":";
    writeJsonString(out, has_diagnostic ? diagnostic->failure_reason : std::string{});
    out << ",\"communication_state\":";
    writeJsonString(out, has_diagnostic ? diagnostic->communication_state : std::string{});
    out << ",\"header\":{\"frame_id\":";
    writeJsonString(out, has_diagnostic ? diagnostic->header.frame_id : std::string{});
    out << ",\"timestamp\":";
    if (has_diagnostic && timestampValid(diagnostic->header.stamp)) {
      writeFinite(out, static_cast<double>(diagnostic->header.stamp.sec) +
          1e-9 * static_cast<double>(diagnostic->header.stamp.nanosec));
    } else {
      out << "null";
    }
    out << "}}";
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
         << ",\"maicp\":{\"case\":\"" << maicp_case_
         << "\",\"logical_time\":" << static_cast<double>(step_) * config_.control_dt
         << ",\"margin_ugv\":" << cbf_config_.ugv_margin
         << ",\"margin_uav\":" << cbf_config_.uav_margin
         << ",\"returned_home\":{";
    for (std::size_t i = 0; i < std::size(kControlled); ++i) {
      if (i) log_ << ',';
      const std::string id = kControlled[i];
      const auto truth_position = agentState(states_.at(id)).position;
      if (periodicMission()) {
        maicp_travel_[id] += (truth_position.head<2>() - maicp_last_truth_.at(id).head<2>()).norm();
        maicp_last_truth_[id] = truth_position;
      }
      const double required_travel = id.rfind("Husky", 0) == 0
          ? 2.8 : 0.7 * 2.0 * std::acos(-1.0) * maicp_loop_radius_;
      const bool home = periodicMission() && maicp_returning_.at(id) &&
          maicp_travel_.at(id) >= required_travel &&
          (truth_position.head<2>() - maicp_truth_home_.at(id).head<2>()).norm() < 0.8;
      log_ << '\"' << id << "\":" << (home ? "true" : "false");
    }
    log_ << "}},\"vehicle_types\":{";
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
      log_ << ",\"yaw\":" << state.yaw << ",\"source_timestamp\":"
           << hercules_tracking_ros::timeSeconds(states_.at(kAll[i]).header.stamp) << '}';
    }
    log_ << "},\"localization\":{\"algorithm\":\"" << localization_algorithm_
         << "\",\"agents\":{";
    for (std::size_t i = 0; i < std::size(kControlled); ++i) {
      if (i) log_ << ',';
      const std::string id = kControlled[i];
      const auto estimate = localization_estimates_.find(id);
      const auto receipt = localization_receipts_.find(id);
      const bool fresh = localizationFresh(id);
      const auto diagnostic = localization_diagnostics_.find(id);
      const auto* estimate_value = estimate == localization_estimates_.end()
          ? nullptr : &estimate->second;
      const auto* diagnostic_value = diagnostic == localization_diagnostics_.end()
          ? nullptr : &diagnostic->second;
      const bool stale = !fresh;
      const bool initialized = estimate_value != nullptr && estimate_value->initialized;
      const bool valid = estimate_value != nullptr && estimate_value->valid;
      const auto algorithm = estimate_value != nullptr && !estimate_value->algorithm.empty()
          ? estimate_value->algorithm : localization_algorithm_;
      const auto counter = [diagnostic_value](std::uint64_t value) {
        return diagnostic_value == nullptr ? std::uint64_t{0} : value;
      };
      const auto real = [diagnostic_value](double value) {
        return diagnostic_value == nullptr
            ? std::numeric_limits<double>::quiet_NaN() : value;
      };
      log_ << '"' << id << "\":{\"agent_id\":";
      writeJsonString(log_, id);
      log_ << ",\"algorithm\":";
      writeJsonString(log_, algorithm);
      log_ << ",\"frame_id\":";
      if (estimate_value != nullptr) writeJsonString(log_, estimate_value->frame_id);
      else log_ << "null";
      log_ << ",\"initialized\":" << (initialized ? "true" : "false")
           << ",\"valid\":" << (valid ? "true" : "false")
           << ",\"stale\":" << (stale ? "true" : "false")
           << ",\"pose\":";
      if (estimate_value != nullptr) {
        log_ << '[';
        writeNullable(log_, estimate_value->position[0]);
        log_ << ',';
        writeNullable(log_, estimate_value->position[1]);
        log_ << ',';
        writeNullable(log_, estimate_value->yaw);
        log_ << ']';
      } else {
        writeNullArray(log_, 3);
      }
      log_ << ",\"planar_velocity\":";
      if (estimate_value != nullptr) {
        log_ << '[';
        writeNullable(log_, estimate_value->velocity[0]);
        log_ << ',';
        writeNullable(log_, estimate_value->velocity[1]);
        log_ << ']';
      } else {
        writeNullArray(log_, 2);
      }
      log_ << ",\"covariance\":";
      if (estimate_value != nullptr) {
        writeNullableArray(log_, estimate_value->covariance, true);
      } else {
        writeNullArray(log_, 9);
      }
      log_ << ",\"timestamp\":";
      if (estimate_value != nullptr && timestampValid(estimate_value->header.stamp)) {
        writeFinite(log_, static_cast<double>(estimate_value->header.stamp.sec) +
            1e-9 * static_cast<double>(estimate_value->header.stamp.nanosec));
      } else {
        log_ << "null";
      }
      log_ << ",\"receipt_age\":";
      writeNullable(log_, receipt == localization_receipts_.end()
          ? std::numeric_limits<double>::quiet_NaN() : seconds(receipt->second));
      log_ << ",\"sequence\":"
           << (estimate_value != nullptr ? estimate_value->sequence : 0);

      // Keep the historical flat aliases while also writing one complete,
      // nested diagnostics object.  Consumers can migrate without needing
      // special handling for records emitted before diagnostics arrived.
      log_ << ",\"updates\":" << counter(diagnostic_value ? diagnostic_value->updates : 0)
           << ",\"measurements_received\":"
           << counter(diagnostic_value ? diagnostic_value->measurements_received : 0)
           << ",\"measurements_rejected\":"
           << counter(diagnostic_value ? diagnostic_value->measurements_rejected : 0)
           << ",\"peer_estimates_received\":"
           << counter(diagnostic_value ? diagnostic_value->peer_estimates_received : 0)
           << ",\"peer_estimates_rejected\":"
           << counter(diagnostic_value ? diagnostic_value->peer_estimates_rejected : 0)
           << ",\"accepted_relative_updates\":"
           << counter(diagnostic_value ? diagnostic_value->accepted_relative_updates : 0)
           << ",\"accepted_communication_updates\":"
           << counter(diagnostic_value ? diagnostic_value->accepted_communication_updates : 0)
           << ",\"rejected_relative_updates\":"
           << counter(diagnostic_value ? diagnostic_value->rejected_relative_updates : 0)
           << ",\"rejected_communication_updates\":"
           << counter(diagnostic_value ? diagnostic_value->rejected_communication_updates : 0)
           << ",\"duplicate_updates\":"
           << counter(diagnostic_value ? diagnostic_value->duplicate_updates : 0)
           << ",\"communication_rejections\":"
           << counter(diagnostic_value ? diagnostic_value->communication_rejections : 0)
           << ",\"innovation_norm\":";
      writeNullable(log_, real(diagnostic_value ? diagnostic_value->innovation_norm :
                               std::numeric_limits<double>::quiet_NaN()));
      log_ << ",\"covariance_trace\":";
      writeNullable(log_, real(diagnostic_value ? diagnostic_value->covariance_trace :
                               std::numeric_limits<double>::quiet_NaN()));
      log_ << ",\"sensor_age\":";
      writeNullable(log_, real(diagnostic_value ? diagnostic_value->sensor_age :
                               std::numeric_limits<double>::quiet_NaN()));
      log_ << ",\"communication_age\":";
      writeNullable(log_, real(diagnostic_value ? diagnostic_value->communication_age :
                               std::numeric_limits<double>::quiet_NaN()));
      log_ << ",\"covariance_spd\":"
           << ((diagnostic_value != nullptr && diagnostic_value->covariance_spd)
                   ? "true" : "false")
           << ",\"robustness_margin\":";
      writeNullable(log_, real(diagnostic_value ? diagnostic_value->robustness_margin :
                               std::numeric_limits<double>::quiet_NaN()));
      log_ << ",\"update_latency_ms\":";
      writeNullable(log_, real(diagnostic_value ? diagnostic_value->update_latency_ms :
                               std::numeric_limits<double>::quiet_NaN()));
      log_ << ",\"status\":";
      writeJsonString(log_, diagnostic_value != nullptr && !diagnostic_value->status.empty()
          ? diagnostic_value->status : (stale ? "stopped_stale_localization" : "missing"));
      log_ << ",\"diagnostics\":";
      writeLocalizationDiagnostics(log_, id, estimate_value, diagnostic_value, stale);
      log_ << '}';
    }
    log_ << "}},\"calibrated_origins\":{";
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
  std::string localization_algorithm_{"recursive_decentralized"};
  std::string localization_source_{"estimate"};
  std::string actuation_profile_{"current_ros"};
  double duration_{30.0};
  double startup_timeout_{30.0};
  double freshness_timeout_{0.5};
  double localization_stale_after_{0.5};
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
  std::string maicp_case_;
  int maicp_steps_{100};
  double maicp_loop_radius_{2.5};
  std::map<std::string, Eigen::Vector3d> maicp_home_;
  std::map<std::string, Eigen::Vector3d> maicp_velocity_;
  std::map<std::string, Eigen::Vector3d> maicp_truth_home_, maicp_last_truth_;
  std::map<std::string, double> maicp_travel_;
  std::map<std::string, bool> maicp_returning_;
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
  std::map<std::string, hercules_interfaces::msg::LocalizationEstimate> localization_estimates_;
  std::map<std::string, hercules_interfaces::msg::LocalizationDiagnostics> localization_diagnostics_;
  std::map<std::string, Clock::time_point> localization_receipts_;
  std::map<std::string, LocalizationOrderKey> localization_order_;
  std::map<std::string, std::uint64_t> localization_rejections_;
  std::map<std::string, hercules_interfaces::msg::GroundTruthState> last_localized_states_;
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
  std::vector<rclcpp::Subscription<hercules_interfaces::msg::LocalizationEstimate>::SharedPtr>
      localization_subscriptions_;
  std::vector<rclcpp::Subscription<hercules_interfaces::msg::LocalizationDiagnostics>::SharedPtr>
      localization_diagnostic_subscriptions_;
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
