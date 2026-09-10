#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
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
#include <hercules_interfaces/msg/tracking_diagnostics.hpp>
#include <hercules_interfaces/msg/tracking_epoch.hpp>
#include <hercules_mission_core/formation_controller.hpp>
#include <hercules_mission_core/mission_config.hpp>
#include <hercules_mission_core/target_formation.hpp>
#include <hercules_mission_core/target_motion.hpp>
#include <rclcpp/rclcpp.hpp>

#include "hercules_mission_ros/mission_actuation.hpp"
#include "hercules_mission_ros/mission_gate.hpp"
#include "hercules_mission_ros/target_source.hpp"
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
    uav_velocity_limit_ = declare_parameter<double>("uav_velocity_limit", 3.0);
    log_path_ = declare_parameter<std::string>(
        "log_path", "ros2/validation/rural_nominal/artifacts/mission.jsonl");
    distributed_target_source_ = std::make_unique<DistributedTargetSource>(config_.tracking.window_seconds);

    for (const char* id : kAll) {
      state_subscriptions_.push_back(
          create_subscription<hercules_interfaces::msg::GroundTruthState>(
              std::string("/hercules_mission/ground_truth/") + id, 10,
              [this, id](hercules_interfaces::msg::GroundTruthState::ConstSharedPtr msg) {
                if (msg->valid && msg->header.frame_id == "airsim_world_ned") {
                  states_[id] = *msg;
                  receipts_[id] = Clock::now();
                }
              }));
      origin_subscriptions_.push_back(
          create_subscription<geometry_msgs::msg::PointStamped>(
              std::string("/hercules_mission/calibrated_origin/") + id,
              rclcpp::QoS(1).transient_local().reliable(),
              [this, id](geometry_msgs::msg::PointStamped::ConstSharedPtr msg) {
                origins_[id] = {msg->point.x, msg->point.y, msg->point.z};
              }));
    }
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
    motion.center = target.position;
    motion.center.z() = config_.target_ground_z;
    motion.route_heading = route_heading_;
    target_controller_ =
        std::make_unique<hercules_mission_core::FigureEightTargetController>(motion);
    target_controller_->setIndex(config_.target_start_sample_index);
    target_controller_->placeStartAt(target.position);
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

  hercules_mission_core::TargetEstimate controlEstimate(
      const std::string& id, const hercules_mission_core::AgentState& target,
      double mission_time) const {
    if (target_source_name_ == "truth") return target_source_.estimate(target);
    const auto found = local_estimates_.find(id);
    const auto receipt = estimate_receipts_.find(id);
    if (found == local_estimates_.end() || receipt == estimate_receipts_.end()) {
      return distributed_target_source_->estimate(std::nullopt, mission_time);
    }
    TimestampedTargetEstimate input;
    input.estimate.position = {found->second.position[0], found->second.position[1]};
    input.estimate.velocity = {found->second.velocity[0], found->second.velocity[1]};
    input.estimate.active = found->second.active;
    input.estimator_timestamp = hercules_tracking_ros::timeSeconds(found->second.stamp);
    input.receipt_age = seconds(receipt->second);
    return distributed_target_source_->estimate(input, mission_time);
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

    for (const char* id : kDrones) {
      const auto agent = agentState(states_.at(id));
      const auto target_estimate = controlEstimate(id, target, mission_time);
      control_estimates[id] = target_estimate;
      const auto acceleration = controller_.targetNominalControl(
          agent, target_estimate, route_heading_, config_.target_ground_z,
          config_.target_ugv_circumradius);
      const auto velocity = target_estimate.active
          ? uavVelocityCommand(agent.velocity, acceleration, config_.control_dt,
                               uav_velocity_limit_)
          : Eigen::Vector3d::Zero();
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
      const auto target_estimate = controlEstimate(id, target, mission_time);
      control_estimates[id] = target_estimate;
      const auto command = controller_.targetNominalUnicycleControl(
          agent, target_estimate, route_heading_, config_.target_ground_z,
          config_.target_ugv_circumradius);
      commands[id] = {command.x(), command.y(), 0.0};
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
      if (!dry_run_ && enable_formation_ && target_estimate.active) {
        publishCar(id, ugvCarCommand(command.x(), command.y(),
                                     agent.velocity.head<2>().norm(),
                                     config_.formation.ugv_max_yaw_rate));
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
                control_estimates, step_max_uav, step_max_ugv);
    ++step_;
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

  void writeRecord(const hercules_mission_core::AgentState& target,
                   const Eigen::Vector2d& target_command,
                   const std::map<std::string, Eigen::Vector3d>& commands,
                   const std::map<std::string, Eigen::Vector3d>& desired_slots,
                   const std::map<std::string, double>& slot_errors,
                   const std::map<std::string, hercules_mission_core::TargetEstimate>& control_estimates,
                   double uav_error, double ugv_error) {
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
         << ",\"collision\":{\"relevant\":false},\"pattern\":{\"type\":\"figure_eight\",\"center\":";
    writeVector(log_, target_controller_->config().center);
    log_ << ",\"route_heading\":" << route_heading_
         << ",\"longitudinal_span\":" << config_.target_motion.longitudinal_span
         << ",\"lateral_span\":" << config_.target_motion.lateral_span << "}},";
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
    log_ << "],"
            "\"safety_communication_links\":[],\"collisions\":{}}\n";
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
  double duration_{30.0};
  double startup_timeout_{30.0};
  double freshness_timeout_{0.5};
  double route_heading_{0.0};
  double uav_velocity_limit_{3.0};
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
  rclcpp::Publisher<hercules_interfaces::msg::TrackingEpoch>::SharedPtr epoch_publisher_;
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
