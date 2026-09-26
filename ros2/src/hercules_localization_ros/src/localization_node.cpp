#include "hercules_localization_ros/localization_adapter.hpp"

#include <airsim_interfaces/msg/gps_yaw.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <hercules_interfaces/msg/global_ci_belief.hpp>
#include <hercules_interfaces/srv/recursive_localization_pair.hpp>
#include <hercules_interfaces/srv/reset_localization.hpp>
#include <hercules_interfaces/srv/set_localization_algorithm.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <stdexcept>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>

namespace hercules_localization_ros {
namespace {

using SetLocalizationAlgorithm = hercules_interfaces::srv::SetLocalizationAlgorithm;
using ResetLocalization = hercules_interfaces::srv::ResetLocalization;
using RecursiveLocalizationPair = hercules_interfaces::srv::RecursiveLocalizationPair;
using Clock = std::chrono::steady_clock;

double yawFromQuaternion(double x, double y, double z, double w) {
  return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

bool finite(double value) { return std::isfinite(value); }

template <std::size_t Size>
bool finiteArray(const std::array<double, Size> &values) {
  return std::all_of(values.begin(), values.end(), finite);
}

Eigen::Matrix3d matrix3FromRowMajor(const std::array<double, 9> &values) {
  Eigen::Matrix3d result;
  for (Eigen::Index row = 0; row < 3; ++row) {
    for (Eigen::Index column = 0; column < 3; ++column) {
      result(row, column) = values[static_cast<std::size_t>(3 * row + column)];
    }
  }
  return result;
}

std::array<double, 9> matrix3ToRowMajor(const Eigen::Matrix3d &matrix) {
  std::array<double, 9> result{};
  for (Eigen::Index row = 0; row < 3; ++row) {
    for (Eigen::Index column = 0; column < 3; ++column) {
      result[static_cast<std::size_t>(3 * row + column)] = matrix(row, column);
    }
  }
  return result;
}

bool positiveDefinite(const Eigen::Matrix3d &matrix) {
  if (!matrix.allFinite() || !matrix.isApprox(matrix.transpose(), 1e-9)) return false;
  return Eigen::LLT<Eigen::Matrix3d>(matrix).info() == Eigen::Success;
}

bool positiveDefiniteRowMajor(const std::vector<double> &values,
                              std::size_t dimension) {
  if (dimension == 0U || values.size() != dimension * dimension ||
      !std::all_of(values.begin(), values.end(), finite)) return false;
  Eigen::MatrixXd matrix(static_cast<Eigen::Index>(dimension),
                         static_cast<Eigen::Index>(dimension));
  for (std::size_t row = 0; row < dimension; ++row) {
    for (std::size_t column = 0; column < dimension; ++column) {
      matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) =
          values[row * dimension + column];
    }
  }
  if (!matrix.isApprox(matrix.transpose(), 1e-9)) return false;
  return Eigen::LLT<Eigen::MatrixXd>(matrix).info() == Eigen::Success;
}

hercules_localization::PoseEstimate poseEstimate(
    const std::array<double, 3> &pose,
    const std::array<double, 9> &covariance) {
  hercules_localization::PoseEstimate result;
  result.mean = hercules_localization::Pose2D(pose[0], pose[1], pose[2]);
  result.covariance = matrix3FromRowMajor(covariance);
  result.valid = finiteArray(pose) && positiveDefinite(result.covariance);
  return result;
}

std::array<double, 3> poseArray(const hercules_localization::PoseEstimate &estimate) {
  return {estimate.mean.position.x(), estimate.mean.position.y(), estimate.mean.yaw};
}

std::string withDefault(const std::string &value, const std::string &fallback) {
  return value.empty() ? fallback : value;
}

hercules_localization::MaicpClass maicpClassFromParameter(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  if (value.empty() || value == "unknown") {
    return hercules_localization::MaicpClass::kUnknown;
  }
  if (value == "ugv" || value == "ground" || value == "ground_vehicle") {
    return hercules_localization::MaicpClass::kUgv;
  }
  if (value == "uav" || value == "drone" || value == "air") {
    return hercules_localization::MaicpClass::kUav;
  }
  throw std::invalid_argument("maicp_class must be ugv, uav, or unknown");
}

}  // namespace

class LocalizationNode final : public rclcpp::Node {
public:
  explicit LocalizationNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
      : rclcpp::Node("localization_node", options),
        agent_id_(declare_parameter<std::string>("agent_id", "agent")),
        adapter_(agent_id_, declare_parameter<std::string>("algorithm", kRecursiveDecentralized)) {
    adapter_.setAnchorAgent(declare_parameter<std::string>("anchor_agent", "Drone1"));
    const std::string base = "/hercules_localization/" + agent_id_;
    const std::string vehicle_type = declare_parameter<std::string>("vehicle_type", "");
    const std::string vehicle_prefix = vehicle_type.empty()
        ? "/" + agent_id_
        : "/hercules_" + vehicle_type + "/" + agent_id_;
    const std::string default_measurement = base + "/measurement";
    const std::string default_odom = vehicle_prefix + "/ground_truth/odom_local";
    const std::string default_gps = vehicle_prefix + "/global_gps";

    measurement_topic_ = withDefault(
        declare_parameter<std::string>("measurement_topic", ""), default_measurement);
    const auto odom_local_topic = declare_parameter<std::string>("odom_local_topic", "");
    odometry_topic_ = withDefault(
        odom_local_topic,
        withDefault(declare_parameter<std::string>("odometry_topic", ""), default_odom));
    const auto global_gps_topic = declare_parameter<std::string>("global_gps_topic", "");
    gps_topic_ = withDefault(
        global_gps_topic, withDefault(declare_parameter<std::string>("gps_topic", ""), default_gps));
    odom_origin_topic_ = declare_parameter<std::string>("odom_origin_topic", "");
    origin_topic_ = declare_parameter<std::string>(
        "gps_origin_topic", "/hercules_drone/origin_geo_point");
    secondary_origin_topic_ = declare_parameter<std::string>(
        "gps_origin_topic_secondary", "/hercules_ugv/origin_geo_point");
    gps_reference_latitude_ = declare_parameter<double>(
        "gps_origin_latitude", std::numeric_limits<double>::quiet_NaN());
    gps_reference_longitude_ = declare_parameter<double>(
        "gps_origin_longitude", std::numeric_limits<double>::quiet_NaN());
    gps_reference_altitude_ = declare_parameter<double>(
        "gps_origin_altitude", std::numeric_limits<double>::quiet_NaN());
    gps_reference_configured_ = finite(gps_reference_latitude_) &&
                                finite(gps_reference_longitude_) &&
                                finite(gps_reference_altitude_);
    gps_reference_set_ = gps_reference_configured_;
    relative_topic_ = withDefault(
        declare_parameter<std::string>("relative_topic", ""), "/hercules_localization/relative");
    global_ci_topic_ = withDefault(
        declare_parameter<std::string>("global_ci_topic", ""), "/hercules_localization/gs_ci");
    global_agent_ids_ = declare_parameter<std::vector<std::string>>(
        "global_agent_ids", std::vector<std::string>{agent_id_});
    const std::set<std::string> unique_global_ids(
        global_agent_ids_.begin(), global_agent_ids_.end());
    if (global_agent_ids_.empty() || unique_global_ids.size() != global_agent_ids_.size() ||
        unique_global_ids.count(agent_id_) == 0U || unique_global_ids.count("Target1") != 0U ||
        unique_global_ids.count("") != 0U) {
      throw std::invalid_argument(
          "global_agent_ids must be unique, exclude Target1, and include agent_id");
    }
    if (!adapter_.setGlobalAgentIds(global_agent_ids_)) {
      throw std::invalid_argument("failed to configure global_agent_ids");
    }
    peer_topic_ = withDefault(
        declare_parameter<std::string>("peer_topic", ""), "/hercules_localization/peer_estimate");
    estimate_topic_ = withDefault(
        declare_parameter<std::string>("estimate_topic", ""), base + "/estimate");
    diagnostics_topic_ = withDefault(
        declare_parameter<std::string>("diagnostics_topic", ""), base + "/diagnostics");
    peer_output_topic_ = withDefault(
        declare_parameter<std::string>("peer_output_topic", ""), "/hercules_localization/peer_estimate");
    hercules_localization::LocalizationConfig localization_config;
    localization_config.covariance_floor = declare_parameter<double>("covariance_floor", 1e-9);
    localization_config.process_covariance_floor = declare_parameter<double>(
        "process_covariance_floor", localization_config.covariance_floor);
    localization_config.measurement_covariance_floor = declare_parameter<double>(
        "measurement_covariance_floor", localization_config.covariance_floor);
    localization_config.lambda = declare_parameter<double>("dcl_lambda", 1.0);
    localization_config.ci_self_weight = declare_parameter<double>("ci_self_weight", 0.8);
    localization_config.max_iterations = declare_parameter<int>("ci_max_iterations", 10);
    localization_config.unknown_motion_variance = declare_parameter<double>(
        "unknown_motion_variance", 0.25);
    localization_config.communication_timeout_sec = declare_parameter<double>(
        "pair_transaction_timeout_sec", 0.5);
    localization_config.maicp_enabled = declare_parameter<bool>("maicp_enabled", false);
    localization_config.maicp_margin = declare_parameter<double>("maicp_margin", 0.0);
    localization_config.maicp_covariance_gain = declare_parameter<double>(
        "maicp_covariance_gain", 0.0);
    std::string maicp_class = declare_parameter<std::string>(
        "maicp_class", vehicle_type.empty() ? "unknown" : vehicle_type);
    if ((maicp_class.empty() || maicp_class == "unknown") && !vehicle_type.empty()) {
      maicp_class = vehicle_type;
    }
    localization_config.maicp_class = maicpClassFromParameter(maicp_class);
    if (!finite(localization_config.process_covariance_floor) ||
        localization_config.process_covariance_floor <= 0.0 ||
        !finite(localization_config.measurement_covariance_floor) ||
        localization_config.measurement_covariance_floor <= 0.0 ||
        !finite(localization_config.unknown_motion_variance) ||
        localization_config.unknown_motion_variance < 0.0 ||
        !finite(localization_config.communication_timeout_sec) ||
        localization_config.communication_timeout_sec <= 0.0 ||
        !finite(localization_config.maicp_margin) ||
        localization_config.maicp_margin < 0.0 ||
        !finite(localization_config.maicp_covariance_gain) ||
        localization_config.maicp_covariance_gain < 0.0) {
      throw std::invalid_argument("invalid localization covariance or communication timeout");
    }
    adapterConfigTimeout_ = localization_config.communication_timeout_sec;
    communication_rate_hz_ = declare_parameter<double>("communication_rate_hz", 2.0);
    if (!finite(communication_rate_hz_) || communication_rate_hz_ <= 0.0) {
      throw std::invalid_argument("communication_rate_hz must be positive");
    }
    // Transport timing/range parameters are declared at the node boundary so
    // launch files and recorded runs have one typed configuration surface.
    // The relative observer owns the actual camera scheduling.
    stale_after_sec_ = declare_parameter<double>("stale_after_sec", 0.5);
    if (!finite(stale_after_sec_) || stale_after_sec_ <= 0.0) {
      throw std::invalid_argument("stale_after_sec must be positive");
    }
    declare_parameter<double>("camera_range_m", 100.0);
    declare_parameter<double>("camera_rate_hz", 2.0);
    localization_config_ = localization_config;
    adapter_.setConfig(localization_config_);

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(20)).reliable();
    measurement_subscription_ = create_subscription<LocalizationMeasurement>(
        measurement_topic_, qos,
        [this](LocalizationMeasurement::ConstSharedPtr message) { onMeasurement(*message); });
    odometry_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
        odometry_topic_, qos,
        [this](nav_msgs::msg::Odometry::ConstSharedPtr message) { onOdometry(*message); });
    gps_subscription_ = create_subscription<sensor_msgs::msg::NavSatFix>(
        gps_topic_, qos,
        [this](sensor_msgs::msg::NavSatFix::ConstSharedPtr message) { onGps(*message); });
    if (!odom_origin_topic_.empty()) {
      odom_origin_subscription_ = create_subscription<geometry_msgs::msg::PointStamped>(
          odom_origin_topic_, rclcpp::QoS(1).transient_local().reliable(),
          [this](geometry_msgs::msg::PointStamped::ConstSharedPtr message) {
            if (!message ||
                (!message->header.frame_id.empty() &&
                 message->header.frame_id != "airsim_world_ned") ||
                !finite(message->point.x) || !finite(message->point.y) ||
                !finite(message->point.z)) {
              return;
            }
            odom_origin_ned_ = {message->point.x, message->point.y, message->point.z};
          });
    }
    const auto on_origin = [this](airsim_interfaces::msg::GPSYaw::ConstSharedPtr message) {
          if (!message || gps_reference_configured_ ||
              !finite(message->latitude) || !finite(message->longitude) ||
              !finite(message->altitude)) {
            return;
          }
          gps_reference_latitude_ = message->latitude;
          gps_reference_longitude_ = message->longitude;
          gps_reference_altitude_ = message->altitude;
          gps_reference_set_ = true;
        };
    origin_subscription_ = create_subscription<airsim_interfaces::msg::GPSYaw>(
        origin_topic_, qos, on_origin);
    if (!secondary_origin_topic_.empty() && secondary_origin_topic_ != origin_topic_) {
      secondary_origin_subscription_ = create_subscription<airsim_interfaces::msg::GPSYaw>(
          secondary_origin_topic_, qos, on_origin);
    }
    relative_subscription_ = create_subscription<PlanarRelativeMeasurement>(
        relative_topic_, qos,
        [this](PlanarRelativeMeasurement::ConstSharedPtr message) { onRelative(*message); });
    peer_subscription_ = create_subscription<LocalizationPeerEstimate>(
        peer_topic_, qos,
        [this](LocalizationPeerEstimate::ConstSharedPtr message) { onPeer(*message); });
    global_ci_subscription_ = create_subscription<hercules_interfaces::msg::GlobalCiBelief>(
        global_ci_topic_, qos,
        [this](hercules_interfaces::msg::GlobalCiBelief::ConstSharedPtr message) {
          onGlobalCi(*message);
        });

    estimate_publisher_ = create_publisher<LocalizationEstimate>(estimate_topic_, qos);
    diagnostics_publisher_ = create_publisher<LocalizationDiagnostics>(diagnostics_topic_, qos);
    peer_publisher_ = create_publisher<LocalizationPeerEstimate>(peer_output_topic_, qos);
    global_ci_publisher_ = create_publisher<hercules_interfaces::msg::GlobalCiBelief>(
        global_ci_topic_, qos);
    freshness_timer_ = create_wall_timer(std::chrono::milliseconds(100),
                                         [this]() { publishFreshness(); });

    algorithm_service_ = create_service<SetLocalizationAlgorithm>(
        "~/set_algorithm",
        [this](const std::shared_ptr<SetLocalizationAlgorithm::Request> request,
               std::shared_ptr<SetLocalizationAlgorithm::Response> response) {
          const auto requested = LocalizationAdapter::canonicalAlgorithm(request->algorithm);
          const bool switching = !requested.empty() && requested != adapter_.algorithm();
          // Algorithm selection is a launch-time choice.  A service call may
          // acknowledge the already-active value, but changing algorithms in
          // a running node would mix estimator state and is intentionally
          // rejected; restart with a fresh launch for a clean state.
          const bool accepted = !switching && adapter_.setAlgorithm(request->algorithm);
          response->accepted = accepted;
          response->active_algorithm = adapter_.algorithm();
          response->message = accepted ? "algorithm selected"
                                       : (switching ? "restart before switching algorithms"
                                                         : "unsupported algorithm");
        });
    reset_service_ = create_service<ResetLocalization>(
        "~/reset",
        [this](const std::shared_ptr<ResetLocalization::Request> /*request*/,
               std::shared_ptr<ResetLocalization::Response> response) {
          peer_estimates_.clear();
          global_ci_beliefs_.clear();
          relative_measurements_.clear();
          pair_events_.clear();
          cached_pair_responses_.clear();
          pending_recursive_pairs_.clear();
          inflight_recursive_pairs_.clear();
          sent_recursive_sequences_.clear();
          peer_receipts_.clear();
          global_ci_receipts_.clear();
          relative_receipts_.clear();
          duplicate_updates_ = 0;
          communication_rejections_ = 0;
          accepted_communication_updates_ = 0;
          rejected_relative_updates_ = 0;
          rejected_communication_updates_ = 0;
          last_global_ci_publish_ = Clock::time_point{};
          have_last_output_ = false;
          have_odometry_sample_ = false;
          pending_gps_.reset();
          gps_reference_set_ = gps_reference_configured_;
          adapter_.reset();
          algorithm_locked_ = false;
          response->success = true;
          response->message = "localization transport reset";
        });
    pair_service_ = create_service<RecursiveLocalizationPair>(
        "~/recursive_pair",
        [this](const std::shared_ptr<RecursiveLocalizationPair::Request> request,
               std::shared_ptr<RecursiveLocalizationPair::Response> response) {
          if (adapter_.algorithm() != kRecursiveDecentralized) {
            ++communication_rejections_;
            ++rejected_communication_updates_;
            response->accepted = false;
            response->duplicate = false;
            response->message = "recursive pair transactions require recursive_decentralized";
            response->sequence = adapter_.sequence();
            return;
          }
          auto measurement = request->measurement;
          if ((!request->source_id.empty() && !measurement.source_id.empty() &&
               request->source_id != measurement.source_id) ||
              (!request->target_id.empty() && !measurement.target_id.empty() &&
               request->target_id != measurement.target_id)) {
            ++communication_rejections_;
            ++rejected_relative_updates_;
            response->message = "conflicting pair identifiers";
            return;
          }
          if (measurement.source_id.empty()) measurement.source_id = request->source_id;
          if (measurement.target_id.empty()) measurement.target_id = request->target_id;
          if (measurement.observer_id.empty()) {
            measurement.observer_id = measurement.source_id;
          }
          if (measurement.observed_id.empty()) {
            measurement.observed_id = measurement.target_id;
          }
          if (request->pair_sequence != 0U) {
            if (measurement.sequence != 0U &&
                measurement.sequence != request->pair_sequence) {
              ++communication_rejections_;
              ++rejected_relative_updates_;
              response->accepted = false;
              response->duplicate = false;
              response->message = "conflicting pair sequence fields";
              response->sequence = adapter_.sequence();
              return;
            }
            measurement.sequence = request->pair_sequence;
          }
          const std::string event_id = request->event_id.empty()
              ? measurement.capture_id : request->event_id;
          const auto key = pairKey(measurement);
          if (event_id.empty() && measurement.sequence == 0U) {
            ++communication_rejections_;
            ++rejected_relative_updates_;
            response->message = "pair transaction requires event_id or nonzero sequence";
            return;
          }
          const std::string cache_key = key + "#" +
              (!event_id.empty() ? event_id : std::to_string(measurement.sequence));
          const auto cached = cached_pair_responses_.find(cache_key);
          if (cached != cached_pair_responses_.end()) {
            ++duplicate_updates_;
            *response = cached->second;
            response->duplicate = true;
            response->message = "cached pair transaction response";
            response->cached_event_id = event_id;
            return;
          }

          Eigen::Matrix2d measurement_covariance;
          measurement_covariance << measurement.covariance[0], measurement.covariance[1],
              measurement.covariance[2], measurement.covariance[3];
          const Eigen::LLT<Eigen::Matrix2d> measurement_factor(measurement_covariance);
          const bool addressed_to_observed = measurement.target_id == agent_id_ &&
              measurement.source_id != agent_id_ && measurement.source_id != "Target1";
          const auto &local_state = adapter_.recursiveState();
          const bool valid_wire = addressed_to_observed && measurement.valid &&
              finiteArray(measurement.relative_position) &&
              finite(measurement.range) && measurement.range > 0.0 &&
              finite(measurement.bearing) && measurement_covariance.allFinite() &&
              measurement_covariance.isApprox(measurement_covariance.transpose(), 1e-9) &&
              measurement_factor.info() == Eigen::Success &&
              finiteArray(request->observer_pose) && finiteArray(request->observer_covariance) &&
              finiteArray(request->correlation_factor) && local_state.has_value();
          const auto observer = poseEstimate(request->observer_pose, request->observer_covariance);
          if (!valid_wire || !observer.valid) {
            ++communication_rejections_;
            ++rejected_relative_updates_;
            response->accepted = false;
            response->duplicate = false;
            response->message = local_state.has_value()
                ? "invalid recursive pair transaction" : "recursive state unavailable";
            response->sequence = measurement.sequence;
            return;
          }

          const auto reciprocal = local_state->correlationFactors().find(measurement.source_id);
          const Eigen::Matrix3d reciprocal_factor = reciprocal ==
                  local_state->correlationFactors().end()
              ? Eigen::Matrix3d::Zero() : reciprocal->second;
          hercules_localization::RecursiveDecentralizedState evaluator(
              measurement.source_id, observer, localization_config_);
          evaluator.initializePeer(agent_id_, matrix3FromRowMajor(request->correlation_factor));
          hercules_localization::RecursivePairTransaction transaction;
          transaction.event_id = event_id;
          transaction.observer_id = measurement.source_id;
          transaction.observed_id = agent_id_;
          transaction.sequence = measurement.sequence;
          transaction.measurement.neighbor_id = agent_id_;
          transaction.measurement.range = measurement.range;
          transaction.measurement.bearing = measurement.bearing;
          transaction.measurement.covariance = measurement_covariance;
          transaction.measurement.timestamp = measurement.timestamp;
          transaction.measurement.valid = true;
          transaction.observed = local_state->estimate();
          transaction.correlation_factor = matrix3FromRowMajor(request->correlation_factor);
          transaction.reciprocal_factor = reciprocal_factor;
          const auto pair_result = evaluator.applyPairTransaction(transaction);
          if (!pair_result.accepted || pair_result.duplicate) {
            ++communication_rejections_;
            ++rejected_relative_updates_;
            response->message = pair_result.status;
            response->sequence = measurement.sequence;
            return;
          }
          const auto installed = adapter_.installRecursivePairPosterior(
              measurement.source_id, pair_result.observed_estimate,
              pair_result.reciprocal_factor, event_id, measurement.sequence);
          if (!installed.accepted) {
            ++communication_rejections_;
            ++rejected_relative_updates_;
            response->message = installed.status;
            response->sequence = measurement.sequence;
            return;
          }

          response->accepted = true;
          response->duplicate = false;
          response->message = "recursive pair transaction applied";
          response->sequence = measurement.sequence;
          response->cached_event_id = event_id;
          response->observer_posterior_pose = poseArray(pair_result.result.estimate);
          response->observer_posterior_covariance = matrix3ToRowMajor(
              pair_result.result.estimate.covariance);
          response->observed_posterior_pose = poseArray(pair_result.observed_estimate);
          response->observed_posterior_covariance = matrix3ToRowMajor(
              pair_result.observed_estimate.covariance);
          response->correlation_factor = matrix3ToRowMajor(pair_result.correlation_factor);
          response->reciprocal_factor = matrix3ToRowMajor(pair_result.reciprocal_factor);
          cached_pair_responses_[cache_key] = *response;
          ++accepted_communication_updates_;
          if (!event_id.empty()) pair_events_.insert(event_id);
        });

    RCLCPP_INFO(get_logger(), "localization transport for %s using %s", agent_id_.c_str(),
                adapter_.algorithm().c_str());
  }

private:
  static std::string pairKey(const PlanarRelativeMeasurement &measurement) {
    return measurement.source_id + "->" + measurement.target_id;
  }

  bool dispatchRecursivePair(const PlanarRelativeMeasurement &measurement) {
    const std::string key = pairKey(measurement);
    if (inflight_recursive_pairs_.count(key) != 0U) return false;
    const auto &state = adapter_.recursiveState();
    if (!state.has_value() || !state->estimate().valid) return false;
    auto &client = pair_clients_[measurement.target_id];
    if (!client) {
      client = create_client<RecursiveLocalizationPair>(
          "/localization_" + measurement.target_id + "/recursive_pair");
    }
    if (!client->service_is_ready()) return false;

    auto request = std::make_shared<RecursiveLocalizationPair::Request>();
    request->source_id = agent_id_;
    request->target_id = measurement.target_id;
    request->event_id = measurement.capture_id.empty()
        ? key + ":" + std::to_string(measurement.sequence)
        : measurement.capture_id;
    request->measurement = measurement;
    request->pair_sequence = measurement.sequence;
    request->observer_pose = poseArray(state->estimate());
    request->observer_covariance = matrix3ToRowMajor(state->estimate().covariance);
    const auto factor = state->correlationFactors().find(measurement.target_id);
    request->correlation_factor = matrix3ToRowMajor(
        factor == state->correlationFactors().end()
            ? Eigen::Matrix3d::Zero() : factor->second);
    const auto peer = peer_estimates_.find(measurement.target_id);
    if (peer != peer_estimates_.end()) {
      request->observed_pose = {peer->second.position[0], peer->second.position[1],
                                peer->second.yaw};
      request->observed_covariance = peer->second.covariance;
    }

    inflight_recursive_pairs_.insert(key);
    pending_recursive_pairs_.erase(key);
    client->async_send_request(
        request,
        [this, key, measurement, event_id = request->event_id](
            rclcpp::Client<RecursiveLocalizationPair>::SharedFuture future) {
          inflight_recursive_pairs_.erase(key);
          try {
            const auto response = future.get();
            if (response->accepted) {
              auto posterior = poseEstimate(response->observer_posterior_pose,
                                            response->observer_posterior_covariance);
              posterior.timestamp = measurement.timestamp;
              const auto installed = adapter_.installRecursivePairPosterior(
                  measurement.target_id, posterior,
                  matrix3FromRowMajor(response->correlation_factor),
                  event_id, measurement.sequence);
              if (installed.accepted) {
                sent_recursive_sequences_[key] = measurement.sequence;
                ++accepted_communication_updates_;
                if (!event_id.empty()) pair_events_.insert(event_id);
              } else {
                ++communication_rejections_;
                ++rejected_relative_updates_;
              }
            } else {
              ++communication_rejections_;
              ++rejected_relative_updates_;
              const auto receipt = relative_receipts_.find(key);
              if (response->message == "recursive state unavailable" &&
                  receipt != relative_receipts_.end() &&
                  std::chrono::duration<double>(Clock::now() - receipt->second).count() <=
                      adapterCommunicationTimeout()) {
                pending_recursive_pairs_[key] = measurement;
              }
            }
          } catch (const std::exception &) {
            ++communication_rejections_;
            ++rejected_communication_updates_;
            const auto receipt = relative_receipts_.find(key);
            if (receipt != relative_receipts_.end() &&
                std::chrono::duration<double>(Clock::now() - receipt->second).count() <=
                    adapterCommunicationTimeout()) {
              pending_recursive_pairs_[key] = measurement;
            }
          }
          const auto pending = pending_recursive_pairs_.find(key);
          if (pending != pending_recursive_pairs_.end()) {
            dispatchRecursivePair(pending->second);
          }
        });
    return true;
  }

  LocalizationMeasurement measurementFromOdometry(const nav_msgs::msg::Odometry &message) const {
    LocalizationMeasurement measurement;
    measurement.header = message.header;
    measurement.header.frame_id = "airsim_world_ned";
    measurement.agent_id = agent_id_;
    measurement.source_id = "odometry";
    measurement.frame_id = "airsim_world_ned";
    const auto &position = message.pose.pose.position;
    const auto &orientation = message.pose.pose.orientation;
    const auto &velocity = message.twist.twist.linear;
    // The AirSim wrapper publishes ROS odometry with Y/Z negated.  Convert
    // back to the canonical AirSim NED convention at this boundary, matching
    // the inverse used by the mission state adapter.
    measurement.position = {position.x, -position.y, -position.z};
    // Mission state calibration publishes a static per-agent NED translation.
    // Apply the same translation to every sample.  The adapter derives motion
    // from consecutive odometry positions, so dropping it after initialization
    // would turn a constant frame offset into a false position jump.
    if (odom_origin_ned_.has_value()) {
      for (std::size_t index = 0; index < measurement.position.size(); ++index) {
        measurement.position[index] += (*odom_origin_ned_)[index];
      }
    }
    measurement.velocity = {velocity.x, -velocity.y, -velocity.z};
    measurement.orientation = {orientation.w, orientation.x, -orientation.y, -orientation.z};
    measurement.yaw = yawFromQuaternion(orientation.x, -orientation.y,
                                        -orientation.z, orientation.w);
    measurement.yaw_rate = -message.twist.twist.angular.z;
    // nav_msgs uses a 6x6 pose covariance with x/y/z/roll/pitch/yaw ordering.
    // Localization is planar, so retain the x/y/yaw principal block rather
    // than treating z as yaw.  The Y and yaw sign changes also transform the
    // corresponding covariance rows/columns.
    measurement.covariance = {message.pose.covariance[0], -message.pose.covariance[1], -message.pose.covariance[5],
                              -message.pose.covariance[6], message.pose.covariance[7], message.pose.covariance[11],
                              -message.pose.covariance[30], message.pose.covariance[31], message.pose.covariance[35]};
    if (!finite(measurement.covariance[0]) || measurement.covariance[0] <= 0.0 ||
        !finite(measurement.covariance[4]) || measurement.covariance[4] <= 0.0 ||
        !finite(measurement.covariance[8]) || measurement.covariance[8] <= 0.0) {
      // AirSim's simulator odometry intentionally reports zero covariance;
      // use a small finite floor so it remains a valid motion input while
      // the estimator still adds its configured process uncertainty.
      measurement.covariance[0] = 1e-4;
      measurement.covariance[4] = 1e-4;
      measurement.covariance[8] = 1e-4;
    }
    measurement.valid = finite(position.x) && finite(position.y) && finite(position.z) &&
                        finite(velocity.x) && finite(velocity.y) && finite(velocity.z);
    return measurement;
  }

  LocalizationMeasurement measurementFromGps(const sensor_msgs::msg::NavSatFix &message) {
    LocalizationMeasurement measurement;
    measurement.header = message.header;
    measurement.header.frame_id = "airsim_world_ned";
    measurement.agent_id = agent_id_;
    measurement.source_id = "gps";
    measurement.frame_id = "airsim_world_ned";
    // AirSim reports latitude/longitude in degrees. Convert to a local NED
    // frame around the common AirSim origin.  Never adopt a per-agent first
    // fix as an origin: doing so would put otherwise valid estimates in
    // different local frames when origin delivery is delayed.
    constexpr double kPi = 3.141592653589793238462643383279502884;
    constexpr double kEarthRadiusMetres = 6378137.0;
    const double latitude_scale = kEarthRadiusMetres * kPi / 180.0;
    const double longitude_scale = latitude_scale *
        std::cos(gps_reference_latitude_ * kPi / 180.0);
    measurement.position = {
        (message.latitude - gps_reference_latitude_) * latitude_scale,
        (message.longitude - gps_reference_longitude_) * longitude_scale,
        gps_reference_altitude_ - message.altitude};
    measurement.orientation = {1.0, 0.0, 0.0, 0.0};
    // NavSatFix covariance is ENU.  Planar AirSim localization is NED, so x
    // is north and y is east.  The third estimator component is yaw rather
    // than vertical position and therefore has no GPS cross-covariance.
    measurement.covariance = {message.position_covariance[4], message.position_covariance[3], 0.0,
                              message.position_covariance[1], message.position_covariance[0], 0.0,
                              0.0, 0.0, 1.0};
    // Some AirSim versions publish an all-zero/unknown NavSatFix covariance.
    // Keep the wrapper's EPH/EPV values when present, but retain the same
    // conservative 0.5 m horizontal fallback at this ROS boundary too.
    const bool horizontal_covariance_valid =
        finite(measurement.covariance[0]) && finite(measurement.covariance[4]) &&
        measurement.covariance[0] > 0.0 && measurement.covariance[4] > 0.0;
    if (!horizontal_covariance_valid) {
      measurement.covariance[0] = 0.25;
      measurement.covariance[4] = 0.25;
    }
    // NavSatFix has no heading covariance; the final planar state component
    // is yaw supplied by odometry, so do not reinterpret vertical GPS error as
    // heading uncertainty.
    measurement.covariance[8] = 1.0;
    measurement.valid = gps_reference_set_ &&
                        message.status.status >= sensor_msgs::msg::NavSatStatus::STATUS_FIX &&
                        finite(message.latitude) && finite(message.longitude) && finite(message.altitude);
    return measurement;
  }

  void onMeasurement(const LocalizationMeasurement &measurement) {
    const std::vector<LocalizationPeerEstimate> peers = [&]() {
      std::vector<LocalizationPeerEstimate> values;
      values.reserve(peer_estimates_.size());
      for (const auto &entry : peer_estimates_) {
        const auto receipt = peer_receipts_.find(entry.first);
        if (receipt != peer_receipts_.end() &&
            std::chrono::duration<double>(Clock::now() - receipt->second).count() >
                adapterCommunicationTimeout()) continue;
        values.push_back(entry.second);
      }
      return values;
    }();
    const std::vector<PlanarRelativeMeasurement> relative = [&]() {
      std::vector<PlanarRelativeMeasurement> values;
      values.reserve(relative_measurements_.size());
      for (const auto &entry : relative_measurements_) {
        const auto receipt = relative_receipts_.find(entry.first);
        if (receipt != relative_receipts_.end() &&
            std::chrono::duration<double>(Clock::now() - receipt->second).count() >
                adapterCommunicationTimeout()) continue;
        values.push_back(entry.second);
      }
      return values;
    }();
    const std::vector<GlobalCiBelief> global_beliefs = [&]() {
      std::vector<GlobalCiBelief> values;
      values.reserve(global_ci_beliefs_.size());
      for (const auto &entry : global_ci_beliefs_) {
        const auto receipt = global_ci_receipts_.find(entry.first);
        if (receipt != global_ci_receipts_.end() &&
            std::chrono::duration<double>(Clock::now() - receipt->second).count() >
                adapterCommunicationTimeout()) continue;
        values.push_back(entry.second);
      }
      return values;
    }();
    const auto output = adapter_.update(measurement, peers, relative, global_beliefs);
    algorithm_locked_ = algorithm_locked_ || output.estimate.initialized;
    auto diagnostics = output.diagnostics;
    diagnostics.duplicate_updates = duplicate_updates_;
    diagnostics.communication_rejections = communication_rejections_;
    diagnostics.accepted_communication_updates = accepted_communication_updates_;
    diagnostics.rejected_relative_updates = rejected_relative_updates_;
    diagnostics.rejected_communication_updates = rejected_communication_updates_;
    diagnostics.communication_age = communicationAge();
    last_estimate_ = output.estimate;
    last_diagnostics_ = diagnostics;
    last_output_receipt_ = Clock::now();
    have_last_output_ = true;
    estimate_publisher_->publish(output.estimate);
    diagnostics_publisher_->publish(diagnostics);
    if (adapter_.algorithm() == kGsCi) {
      peer_publisher_->publish(LocalizationAdapter::toPeerEstimate(output.estimate, agent_id_));
    }
    publishGlobalCiBelief(output.estimate);
  }

  void onOdometry(const nav_msgs::msg::Odometry &message) {
    // A configured transient-local origin is required before accepting the
    // first odometry sample.  Otherwise a raw sample could initialize the
    // adapter (or become its previous-position baseline) before the static
    // translation arrives, after which applying the translation would create
    // an artificial jump.
    if (!odom_origin_topic_.empty() && !odom_origin_ned_.has_value()) return;
    have_odometry_sample_ = true;
    onMeasurement(measurementFromOdometry(message));
    if (pending_gps_.has_value()) {
      const auto pending = *pending_gps_;
      pending_gps_.reset();
      onMeasurement(measurementFromGps(pending));
    }
  }

  void onGps(const sensor_msgs::msg::NavSatFix &message) {
    if (!have_odometry_sample_) {
      pending_gps_ = message;
      return;
    }
    onMeasurement(measurementFromGps(message));
  }

  void publishGlobalCiBelief(const LocalizationEstimate &estimate) {
    const auto publish_time = Clock::now();
    if (adapter_.algorithm() != kGsCi || !estimate.valid || estimate.stale ||
        (last_global_ci_publish_ != Clock::time_point{} &&
         std::chrono::duration<double>(publish_time - last_global_ci_publish_).count() <
             1.0 / communication_rate_hz_)) return;
    hercules_interfaces::msg::GlobalCiBelief belief;
    belief.header = estimate.header;
    belief.agent_id = agent_id_;
    belief.sender_id = agent_id_;
    belief.source_id = agent_id_;
    belief.frame_id = estimate.frame_id;
    belief.ordered_agent_ids = global_agent_ids_;
    const std::size_t global_dimension = 2U * global_agent_ids_.size() + 1U;
    belief.global_mean.assign(global_dimension, 0.0);
    belief.global_covariance.assign(global_dimension * global_dimension, 0.0);
    const double unknown_variance = 1.0e6;
    for (std::size_t index = 0; index < global_agent_ids_.size(); ++index) {
      const std::string &id = global_agent_ids_[index];
      std::array<double, 2> position{0.0, 0.0};
      std::array<double, 4> covariance{unknown_variance, 0.0, 0.0, unknown_variance};
      if (id == agent_id_) {
        position = {estimate.position[0], estimate.position[1]};
        covariance = {estimate.covariance[0], estimate.covariance[1],
                      estimate.covariance[3], estimate.covariance[4]};
      }
      const std::size_t offset = 2U * index;
      belief.global_mean[offset] = position[0];
      belief.global_mean[offset + 1U] = position[1];
      belief.global_covariance[offset * global_dimension + offset] = covariance[0];
      belief.global_covariance[offset * global_dimension + offset + 1U] = covariance[1];
      belief.global_covariance[(offset + 1U) * global_dimension + offset] = covariance[2];
      belief.global_covariance[(offset + 1U) * global_dimension + offset + 1U] = covariance[3];
    }
    belief.global_mean.back() = estimate.yaw;
    belief.global_covariance.back() = estimate.covariance[8];
    const auto &core_belief = adapter_.globalBelief();
    if (core_belief.has_value() && core_belief->valid &&
        core_belief->ordered_agent_ids == global_agent_ids_ &&
        core_belief->mean.size() == static_cast<Eigen::Index>(global_dimension) &&
        core_belief->covariance.rows() == static_cast<Eigen::Index>(global_dimension) &&
        core_belief->covariance.cols() == static_cast<Eigen::Index>(global_dimension)) {
      belief.global_mean.assign(core_belief->mean.data(),
                                core_belief->mean.data() + global_dimension);
      for (std::size_t row = 0; row < global_dimension; ++row) {
        for (std::size_t column = 0; column < global_dimension; ++column) {
          belief.global_covariance[row * global_dimension + column] =
              core_belief->covariance(static_cast<Eigen::Index>(row),
                                      static_cast<Eigen::Index>(column));
        }
      }
      belief.timestamp = core_belief->timestamp;
    } else {
      belief.timestamp = static_cast<double>(estimate.header.stamp.sec) +
                         1e-9 * static_cast<double>(estimate.header.stamp.nanosec);
    }
    belief.mean = {estimate.position[0], estimate.position[1]};
    belief.covariance = {estimate.covariance[0], estimate.covariance[1],
                         estimate.covariance[3], estimate.covariance[4]};
    belief.confidence_weight = localization_config_.ci_self_weight;
    belief.sequence = estimate.sequence;
    belief.valid = true;
    global_ci_publisher_->publish(belief);
    last_global_ci_publish_ = publish_time;
  }

  void publishFreshness() {
    if (have_last_output_) publishGlobalCiBelief(last_estimate_);
    if (adapter_.algorithm() == kRecursiveDecentralized) {
      std::vector<std::pair<std::string, PlanarRelativeMeasurement>> pending(
          pending_recursive_pairs_.begin(), pending_recursive_pairs_.end());
      std::vector<std::string> expired;
      for (const auto &[key, measurement] : pending) {
        const auto receipt = relative_receipts_.find(key);
        if (receipt != relative_receipts_.end() &&
            std::chrono::duration<double>(Clock::now() - receipt->second).count() >
                adapterCommunicationTimeout()) {
          expired.push_back(key);
          ++communication_rejections_;
          ++rejected_communication_updates_;
        } else {
          dispatchRecursivePair(measurement);
        }
      }
      for (const auto &key : expired) pending_recursive_pairs_.erase(key);
    }
    if (!have_last_output_) return;
    const double age = std::chrono::duration<double>(Clock::now() - last_output_receipt_).count();
    if (age <= stale_after_sec_) return;
    auto estimate = last_estimate_;
    auto diagnostics = last_diagnostics_;
    estimate.stale = true;
    estimate.valid = false;
    diagnostics.stale = true;
    diagnostics.valid = false;
    diagnostics.sensor_age = age;
    diagnostics.status = "stale";
    diagnostics.failure_reason = "localization input exceeded freshness timeout";
    estimate_publisher_->publish(estimate);
    diagnostics_publisher_->publish(diagnostics);
  }

  void onRelative(const PlanarRelativeMeasurement &message) {
    auto normalized = message;
    if (normalized.source_id.empty()) normalized.source_id = normalized.observer_id;
    if (normalized.target_id.empty()) normalized.target_id = normalized.observed_id;
    if (normalized.observer_id.empty()) normalized.observer_id = normalized.source_id;
    if (normalized.observed_id.empty()) normalized.observed_id = normalized.target_id;
    Eigen::Matrix2d covariance;
    covariance << normalized.covariance[0], normalized.covariance[1],
        normalized.covariance[2], normalized.covariance[3];
    const bool covariance_valid = covariance.allFinite() &&
        covariance.isApprox(covariance.transpose(), 1e-9) &&
        Eigen::LLT<Eigen::Matrix2d>(covariance).info() == Eigen::Success;
    if (normalized.source_id != agent_id_ || normalized.target_id.empty() ||
        normalized.target_id == agent_id_ || normalized.target_id == "Target1" ||
        !normalized.valid || !finiteArray(normalized.relative_position) ||
        !finite(normalized.range) || normalized.range <= 0.0 ||
        !finite(normalized.bearing) || !covariance_valid) {
      ++communication_rejections_;
      ++rejected_relative_updates_;
      return;
    }
    const auto key = pairKey(normalized);
    if (adapter_.algorithm() == kRecursiveDecentralized) {
      if (normalized.sequence == 0U && normalized.capture_id.empty()) {
        ++communication_rejections_;
        ++rejected_relative_updates_;
        return;
      }
      const auto applied = sent_recursive_sequences_.find(key);
      if ((applied != sent_recursive_sequences_.end() && normalized.sequence != 0U &&
           normalized.sequence <= applied->second) ||
          (!normalized.capture_id.empty() && pair_events_.count(normalized.capture_id) != 0U)) {
        ++duplicate_updates_;
        return;
      }
      pending_recursive_pairs_[key] = normalized;
      relative_receipts_[key] = Clock::now();
      dispatchRecursivePair(normalized);
      return;
    }
    const auto found = relative_measurements_.find(key);
    if (found != relative_measurements_.end() && normalized.sequence <= found->second.sequence) {
      ++duplicate_updates_;
      return;
    }
    relative_measurements_[key] = normalized;
    relative_receipts_[key] = Clock::now();
    ++accepted_communication_updates_;
  }

  void onPeer(const LocalizationPeerEstimate &message) {
    const Eigen::Matrix3d covariance = matrix3FromRowMajor(message.covariance);
    if (message.sender_id.empty() || message.sender_id == agent_id_ ||
        message.sender_id == "Target1" || !message.valid ||
        !finiteArray(message.position) || !finiteArray(message.velocity) ||
        !finiteArray(message.orientation) || !finite(message.yaw) ||
        !finite(message.yaw_rate) || !positiveDefinite(covariance)) {
      ++communication_rejections_;
      ++rejected_communication_updates_;
      return;
    }
    if (!message.receiver_id.empty() && message.receiver_id != agent_id_) {
      ++communication_rejections_;
      ++rejected_communication_updates_;
      return;
    }
    if (!message.algorithm.empty() && message.algorithm != adapter_.algorithm()) {
      ++communication_rejections_;
      ++rejected_communication_updates_;
      return;
    }
    const auto found = peer_estimates_.find(message.sender_id);
    if (found != peer_estimates_.end() && message.sequence <= found->second.sequence) {
      ++duplicate_updates_;
      return;
    }
    peer_estimates_[message.sender_id] = message;
    peer_receipts_[message.sender_id] = Clock::now();
    ++accepted_communication_updates_;
  }

  void onGlobalCi(const hercules_interfaces::msg::GlobalCiBelief &message) {
    const std::string sender_id = message.sender_id.empty() ? message.agent_id : message.sender_id;
    if (adapter_.algorithm() != kGsCi || sender_id.empty() || sender_id == agent_id_ || !message.valid ||
        message.sequence == 0) {
      ++communication_rejections_;
      ++rejected_communication_updates_;
      return;
    }
    const std::size_t global_dimension = 2U * message.ordered_agent_ids.size() + 1U;
    const bool sender_is_ordered = std::find(
        message.ordered_agent_ids.begin(), message.ordered_agent_ids.end(), sender_id) !=
        message.ordered_agent_ids.end();
    if (message.ordered_agent_ids != global_agent_ids_ ||
        std::find(message.ordered_agent_ids.begin(), message.ordered_agent_ids.end(), "Target1") !=
            message.ordered_agent_ids.end() || !sender_is_ordered ||
        message.global_mean.size() != global_dimension ||
        message.global_covariance.size() != global_dimension * global_dimension ||
        !std::all_of(message.global_mean.begin(), message.global_mean.end(), finite) ||
        !positiveDefiniteRowMajor(message.global_covariance, global_dimension)) {
      ++communication_rejections_;
      ++rejected_communication_updates_;
      return;
    }
    const auto found = global_ci_beliefs_.find(sender_id);
    if (found != global_ci_beliefs_.end() && message.sequence <= found->second.sequence) {
      ++duplicate_updates_;
      return;
    }
    std::array<double, 3> position{message.mean[0], message.mean[1], 0.0};
    std::array<double, 9> covariance{message.covariance[0], message.covariance[1], 0.0,
                                     message.covariance[2], message.covariance[3], 0.0,
                                     0.0, 0.0, 1.0};
    double peer_yaw = 0.0;
    auto ordered = std::find(message.ordered_agent_ids.begin(),
                             message.ordered_agent_ids.end(), sender_id);
    if (ordered != message.ordered_agent_ids.end()) {
      const std::size_t index = static_cast<std::size_t>(
          std::distance(message.ordered_agent_ids.begin(), ordered));
      const std::size_t offset = 2U * index;
      if (message.global_mean.size() >= offset + 2U) {
        position[0] = message.global_mean[offset];
        position[1] = message.global_mean[offset + 1U];
        const std::size_t dimension = message.global_mean.size();
        if (dimension > 0U && message.global_covariance.size() >= dimension * dimension) {
          covariance[0] = message.global_covariance[offset * dimension + offset];
          covariance[1] = message.global_covariance[offset * dimension + offset + 1U];
          covariance[3] = message.global_covariance[(offset + 1U) * dimension + offset];
          covariance[4] = message.global_covariance[(offset + 1U) * dimension + offset + 1U];
          if (dimension >= offset + 3U) {
            peer_yaw = message.global_mean.back();
            covariance[8] = message.global_covariance.back();
          }
        }
      }
    }
    if (!std::all_of(position.begin(), position.end(), [](double value) { return std::isfinite(value); })) {
      ++communication_rejections_;
      ++rejected_communication_updates_;
      return;
    }
    LocalizationPeerEstimate peer;
    peer.header = message.header;
    peer.sender_id = sender_id;
    peer.frame_id = message.frame_id;
    peer.position = position;
    peer.velocity = {0.0, 0.0, 0.0};
    peer.orientation = {1.0, 0.0, 0.0, 0.0};
    peer.yaw = peer_yaw;
    peer.yaw_rate = 0.0;
    peer.covariance = covariance;
    peer.algorithm = kGsCi;
    peer.sequence = message.sequence;
    peer.valid = true;
    peer_estimates_[sender_id] = peer;
    peer_receipts_[sender_id] = Clock::now();
    global_ci_beliefs_[sender_id] = message;
    global_ci_receipts_[sender_id] = Clock::now();
    ++accepted_communication_updates_;
  }

  double adapterCommunicationTimeout() const {
    return std::max(1e-3, adapterConfigTimeout_);
  }

  double communicationAge() const {
    double age = 0.0;
    const auto now_time = Clock::now();
    for (const auto &[key, receipt] : relative_receipts_) {
      (void)key;
      age = std::max(age, std::chrono::duration<double>(now_time - receipt).count());
    }
    for (const auto &[key, receipt] : peer_receipts_) {
      (void)key;
      age = std::max(age, std::chrono::duration<double>(now_time - receipt).count());
    }
    return age;
  }

  std::string agent_id_;
  LocalizationAdapter adapter_;
  hercules_localization::LocalizationConfig localization_config_;
  bool algorithm_locked_{false};
  std::string measurement_topic_;
  std::string odometry_topic_;
  std::string gps_topic_;
  std::string odom_origin_topic_;
  std::string origin_topic_;
  std::string secondary_origin_topic_;
  std::string relative_topic_;
  std::string global_ci_topic_;
  std::string peer_topic_;
  std::string estimate_topic_;
  std::string diagnostics_topic_;
  std::string peer_output_topic_;
  std::vector<std::string> global_agent_ids_;
  std::optional<std::array<double, 3>> odom_origin_ned_;
  double communication_rate_hz_{2.0};
  double adapterConfigTimeout_{0.5};
  double stale_after_sec_{0.5};
  Clock::time_point last_global_ci_publish_{};
  Clock::time_point last_output_receipt_{};
  bool have_last_output_{false};
  bool have_odometry_sample_{false};
  std::optional<sensor_msgs::msg::NavSatFix> pending_gps_;
  LocalizationEstimate last_estimate_;
  LocalizationDiagnostics last_diagnostics_;

  rclcpp::Subscription<LocalizationMeasurement>::SharedPtr measurement_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr odom_origin_subscription_;
  rclcpp::Subscription<airsim_interfaces::msg::GPSYaw>::SharedPtr origin_subscription_;
  rclcpp::Subscription<airsim_interfaces::msg::GPSYaw>::SharedPtr secondary_origin_subscription_;
  rclcpp::Subscription<PlanarRelativeMeasurement>::SharedPtr relative_subscription_;
  rclcpp::Subscription<LocalizationPeerEstimate>::SharedPtr peer_subscription_;
  rclcpp::Subscription<hercules_interfaces::msg::GlobalCiBelief>::SharedPtr
      global_ci_subscription_;
  rclcpp::Publisher<LocalizationEstimate>::SharedPtr estimate_publisher_;
  rclcpp::Publisher<LocalizationDiagnostics>::SharedPtr diagnostics_publisher_;
  rclcpp::Publisher<LocalizationPeerEstimate>::SharedPtr peer_publisher_;
  rclcpp::Publisher<hercules_interfaces::msg::GlobalCiBelief>::SharedPtr
      global_ci_publisher_;
  rclcpp::Service<SetLocalizationAlgorithm>::SharedPtr algorithm_service_;
  rclcpp::Service<ResetLocalization>::SharedPtr reset_service_;
  rclcpp::Service<RecursiveLocalizationPair>::SharedPtr pair_service_;
  std::map<std::string, rclcpp::Client<RecursiveLocalizationPair>::SharedPtr> pair_clients_;
  rclcpp::TimerBase::SharedPtr freshness_timer_;
  std::map<std::string, LocalizationPeerEstimate> peer_estimates_;
  std::map<std::string, Clock::time_point> peer_receipts_;
  std::map<std::string, GlobalCiBelief> global_ci_beliefs_;
  std::map<std::string, Clock::time_point> global_ci_receipts_;
  std::map<std::string, PlanarRelativeMeasurement> relative_measurements_;
  std::map<std::string, Clock::time_point> relative_receipts_;
  std::set<std::string> pair_events_;
  std::map<std::string, RecursiveLocalizationPair::Response> cached_pair_responses_;
  std::map<std::string, PlanarRelativeMeasurement> pending_recursive_pairs_;
  std::set<std::string> inflight_recursive_pairs_;
  std::map<std::string, std::uint64_t> sent_recursive_sequences_;
  std::uint64_t duplicate_updates_{0};
  std::uint64_t communication_rejections_{0};
  std::uint64_t accepted_communication_updates_{0};
  std::uint64_t rejected_relative_updates_{0};
  std::uint64_t rejected_communication_updates_{0};
  bool gps_reference_set_{false};
  bool gps_reference_configured_{false};
  double gps_reference_latitude_{std::numeric_limits<double>::quiet_NaN()};
  double gps_reference_longitude_{std::numeric_limits<double>::quiet_NaN()};
  double gps_reference_altitude_{std::numeric_limits<double>::quiet_NaN()};
};

}  // namespace hercules_localization_ros

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<hercules_localization_ros::LocalizationNode>());
  rclcpp::shutdown();
  return 0;
}
