#include "hercules_control/core.hpp"
#include "hercules_control/ros_adapter.hpp"
#include <rclcpp/rclcpp.hpp>
#include <airsim_interfaces/srv/takeoff.hpp>
#include <airsim_interfaces/srv/land.hpp>
#include <chrono>
#include <csignal>
#include <future>
#include <iostream>

namespace hc = hercules_control;
using Clock = std::chrono::steady_clock;
static volatile std::sig_atomic_t interrupted = 0;
static void signal_handler(int) { interrupted = 1; }
static double seconds() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

class SmokeNode : public rclcpp::Node {
 public:
  SmokeNode() : Node("hercules_smoke"), started_(seconds()) {
    const auto type = declare_parameter<std::string>("vehicle_type", "drone");
    if (type != "drone" && type != "ugv") throw std::invalid_argument("vehicle_type must be drone or ugv");
    drone_ = type == "drone";
    wait_on_flight_task_ = declare_parameter<bool>("wait_on_flight_task", false);
    mode_ = declare_parameter<std::string>("mode", "observe");
    if (mode_ != "observe" && mode_ != "motion" && mode_ != "benchmark")
      throw std::invalid_argument("mode must be observe, motion or benchmark");
    const auto vehicle = declare_parameter<std::string>("vehicle_name", drone_ ? "Drone1" : "Husky1");
    const auto prefix = declare_parameter<std::string>("wrapper_prefix", "/hercules_" + type + "/" + vehicle);
    const auto odom = declare_parameter<std::string>("odom_topic", prefix + "/ground_truth/odom_local");
    const auto command = declare_parameter<std::string>("command_topic", prefix + (drone_ ? "/vel_cmd_world_frame" : "/car_cmd"));
    rate_ = declare_parameter<double>("command_rate_hz", 20.0);
    duration_ = declare_parameter<double>("duration_sec", mode_ == "observe" ? 30.0 : 15.0);
    warmup_ = declare_parameter<double>("warmup_sec", mode_ == "benchmark" ? 5.0 : 0.0);
    if (!std::isfinite(rate_) || rate_ < 1 || rate_ > 100 || !std::isfinite(duration_) || duration_ <= 0 ||
        !std::isfinite(warmup_) || warmup_ < 0) throw std::invalid_argument("Invalid rate/duration/warmup");
    if (mode_ == "motion" && rate_ < 20) throw std::invalid_argument("Motion checks require at least 20 Hz");
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(odom, rclcpp::SensorDataQoS().keep_last(5),
      [this](nav_msgs::msg::Odometry::ConstSharedPtr message) {
        const double t = seconds();
        receiver_.receive(hc::from_odometry(*message), t);
        adapter_seconds_ += seconds() - t;
      });
    if (mode_ != "observe") {
      if (drone_) {
        drone_pub_ = create_publisher<airsim_interfaces::msg::VelCmd>(command, 1);
        takeoff_ = create_client<airsim_interfaces::srv::Takeoff>(declare_parameter<std::string>("takeoff_service", prefix + "/takeoff"));
        land_ = create_client<airsim_interfaces::srv::Land>(declare_parameter<std::string>("land_service", prefix + "/land"));
      } else ground_pub_ = create_publisher<airsim_interfaces::msg::CarControls>(command, 1);
    }
    sequence_ = std::make_unique<hc::SmokeSequence>(drone_, started_);
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate_), [this] { tick(); });
    RCLCPP_INFO(get_logger(), "vehicle=%s mode=%s odom=%s command=%s", vehicle.c_str(), mode_.c_str(), odom.c_str(), command.c_str());
  }
  bool done() const { return done_; }
  int result() const { return result_; }
 private:
  bool wait_on_flight_task_ = false;
  bool ready() const {
    if (mode_ == "observe") return true;
    if (!drone_) return ground_pub_->get_subscription_count() > 0;
    return drone_pub_->get_subscription_count() > 0 && (mode_ == "benchmark" ||
      (takeoff_->service_is_ready() && land_->service_is_ready()));
  }
  void publish(const hc::Step &step) {
    if (drone_) drone_pub_->publish(hc::to_drone_message(step.velocity));
    else ground_pub_->publish(hc::to_ground_message(step.ground));
    ++published_;
  }
  void finish(bool success, const std::string &reason) {
    done_ = true; result_ = success ? 0 : 1;
    RCLCPP_INFO(get_logger(), "SMOKE_RESULT success=%s reason=%s messages=%llu distinct_stamps=%llu commands=%llu motion_m=%.6f",
      success ? "true" : "false", reason.c_str(), (unsigned long long)receiver_.messages,
      (unsigned long long)receiver_.distinct_stamps, (unsigned long long)published_, sequence_->motion_distance_m());
  }
  void tick() {
    const double now = seconds();
    if (done_) return;
    if (now - last_report_ >= 1.0) {
      RCLCPP_INFO(get_logger(), "SMOKE_METRICS steady_s=%.6f messages=%llu unique_stamps=%llu invalid=%llu commands=%llu adapter_total_us=%.3f fresh=%s",
        now, (unsigned long long)receiver_.messages, (unsigned long long)receiver_.distinct_stamps,
        (unsigned long long)receiver_.invalid_messages, (unsigned long long)published_, adapter_seconds_ * 1e6,
        receiver_.fresh(now) ? "true" : "false");
      last_report_ = now;
      if (receiver_.state()) {
        const auto &s = *receiver_.state();
        RCLCPP_INFO(get_logger(), "SMOKE_STATE x=%.6f y=%.6f z=%.6f speed_mps=%.6f",
                    s.position.x(), s.position.y(), s.position.z(), s.velocity.norm());
      }
    }
    if (mode_ == "motion") {
      if (takeoff_future_ && takeoff_future_->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        sequence_->service_result(takeoff_future_->get()->success); takeoff_future_.reset();
      }
      if (land_future_ && land_future_->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        sequence_->service_result(land_future_->get()->success); land_future_.reset();
      }
      const auto step = sequence_->tick(now, receiver_, ready(), interrupted != 0);
      if (step.action == hc::Action::Move || step.action == hc::Action::Stop) publish(step);
      else if (step.action == hc::Action::Takeoff) {
        auto request = std::make_shared<airsim_interfaces::srv::Takeoff::Request>(); request->wait_on_last_task = wait_on_flight_task_;
        takeoff_future_ = takeoff_->async_send_request(request).future;
      } else if (step.action == hc::Action::Land) {
        // A late takeoff reply must not be interpreted as a landing result.
        takeoff_future_.reset();
        auto request = std::make_shared<airsim_interfaces::srv::Land::Request>(); request->wait_on_last_task = wait_on_flight_task_;
        land_future_ = land_->async_send_request(request).future;
      } else if (step.action == hc::Action::Finished) finish(sequence_->success(), sequence_->error());
      return;
    }
    if (measurement_start_ < 0) {
      if (receiver_.fresh(now) && ready()) { measurement_start_ = now + warmup_; benchmark_origin_ = receiver_.state()->position; }
      else if (now - started_ > 10 || interrupted) finish(false, "state/interface unavailable or interrupted");
      return;
    }
    if (mode_ == "benchmark") publish(hc::Step{}); // zero velocity / full brake
    if (!receiver_.fresh(now) || interrupted || (mode_ == "benchmark" &&
        (receiver_.state()->position - benchmark_origin_).norm() > 0.5)) {
      finish(false, "invalid/stale state, unexpected motion or interruption"); return;
    }
    if (now >= measurement_start_ && !measuring_) {
      measuring_ = true; measurement_start_ = now;
      base_messages_ = receiver_.messages; base_stamps_ = receiver_.distinct_stamps;
      base_commands_ = published_; base_adapter_ = adapter_seconds_;
      RCLCPP_INFO(get_logger(), "MEASUREMENT_START steady_s=%.6f", now);
    }
    if (measuring_ && now - measurement_start_ >= duration_) {
      const double elapsed = now - measurement_start_;
      const auto count = receiver_.messages - base_messages_;
      RCLCPP_INFO(get_logger(), "BENCHMARK_RESULT start_s=%.6f end_s=%.6f elapsed_s=%.6f callback_hz=%.3f unique_stamp_hz=%.3f publish_hz=%.3f adapter_mean_us=%.3f",
        measurement_start_, now, elapsed, count / elapsed, (receiver_.distinct_stamps - base_stamps_) / elapsed,
        (published_ - base_commands_) / elapsed, count ? (adapter_seconds_ - base_adapter_) * 1e6 / count : 0);
      finish(true, "state reception verified");
    }
  }
  bool drone_ = true, done_ = false, measuring_ = false;
  int result_ = 1;
  std::string mode_;
  double started_, rate_, duration_, warmup_, last_report_ = 0, measurement_start_ = -1;
  double adapter_seconds_ = 0, base_adapter_ = 0;
  uint64_t published_ = 0, base_messages_ = 0, base_stamps_ = 0, base_commands_ = 0;
  Eigen::Vector3d benchmark_origin_ = Eigen::Vector3d::Zero();
  hc::StateReceiver receiver_;
  std::unique_ptr<hc::SmokeSequence> sequence_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<airsim_interfaces::msg::VelCmd>::SharedPtr drone_pub_;
  rclcpp::Publisher<airsim_interfaces::msg::CarControls>::SharedPtr ground_pub_;
  rclcpp::Client<airsim_interfaces::srv::Takeoff>::SharedPtr takeoff_;
  rclcpp::Client<airsim_interfaces::srv::Land>::SharedPtr land_;
  std::optional<rclcpp::Client<airsim_interfaces::srv::Takeoff>::SharedFuture> takeoff_future_;
  std::optional<rclcpp::Client<airsim_interfaces::srv::Land>::SharedFuture> land_future_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, signal_handler); std::signal(SIGTERM, signal_handler);
  int result = 1;
  try {
    auto node = std::make_shared<SmokeNode>();
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    while (rclcpp::ok() && !node->done()) executor.spin_once(std::chrono::milliseconds(50));
    result = node->result();
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; }
  rclcpp::shutdown();
  return result;
}
