#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <optional>
#include <string>

namespace hercules_control {
// Position is wrapper-start-relative, in metres, with right-handed X/Y-left/Z-up
// axes. Velocity retains the wrapper's fixed-axis convention. See README caveats.
struct State {
  int64_t stamp_ns = 0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
};
struct VelocityCommand { Eigen::Vector3d velocity = Eigen::Vector3d::Zero(); double yaw_rate = 0; };
struct GroundCommand { double throttle = 0; double steering = 0; double brake = 1; };
bool valid(const State &state);

class StateReceiver {
 public:
  bool receive(const State &state, double now);
  bool fresh(double now) const;
  const std::optional<State> &state() const { return state_; }
  uint64_t messages = 0, distinct_stamps = 0, invalid_messages = 0;
 private:
  std::optional<State> state_;
  double received_ = 0, advanced_ = 0;
  bool invalid_ = false;
};

enum class Action { None, Takeoff, Move, Stop, Land, Finished };
enum class Phase { Waiting, TakingOff, Moving, Stopping, Landing, Finished };
struct Step { Action action = Action::None; VelocityCommand velocity; GroundCommand ground; };

// A finite smoke-test sequence, not a feedback controller. Times are steady seconds.
class SmokeSequence {
 public:
  explicit SmokeSequence(bool drone, double now) : drone_(drone), phase_start_(now) {}
  Step tick(double now, const StateReceiver &receiver, bool ready, bool interrupt = false);
  void service_result(bool success) { service_result_ = success; }
  bool done() const { return phase_ == Phase::Finished; }
  bool success() const { return done() && error_.empty(); }
  const std::string &error() const { return error_; }
  Phase phase() const { return phase_; }
  double motion_distance_m() const { return max_motion_; }
 private:
  void transition(Phase phase, double now);
  void fail(const std::string &reason, double now);
  bool drone_, flight_attempted_ = false, pulse_started_ = false;
  Phase phase_ = Phase::Waiting;
  double phase_start_, settled_since_ = -1;
  Eigen::Vector3d start_ = Eigen::Vector3d::Zero(), pulse_start_ = Eigen::Vector3d::Zero();
  std::optional<bool> service_result_;
  double max_motion_ = 0;
  std::string error_;
};
}  // namespace hercules_control
