#include "hercules_control/core.hpp"
#include <algorithm>
#include <cmath>

namespace hercules_control {
bool valid(const State &s) {
  return s.stamp_ns >= 0 && s.position.allFinite() && s.velocity.allFinite() &&
    s.angular_velocity.allFinite() && s.orientation.coeffs().allFinite() &&
    std::abs(s.orientation.norm() - 1.0) < 0.05;
}
bool StateReceiver::receive(const State &s, double now) {
  ++messages;
  if (!valid(s) || (state_ && s.stamp_ns < state_->stamp_ns)) {
    ++invalid_messages; invalid_ = true; return false;
  }
  if (!state_ || s.stamp_ns > state_->stamp_ns) { advanced_ = now; ++distinct_stamps; }
  received_ = now; state_ = s; invalid_ = false; return true;
}
bool StateReceiver::fresh(double now) const {
  return state_ && !invalid_ && distinct_stamps >= 2 && now - received_ < 0.5 && now - advanced_ < 0.5;
}
void SmokeSequence::transition(Phase p, double now) {
  phase_ = p; phase_start_ = now; settled_since_ = -1; service_result_.reset();
  if (p == Phase::Moving) pulse_started_ = true;
}
void SmokeSequence::fail(const std::string &why, double now) {
  if (error_.empty()) error_ = why;
  if (phase_ != Phase::Stopping && phase_ != Phase::Landing && phase_ != Phase::Finished)
    transition(Phase::Stopping, now);
}
Step SmokeSequence::tick(double now, const StateReceiver &r, bool ready, bool interrupt) {
  Step out;
  if (done()) { out.action = Action::Finished; return out; }
  if (interrupt) fail("interrupted", now);
  if (r.invalid_messages > 0) fail("invalid odometry received", now);
  if (phase_ == Phase::Waiting) {
    if (now - phase_start_ > 10) fail("state or command interface unavailable", now);
    else if (r.fresh(now) && ready) {
      start_ = r.state()->position; pulse_start_ = start_;
      if (drone_) {
        flight_attempted_ = true; transition(Phase::TakingOff, now);
        out.action = Action::Takeoff; return out;
      }
      transition(Phase::Moving, now);
    } else return out;
  }
  const bool fresh = r.fresh(now);
  if (phase_ != Phase::Stopping && phase_ != Phase::Landing && !fresh)
    fail("invalid, stale or non-advancing odometry", now);
  if (fresh) {
    if (pulse_started_)
      max_motion_ = std::max(max_motion_, (r.state()->position - pulse_start_).head<2>().norm());
    if ((r.state()->position - start_).head<2>().norm() > 0.5)
      fail("horizontal displacement bound exceeded", now);
    if (drone_ && r.state()->position.z() - start_.z() > 5.0)
      fail("altitude ceiling exceeded", now);
  }
  const double elapsed = now - phase_start_;
  if (phase_ == Phase::TakingOff) {
    if ((service_result_ && !*service_result_) || elapsed > 20) fail("takeoff failed or timed out", now);
    else if (service_result_ && *service_result_ && fresh &&
             r.state()->position.z() - start_.z() > 0.5 && r.state()->velocity.norm() < 0.2) {
      if (settled_since_ < 0) settled_since_ = now;
      if (now - settled_since_ >= 0.5) {
        pulse_start_ = r.state()->position; transition(Phase::Moving, now);
      }
    } else settled_since_ = -1;
  }
  if (phase_ == Phase::Moving) {
    const double moved = (r.state()->position - pulse_start_).head<2>().norm();
    max_motion_ = std::max(max_motion_, moved);
    if (now - phase_start_ >= 1.0 || (!drone_ &&
        (moved >= 0.25 || r.state()->velocity.head<2>().norm() >= 0.5))) {
      if (max_motion_ < 0.02) error_ = "no measurable response to motion command";
      transition(Phase::Stopping, now);
    } else {
      out.action = Action::Move;
      out.velocity.velocity.x() = 0.2;
      out.ground = {0.15, 0.0, 0.0};
      return out;
    }
  }
  if (phase_ == Phase::Stopping) {
    out.action = Action::Stop;
    const bool stopped = fresh && r.state()->velocity.norm() < 0.1;
    if (now - phase_start_ >= 1.0 && stopped) {
      if (drone_ && flight_attempted_) { transition(Phase::Landing, now); out.action = Action::Land; }
      else { transition(Phase::Finished, now); out.action = Action::Finished; }
    } else if (now - phase_start_ > 5.0) {
      if (error_.empty()) error_ = "could not verify stopping";
      if (drone_ && flight_attempted_) { transition(Phase::Landing, now); out.action = Action::Land; }
      else { transition(Phase::Finished, now); out.action = Action::Finished; }
    }
    return out;
  }
  if (phase_ == Phase::Landing) {
    // Never publish velocity here: it would cancel the simulator's landing task.
    if ((service_result_ && !*service_result_) || now - phase_start_ > 60.0) {
      if (error_.empty()) error_ = "landing failed or timed out";
      transition(Phase::Finished, now); out.action = Action::Finished;
    } else if (service_result_ && *service_result_ && fresh &&
               std::abs(r.state()->position.z() - start_.z()) < 0.3 && r.state()->velocity.norm() < 0.1) {
      transition(Phase::Finished, now); out.action = Action::Finished;
    }
  }
  return out;
}
}  // namespace hercules_control
