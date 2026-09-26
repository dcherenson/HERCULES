#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Core>

namespace hercules_mission_core {

// A deterministic closed-loop reference used by short mission runs.  The
// phase starts at the supplied home point and moves in the +Y tangent
// direction, which gives a well-defined initial heading of pi/2.
struct PeriodicMissionConfig {
  Eigen::Vector3d home{Eigen::Vector3d::Zero()};
  double radius{2.5};
  double duration{10.0};
};

struct PeriodicMissionState {
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  Eigen::Vector3d acceleration{Eigen::Vector3d::Zero()};
};

// Short name used by mission adapters that consume a reference directly.
using PeriodicReference = PeriodicMissionState;

inline PeriodicMissionState periodicMissionReference(
    const PeriodicMissionConfig& config, double elapsed_seconds) {
  const double duration = std::max(config.duration, 1e-9);
  const double radius = std::max(0.0, config.radius);
  const double elapsed = std::clamp(elapsed_seconds, 0.0, duration);
  const double pi = std::acos(-1.0);
  const double omega = 2.0 * pi / duration;
  // Use phase zero at the clamped endpoint as well.  This makes the returned
  // home pose and tangent exactly equal at t=0 and t=T despite sin(2*pi)
  // roundoff, while leaving all interior samples on the same circle.
  const double theta = elapsed >= duration ? 0.0 : omega * elapsed;
  const double sine = std::sin(theta);
  const double cosine = std::cos(theta);

  PeriodicMissionState state;
  state.position = config.home + Eigen::Vector3d(
      radius * (1.0 - cosine), radius * sine, 0.0);
  state.velocity = Eigen::Vector3d(
      radius * omega * sine, radius * omega * cosine, 0.0);
  state.acceleration = Eigen::Vector3d(
      radius * omega * omega * cosine,
      -radius * omega * omega * sine, 0.0);
  return state;
}

inline PeriodicReference periodicMissionReference(
    const Eigen::Vector3d& home, double radius, double duration,
    double elapsed_seconds) {
  return periodicMissionReference(
    PeriodicMissionConfig{home, radius, duration}, elapsed_seconds);
}

// A closed out-and-back reference for short missions.  The vehicle starts at
// home, travels length metres along +Y, then returns to home in duration
// seconds.  Clamping the phase to zero at the completed endpoint keeps the
// endpoint state exactly equal to the initial state despite floating-point
// roundoff in 2*pi.
inline PeriodicReference outAndBackMissionReference(
    const Eigen::Vector3d& home, double length, double duration,
    double elapsed_seconds) {
  const double period = std::max(duration, 1e-9);
  const double travel = std::max(0.0, length);
  const double elapsed = std::clamp(elapsed_seconds, 0.0, period);
  const double pi = std::acos(-1.0);
  const double omega = 2.0 * pi / period;
  const double theta = elapsed >= period ? 0.0 : omega * elapsed;
  const double sine = std::sin(theta);
  const double cosine = std::cos(theta);

  PeriodicReference state;
  state.position = home + Eigen::Vector3d(
      0.0, 0.5 * travel * (1.0 - cosine), 0.0);
  state.velocity = Eigen::Vector3d(
      0.0, travel * pi / period * sine, 0.0);
  state.acceleration = Eigen::Vector3d(
      0.0, 2.0 * travel * pi * pi / (period * period) * cosine, 0.0);
  return state;
}

}  // namespace hercules_mission_core
