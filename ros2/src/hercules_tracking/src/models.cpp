#include "hercules_tracking/models.hpp"

#include <algorithm>

namespace hercules_tracking {

StateMatrix constantVelocityTransition(double dt) {
  const double delta = std::max(0.0, dt);
  StateMatrix transition = StateMatrix::Identity();
  transition(0, 2) = delta;
  transition(1, 3) = delta;
  return transition;
}

StateMatrix constantAccelerationProcessNoise(double dt, double spectral_density) {
  const double delta = std::max(0.0, dt);
  const double q = std::max(0.0, spectral_density);
  StateMatrix process = StateMatrix::Zero();
  const double position = delta * delta * delta / 3.0;
  const double cross = delta * delta / 2.0;
  process(0, 0) = position;
  process(1, 1) = position;
  process(0, 2) = process(2, 0) = cross;
  process(1, 3) = process(3, 1) = cross;
  process(2, 2) = delta;
  process(3, 3) = delta;
  return q * process;
}

Eigen::Matrix<double, 2, kStateDim> positionMeasurementMatrix() {
  Eigen::Matrix<double, 2, kStateDim> measurement =
      Eigen::Matrix<double, 2, kStateDim>::Zero();
  measurement(0, 0) = 1.0;
  measurement(1, 1) = 1.0;
  return measurement;
}

}  // namespace hercules_tracking
