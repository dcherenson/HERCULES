#pragma once

#include <Eigen/Core>

namespace hercules_tracking {

inline constexpr int kStateDim = 4;
inline constexpr double kEigenvalueFloor = 1e-8;

using State = Eigen::Matrix<double, kStateDim, 1>;
using StateMatrix = Eigen::Matrix<double, kStateDim, kStateDim>;
using Position = Eigen::Vector2d;
using PositionMatrix = Eigen::Matrix2d;

StateMatrix constantVelocityTransition(double dt);
StateMatrix constantAccelerationProcessNoise(double dt, double spectral_density = 1.0);
Eigen::Matrix<double, 2, kStateDim> positionMeasurementMatrix();

}  // namespace hercules_tracking
