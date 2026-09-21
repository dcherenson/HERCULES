#include "hercules_localization/covariance.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Eigenvalues>

namespace hercules_localization {
namespace {

double validFloor(double floor) {
  return std::isfinite(floor) && floor > 0.0 ? floor : 1e-12;
}

Eigen::MatrixXd finiteSymmetric(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  if (matrix.rows() != matrix.cols() || matrix.rows() == 0) {
    throw std::invalid_argument("covariance must be a non-empty square matrix");
  }
  Eigen::MatrixXd result = matrix;
  for (Eigen::Index row = 0; row < result.rows(); ++row) {
    for (Eigen::Index col = 0; col < result.cols(); ++col) {
      if (!std::isfinite(result(row, col))) result(row, col) = 0.0;
    }
  }
  return 0.5 * (result + result.transpose());
}

Eigen::MatrixXd regularizeDynamic(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                                  double floor) {
  const Eigen::MatrixXd symmetric = finiteSymmetric(matrix);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetric);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
    return Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols()) * validFloor(floor);
  }
  const double eigenvalue_floor = validFloor(floor);
  const Eigen::VectorXd values = solver.eigenvalues().unaryExpr(
      [eigenvalue_floor](double value) {
        return std::isfinite(value) ? std::max(value, eigenvalue_floor)
                                    : eigenvalue_floor;
      });
  const Eigen::MatrixXd result =
      solver.eigenvectors() * values.asDiagonal() * solver.eigenvectors().transpose();
  return 0.5 * (result + result.transpose());
}

Eigen::MatrixXd inverseDynamic(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                               double floor) {
  const Eigen::MatrixXd regularized = regularizeDynamic(matrix, floor);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(regularized);
  if (solver.info() != Eigen::Success) {
    return Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols()) / validFloor(floor);
  }
  const Eigen::VectorXd inverse_values = solver.eigenvalues().unaryExpr(
      [floor](double value) { return 1.0 / std::max(value, validFloor(floor)); });
  const Eigen::MatrixXd result = solver.eigenvectors() * inverse_values.asDiagonal() *
                                 solver.eigenvectors().transpose();
  return 0.5 * (result + result.transpose());
}

}  // namespace

Eigen::MatrixXd symmetrize(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("matrix must be square");
  }
  return 0.5 * (matrix + matrix.transpose());
}

bool isFiniteSquare(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  return matrix.rows() == matrix.cols() && matrix.rows() > 0 && matrix.allFinite();
}

Eigen::MatrixXd regularizeCovariance(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                                     double floor) {
  return regularizeDynamic(matrix, floor);
}

Eigen::Matrix2d regularizeCovariance(const Eigen::Matrix2d& matrix, double floor) {
  const Eigen::MatrixXd dynamic = matrix;
  return regularizeDynamic(dynamic, floor);
}

Eigen::Matrix3d regularizeCovariance(const Eigen::Matrix3d& matrix, double floor) {
  const Eigen::MatrixXd dynamic = matrix;
  return regularizeDynamic(dynamic, floor);
}

Eigen::MatrixXd safeInverse(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                            double floor) {
  return inverseDynamic(matrix, floor);
}

Eigen::Matrix2d safeInverse(const Eigen::Matrix2d& matrix, double floor) {
  const Eigen::MatrixXd dynamic = matrix;
  return inverseDynamic(dynamic, floor);
}

Eigen::Matrix3d safeInverse(const Eigen::Matrix3d& matrix, double floor) {
  const Eigen::MatrixXd dynamic = matrix;
  return inverseDynamic(dynamic, floor);
}

double logDeterminant(const Eigen::Ref<const Eigen::MatrixXd>& matrix, double floor) {
  const Eigen::MatrixXd regularized = regularizeCovariance(matrix, floor);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(regularized);
  if (solver.info() != Eigen::Success) return std::numeric_limits<double>::infinity();
  double result = 0.0;
  for (const double value : solver.eigenvalues()) {
    result += std::log(std::max(value, validFloor(floor)));
  }
  return result;
}

double mahalanobisSquared(const Eigen::Ref<const Eigen::VectorXd>& error,
                          const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                          double floor) {
  if (covariance.rows() != covariance.cols() || covariance.rows() != error.size()) {
    throw std::invalid_argument("error and covariance dimensions do not agree");
  }
  if (!error.allFinite()) return std::numeric_limits<double>::infinity();
  const Eigen::MatrixXd inverse = safeInverse(covariance, floor);
  const double value = error.dot(inverse * error);
  return std::isfinite(value) ? std::max(0.0, value)
                              : std::numeric_limits<double>::infinity();
}

Eigen::MatrixXd inflateCovariance(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                                  double standard_deviation, double floor) {
  Eigen::MatrixXd result = regularizeCovariance(matrix, floor);
  const double sigma = std::isfinite(standard_deviation)
                           ? std::max(0.0, standard_deviation)
                           : 0.0;
  result.diagonal().array() += sigma * sigma;
  return regularizeDynamic(result, floor);
}

}  // namespace hercules_localization
