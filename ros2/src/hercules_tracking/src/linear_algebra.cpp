#include "hercules_tracking/linear_algebra.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace hercules_tracking {

Eigen::MatrixXd positiveDefinite(const Eigen::Ref<const Eigen::MatrixXd>& matrix, double floor) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("matrix must be square");
  }
  if (!matrix.allFinite() || !std::isfinite(floor)) {
    throw std::invalid_argument("matrix and floor must be finite");
  }
  const Eigen::MatrixXd symmetric = 0.5 * (matrix + matrix.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetric);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("symmetric eigendecomposition failed");
  }
  const Eigen::VectorXd values = solver.eigenvalues().cwiseMax(floor);
  Eigen::MatrixXd result = solver.eigenvectors() * values.asDiagonal() * solver.eigenvectors().transpose();
  return 0.5 * (result + result.transpose());
}

Eigen::MatrixXd safeInverse(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  const Eigen::MatrixXd regularized = positiveDefinite(matrix);
  Eigen::FullPivLU<Eigen::MatrixXd> lu(regularized);
  if (lu.isInvertible()) {
    return lu.inverse();
  }
  return pseudoInverse(regularized);
}

Eigen::MatrixXd pseudoInverse(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  if (!matrix.allFinite()) {
    throw std::invalid_argument("matrix must be finite");
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const double tolerance = std::numeric_limits<double>::epsilon() *
                           static_cast<double>(std::max(matrix.rows(), matrix.cols())) *
                           (svd.singularValues().size() ? svd.singularValues()(0) : 0.0);
  Eigen::VectorXd inverse_values = svd.singularValues();
  for (Eigen::Index index = 0; index < inverse_values.size(); ++index) {
    inverse_values(index) = inverse_values(index) > tolerance ? 1.0 / inverse_values(index) : 0.0;
  }
  return svd.matrixV() * inverse_values.asDiagonal() * svd.matrixU().transpose();
}

Eigen::VectorXd solveBlockTridiagonal(const Eigen::Ref<const Eigen::MatrixXd>& system,
                                      const Eigen::Ref<const Eigen::VectorXd>& vector,
                                      int block_dim) {
  if (block_dim <= 0) {
    throw std::invalid_argument("block_dim must be positive");
  }
  if (system.rows() != system.cols() || system.rows() != vector.size()) {
    throw std::invalid_argument("system and vector dimensions do not agree");
  }
  if (system.rows() == 0 || system.rows() % block_dim != 0) {
    throw std::invalid_argument("system dimension must be a nonzero multiple of block_dim");
  }
  if (!system.allFinite() || !vector.allFinite()) {
    throw std::invalid_argument("system and vector must be finite");
  }

  const Eigen::Index count = system.rows() / block_dim;
  std::vector<Eigen::MatrixXd> lower;
  std::vector<Eigen::MatrixXd> factors;
  factors.reserve(static_cast<std::size_t>(count));
  if (count > 1) lower.reserve(static_cast<std::size_t>(count - 1));

  for (Eigen::Index index = 0; index < count; ++index) {
    const Eigen::Index start = index * block_dim;
    Eigen::MatrixXd diagonal = system.block(start, start, block_dim, block_dim);
    if (index > 0) diagonal -= lower.back() * lower.back().transpose();
    const Eigen::MatrixXd regularized = positiveDefinite(diagonal);
    Eigen::LLT<Eigen::MatrixXd> llt(regularized);
    if (llt.info() != Eigen::Success) throw std::runtime_error("block Cholesky factorization failed");
    factors.push_back(llt.matrixL());
    if (index < count - 1) {
      const Eigen::MatrixXd off_diagonal =
          system.block(start + block_dim, start, block_dim, block_dim);
      // Python: off_diagonal @ inv(factor.T). Solve the transposed identity
      // formulation to avoid explicitly inverting in production.
      lower.push_back(factors.back().triangularView<Eigen::Lower>()
                          .solve(off_diagonal.transpose()).transpose());
    }
  }

  Eigen::VectorXd forward = Eigen::VectorXd::Zero(vector.size());
  for (Eigen::Index index = 0; index < count; ++index) {
    const Eigen::Index start = index * block_dim;
    Eigen::VectorXd value = vector.segment(start, block_dim);
    if (index > 0) value -= lower[static_cast<std::size_t>(index - 1)] *
                           forward.segment(start - block_dim, block_dim);
    forward.segment(start, block_dim) =
        factors[static_cast<std::size_t>(index)].triangularView<Eigen::Lower>().solve(value);
  }

  Eigen::VectorXd solution = Eigen::VectorXd::Zero(vector.size());
  for (Eigen::Index index = count; index-- > 0;) {
    const Eigen::Index start = index * block_dim;
    Eigen::VectorXd value = forward.segment(start, block_dim);
    if (index < count - 1) value -= lower[static_cast<std::size_t>(index)].transpose() *
                                   solution.segment(start + block_dim, block_dim);
    solution.segment(start, block_dim) = factors[static_cast<std::size_t>(index)]
        .transpose().triangularView<Eigen::Upper>().solve(value);
  }
  return solution;
}

Eigen::VectorXd denseInformationSolution(const Eigen::Ref<const Eigen::MatrixXd>& information,
                                         const Eigen::Ref<const Eigen::VectorXd>& information_vector) {
  if (information.rows() != information.cols() || information.rows() != information_vector.size()) {
    throw std::invalid_argument("information dimensions do not agree");
  }
  return positiveDefinite(information).fullPivLu().solve(information_vector);
}

}  // namespace hercules_tracking
