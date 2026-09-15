#pragma once

#include <Eigen/Core>

#include "hercules_tracking/models.hpp"

namespace hercules_tracking {

Eigen::MatrixXd positiveDefinite(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                                 double floor = kEigenvalueFloor);
Eigen::MatrixXd safeInverse(const Eigen::Ref<const Eigen::MatrixXd>& matrix);
Eigen::MatrixXd pseudoInverse(const Eigen::Ref<const Eigen::MatrixXd>& matrix);
Eigen::VectorXd solveBlockTridiagonal(const Eigen::Ref<const Eigen::MatrixXd>& system,
                                      const Eigen::Ref<const Eigen::VectorXd>& vector,
                                      int block_dim = kStateDim);
Eigen::VectorXd denseInformationSolution(const Eigen::Ref<const Eigen::MatrixXd>& information,
                                         const Eigen::Ref<const Eigen::VectorXd>& information_vector);

}  // namespace hercules_tracking
