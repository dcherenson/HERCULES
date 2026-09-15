#pragma once

#include "hercules_cbf/constraints.hpp"

namespace hercules_cbf {

CBFResult filter(const CBFRequest& request, const CBFConfig& config);
Eigen::VectorXd projectedCorrection(Eigen::VectorXd control,
                                    const ConstraintSet& constraints,
                                    const CBFConfig& config,
                                    int* rounds);
double maximumRowViolation(const Eigen::VectorXd& control,
                           const ConstraintSet& constraints);
double maximumBoundViolation(const Eigen::VectorXd& control,
                             const ConstraintSet& constraints);

}  // namespace hercules_cbf
