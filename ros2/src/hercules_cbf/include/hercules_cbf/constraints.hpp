#pragma once

#include "hercules_cbf/types.hpp"

namespace hercules_cbf {

int controlDimension(const CBFRequest& request, const CBFConfig& config);
ConstraintSet buildConstraints(const CBFRequest& request, const CBFConfig& config);
double uncertainty(const CBFRequest& request, const CBFConfig& config);
bool isUnicycle(const CBFRequest& request, const CBFConfig& config);

// Direct Wang equation helpers.  They use planar position/velocity and are
// useful to callers that need to audit a generated row without reconstructing
// the QP.  Both throw no exceptions; invalid geometry is represented by NaN.
double wangBarrier(const Eigen::Vector3d& position_i,
                   const Eigen::Vector3d& position_k,
                   const Eigen::Vector3d& velocity_i,
                   const Eigen::Vector3d& velocity_k,
                   double acceleration_limit_i,
                   double acceleration_limit_k,
                   double safe_distance);
double wangRhs(const Eigen::Vector3d& position_i,
               const Eigen::Vector3d& position_k,
               const Eigen::Vector3d& velocity_i,
               const Eigen::Vector3d& velocity_k,
               double acceleration_limit_i,
               double acceleration_limit_k,
               double safe_distance,
               double gamma);
double strategyBAuthorityWeight(double own_acceleration_limit,
                                double other_acceleration_limit);

}  // namespace hercules_cbf
