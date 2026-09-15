#pragma once

#include "hercules_cbf/types.hpp"

namespace hercules_cbf {

int controlDimension(const CBFRequest& request, const CBFConfig& config);
ConstraintSet buildConstraints(const CBFRequest& request, const CBFConfig& config);
double uncertainty(const CBFRequest& request, const CBFConfig& config);
bool isUnicycle(const CBFRequest& request, const CBFConfig& config);

}  // namespace hercules_cbf
