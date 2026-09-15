# Abstract planar constant-velocity estimate.
string target_id
builtin_interfaces/Time stamp
float64[2] position             # [x, y]
float64[2] velocity             # [vx, vy]
float64[16] state_covariance    # row-major 4x4 for [x, y, vx, vy]
bool active
uint32 consensus_iterations
float64 consensus_residual
bool has_measurement_residual
float64[2] measurement_residual
