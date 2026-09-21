# hercules_localization

`hercules_localization` is a simulator- and ROS-independent C++17/Eigen core
for cooperative planar pose localization.  It contains the shared SE(2) and
range/bearing math, covariance sanitization and inversion, a recursive
decentralized EKF-style update, and a Gauss--Seidel covariance-intersection
(GS-CI) update.

The package intentionally has no ROS, AirSim, DDS, tracking, or mission
dependencies. The ROS adapter translates messages into the types in
`include/hercules_localization/types.hpp` and calls the algorithms through
`include/hercules_localization/localization.hpp`.

All covariance inputs are symmetrized and eigenvalue-clamped before use.  This
keeps singular, slightly indefinite, and round-off-corrupted covariances from
turning into NaNs while retaining an explicit configurable floor.  Invalid
measurements are rejected with a diagnostic rather than contaminating the
estimate.  `LocalizationConfig::lambda` optionally damps the projected
neighbor covariance in the recursive update (the default value of one keeps
the full propagated uncertainty).

The `global_ci.hpp` helpers expose GS-CI's ordered full state
`[p1 ... pn, yaw_self]`: propagation updates the receiver block and inflates
remote position blocks, while communication removes each sender yaw and
inserts the receiver yaw before deterministic batch covariance intersection.
