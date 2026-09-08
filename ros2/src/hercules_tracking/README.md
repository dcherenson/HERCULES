# HERCULES tracking numerical core

This package ports the numerical target-tracking behavior from
`PythonClient/distributed_mission/modules/target_tracking.py` at repository
revision `034b387df38c8a63798f3d09607956e280c87d21`. It is a C++17/Eigen library
with no ROS, AirSim, Unreal, rpclib, or simulator dependency. The separate
`hercules_interfaces` package defines future typed ROS transport schemas; no
tracking node, publisher, subscriber, DDS protocol, or odometry adapter exists yet.

## Ported behavior

The state is the abstract planar vector `[x,y,vx,vy]`. The implementation includes
the constant-velocity transition, the Python white-acceleration process covariance,
linear XY observations, eigenvalue-floor positive-definite treatment, safe inverse,
forward/backward block-Cholesky solve, rolling-window information assembly, local
tracks, synchronous one-round ADMM updates, prediction, information handoff, and a
deterministic centralized harness that simulates the Python synchronous protocol.

The rolling window has duration `window_seconds`; samples are kept in timestamp order
and pruning preserves the same Python boundary rule. A valid asynchronous capture is
placed at `min(estimator_epoch, capture_time)`. Prior and process covariances are
scaled by the total active tracker count supplied to an epoch. Consensus uses the
supplied adjacency graph, round-start snapshots, and a global maximum primal-change
residual. Neighbor trajectories are interpolated when a complete time grid is
provided; otherwise equal-length trajectories are used directly, and incompatible
lengths repeat the latest neighbor state.

## Inherited assumptions and limits

- constant-velocity planar target dynamics and white-acceleration process noise;
- positive-definite covariance/information matrices after an eigenvalue floor of
  `1e-8`;
- synchronous consensus rounds over a caller-supplied adjacency graph;
- no modeled network delay, packet loss, or out-of-order consensus messages;
- no ROS transport or simulator coordinate-frame conversion;
- no cooperative-localization, conformal-prediction, or CBF coupling.

Python prediction behavior is deliberately preserved even though it is mathematically
unconventional: the state is propagated with `F`, but only the reported 2x2 position
covariance receives the position block of `Q`; the full state covariance is not
propagated with `F P F^T + Q`. Consensus covariance is carried as support metadata and
is not re-added as a measurement factor. Handoff information is accumulated separately
and consumed once by the next estimator epoch.

The accompanying Python documentation also differs from the executable source in two
places. It names `TARGET_CBF_SIGMA_MULTIPLIER` as `4.0`, while the referenced source sets
it to `2.0`; that CBF-only constant is outside this port. Its phrase "expires after a
full window" is only approximate: the source expires a track when elapsed support time
is strictly greater than the window, while handoff becomes eligible at the window
boundary. The C++ implementation preserves those executable boundary conditions.

Eigen and NumPy/LAPACK may choose different eigenvector signs, factorization paths,
and pseudoinverse thresholds. Tests therefore compare resulting matrices, states,
covariances, trajectories, duals, and residuals with explicit tolerances rather than
requiring bit-for-bit identity.

The tests establish regression parity for deterministic cases. They do **not** prove
ADMM convergence for arbitrary or time-varying graphs, and positive-definite covariance
checks do **not** establish statistical consistency.

Python-only metadata is intentionally omitted from the production C++ type because it
is neither used by the estimator nor safely representable as a typed algorithmic
interface. The typed future ROS measurement retains target/source/capture/sensor IDs,
visibility, validity, timestamp, position, and covariance.
