# HERCULES nominal target-mission core

This package freezes the deterministic target motion, target-centered slot
geometry, and nominal target-following control used by the RuralAustralia
Python mission at repository revision
`2ef27d8ddc7f019f4b1af0196f7fbb4d91ce595f`. It is a pure C++17/Eigen library
with no ROS, AirSim, `airsim_interfaces`, AirLib, rpclib, Unreal, CBF,
perception, or simulator-command dependency.

The port contains the sampled route-aligned Gerono figure-eight and its exact
forward-only route-index progression, the five named UAV slots, the three
named UGV triangle slots, and the target-following methods from the Python
`FormationController`. Commands remain model-level acceleration for UAVs and
`[speed,yaw_rate]` for UGVs. Throttle, brake, steering, frame conversion,
networking, tracking transport, and vehicle actuation remain adapter work.

`config/rural_target_tracking.yaml` is the frozen future-integration fixture.
Its algorithm sections define timing, motion, slot geometry, nominal gains,
and the already-ported tracking parameters. Its explicitly separated
`simulator_adapter` section is descriptive and is not consumed by this
library. CBF and perception parameters are deliberately absent.

## Executable-source values

Documentation and the Python runtime test follow the executable source:
RuralAustralia translates the target route 5 m back toward the robot launch
point; `TARGET_CBF_SIGMA_MULTIPLIER` is 2.0; the fixed-goal UGV hold radius is
2.0 m; and the target-centered UGV hold radius is 0.5 m. The generic parser's
target-speed default remains 0.5 m/s, while the tested RuralAustralia command
explicitly selects 0.10 m/s. The frozen fixture therefore retains 0.10 m/s as
the reproduction value rather than presenting it as the generic default.

## Deliberately preserved behavior

The target controller accepts `dt` but does not use it. Startup resets the
configured index to one eighth of the sample count before the orchestrator
explicitly selects sample 5. Route progress may jump only into a short forward
candidate section and uses a fixed 0.05 m improvement gate. At low target
velocity, slot orientation stays at the fallback route heading; it changes
only when planar speed is strictly greater than 2.5 m/s. Unknown UAV IDs use a
zero offset and unknown UGV IDs use the phase-zero vertex, matching Python.
The target-centered UGV controller keeps at least 25 percent forward alignment
outside its 0.5 m hold radius, including when facing away from its slot.

The offline eight-agent test uses deliberately simple kinematics only to catch
identity, slot, sign, heading, and configuration drift. It is not a simulator
model or a closed-loop stability result.
