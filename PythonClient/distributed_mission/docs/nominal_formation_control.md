# Nominal leader-follower formation control

The nominal policy is intentionally obstacle-unaware so safety behavior can be
diagnosed independently from planning.

`Husky1` follows the mission goal. Every other vehicle has a body-frame slot
`delta_i` and receives

```text
p_i_ref = p_leader + R(yaw_leader) delta_i
v_i_ref = v_leader + omega_leader x R(yaw_leader) delta_i
```

UAVs and Wang-mode Huskies use bounded position/velocity PD acceleration.
Mestres-mode Huskies use bounded linear speed and heading-error yaw rate.
The default slots and gains are in `modules/formation_control.py` and are
intended to be changed through configuration later. The default UAV slots are
the four corners and center of a 4 m x 4 m horizontal box. The UGV slots are
an equilateral triangle of side length 4 m, with `Husky1` at the front vertex.
UAV altitude is held at a fixed global AirSim NED reference (`-5 m` by
default), rather than being offset from the Husky body origin. The UAV CBF
altitude floor is separately calibrated to one metre above the map ground
(`z=-1` for FlyingCPP and `z=+1` for RuralAustralia by default), allowing
temporary downward avoidance beneath tree canopies. Override it with
`--uav-altitude-floor` when testing another map frame.

For the deterministic FlyingCPP block course, the mission goal is behind two
staggered mixed-height obstacle rows. There is no bypass waypoint: the nominal
leader-follower controller continues toward the shifted goal and the CBF layer
is expected to produce the small formation deformation and heading change
needed to pass through both central openings. A floating block in the first
passage also tests UAV altitude avoidance. No online waypoint generation is
used.

Mestres' unicycle nominal control retains a small crawl speed while making a
large heading correction, because an AirSim car cannot rotate in place. It
stops inside a 0.75 m target deadband. Wang-mode Husky acceleration is mapped
to AirSim throttle, brake, and steering in the orchestrator adapter.
