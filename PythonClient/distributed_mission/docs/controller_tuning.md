# Controller tuning

The nominal controller structure and the fixed formations are unchanged. The
nominal defaults are the same for FlyingCPP and RuralAustralia:

- nominal speed: `1.0 m/s`
- nominal position gain: `0.5`
- nominal velocity gain: `3.0`
- UGV heading gain: `1.0`
- UGV maximum yaw rate: `1.0 rad/s`
- UAV velocity limit: `3.0 m/s`
- UGV speed limit: `3.0 m/s`
- UAV acceleration limit: `6.0 m/s²`
- UGV acceleration limit: `3.0 m/s²`

These are exposed as command-line parameters so they can be swept without
editing the controller equations. `--leader-nominal-speed` can be set below
`--nominal-speed` so followers have catch-up authority without making the
leader more aggressive. CBF gains (`k1`, `k2`, and `alpha`) remain
at their existing values; a permissive-CBF experiment caused an early
inter-agent collision and was not adopted.

For acceptance, altitude is intentionally excluded from UAV formation error:
the diagnostics report `formation_xy_rms_error` and
`formation_xy_max_error`. A successful formation-convergence check means the
leader is within 2 m XY of the goal and every other vehicle is within 2 m XY
of its fixed box/triangle slot relative to the leader. In the validated
FlyingCPP run this occurred at approximately 17.2 s with zero relevant
collisions.

Example:

```bash
PYTHONPATH=. python PythonClient/distributed_mission/orchestrator.py \
  --launch-mode headless --use-truth-obstacles --steps 200 --dt 0.1
```

The target-centered UGV triangle defaults to a 5 m circumradius; override it
with `--target-ugv-circumradius` when another map needs a different clearance.

The latest FlyingCPP oscillation was not caused by a different nominal
controller tuning. Its log used the same nominal gains as the RuralAustralia
run, but used the Wang UAV CBF method and the looser FlyingCPP perception
settings. The UAV detector was inserting self/neighbor returns as static
obstacles, producing repeated infeasible CBF solves and saturated vertical
commands. Those controlled-agent perception proxies are now filtered before
the CBF; the CBF equations and nominal controller were not changed.

Override only the parameters under test, for example
`--nominal-speed 5 --leader-nominal-speed 3 --ugv-speed-limit 5` or
`--nominal-position-gain 0.8`. RuralAustralia perception uses the existing
memoryless detector, rejects under-supported patches below 32 points, bounds
detector proxy radii at 1.0 m, allows two delayed capture cycles, and caps
age-derived proxy growth at 0.25 m. Other maps retain their prior detector
settings. A zero-speed UGV command applies the existing AirSim brake field so
a settled Husky does not coast past its slot or goal. Do not add online
waypoints or alter the CBF/formation controller structure as part of this
tuning stage.

Validation status: the detector sidecar now shows no zero LiDAR returns enter
clustering, sensor ages are nonnegative, and unsupported one-to-nine-point
patches are rejected. The leader reaches the RuralAustralia actor goal in
current Mestres and Wang runs, but both methods can stop at the same real tree
row and do not satisfy the strict all-agent 2 m XY criterion by 20 s. This is
now a route-feasibility limitation of direct reactive CBF control on the
foliage map, not a frame/timing defect or a method-specific CBF failure. No
online waypoint or CBF-equation change was added. FlyingCPP truth-course
validation remains the appropriate controlled test; RuralAustralia has no
orchestrator truth-obstacle set, so map meshes are not reported as truth boxes.
