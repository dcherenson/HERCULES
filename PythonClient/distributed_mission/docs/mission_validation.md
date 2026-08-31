# Mission validation results

## FlyingCPP through-gap truth test

The deterministic course uses a goal `(16, 1, -1)` in AirSim NED. There is no
intermediate bypass waypoint. Two rows of four ground blocks are placed at
`x=7.5` and `x=12.5`. The first central opening is centered near `y=0`; the
second is shifted near `y=1`, leaving a physically passable but staggered
corridor. The spherical CBF proxies make the second corridor slightly tighter than
the nominal formation, so the team must compress and turn. A floating block
at `(10.5, 0, -6.5)` tests a UAV altitude change.

The final staggered truth-obstacle run on 2026-08-31 completed 800 control
steps with zero relevant Unreal collisions. Husky1 and all UAVs crossed the
two rows through the central route. Husky2 and Husky3 also began crossing the
second row through its shifted central opening rather than taking a side gap.
The UGV and UAV lateral paths changed between rows, and the followers
compressed modestly around the shifted corridor. The center UAV changed
altitude near the floating block. The followers remain behind Husky1 at the
end because they retain their formation offsets; that is not a side-route
failure.

Command used:

```bash
../../herculesvenv/bin/python orchestrator.py --launch-mode headless \
  --map flyingcpp --cbf-method mestres --steps 800 --no-animation \
  --use-truth-obstacles --debug-dir /tmp/hercules_truth_through_gap_tight_20260831
```

## FlyingCPP live-perception test

The final 800-step course with live depth/LiDAR perception also completed with
zero relevant Unreal collisions and used the central route. The perception
report still identifies approximately 4--5 m matched-proxy association jumps
on many agents, although the final run stayed safe and retained modest final
formation error. This remains a perception-quality limitation, not a change
to the CBF equations. The implementation remains memoryless as intended;
tracking and smoothing should only be considered after the capture
diagnostics are reviewed.

## RuralAustralia early-stop result

An 800-step no-spawn-obstacles run was attempted on RuralAustralia Example 1.
The map's foliage generated repeated authoritative collisions for Husky2
with `InstancedFoliageActor_0`, and the UGV formation degraded substantially
(Husky2 ended far behind and laterally displaced). The run is therefore not
treated as a successful perception validation. The orchestrator truth list is
correctly empty on this map, so the result reflects map geometry and local
perception rather than the deterministic test boxes.

Reasonable next options are:

1. Exclude or simplify foliage collision geometry in the Unreal test map while
   retaining buildings and large obstacles.
2. Add a map-specific sensor calibration/ground-filter configuration after
   inspecting the generated point-cloud overlays.
3. Reduce the nominal UGV speed and/or use a larger formation safety margin
   for RuralAustralia only, without changing the CBF equations.
4. Add an explicit map-truth obstacle export for controlled validation before
   requiring perception to represent every RuralAustralia mesh.

## RuralAustralia perception calibration follow-up

The detector now composes each Husky LiDAR mount with the vehicle pose, removes
zero returns before clustering, and uses the nearest measured surface point
for sparse planar foliage patches. A short 100-step run with the existing
`--obstacle-margin 1.0` completed with zero relevant Unreal collisions and no
invalid or negative-age sensor observations. The same margin is now the
RuralAustralia default; FlyingCPP remains at its previous 0 m default, and an
explicit `--obstacle-margin 0` still provides an uncalibrated baseline.

A subsequent clean-start run exposed a `Spline_Fence_2` contact while the
Husky formation was placed several metres from its requested startup pose.
That is recorded as a startup-placement issue rather than evidence that the
LiDAR transform is wrong. The run with the more conservative existing 2 m
margin had zero relevant collisions, but the 1 m default remains the selected
calibration because it allows substantially more progress. The generated
`*_perception_report.md`, timeline, and per-agent point overlays should be
reviewed for any future fence/foliage contact before increasing the margin.

The final default-calibration smoke run completed with all three Huskies near
their requested startup formation, zero relevant Unreal collisions, no invalid
sensor observations, and no negative sensor ages. The earlier startup/fence
contact was not reproduced.

## RuralAustralia ground-height correction

RuralAustralia Example 1 uses a ground reference approximately 2 m lower than
the FlyingCPP course in the mission frame. Ground-referenced Rural geometry is
therefore shifted by `+2 m` in NED: the goal and route markers use `z=+1`
instead of `z=-1`, and Husky startup poses use `z=+1` instead of being dropped
from `z=-1`. UAV flight altitude remains the configured `-5 m` NED value.

The post-correction 100-step Rural smoke run logged goal `[1, -16, 1]`, Husky
startup heights around `z=0.6–0.8` after settling on the terrain, and zero
relevant Unreal collisions. FlyingCPP continues to use its original `z=-1`
goal and startup references.

## Unreal route markers

The orchestrator calls AirSim's persistent `simPlotPoints` API after connecting
and spawning the course. The current route contains the goal marker; any
future fixed course waypoint is plotted as well, with a line connecting the
markers when there is more than one. Marker failure only emits a warning and
does not affect mission execution. Online waypoint generation is not used.

## User-saved course test

The user-saved `debug/course.json` was first run unchanged with truth
obstacles. It registered zero relevant collisions, but the team stalled near
the central obstacle because its conservative spherical proxy occupied the
nominal corridor. The original file was preserved.

The tested adjusted copy is `debug/course_adjusted.json`. Only two centers
were moved, each by no more than 1 m:

| obstacle | original center (NED) | adjusted center (NED) | displacement |
| --- | --- | --- | --- |
| `obstacle_0` | `(9.455, -0.069, -2)` | `(10.455, -1.000, -2)` | `(+1.000, -0.931, 0)` |
| `obstacle_2` | `(17.621, 7.504, -3)` | `(17.621, 8.504, -3)` | `(0, +1.000, 0)` |

The other four obstacle centers and all dimensions are unchanged. The
adjusted truth run and the adjusted live-perception run both completed 800
steps with zero relevant Unreal collisions. The team made progress through
the saved layout, but the UGV follower paths and some UAV paths still have
large excursions. The perception report continues to flag 3--5 m proxy
association jumps, so this course is safe in these runs but is not yet a
clean formation-tracking benchmark.
