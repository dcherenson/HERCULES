# AirSim launch, maps, and debugging

Both `start_formation.py` and `orchestrator.py` use the shared launcher.

```bash
./herculesvenv/bin/python start_formation.py --launch-mode visible --map rural_australia
./herculesvenv/bin/python PythonClient/distributed_mission/orchestrator.py \
  --launch-mode headless --map flyingcpp --cbf-method mestres
```

The orchestrator defaults to `--timing-mode realtime`, which leaves AirSim
running and schedules the control loop against a 10 Hz monotonic deadline.
Use `--timing-mode stepped` when deterministic `simPause`/
`simContinueForTime` execution is needed. Obstacle perception defaults to
2.5 Hz per vehicle and is staggered across the team; UAV captures use the
existing three-camera depth fan while cached proxies are inflated by age and
expire into the safety fail-safe. Use the existing `--sensor-rate` option to
trade capture cost against observation freshness.

The control loop is scheduled against a 10 Hz deadline, but live sensor
perception is the expensive path. On the current macOS/AirSim setup, a
three-camera UAV obstacle fan can take hundreds of milliseconds per cycle even
when captures are staggered; deadline misses are recorded in the JSONL timing
fields. Truth-obstacle mode skips sensor captures and is useful for isolating
CBF/formation behavior from this perception cost. Keep this distinction in
mind when judging chase-video timing versus controller real-time performance.

Launch modes are `visible`, `headless`, and `existing`. FlyingCPP is the
default orchestrator map. It uses deterministic `1M_Cube_Chamfer` blocks whose
names begin with `distributed_cbf_block_`; only those objects are deleted on
cleanup. RuralAustralia Example 1 remains selectable, but is not yet a
safety-certification scenario.

The default FlyingCPP course places the mission goal at `(16, 1, -1)` in
AirSim NED, beyond two staggered mixed-height obstacle rows. It has no bypass
waypoint: the first central gap is near `y=0` and the second is shifted near
`y=1`, so the conservative CBF corridors require a turn and slight formation
compression. A floating block tests UAV altitude changes.
Fixed route markers (the goal and any future
course waypoint) are drawn in Unreal with AirSim's persistent plot API when
supported. Use `--no-spawn-obstacles` to disable this course.

For `--map rural_australia`, the goal actor determines the route heading and
the optional `--initial-heading-offset-deg` rotates the startup formation and
target pattern around that route. The validated target-tracking configuration
uses `--initial-heading-offset-deg 90 --target-ugv-circumradius 5`; this keeps
the deterministic figure-eight clear of the observed foliage/vehicle contact
region. FlyingCPP uses the unoffset route and a 5 m target-centered UGV radius
by default. These are route/initialization choices, not changes to the CBF
equations.
RuralAustralia ground-referenced goal, waypoint, and obstacle geometry receives
a +2 m NED correction because its terrain is approximately 2 m lower than the
FlyingCPP reference. UAV flight altitude is not shifted. The UAV CBF altitude
floor defaults to one metre above the calibrated ground (`z=+1` NED on
RuralAustralia, versus `z=-1` on FlyingCPP), so the drones can descend beneath
tree canopies when the safety filter needs the additional clearance. Use
`--uav-altitude-floor` to override this test parameter.

In managed `visible`/`headless` launches, the three controlled Huskies are
deliberately not auto-created from `settings.json`. The map's auto-created
CPHusky bodies can begin falling before the client pauses physics; on
RuralAustralia that can produce invalid Chaos actors and a later Unreal crash.
The launcher copies the configured `Lidar1` profile into the temporary
`DefaultSensors` section and the orchestrator runtime-spawns each Husky while
paused at `z=-1` NED, above the uneven terrain, before applying the handbrake.
The source settings file is never modified. Existing-launch mode cannot apply
this protection, so use a managed launch when reliable UGV startup is needed.

For RuralAustralia, the launch-time override rotates the existing default
CameraDirector XY position and yaw with the mission, so the original camera
height and pitch are preserved while it views the agents from behind. When
`--top-down-camera` is selected, the external camera instead uses the explicit
rear-side position and a `+90` degree top-down yaw so the agents' `-Y` travel
direction runs away from the camera (toward the top of the view). FlyingCPP
keeps its existing camera placement.

Startup launches all UAV takeoff commands concurrently, then launches all UAV
altitude commands concurrently. FlyingCPP obstacle blocks are spawned before
vehicle motion begins. The CBF altitude/ground guard applies only to UAVs;
UGV contact with Unreal ground, landscape, terrain, or floor actors is ignored
for collision warnings and plot markers, while UGV contact with other objects
remains reportable.

The external simulator camera keeps its original AirSim/Unreal angle by
default. Add `--top-down-camera` to opt into a launch-time top-down view,
centered near the course at `(x=6, y=0)` and 30 m above the NED origin; adjust
it with `--camera-height`, `--camera-x`, and `--camera-y`. The launcher applies
this through a temporary `CameraDirector` settings override so the Hero-mode
camera selector cannot move Drone1's vehicle-mounted perception camera. The
The camera roll is forced to exactly zero in both launch-time and runtime
poses. The older `--no-top-down-camera` flag remains accepted as an explicit disable. With
`--launch-mode existing`, configure the CameraDirector before starting Unreal
because a running world cannot safely receive this override.

Target tracking uses a 0.10 m/s figure-eight by default and the existing UGV
CBF control-point lookahead defaults to 0.1 m; both remain command-line tuning
parameters. It is active only for `--mission-objective track-target`. With
`--mission-objective fixed-goal`, the target actor is not spawned and the
tracking cameras, worker, estimator, target CBF proxy, target commands, and
target collision polling are all disabled; the original goal-directed
formation controller is used. As a first target-centered CBF baseline, use
truth observations with `--tracking-measurement-std 0.25` and the default
target speed; this is a tuning command, not a new controller structure. Each run writes JSONL records containing configuration, states, formation
errors, nominal/safe commands, barrier values, solver status, robust terms,
sensor/proxy counts, and fallback events. The AirSim-specific tests are marked
for explicit execution because they require a running Unreal instance.
