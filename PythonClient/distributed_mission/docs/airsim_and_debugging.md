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
2.5 Hz per vehicle and is staggered across the team; cached proxies are
inflated by age and expire into the safety fail-safe.

The control loop is scheduled against a 10 Hz deadline, but live sensor
perception is the expensive path: a recent 800-step FlyingCPP run averaged
about 115 ms per cycle, with perception averaging about 67 ms and occasional
longer RPC/depth calls. Deadline misses are recorded in the JSONL timing
fields. Truth-obstacle mode skips sensor captures and is useful for isolating
CBF/formation behavior from this perception cost.

Launch modes are `visible`, `headless`, and `existing`. FlyingCPP is the
default orchestrator map. It uses deterministic `1M_Cube_Chamfer` blocks whose
names begin with `distributed_cbf_block_`; only those objects are deleted on
cleanup. RuralAustralia Example 1 remains selectable, but is not yet a
safety-certification scenario.

The default FlyingCPP course places the mission goal at x=16 m, beyond a
deterministic mixed-height obstacle course. It uses one fixed bypass waypoint
`(14, -14, -1)` in AirSim NED and a 3 m transition radius. UAV references use
that point as a temporary virtual formation center while the UGVs retain their
leader-follower formation; no online waypoint generation is used. The first
row has lateral passages smaller than the UAV box formation, and floating
blocks create additional altitude changes. Use `--no-spawn-obstacles` to
disable this course.

Startup launches all UAV takeoff commands concurrently, then launches all UAV
altitude commands concurrently. FlyingCPP obstacle blocks are spawned before
vehicle motion begins. The CBF altitude/ground guard applies only to UAVs;
UGV contact with Unreal ground, landscape, terrain, or floor actors is ignored
for collision warnings and plot markers, while UGV contact with other objects
remains reportable.

The external simulator camera is configured top-down at launch, centered near
the course at `(x=6, y=0)` and 30 m above the NED origin. Adjust it with
`--camera-height`, `--camera-x`, and `--camera-y`, or disable it with
`--no-top-down-camera`. The launcher applies this through a temporary
`CameraDirector` settings override so the Hero-mode camera selector cannot
move Drone1's vehicle-mounted perception camera. With `--launch-mode existing`,
configure the CameraDirector before starting Unreal because a running world
cannot safely receive this override.

Each run writes JSONL records containing configuration, states, formation
errors, nominal/safe commands, barrier values, solver status, robust terms,
sensor/proxy counts, and fallback events. The AirSim-specific tests are marked
for explicit execution because they require a running Unreal instance.
