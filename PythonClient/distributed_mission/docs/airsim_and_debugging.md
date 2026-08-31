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

On the reference Mac, the optimized FlyingCPP headless run achieved roughly
59--60 ms mean cycle time and 93--94 ms p95 over 60 control steps for both
CBF methods. A small number of deadline misses can still occur from Unreal
or RPC jitter; they are recorded in the JSONL timing fields.

Launch modes are `visible`, `headless`, and `existing`. FlyingCPP is the
default orchestrator map. It uses deterministic `1M_Cube_Chamfer` blocks whose
names begin with `distributed_cbf_block_`; only those objects are deleted on
cleanup. RuralAustralia Example 1 remains selectable, but is not yet a
safety-certification scenario.

The default FlyingCPP course places the mission goal at x=16 m, beyond a
deterministic mixed-height obstacle course. The first wall has narrow lateral
passages that are smaller than the UAV box formation, and floating blocks in
the passages create additional altitude changes. Use `--no-spawn-obstacles` to
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
