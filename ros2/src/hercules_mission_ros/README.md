# HERCULES mission ROS adapter

`hercules_mission_ros` converts the existing wrapper's vehicle-local odometry
into the common frame used by the working Python mission. The package also has
a separate, bounded RuralAustralia nominal-mission runner. Distributed target
tracking lives in `hercules_tracking_ros`; this package consumes each agent's
local estimate and owns mission orchestration, origin calibration, and actuation.
The optional CBF core/ROS adapter and read-only obstacle/collision observers live
in `hercules_cbf` and `hercules_cbf_ros`, and the mission node invokes them only
when enabled.

## Comparing the Python and ROS paths

Use the no-CBF ROS configuration as the first comparison baseline:

```bash
./docker/ros2/reproduce_cbf.sh --mode no_cbf --obstacles none --dry-run
```

Run this live harness only when a supported Linux AirSim/Unreal simulator is
reachable; `dry_run` disables actuation but still requires ROS state and origin
discovery. A Mac-only setup should use the Docker build/test gate and the
Python/C++ replay tests first, then perform live mode comparisons on Linux.

The `rural_tracking_mestres.yaml` and `rural_tracking_wang.yaml` profiles enable
the corresponding native filter for later experiments. CBF mode comparison is
separate from the default tracking reproduction gate. The perception-backed
observer reuses Python capture/detection helpers, but its canonical camera pose,
vehicle-origin, and orientation contract must be validated before a camera CBF
run is treated as equivalent to the Python-only implementation. Compare the
deterministic Python/C++ replay and no-CBF ROS artifacts first, then compare live
truth-observation runs, and only then evaluate camera/perception CBF behavior.

## Canonical frame

The canonical frame ID is `airsim_world_ned`. It is intentionally not called
ENU and is not claimed to be REP-103 compliant:

- +X is AirSim world +X (north), in metres.
- +Y is AirSim world +Y (east), in metres.
- +Z is down, in metres.
- velocity uses the same fixed world-NED axes, in m/s.
- orientation is the AirSim NED quaternion in `[w,x,y,z]` order.
- yaw is the ZYX yaw in radians, wrapped to `[-pi,pi)`, and yaw rate is the
  AirSim NED angular-velocity Z component in rad/s.

## Executable wrapper semantics and inverse

The launched `hercules_node` is built from
`airsim_ros_pkgs/src/hercules_ros_wrapper.cpp`. Its
`get_odom_msg_from_kinematic_state` publishes `kinematics_estimated` after:

```text
p_wrapper = [ p_local_ned.x, -p_local_ned.y, -p_local_ned.z ]
v_wrapper = [ v_ned.x,       -v_ned.y,       -v_ned.z       ]
w_wrapper = [ w_ned.x,       -w_ned.y,       -w_ned.z       ]
q_wrapper = [ q_ned.w, q_ned.x, -q_ned.y, -q_ned.z ]
```

The adapter deliberately performs the exact inverse:

```text
p_local_ned = [ p_wrapper.x, -p_wrapper.y, -p_wrapper.z ]
p_world_ned = vehicle_origin_world_ned + p_local_ned
v_world_ned = [ v_wrapper.x, -v_wrapper.y, -v_wrapper.z ]
w_world_ned = [ w_wrapper.x, -w_wrapper.y, -w_wrapper.z ]
q_ned       = [ q_wrapper.w, q_wrapper.x, -q_wrapper.y, -q_wrapper.z ]
```

There is no origin rotation. Translation-only behavior is supported by the
Python reference's `actor_position - kinematics_position` calculation and by
the live motion comparison against `simGetObjectPose(name, True)`: both
displacement vectors had cosine indistinguishable from 1. Wrapper topic/frame
labels do not make the published local pose a common multi-agent world pose.

The topic name contains `ground_truth`, but the launched executable actually
publishes `kinematics_estimated`. This label/data mismatch matters for origins:
in the live setup, the wrapper-local estimate was zero after settling while
`simGetGroundTruthKinematics().position` retained the settings spawn frame.

## Origins, timestamps, and validity

Origins are explicit parameters. The settings requested Drone1
`[0,-3,-0.2]` and Husky1 `[0,3,-0.2]`, but live evidence showed these are spawn
requests, not the world positions of the wrapper's local odometry zeros. Using
them directly produced constant 1.023900 m (Drone1) and 0.916258 m (Husky1)
3-D position errors. Actor-minus-ground-truth-kinematics still recovered the
requested spawn origins within 1.5 mm, so the disagreement is specifically
between wrapper `kinematics_estimated` and the direct ground-truth kinematics.

For the validated run, the observed odometry origins were:

```text
Drone1 [0.000000000, -3.000000000, 0.823899925]
Husky1 [0.004974978,  3.001892567, 0.716242492]
```

`config/hero_smoke_state.yaml` records those explicit mapping origins and also
documents the requested settings poses. The adapter code contains no
vehicle-origin constants. The Rural mission launch instead calibrates all nine
origins after runtime spawn/settling from synchronized wrapper-local and direct
world-NED poses. Requested spawn coordinates alone remain insufficient.

The simulator odometry stamp remains the canonical sample stamp. A separate
steady-clock receipt time is used only by `StateCache` for freshness. Empty or
nonfinite state, nonpositive stamps, invalid quaternions, and regressing stamps
are rejected. Repeated stamps are not republished and cannot keep a state fresh:
freshness requires two distinct samples, a recent receipt, and a recently
advanced simulator stamp.

## Running

The original two-vehicle read-only adapter remains available with:

```bash
./docker/ros2/exec.sh ros2 launch hercules_mission_ros state.launch.py
```

The configured outputs are:

```text
/hercules_mission/ground_truth/Drone1
/hercules_mission/ground_truth/Husky1
```

See `ros2/validation/mission_state/README.md` for direct-AirSim comparison,
CSV capture, plots, metrics, and the optional screenshot command.

For the full mission, first start Unreal on the host:

```bash
./docker/ros2/launch_rural_mission_sim.sh
```

Then run the nine-vehicle wrapper, runtime calibration, and controller with a
single ROS launch. Start with the mandatory no-actuation preflight:

```bash
./docker/ros2/exec.sh ros2 launch hercules_mission_ros rural_nominal.launch.py dry_run:=true duration_sec:=5
```

Set `dry_run:=false` only after that succeeds. The live path derives every
origin from synchronized wrapper-local and direct `simGetObjectPose(name,
True)` samples; it does not reuse the measured smoke origins below.

The default target path is the production distributed-camera path:

```bash
./docker/ros2/exec.sh ros2 launch hercules_mission_ros rural_nominal.launch.py \
  dry_run:=false target_source:=distributed_tracking \
  target_observation_source:=camera duration_sec:=30
```

This launch starts eight independent C++ tracker processes and the
asynchronous Python camera observer from `hercules_tracking_ros`. Drone1,
Drone2, SimpleFlight, Drone4, and Drone5 use `target_bottom`; Husky1, Husky2,
and Husky3 use `front_center`. Each formation controller consumes only its own
`/hercules_tracking/<agent>/Target1/estimate`. A fresh active estimate is
predicted to the control time; a missing, inactive, future-dated, or stale
estimate produces zero UAV acceleration or a hard-stop UGV command. There is
no truth fallback.

The default launch leaves CBF disabled (`cbf_enabled:=false` and
`cbf_obstacle_source:=none`). This preserves a clean ROS-versus-Python tracking
comparison without safety-filter intervention. Enable CBF explicitly with the
mode profiles or launch arguments only after the no-CBF run and the required
origin/capture validation succeed.

For the deterministic observation gate, change only
`target_observation_source:=truth`. To reproduce the historical direct-truth
control path, set `target_source:=truth`; tracker/observer nodes are then not
started. Camera mode requires a rendered Unreal RHI. Do not use `-nullrhi`;
use the ordinary visible launch or `-RenderOffscreen` when a noninteractive
rendered session is needed.

The staged live gates are launch arguments. Target-only validation uses
`dry_run:=false enable_target:=true enable_formation:=false`; the complete run
uses both enable arguments set to `true`. Logs are rendered with:

```bash
../.venvs/hercules-python310/bin/python ros2/validation/rural_nominal/render_validation.py \
  ros2/validation/rural_nominal/artifacts/mission.jsonl
```

## Validation boundary

Deterministic conversion, origin separation, orientation/yaw, invalid values,
timestamp freshness, mission gating, truth-target injection, and actuator
conversion are covered by unit and synthetic ROS tests. The historical report
below covers the two-vehicle smoke configuration. The separate nine-vehicle
headless live report is under `ros2/validation/rural_nominal/`; its generated
outputs are in the Git-ignored `artifacts/` subdirectory. Neither experiment
is safety or stability evidence.

### Live results (2026-09-08)

The recorder sampled direct AirSim immediately from each canonical-state
callback. Position used `simGetObjectPose(name, True)`; velocity and yaw used
`simGetGroundTruthKinematics`. It recorded pre-motion stationary samples,
observed the unchanged existing smoke motion test, then recorded post-motion
stationary samples. Pointwise motion position is expected to contain small RPC
sampling skew, so displacement/direction and tight stationary comparisons are
the primary checks.

| metric | tolerance | Drone1 | Husky1 |
|---|---:|---:|---:|
| max stationary position error | 0.05 m | 0.000000060 m | 0.001093825 m |
| post-motion position error | 0.05 m | 0.000000060 m | 0.001093823 m |
| max asynchronously sampled motion error | reported | 0.003343232 m | 0.002666608 m |
| displacement error | 0.05 m | 0.000000060 m | 0.000000016 m |
| stationary velocity error | 0.05 m/s | 0.000000000 m/s | 0.000000000 m/s |
| yaw error | 0.02 rad | 0.0000000004 rad | 0.002561008 rad |
| settings spawn-origin recovery error | 0.05 m | 0.001483989 m | 0.000980743 m |
| XY displacement direction cosine | at least 0.95 | 1.000000000 | 1.000000000 |
| moving velocity direction agreement | at least 95% | 100% | 100% |

Drone1 and Husky1 moved 0.201242 m and 0.217486 m respectively. All numerical
tolerances passed, there were no reference RPC errors, and the top-down plots
show the canonical and direct XY trajectories overlaid without reflection,
rotation, swapped axes, translation, reversal, or incorrect inter-vehicle
separation.

The ordinary visible Vulkan launch failed before AirSim startup with
`VK_ERROR_OUT_OF_DEVICE_MEMORY`; NVML also reported a driver/library version
mismatch. Validation therefore used Unreal `-nullrhi`. This preserved live
physics/RPC/ROS validation and plot generation, but a visible Unreal screenshot
could not be produced in this environment. Screenshots remain supplementary,
not a replacement for the successful numerical and trajectory comparison.

That paragraph is historical Linux evidence. On Apple Silicon, the
platform-aware launch helpers select the native UE 5.2.1 Metal editor and omit
the Linux NVIDIA/Vulkan variables. When Docker Desktop host networking is not
enabled, use `HERCULES_NETWORK_MODE=bridge`,
`HERCULES_RPC_BIND_IP=0.0.0.0`, and `AIRSIM_HOST=host.docker.internal`, with
the same host and ports passed to every ROS and Python client.

These results validate only the two settings-defined smoke vehicles and the
observed origins above. See the Rural nominal validation report for the later
nine-vehicle runtime-calibrated experiment.

## Separate ROS mission media

Set `record_video:=true` on `rural_nominal.launch.py` to capture the same three
streams used by the Python distributed mission: the moving `mission_follow`
chase camera, a selected UAV `front_center`, and a selected UGV
`front_center`. Capture runs in a separate Python node and camera frames are
encoded only after the mission control loop finishes. Each run writes its own
`topdown.mp4`/`topdown.gif`, `mission_chase.mp4`/`.gif`, FPV files, and
`media_manifest.json` under `video_output_dir`.

The Unreal settings must contain the recording-only `mission_follow` external
camera before startup. Prepare a copy of the normal Rural Australia settings,
then point the host launcher at that copy:

```bash
../.venvs/hercules-python310/bin/python ros2/src/hercules_mission_ros/scripts/prepare_rural_video_settings.py \
  --source docker/ros2/settings.rural-nominal.json \
  --output /tmp/settings.rural-video.json
HERCULES_UNREAL_SETTINGS=/tmp/settings.rural-video.json \
  ./docker/ros2/launch_rural_mission_sim.sh -noraytracing -NoLumenReflections
```

The helper adds only the recording cameras and leaves the mounted perception
cameras unchanged. An already-running Unreal session cannot create the
external camera from the ROS launch file.

For an existing ROS JSONL and captured frame directory, export media without
rerunning the mission:

```bash
./docker/ros2/exec.sh ros2 run hercules_mission_ros render_ros_media \
  --log /workspaces/hercules/ros2/validation/rural_nominal/artifacts/mission.jsonl \
  --output-dir /workspaces/hercules/ros2/validation/rural_nominal/artifacts/mission-media \
  --staging-dir /workspaces/hercules/ros2/validation/rural_nominal/artifacts/mission-frames \
  --route-heading -1.5083775167989393 --playback-speed 2
```

To make separate Python and ROS top-down animations use identical bounds,
first compute a shared display file with `--reference-log`, then pass that
file to each export using `--display-config`. The files remain separate; the
shared JSON only fixes route heading, origin, and plot limits.
