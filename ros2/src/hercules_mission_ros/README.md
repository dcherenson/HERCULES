# HERCULES read-only mission-state adapter

`hercules_mission_ros` converts the existing wrapper's vehicle-local odometry
into the common frame used by the working Python mission. It observes state
only: the package has no command publishers, AirSim RPC client, takeoff/land
client, spawning code, controller, tracker transport, CBF, or perception.

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
vehicle-origin constants. A future Rural mission manager must establish and
pass authoritative wrapper-odometry origins after runtime spawn/settling for
every vehicle, including Target1; requested spawn coordinates alone are not
sufficient and that case is not yet validated.

The simulator odometry stamp remains the canonical sample stamp. A separate
steady-clock receipt time is used only by `StateCache` for freshness. Empty or
nonfinite state, nonpositive stamps, invalid quaternions, and regressing stamps
are rejected. Repeated stamps are not republished and cannot keep a state fresh:
freshness requires two distinct samples, a recent receipt, and a recently
advanced simulator stamp.

## Running

With the native simulator and existing wrapper running:

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

## Validation boundary

Deterministic conversion, origin separation, orientation/yaw, invalid values,
and timestamp freshness are validated in unit and synthetic ROS tests. The
live report below is specifically for settings-defined Drone1 and Husky1 in
`docker/ros2/settings.hero-smoke.json`. Runtime-spawned eight-agent
RuralAustralia origins, nonzero spawn yaw, and the eventual live mission remain
unvalidated and must not be inferred from this two-vehicle smoke setup.

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

These results validate only the current two settings-defined vehicles and the
observed origins above. Runtime-spawned eight-agent RuralAustralia origins,
nonzero spawn yaw, and the eventual live mission remain unvalidated.
