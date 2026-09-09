# RuralAustralia nominal live validation report

Date: 2026-09-09

This report covers the staged, nine-vehicle, truth-target nominal mission. It
does not enable distributed tracking, CBF safety filtering, perception, or
conformal prediction, and it makes no claim about closed-loop stability.

## Staged runs

- Dry run: all nine AirSim names were discovered and all runtime origins
  calibrated; 50 control records were produced at 10 Hz with actuation gated
  off.
- Target only: Target1 moved 0.217 m in 10 seconds and its selected figure-eight
  sample advanced from 7 to 8. Formation commands remained gated off.
- Full nominal mission: 299 samples were recorded in 30 seconds. Mean loop rate
  was 9.9998 Hz, minimum instantaneous rate was 9.6694 Hz, the largest update
  gap was 0.10342 s, and the 95th-percentile gap was 0.10011 s. There were zero
  invalid/stale-state events, rejected commands, or origin-validation errors.

The full-run target moved 0.877 m and progressed through 4.6875% of the sampled
path. End-of-run slot errors were 0.566, 0.557, 0.104, 0.588, and 0.103 m for
Drone1, Drone2, SimpleFlight, Drone4, and Drone5; and 0.436, 5.794, and 0.725 m
for Husky1, Husky2, and Husky3.

Husky2's error plateaued near 5.8 m. The plotted route shows its nominal path
crossing the tightly grouped agents/target, while CBF collision avoidance is
explicitly disabled for this task. The controller math and frozen starting
geometry were therefore not retuned to hide this result. This staged run passes
transport, calibration, timing, naming, frame-direction, and command-delivery
checks, but it does not demonstrate that every UGV reaches its slot safely in
the nominal-only experiment.

The smallest planar separation among controlled agents was 0.0130 m. It occurs
at the initial SimpleFlight/Husky planar overlap before the UAV climbs, and is
another reason these data must not be treated as a safety result.

## Visual inspection

`artifacts/full_mission_trajectories.png` is an equal-axis world-NED XY comparison of all
actual vehicle trajectories, desired target-centered slots, Target1, and the
sampled figure eight. It showed no reflection, 90-degree rotation, unexplained
constant translation, X/Y swap, or reversed target direction.

`artifacts/slot_errors.png`, `artifacts/topdown.mp4`, and
`artifacts/topdown.gif` provide the corresponding error history and existing
Python media-pipeline animation. Exact computed values are in
`artifacts/metrics.json`; the source samples are retained in
`artifacts/mission.jsonl`. The smaller `artifacts/dry_run.jsonl` and
`artifacts/target_only.jsonl` preserve the two preceding stage records. The
entire `artifacts/` directory is ignored by Git.

A visible Unreal screenshot could not be captured. The visible Vulkan launch
failed before AirSim startup with `VK_ERROR_OUT_OF_DEVICE_MEMORY`, and NVML
reported a driver/library version mismatch. Validation used Unreal `-nullrhi`,
which retained physics, RPC, and ROS behavior but provided no visible render.

## Runtime-calibrated origins

The dry-run calibration produced these world-NED translations in metres:

```text
Drone1       [-1.871349, -2.120862, 0.301316]
Drone2       [ 2.120862, -1.871349, 0.301315]
SimpleFlight [ 0.000000,  0.000000, 0.301315]
Drone4       [-2.120862,  1.871349, 0.301315]
Drone5       [ 1.871349,  2.120862, 0.301316]
Husky1       [ 0.017110,  0.004151, 0.800323]
Husky2       [-3.315488, -2.208039, 0.800323]
Husky3       [-3.565002,  1.784171, 0.800323]
Target1      [ 4.945481,  1.306305, 0.800323]
```

The later full run deliberately recalibrated against then-current world poses,
so ground vehicles shifted by their intervening target-only motion. This is
expected for translation calibration and confirms that configured spawn values
are not reused as authoritative live origins.

## Reproduction

Start the RuralAustralia simulator, run each launch gate, then render:

```bash
./docker/ros2/launch_rural_mission_sim.sh -nullrhi -unattended -NoSound -stdout
./docker/ros2/exec.sh ros2 launch hercules_mission_ros rural_nominal.launch.py dry_run:=true duration_sec:=5 log_path:=/workspaces/hercules/ros2/validation/rural_nominal/artifacts/dry_run.jsonl
./docker/ros2/exec.sh ros2 launch hercules_mission_ros rural_nominal.launch.py dry_run:=false enable_target:=true enable_formation:=false duration_sec:=10 log_path:=/workspaces/hercules/ros2/validation/rural_nominal/artifacts/target_only.jsonl
./docker/ros2/exec.sh ros2 launch hercules_mission_ros rural_nominal.launch.py dry_run:=false enable_target:=true enable_formation:=true duration_sec:=30 log_path:=/workspaces/hercules/ros2/validation/rural_nominal/artifacts/mission.jsonl
herculesvenv/bin/python ros2/validation/rural_nominal/render_validation.py ros2/validation/rural_nominal/artifacts/mission.jsonl
```
