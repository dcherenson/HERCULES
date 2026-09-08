# Validation record — 2026-09-08

Native HERCULES physics and Docker ROS 2 Humble were exercised on the same Ubuntu
22.04 host. Rendering was unavailable: both windowed and offscreen Vulkan attempts
failed a fence assertion in `VulkanCommandBuffer.cpp:503`. Native `-nullrhi` physics
worked. These results do **not** validate rendered performance or camera output.

## Environment and build

- Repository starting commit: `ce742c81`, branch `codex/ros2`.
- CPU: Intel Core i7-13650HX, 20 logical CPUs; 15 GiB RAM, 2 GiB swap.
- Native Unreal: 5.2.1, Blocks project, RuralAustralia example map.
- Container: Humble/Jammy, GCC 11.4.0, CMake 3.22.1, host networking,
  ROS domain 42, developer UID/GID matching the host (1001).
- Base: `ros:humble-ros-base-jammy@sha256:75dd3aba34a3838dadbb31a9f7bef769bdfa8713e6cec686fc868db2981b0987`.
- Built image: `sha256:01e355483d2557249f8c7e060d7099c8373b424d41502b7e529def8e9e408eb7`.
- Image build passed; its manifest-only context was about 16 KB. `rosdep check`
  reports all system dependencies satisfied. No host ROS or native AirSim binaries
  were used to build the ROS overlay; AirLib/MavLinkCom/rpclib compile inside Docker.
- RelWithDebInfo, sequential packages, two compiler jobs by default. The independent
  build directory started empty with one job; it was paused to free memory during
  live validation, then resumed with two jobs after stopping Unreal.

The repository has **11 existing packages**, not 12 as estimated in the plan.
With `hercules_control`, all 12 packages are included: airsim_interfaces,
airsim_ros_pkgs, hercules-ros2, hercules_control, imu_complementary_filter,
imu_filter_madgwick, imu_tools, octomap_mapping, octomap_ros,
octomap_rviz_plugins, octomap_server, and rviz_imu_plugin. No package or target
was disabled. See the README for each manifest/CMake correction and its reason.

The independent full build completed successfully: **12 packages**, with no failed
packages. Its resumed pass took 4 min 19 s; the final test-only addition rebuilt in
12.6 s. Final tests report **30 results, zero errors/failures/skips** (25 individual
test cases plus five CTest suite entries). They cover axis/quaternion conversion,
invalid/stale state, motion bounds and deadlines, cleanup, service failure,
interrupted drone/UGV tests, asynchronous flight requests, and a synthetic ROS
latched-command expiry through the production watchdog.

The core also compiled directly with `g++ -std=c++17`, its own include directory,
and `/usr/include/eigen3`, without ROS or AirSim include paths. Its generated compile
command confirms this separation. CMake caches confirm RelWithDebInfo for both the
wrapper and OctoMap. Existing compiler/deprecation warnings remain; no warnings
were suppressed to pass validation. The normal development overlay was also
updated: all 12 packages built and the same 30 test results passed, so the default
helper commands are ready to use. Release is supported but was not fully built.

## Exact execution commands

From the repository root, image/full-build/test commands:

```bash
./docker/ros2/build.sh
./docker/ros2/build_ws.sh
./docker/ros2/test.sh
./docker/ros2/exec.sh rosdep check --from-paths ros2/src --ignore-src --rosdistro humble
# Separate clean container; initial invocation used BUILD_JOBS=1.
docker compose -p hercules-clean -f docker/ros2/compose.yaml run --rm --no-deps \
  --entrypoint /workspaces/hercules/docker/ros2/build_ws.sh \
  -e HERCULES_BUILD_ROOT=/workspaces/hercules/ros2/.docker/humble/RelWithDebInfo-fresh \
  -e BUILD_JOBS=2 dev
```

Native physics and ROS validation, each persistent command in its own terminal:

```bash
UNREAL_EDITOR=/home/dasc-lab/Desktop/Unreal/Engine/Binaries/Linux/UnrealEditor \
  ./docker/ros2/launch_sim.sh -nullrhi -unattended -NoSound -stdout
./docker/ros2/launch_wrapper.sh
./docker/ros2/exec.sh ros2 node list
./docker/ros2/exec.sh ros2 topic list
./docker/ros2/exec.sh ros2 service list
./docker/ros2/smoke.sh drone observe -p duration_sec:=5.0
./docker/ros2/smoke.sh ugv observe -p duration_sec:=5.0
# Stop the passive wrappers, then:
./docker/ros2/launch_wrapper.sh enable_api_control:=true benchmark_logging:=true
./docker/ros2/smoke.sh drone motion -p wait_on_flight_task:=false
./docker/ros2/smoke.sh ugv motion
# Stop those wrappers; the benchmark manages its own launches:
./docker/ros2/benchmark.sh
```

The final fresh-overlay checks used:

```bash
export HERCULES_BUILD_ROOT=/workspaces/hercules/ros2/.docker/humble/RelWithDebInfo-fresh
./docker/ros2/test.sh
./docker/ros2/launch_wrapper.sh enable_api_control:=true benchmark_logging:=true
# Another terminal with the same HERCULES_BUILD_ROOT:
./docker/ros2/smoke.sh drone motion
./docker/ros2/smoke.sh ugv motion
./docker/ros2/smoke.sh drone observe -p duration_sec:=5.0
./docker/ros2/smoke.sh ugv observe -p duration_sec:=5.0
```

## Live motion and safety findings

Both endpoints connected and exposed the exact node/topic/service names in the
README. API-disabled wrappers produced state without command subscribers. Initial
five-second observe runs delivered 20 Hz callbacks and distinct timestamps for both
vehicles. After all benchmarks, another five-second observation of each vehicle
passed at 20 Hz with zero measured speed; the native simulator remained responsive.

| Test | Result | Measured horizontal motion | Cleanup |
|---|---|---:|---|
| Drone, final fresh overlay/default flight mode | Passed | 0.200602 m | Takeoff/settle, velocity pulse, stop, land verified from odometry |
| Husky, final fresh overlay/0.15 throttle | Passed | 0.219552 m | Early speed guard, zero throttle/full brake, zero speed verified |
| Drone, blocking flight requests | Failed safely | No horizontal pulse | Native takeoff returned false after ~1.5 m climb; stopped and landed |

The blocking failure is retained as a native-plugin limitation, not converted to
success. Blocking service responses now carry the actual RPC task result. The smoke
node defaults to asynchronous submission, then independently checks physical
completion within the original 20/60-second deadlines. Set
`wait_on_flight_task:=true` to reproduce strict blocking-result validation.
No command magnitude or safety bound was increased to obtain the passing result.

An earlier blocking-service run exposed the wrapper holding its command mutex
through the flight wait, stalling state polling. A separate persistent flight RPC
client fixes that problem; subsequent runs maintained fresh state throughout flight.
The Husky wrapper logged one watchdog stop after its command stream ended
(`commands_received=26`, `rpc_dispatched=27`, `watchdog_stops=1`). This live check
expired an already-braked final command. Unit tests and a synthetic ROS publisher/subscriber test separately check that an
expired latched throttle command is replaced by zero throttle and full braking
after 0.5 seconds, without any further message from the command publisher.

## Benchmark measurements

Each passive trial measured 30 seconds. Active trials used five seconds of warm-up
and 15 seconds of measurement. Both wrappers ran throughout, with one smoke node at
a time. Values below are rates per steady wall-clock second. Wrapper samples use
the interior of each measurement window, so small receipt/publisher differences
are sampling-boundary effects. Odom publication/state-acquisition rates are retained
in `summary.csv`; they matched wrapper distinct-stamp rates in these windows.

| Vehicle | Period (s) | Requested Hz | Publisher Hz | Receipt Hz | RPC Hz | Odom callback Hz | Distinct stamp Hz | Adapter µs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| drone | 0.05 | 0 | 0.00 | 0.00 | 0.00 | 20.00 | 20.00 | 0.492 |
| ugv | 0.05 | 0 | 0.00 | 0.00 | 0.00 | 20.00 | 20.00 | 0.465 |
| drone | 0.05 | 10 | 10.00 | 10.03 | 10.03 | 20.00 | 20.00 | 0.556 |
| drone | 0.05 | 20 | 20.00 | 20.00 | 20.00 | 20.00 | 20.00 | 0.597 |
| drone | 0.05 | 50 | 50.00 | 50.00 | 20.00 | 20.00 | 20.00 | 0.481 |
| drone | 0.05 | 100 | 100.00 | 100.00 | 20.00 | 20.00 | 20.00 | 0.494 |
| ugv | 0.05 | 10 | 10.00 | 10.00 | 10.00 | 20.00 | 20.00 | 0.501 |
| ugv | 0.05 | 20 | 20.00 | 20.00 | 20.00 | 20.00 | 20.00 | 0.443 |
| ugv | 0.05 | 50 | 50.00 | 50.00 | 20.00 | 20.00 | 20.00 | 0.431 |
| ugv | 0.05 | 100 | 100.00 | 100.00 | 20.00 | 20.00 | 20.00 | 0.406 |
| drone | 0.01 | 10 | 10.00 | 9.99 | 9.99 | 76.06 | 76.06 | 0.295 |
| drone | 0.01 | 20 | 20.00 | 20.00 | 20.00 | 100.00 | 100.00 | 0.255 |
| drone | 0.01 | 50 | 50.00 | 50.03 | 50.03 | 100.00 | 100.00 | 0.253 |
| drone | 0.01 | 100 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.306 |
| ugv | 0.01 | 10 | 10.00 | 9.95 | 9.95 | 100.00 | 100.00 | 0.254 |
| ugv | 0.01 | 20 | 20.00 | 19.97 | 19.97 | 100.00 | 100.00 | 0.236 |
| ugv | 0.01 | 50 | 50.00 | 50.03 | 50.03 | 100.00 | 100.00 | 0.226 |
| ugv | 0.01 | 100 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.220 |

At the 0.05-second period, receipt kept up with 50/100 Hz publishers while dispatch
stayed near 20 Hz: commands were coalesced in the latest-command cache. At the
0.01-second period, the 100 Hz trials dispatched about 100 commands per second.
The first drone/10 Hz trial polled state at about 76 Hz; later trials reached 100 Hz.
No repeated simulator timestamps were observed within these measured windows.
These measurements do not establish a universal RPC ceiling. Drone dispatch counts
**asynchronous submissions**, not completed simulator actions; Husky dispatch counts
successful synchronous returns, not distinct physics steps.

CPU was sampled with `pidstat -h -u ... 1`. Means below include phase startup and
warm-up, not just the interior measurement windows. Smoke means cover samples while
a smoke process existed. Percent is relative to one logical CPU (100%); native CPU
can exceed 100%. Native sampling started 13 seconds into the passive phase.

| Phase | Drone wrapper CPU % | Husky wrapper CPU % | Smoke CPU % | Native Unreal CPU % |
|---|---:|---:|---:|---:|
| 0.05 s passive | 3.84 | 3.92 | 0.39 | 417.40 |
| 0.05 s command sweep | 3.57 | 3.25 | 0.80 | 455.67 |
| 0.01 s command sweep | 4.38 | 5.41 | 1.17 | 461.13 |

Adapter timing measures odometry conversion plus local state validation/cache work.
It is neither transport latency nor controller computation; no controller exists.
Native CPU was collected using a temporary Docker process with `--pid=host` and
`--entrypoint pidstat`, since host `pidstat` was unavailable. No host package install
was needed. All benchmark command publications were neutral.

## Evidence locations

Untracked logs are retained under:

- `ros2/.docker/humble/RelWithDebInfo/validation/implementation/`: image/build/test,
  motion, simulator, post-benchmark state, and native CPU logs.
- `ros2/.docker/humble/RelWithDebInfo/validation/20260908-011022/`: raw per-trial
  state logs, wrapper boundary counters, container CPU logs, and `summary.csv`.
- `ros2/.docker/humble/RelWithDebInfo-fresh/{build,install,log}/`: independent full
  workspace build and final test results.

The native simulator and benchmark-owned wrappers were stopped after validation.
Existing unrelated Docker services and Python control code were left untouched.
