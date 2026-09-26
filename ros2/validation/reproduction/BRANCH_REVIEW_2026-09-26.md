# Branch review and full-simulator validation — 2026-09-26

The branch can run substantive native Unreal/AirSim missions with Docker ROS, and target-estimation accuracy is close to the Python reference in the measured six-agent trials. It is **not yet a validated, repeatable 100–200-trial research runner**. Ground-vehicle stalls, safety-margin violations, inconsistent localization uncertainty, shutdown reliability, and mixed time domains remain material limitations.

The active fleet is now **Drone1, Drone2, SimpleFlight, Husky1, Husky2, Husky3**, with **Target1 separate**: six controlled robots and seven simulated vehicles. Both the Python and ROS executable defaults use this fleet. Generic historical mathematical fixtures were retained.

## Environment and provenance

- Reviewed branch: `coop_localization`, starting HEAD `66efb6c852775f0ef430da5e7dc5d4b431e6f872`; fixes and this report are uncommitted.
- Native rendered Unreal 5.2.1, Blocks project, `/Game/RuralAustralia/Maps/RuralAustralia_Example_01`, 960×540 window; cameras/depth/LiDAR enabled as required by each case. This was not a mock simulator or ROS-only test.
- Apple M5 Pro, 24 GiB RAM, macOS arm64. ROS Humble runs in Docker Desktop, linux/arm64, `RelWithDebInfo`. Image: `hercules-ros2:humble`, ID `sha256:6b2bc9ff1ce2e27c1a4b53937d9a7902da39f292aa0f91f90fb177dc77a25d4f`.
- AirSim ports 41451/41452, native RPC bind `0.0.0.0`, container host `host.docker.internal`. Final correctness runs use `ClockSpeed=1`; acceleration probes use separate settings copies.
- Python 3.10.21 virtual environment at `/Users/dmrc/git/.venvs/hercules-python310`. `OPENBLAS_NUM_THREADS=1`; ROS launches also use `OMP_NUM_THREADS=1`.
- Python reference mathematics were checked against `main` (`ce742c81`): CBF, formation, and target-motion modules are identical; target tracking differs only in a comment. Runs used the current branch's host I/O and reduced roster, **not an untouched historical checkout**. There is no original Python cooperative-localization baseline.

Raw evidence and executable review helpers are retained locally under [artifacts/branch-review-20260926](artifacts/branch-review-20260926). That directory is Git-ignored, following this validation area's artifact policy. Compact metrics, raw JSONL, launch logs, preparation logs, independent clock samples, and plots remain there; this report retains the key measured results in the repository.

## Method and acceptance limits

The final suite uses real takeoff/hover before each launch, then checks that all three UAVs are airborne in the first mission record. Cases run separately, for approximately 40 wall seconds each. Actual elapsed AirSim time is measured independently from `getMultirotorState(Drone1).timestamp`; JSONL `timestamp` is **not** used as proof of simulated duration. Sampling is approximately 1 Hz, so the reported interval excludes small portions at the beginning/end.

The target-speed setting is 0.5 m/s: the original 0.1 setting barely moved Target1 in a Python trial. Final moving-target cases cover a substantial path. Pose preparation starts from the Python reference's first recorded configuration, but cars subsequently settle onto the terrain and can shift. This is not exact replay of identical initial dynamics, observations, or target paths.

- Combined tracking comparison: distributed truth-generated target observations, truth robot state, Mestres CBF, no static obstacle proxies. Python uses an empty truth-obstacle list. These tests assess target estimation and basic formation behavior, not static-obstacle perception.
- Tracking isolation: distributed target estimation with truth or camera observations; truth robot state; CBF disabled. Contacts in these runs do not indicate a CBF failure.
- Collision isolation: truth robot/target state, stationary target, live static perception, CBF enabled. Selecting Wang changes the UAV controller; **UGVs always use Mestres**, as implemented in `mission_node.cpp`.
- Localization isolation: truth target, estimated robot state for control, CBF disabled; recursive decentralized and GS-CI algorithms each tested with truth-generated and camera relative observations. Truth relative measurements use simulated ground-truth geometry with configured covariance. Camera relative measurements use rendered detection/depth with ground-truth ego kinematics for frame conversion. Startup origins and the GPS/odometry inputs are also simulator-supported. These are not sensor-only or GPS-denied localization tests.

All error figures below are the arithmetic mean of the six per-agent XY RMSEs, computed only from active target estimates or valid/non-stale localization estimates. They are diagnostic single-run results, not statistical equivalence. Collision summaries ignore ground/landscape/terrain/floor contacts for UGVs and Target1, matching Python; UAV ground contacts and other objects remain relevant. Repeated collision timestamps are not counted as separate physical accidents.

| Case | Records | Measured sim s | RTF | Target XY RMSE (m) | Localization XY RMSE (m) |
|---|---:|---:|---:|---:|---:|
| Python: truth tracking + Mestres | 400 | 27.52 | 0.702 | 0.130 | — |
| ROS: truth tracking + Mestres | 400 | 27.88 | 0.714 | 0.129 | — |
| Python: camera tracking + Mestres | 400 | 28.93 | 0.757 | 1.262 | — |
| ROS: truth tracking, CBF off | 400 | 26.97 | 0.709 | 0.130 | — |
| ROS: camera tracking, CBF off | 400 | 26.45 | 0.695 | 1.312 | — |
| ROS: collision isolation, Mestres | 400 | 27.33 | 0.700 | — | — |
| ROS: collision isolation, Wang UAVs / Mestres UGVs | 399 | 25.20 | 0.661 | — | — |
| ROS: recursive localization, truth relative | 400 | 26.17 | 0.670 | — | 0.637 |
| ROS: GS-CI localization, truth relative | 399 | 25.94 | 0.664 | — | 1.214 |
| ROS: recursive localization, camera relative | 400 | 28.31 | 0.725 | — | 1.972 |
| ROS: GS-CI localization, camera relative | 400 | 27.40 | 0.720 | — | 3.724 |

All nine final ROS cases reached mission completion with valid airborne starts; eight had no process-error log entries, and the Wang case had a localization SIGSEGV during teardown. Each final case contains 399–400 control records. “Complete” is not an acceptance verdict.

- **Target tracking:** truth-generated observations give 0.129 m ROS versus 0.130 m Python mean-agent RMSE. Camera observations give 1.312 m ROS versus 1.262 m Python; the ROS camera isolation case has CBF disabled while the Python reference has it enabled. Estimate-active coverage is 99.75% for ROS, 100% Python truth and 98.75% Python camera. This is encouraging estimator agreement, not full closed-loop equivalence. Target paths are 12.93 m ROS truth / 15.38 m Python truth and 11.70 m ROS camera / 12.51 m Python camera.
- **Formation / avoidance:** the two live-perception collision cases have no raw or relevant native contacts. Minimum UGV center separations are 1.787 m (Mestres) and 2.495 m (Wang selection), versus a configured two-radius threshold of 2.50 m: margins −0.713 m and −0.005 m. Mestres Husky1 fell back on 118 records, including 115 primal-infeasible solves. Final ROS maximum slot error is 7.25 m / 7.60 m, so formation convergence is not established. In the combined truth-tracking run Husky3 traveled only 0.034 m and Husky1 reported foliage contact. Terrain-related stalls also occurred in the Python camera run. **Do not mark formation/safety parity passed.**
- **Cooperative localization:** all final localization samples are valid and non-stale, but accuracy and uncertainty are not yet acceptable as a blanket pass. Truth-relative recursive/GS-CI mean XY NEES is approximately 80 / 592, with mean reported trace-based XY sigma about 0.114 / 0.333 m. Camera RMSE rises to 1.97 / 3.72 m; GS-CI camera finishes with mean error above 5 m. CBF is disabled in these cases, and native contacts occur, including Drone1–SimpleFlight in GS-CI camera. **There is no original Python cooperative-localization comparator.**

[Plots (PNG)](artifacts/branch-review-20260926/review-results.png) · [Plots (PDF)](artifacts/branch-review-20260926/review-results.pdf) · [Compact metrics](artifacts/branch-review-20260926/review-results.json)


## Fixes made and verified

1. Applied the explicit 3-UAV/3-UGV roster to AirSim settings, Python orchestration, ROS mission/state configuration, tracking/localization observers, perception, collision reporting, video, and validation runners. UAV spawn Z is zero so a local NED altitude command does not double the intended global altitude offset.
2. Preserved absolute odometry orientation in the AirSim wrapper. The previous shared initial quaternion rotated every vehicle into the first vehicle's initial heading while leaving velocities in world axes. A direct AirSim-versus-ROS probe after the fix showed mean position discrepancies of roughly 0.001–0.036 m and yaw discrepancies up to roughly 0.009 rad.
3. Converted localization odometry back into AirSim NED, including velocity, quaternion, yaw rate, and planar covariance signs. Wired the existing static per-agent origin calibration into localization and gated odometry until it arrives. The same translation is applied to every sample; it does not introduce continuing direct-pose truth fusion.
4. Avoided re-requesting API control on already-controlled UAVs. SimpleFlight resets its active hover goal when API control is requested again; this caused previously prepared UAVs to fall during ROS startup. Final acceptance runs were repeated after this fix. Earlier `fleet6_` ROS correctness runs are diagnostics only; use `final6_` for the final suite.
5. Kept obstacle-observer AirSim clients on one dedicated capture thread. A lock did not protect thread-affine Tornado/msgpack clients when different pool workers reused them. Reused the static camera-FOV cache. Final perception continued through the full missions.
6. Fixed eager `dict.setdefault(...)` construction of a fresh relative-observation RPC client on every capture. The worker now reuses one client per observer, with a regression test for camera round-robin and cleanup.
7. Matched ROS ground-contact classification to Python and made direct-pose bridge shutdown tolerate an already-shutdown ROS context.

The main agent reviewed the actual delegated diffs and tests, corrected an intermediate origin-translation issue before acceptance, inspected native frame/clock code, and personally checked the full-run logs and metrics. No commit, merge, or push was performed.

## Remaining actionable findings

**P1 — Episode/controller time is not simulator time.** `mission_node.cpp:693` advances estimator time by `step * control_dt`; duration and scheduling are based on wall time. Python similarly uses nominal step time for tracking. Changing `ClockSpeed` changes the physical evolution per estimator/control step. `SimModeWorldHero.cpp:40` also sets Unreal time dilation separately from the AirSim steppable clock, with an integer cast when dividing the physics period. The measured UAV AirSim clock therefore does not prove identical timing of every physics subsystem. `simContinueForTime` uses elapsed scheduler time in this implementation and should not be assumed to provide exact simulated stepping. Accelerated performance is a capacity result, not equivalent scientific trials.

**P1 — Collision-free completion does not establish safety or formation parity.** The final collision cases have no relevant native contact reports, but the configured same-type UGV radius margin is violated. Combined formation also has a nearly stationary Husky despite nonzero motion commands. Terrain settling/contact and the ground-vehicle actuator model need repeatable starts and movement checks. CBF only includes same-type neighbors; altitude separation is relied upon for UAV/UGV safety.

**P1 — Cooperative localization is not calibrated.** Valid/non-stale estimates and accepted updates are present, but meter-scale errors and large XY normalized estimation errors remain. Camera observations are substantially worse than synthetic relative observations. GS-CI fusion, relative-measurement geometry/noise, GPS consistency, and clock handling need validation before uncertainty bounds or estimated-state control can be trusted. One localization process segfaulted during shutdown after the Wang mission completed; this is a batch-reliability defect, even though the mission data were already recorded.

**P1 — Camera-relative timestamps corrupt the estimator time domain.** `relative_observation.py:229` captures `time.time()`; `relative_observer_node.py:195` copies it into the scalar measurement timestamp. `localization_adapter.cpp:590` forwards that scalar to the core, while odometry propagation computes `dt` from its simulator header minus the estimator timestamp (`:471–475`). A camera update can move the internal estimate to a host epoch roughly 1.79 billion seconds ahead, causing subsequent odometry `dt` to clamp to zero. The adapter overwrites output headers with odometry headers, hiding this mismatch in normal mission timestamps. This is a verified remaining camera-path defect; its isolated numerical contribution has not been measured. Fix acquisition timestamps and test camera/odom interleaving before accepting camera localization or acceleration.

GS-CI also reapplies cached peer beliefs on local updates. This is a stale-sequence/acceptance-contract concern requiring ablation or sequence-consumption tests; repeated covariance intersection alone is not proof of mathematically invalid covariance. The teardown SIGSEGV cause was not established.

**P2 — Truth static-obstacle mode silently has no fixture geometry.** `mission_node.cpp:258` accepts `cbf_obstacle_source=truth` with `truth_obstacle_fixture=true`, but `runCbf` only loads static proxies in its `perception` branch (`:869`). The observer's truth mode publishes an empty proxy array. These review runs do not claim validation of a deterministic truth static-obstacle fixture.

**P2 — A target proxy masks static-perception invalidity for UGVs.** `mission_node.cpp:947` computes `sensor_valid = static_sensor_valid || target_proxy_active`. A valid target estimate can keep the overall sensor gate open after static perception becomes invalid. Static freshness must be assessed independently; a target detection is not evidence that the static scene was observed.

**P2 — Camera detections need geometric/noise validation.** Bounding-box/depth measurements can accept large outliers with a small covariance floor. A previous diagnostic had approximately 10 m target error from an accepted capture. AirSim's native detection `relative_pose` cannot safely be used as an independent gate yet: `DetectionComponent` produces camera-local coordinates, then `WorldSimApi.cpp` applies an origin-subtracting vehicle `toLocalNed` transform. A read-only probe returned a relative Y near 218 m for a target about 6 m away. No speculative native or camera-filter patch was applied.

**P2 — Existing parity coverage and metadata can overstate evidence.** `compare_python_ros.py:724–797` clamps interpolated endpoints and can fill missing per-agent samples while reporting coverage inside the overall log interval. ROS formation and Python formation fields also use different definitions; ROS's `formation_xy_max_error` includes a 3-D slot norm, and `safety_communication_links` is always empty. This report uses independently computed errors and explicit limitations rather than interpreting those fields as parity passes. `run_trials.py`'s nominal Python commit metadata does not itself check out that revision.

## Throughput and the next implementation work

| Probe | ClockSpeed | Measured native sim s | Sampled wall s | Measured RTF |
|---|---:|---:|---:|---:|
| Python camera | 2 | 20.247 | 18.060 | 1.121× |
| ROS: target camera + relative camera + static perception + CBF | 2 | 18.729 | 19.025 | 0.984× |
| ROS: target camera + relative camera + static perception + CBF | 4 | 27.586 | 19.022 | 1.450× |

**Faster-than-real-time execution was measured:** the full ROS perception workload reached 1.45× by the native AirSim clock at `ClockSpeed=4`, with 27.6 native seconds during 19.0 sampled wall seconds. That run retained airborne starts, had no relevant native contacts, and had 1.07 m mean-agent target RMSE, but still violated the UGV disk margin by about 0.13 m. Robot control used truth state; running the relative-camera observer does not by itself validate estimated-state control. This is not evidence of equivalent accelerated dynamics or a 4× speedup.

At `ClockSpeed=1`, final ROS cases measured 0.66–0.72×. The nine-case suite took approximately 553 wall seconds including preparation and per-case ROS startup/shutdown: about 61.5 s per 25–28-native-second episode. A simple extrapolation is **about 1 h 43 min for 100 or 3 h 25 min for 200 episodes**, excluding retries, long-run degradation, and additional recording. The ClockSpeed4 probe took about 29.9 s for launch/run/shutdown plus approximately 7 s preparation. Its better capacity is promising, but scientific validity must be repaired before extrapolating that setting into an accepted batch.


The C++ CBF solve is already a small fraction of the budget (roughly 0.02–0.03 ms per agent in these runs, versus roughly 0.8–0.9 ms in Python). Further solver micro-optimization is unlikely to be the main end-to-end win. The six-agent tracking microbenchmark took about 0.392 s with one OpenBLAS thread versus 0.783 s with fifteen threads; use one thread for these small matrices. This controlled microbenchmark is separate from the full simulator, where fleet size, scene load, and startup differ.

Prioritize: (1) one validated simulation-time epoch for acquisition, estimation, control, age checks, and episode termination, with wall time reserved for watchdogs; (2) terrain-safe, settled reset poses plus active-flight/UGV-movement checks; (3) localization and safety acceptance thresholds; (4) a warm Unreal/ROS session with explicit per-episode reset of estimator state, observation caches, controllers, counters, and clock epochs; (5) measured camera/RPC scheduling and bounded timeout/recovery. Choose one `/clock` publisher if ROS simulation time is enabled, because separate drone/UGV wrappers otherwise compete. Native UAV and UGV state APIs share the clock source, but sample both timestamps to bound actor-update skew. AirSim's default reset loops vehicle APIs and does not reset every dynamic world actor. The current obstacle worker can still be blocked by a genuinely stalled RPC, and capture exceptions need better diagnostics.

Do not start a 100–200-run study by simply increasing `ClockSpeed` and counting nominal steps. First demonstrate equivalent errors, clearances, and failures at equal actual simulated duration over repeated seeds. No 100–200-trial batch was executed in this review.

## Reproduction and checks

Build/test entrypoints remain `docker/ros2/build_ws.sh`, `docker/ros2/test.sh`, and `docker/ros2/exec.sh`. The final test-result set is **182 ROS tests, zero failures/errors/skips**; the host Python/validation set is **150 passed**. Full ROS tests ran after the frame/roster changes, with affected packages rebuilt/retested after later fixes. Logs: `ros-tests-final.log`, `ros-tests-hover-handoff.log`, and `python-tests-3u3g.log` in the artifact directory. `git diff --check` passes.

From the repository root, launch the simulator in its own terminal:

```sh
HERCULES_RPC_BIND_IP=0.0.0.0 docker/ros2/launch_rural_mission_sim.sh -unattended -NoSound
```

The retained review helpers are `run_python_case.py`, `prepare_case.py`, `run_ros_case.py`, `run_isolated_suite.py`, `summarize_cases.py`, and `plot_review.py`. `run_isolated_suite.py` resets/prepares the simulator before each case and then launches Docker ROS with the mode arguments described above. Its exact launch commands and startup checks are saved in each case's `run.json`; preparation evidence is in `prepare-<case>.log`. The helpers are tied to this artifact layout and the recorded baseline. Use new output case names/directories when reproducing to preserve this evidence.

The stored final invocation was:

```sh
OPENBLAS_NUM_THREADS=1 /Users/dmrc/git/.venvs/hercules-python310/bin/python \
  ros2/validation/reproduction/artifacts/branch-review-20260926/run_isolated_suite.py \
  final6_parity_truth final6_tracking_only_truth final6_tracking_only_camera \
  final6_collision_mestres final6_collision_wang \
  final6_localization_recursive_truth final6_localization_gs_truth \
  final6_localization_recursive_camera final6_localization_gs_camera
```

The mission-complete marker is treated as **execution completion**, not a safety/localization pass or proof of clean shutdown. Full launch logs were inspected for errors separately. Camera/truth modes, enabled controllers, and startup validity must remain explicit when comparing results.
