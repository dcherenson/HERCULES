# Phase 3 validation report

Phase 3 is complete through the reproducible offline/replay gate. The clean
image `hercules-ros2:humble-phase3-final` was built with the pinned perception
stack (NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.7.1, threadpoolctl 3.6.0)
and native OSQP. The selected ROS overlay built successfully and its complete
package gate passed **129 tests, 0 errors, 0 failures, 0 skipped**. The
distributed-mission Python suite passed **135 tests**; the CBF metrics tests
passed **2 tests**.

The deterministic replay under `artifacts/deterministic_replay/` exercises the
three executable modes and generates `compare/metrics.json` and
`compare/cbf_modes.png`. It reports 30% Mestres interventions (maximum 0.2 m/s²),
20% Wang interventions (maximum 0.1 m/s²), no fallbacks or deadline misses, and
the no-CBF disabled baseline. Old logs without CBF fields remain accepted.

AirSim was subsequently started natively from
`docker/ros2/launch_rural_mission_sim.sh` with
`-noraytracing -NoLumenReflections -unattended`. Unreal initialized the Rural
Australia map and listened on `127.0.0.1:41451` and `127.0.0.1:41452`; a
container client confirmed all nine configured vehicles. A 5-second no-CBF
truth-observation smoke run completed 50 steps at 10 Hz. An enabled Mestres
truth-observation run also completed 50 steps at 10 Hz and wrote
`artifacts/live_mestres.jsonl` (50 valid JSONL rows, eight CBF entries per
row). The collision observer reported all nine vehicles available with
`has_collided=false`.

The camera/Wang smoke run reached mission completion, but its obstacle observer
exited during startup because the ROS node attempted to assign rclpy's
read-only `publishers` and `subscriptions` properties. Those handles have now
been renamed in the source. A subsequent 500-tick camera attempt exposed the
next issue: creating a new AirSim facade for every capture caused stale-state
warnings; all 403 detections were invalid, so it is not a valid closed-loop
perception trial. The long compile that followed also caused the first Unreal
process to exit; the simulator has since been restarted and is currently
reachable.

The completed Rural Australia timing/trajectory experiment is documented in
`COMPARISON_500_REPORT.md`. It includes the Python camera run (500 rows), the
Python truth run (500 rows), the ROS camera dry run (500 ticks, all 580 target
detections invalid), and the ROS truth live run (499 logged rows because the
mission checks completion before writing the final row). The matched truth
runs, native/Python CBF timings, tracking counters, translated trajectories,
and numerical RMSE values are in `artifacts/comparison_500.json` and
`artifacts/comparison_500_trajectories.png`. These runs are diagnostic timing
and behavior evidence; different live simulator starting states mean they do
not establish a normalized trajectory or closed-loop safety comparison.

`docker/ros2/reproduce_cbf.sh` remains the reproduction entry point; it records
the resolved configuration, source profile, timing, freshness and collision
artifacts when a full AirSim sequence is run.
