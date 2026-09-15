# Rural Australia 500-step Mestres CBF comparison

These runs used the same running Rural Australia AirSim instance, target `Target1`, Mestres CBF, `dt=0.1 s`, and the eight-agent roster (five UAVs and three Huskies). The Python-only run completed 500 JSONL rows. The ROS mission ran for the requested 50 seconds at 10 Hz, but its completion check occurs before the final write, so it recorded 499 rows; that one-row difference is a source behavior observed and preserved in this measurement.

## Runs and comparability

| Run | Target/static-obstacle source | Actuation | Recorded rows | Mission time | Wall span |
|---|---|---:|---:|---:|---:|
| Python | camera / perception | live | 500 | 49.900 s | 204.010 s |
| Python | truth / truth obstacles | live | 500 | 49.900 s | 136.204 s |
| ROS | camera / no obstacle RPC | dry run | 500 | 50.000 s | 49.902 s |
| ROS | truth / no obstacle RPC | live | 499 | 49.902 s | 49.801 s |

The matched controller/trajectory comparison is Python truth versus ROS truth. They were separate live runs with different simulator poses and calibrated state origins. The trajectory figure therefore translates each vehicle's XY trace to its own start; the RMSE values are aligned trace differences, not repeatability or safety bounds. Absolute world coordinates are not comparable across the two runs.

## Timing

### Python truth run (500 rows)

| Phase | Mean ms | p50 ms | p95 ms | Max ms |
|---|---:|---:|---:|---:|
| State acquisition | 22.941 | 22.960 | 27.134 | 69.938 |
| Target estimate | 0.430 | 0.473 | 0.527 | 0.922 |
| Perception/tracking accounting | 176.872 | 1.767 | 600.844 | 1016.339 |
| CBF phase | 8.142 | 6.962 | 11.763 | 113.076 |
| Actuation | 1.506 | 1.451 | 2.014 | 4.827 |
| Whole cycle | 271.902 | 102.086 | 691.978 | 1107.938 |

The cycle deadline was 100 ms; 290 of 500 cycles missed it. The perception field includes the orchestrator's synchronous tracking/formation work around obstacle handling, so it is not a pure sensor RPC measurement. Its low median and high p95 show intermittent long work rather than a constant 177 ms sensor cost.

End to end, the Python truth run took 136.204 s of wall time for 49.9 s of simulated mission time, versus 49.801 s for the ROS truth run: a 2.735x wall-span ratio. The Python mean cycle was 2.719x the ROS control period, which is consistent with the deadline-miss count and the long perception tail.

### Python camera run (500 rows)

The camera run had a 204.010 s wall span and 407.457 ms mean cycle (340.815 ms p50, 823.454 ms p95, 1224.833 ms max), with 476/500 deadline misses. Phase means were state 22.645 ms, estimate 0.458 ms, perception 342.476 ms, CBF 7.591 ms, and actuation 1.454 ms. Camera capture/perception is the dominant added cost. UAV cameras captured 125–126 frames each; the legacy Husky path captured 500 frames each. The generated perception report records six UAV proxy-association jump anomalies and one Husky radius-change anomaly.

The Python camera tracker still published active estimates on all 500 control rows by retaining/handoffing estimates after invalid or clipped direct detections. That cache/handoff behavior is why its camera run can continue target-coupled CBF requests even when individual camera measurements are rejected. ROS's camera run did not activate any track because every one of its 580 detections was invalid.

### ROS truth run (499 recorded rows)

ROS control periods were 100.002 ms mean, 99.990 ms p50, 100.300 ms p95, and 100.630 ms max. The console reported `mean_frequency=10.000Hz` and `max_gap=0.1002s`. ROS does not log a per-cycle phase decomposition, so there is no defensible ROS state/perception/tracking duration to subtract or invent. It does log the in-process CBF filter time:

| Agent | Mean ms | p95 ms | Max ms |
|---|---:|---:|---:|
| Drone1 | 0.0475 | 0.0607 | 0.1608 |
| Drone2 | 0.0252 | 0.0379 | 0.0549 |
| SimpleFlight | 0.0207 | 0.0321 | 0.0542 |
| Drone4 | 0.0216 | 0.0319 | 0.0771 |
| Drone5 | 0.0205 | 0.0310 | 0.0428 |
| Husky1 | 0.0221 | 0.0319 | 0.1219 |
| Husky2 | 0.0180 | 0.0251 | 0.0496 |
| Husky3 | 0.0175 | 0.0259 | 0.0437 |

The Python OSQP solve-time means for the same agents were Drone1 1.4305 ms, Drone2 0.9897 ms, SimpleFlight 0.9255 ms, Drone4 0.9011 ms, Drone5 0.8820 ms, Husky1 0.8476 ms, Husky2 0.8313 ms, and Husky3 0.8166 ms. These are not an apples-to-apples benchmark: the Python values include the Python OSQP call and Python-side request path, while ROS values are the native in-process filter timing on a different run/build path. The measured native filter is 17–45x lower per agent (7.624 ms versus 0.193 ms when the eight agent means are summed), but that ratio should not be treated as an end-to-end speedup.

### Target tracking and camera observer

The Python truth tracker was active on all 500 rows and reported 19 or 20 iterations per step. The ROS truth tracker reached epoch 200 with eight active tracks, eight direct observations, 33,288 consensus messages, and zero timed-out epochs. Neither ROS log contains a tracker processing-duration field; only counts and epochs are reported.

All 4,000 Python CBF entries and all 3,992 ROS entries reported `success=true` and `fallback=false`. Python reported 3,897 one-round projections and 59 exhausted 20-round projections (the remaining 44 entries took 2–19 rounds). ROS reported 3,595 one-round and 348 20-round projections. ROS's additional diagnostics marked 515 entries `final_feasible=false`; this is the documented post-clip/projection behavior and does not change the success flag. Python does not emit the corresponding final-feasibility flag. The mean reported minimum barrier was 7.258 in Python and 0.875 in ROS, with minima of -6.964 and -7.476 respectively; those values are not normalized across the different live state trajectories.

The ROS camera dry run completed 500 ticks but captured 580 images at 10.981 Hz; all 580 detections were invalid, with zero RPC errors. Mean image RPC duration was 91.190 ms and max 164.580 ms. It produced zero active tracks and zero direct observations, so it is a camera-RPC timing result, not a successful target-tracking result. The attempted ROS camera run with the obstacle observer captured 403 images at 7.752 Hz; all 403 were invalid, and its per-capture AirSim facade creation caused stale-state warnings and prevented a valid perception trial. The observer must reuse a connection/facade before a closed-loop camera comparison is meaningful.

For truth ROS diagnostics, the fields are still named `camera_captures`, `camera_visible_detections`, and related names even though the source was truth. In that run they read 1584 captures, 1584 visible, zero invalid, and zero RPC duration; those are source-count diagnostics, not camera image timings.

## Trajectories

Path length and XY displacement are computed from each run's logged positions. Values are metres.

| Agent | Python path | ROS path | Python displacement | ROS displacement | aligned XY RMSE |
|---|---:|---:|---:|---:|---:|
| Drone1 | 23.234 | 10.182 | 1.814 | 1.530 | 2.916 |
| Drone2 | 18.662 | 14.168 | 1.751 | 9.076 | 9.188 |
| SimpleFlight | 19.026 | 14.272 | 1.728 | 5.393 | 7.453 |
| Drone4 | 19.061 | 10.388 | 1.703 | 6.505 | 7.176 |
| Drone5 | 19.067 | 14.240 | 1.731 | 10.931 | 11.315 |
| Husky1 | 65.305 | 7.487 | 2.147 | 6.933 | 12.691 |
| Husky2 | 57.287 | 19.366 | 6.700 | 10.424 | 14.219 |
| Husky3 | 49.307 | 1.126 | 2.577 | 0.622 | 5.304 |

The traces differ materially, especially Husky1/Husky2 and Drone5. This is consistent with different starting poses/origin calibration, simulator state carried between runs, and the Python run's intermittent long perception cycles. It is not evidence that either controller is intrinsically less safe. The Python truth run recorded 107 relevant collision-observer records, all on Husky3 foliage. The ROS truth run recorded 963 relevant records (Husky3 484, Husky1 339, Target1 140); the observer schema and run state differ, so these counts are diagnostic records rather than a normalized collision-rate comparison.

The translated trajectory plot is `artifacts/comparison_500_trajectories.png`; all raw JSONL and the machine-readable summary are in `artifacts/comparison_500.json`.

## Behaviors intentionally preserved

The comparison uses the literal Python production contract implemented by the plan: OSQP receives barrier rows without box-bound rows and the result is clipped afterward; Mestres applies up to 20 order-dependent projection rounds with a 1e-3 stopping tolerance; invalid/stale sensors return an early zero vector; ordinary solver failure uses the model-specific Python fallback; ordered roster rows and same-type pair exclusions are retained; the executable altitude row uses the +1 NED Rural Australia floor without a vertical-velocity high-order term; mission Huskies remain Mestres unicycles and use the mission lookahead of 0.1 m; and the CLI's conformal uncertainty override remains 0.0 for parity. The ROS mission's pre-write completion check (499 logged rows for a 500-tick duration) and the camera-named truth diagnostics are also reported as observed source behavior instead of being silently “fixed.”
