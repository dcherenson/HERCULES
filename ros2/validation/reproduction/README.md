# Distributed tracking live reproduction

The [2026-09-26 branch review](BRANCH_REVIEW_2026-09-26.md) records full native
Unreal/AirSim + Docker ROS tests with the current 3-UAV/3-UGV fleet, separate
Target1, Python comparisons, unresolved correctness findings, and measured
faster-than-real-time capacity. The older eight-agent result below is historical.

Generated logs, metrics, plots, and animations belong only in the ignored
`artifacts/` child directory. Run the complete deterministic and live workflow
from the repository root with:

```bash
./docker/ros2/reproduce.sh --with-live-video --duration 30
```

Unreal Engine 5.2.1 must already be running RuralAustralia with a rendered
RHI. The ordinary visible launch is preferred. `-RenderOffscreen` is supported
for automated camera validation, but `-nullrhi` is not: AirSim's
`DepthPerspective` RPC requires a render target.

## Bounded live result, 2026-09-10

The final local validation used separately reset 10-second runs at 10 Hz for
the direct-truth baseline, distributed truth-observation mode, and distributed
camera-observation mode. Each produced 100 mission records with zero stale
state, origin-validation, rejected-command, or tracking-epoch-timeout events.

The camera observer produced 183 captures: 60 visible and 123 naturally
invalid/invisible detections, with zero RPC errors. Aggregate capture rate was
15.3713 Hz across eight sequential cameras (about 1.92 Hz per camera), capture
interval p95/max were 0.09659/0.11355 s, and image RPC mean/max were
0.06531/0.11354 s. All eight local tracks stayed active. Seven typed handoffs
were sent and accepted, demonstrating that agents without a current direct
observation retained a distributed estimate. The C++ control timer remained
at 10.0006 Hz mean with a 0.10019 s maximum update gap.

The camera-mode per-agent RMS target position errors were 0.63595 m (Drone1),
0.63490 m (Drone2), 0.63423 m (SimpleFlight), 0.63587 m (Drone4), 0.63846 m
(Drone5), 0.63625 m (Husky1), 0.63954 m (Husky2), and 0.63720 m (Husky3).
The corresponding fractions of tracking epochs active were 1.0 for all eight;
direct-observation fractions were 0.000, 0.200, 0.175, 0.000, 0.225, 0.100,
0.400, and 0.200 in the same order.

`artifacts/mode_comparison.png` and each mode's
`artifacts/<mode>/full_mission_trajectories.png` were visually inspected. They
showed consistent AirSim world-NED axes and target direction, with no obvious
reflection, 90-degree rotation, X/Y swap, or unexplained translation. These
are implementation diagnostics, not evidence of safety or closed-loop
stability; CBF is deliberately absent.
