# Distributed target tracking

The mission now spawns a separate `CPHusky` named `Target1`. It is not a
formation member and is not included in localization or safety-neighbor
messages. It follows a deterministic route-aligned Gerono figure-eight. The
target command is generated from startup path samples; the target does not
run the distributed CBF.

## Run modes

Target-centered tracking is the default:

```bash
cd /Users/dmrc/git/HERCULES/PythonClient/distributed_mission
/Users/dmrc/anaconda3/bin/python orchestrator.py --launch-mode headless --map flyingcpp --steps 600 --target-observation-source truth
```

Use real AirSim named-object detections and the target cameras with a managed
launch for the normal mode:

```bash
/Users/dmrc/anaconda3/bin/python orchestrator.py --launch-mode headless --map flyingcpp --steps 600 --target-observation-source camera
```

`camera` is the default and never falls back to truth. The camera mode adds a
temporary downward `target_bottom` camera to every SimpleFlight vehicle and
uses `front_center` on each Husky. Existing vehicle camera and depth settings
are copied unchanged into the temporary settings override. `truth` mode uses
range-gated, seeded Gaussian position noise and is intended for estimator and
CBF tuning. Existing-launch mode is appropriate for truth mode, or for a
world that was already launched with the tracking cameras configured.

The previous fixed-goal formation remains available:

```bash
/Users/dmrc/anaconda3/bin/python orchestrator.py --mission-objective fixed-goal --target-observation-source truth
```

The target motion and estimator parameters are command-line options. The
defaults are a 10 m longitudinal by 8 m lateral figure-eight at 1.5 m/s, a
5-second rolling window, 4 Hz epochs, `rho=1`, 20 ADMM rounds, and `1e-3`
consensus tolerance. The UGV triangle has a 6 m circumradius. UAV target
obstacle inflation is controlled by `TARGET_CBF_SIGMA_MULTIPLIER = 2.0` near
the top of `modules/target_tracking.py`; it is the requested number of XY
covariance standard deviations.

## Estimator and messages

Each controlled agent owns an independent DRWT state keyed by target ID. The
planar state is `[x, y, vx, vy]`, with a constant-velocity transition and
white-acceleration process covariance. At each tracking epoch the module
assembles its local rolling-window information system, solves its block
tridiagonal primal system with the paper's forward/backward Cholesky passes,
and performs synchronous ADMM rounds over the current distance graph.

Tracking messages contain only target-ID-keyed trajectory estimates and are
routed over all controlled-agent graph edges, so UAV--UGV links are allowed.
Same-type localization/CBF edges are kept separately. A stale local track can
handoff its information matrix and vector to a continuing neighbor after one
full window; a track expires after a full window without direct support or a
valid handoff.

## Target-centered formation

In `track-target` mode the UAV box is centered over the latest local target
estimate, with SimpleFlight at its center and the other four UAVs at the
corners. The three UGVs use an equilateral triangle centered on the target;
Husky1 is the vertex aligned with estimated target motion and Husky2/Husky3
are the ±120° vertices. At low estimated target speed the initial route
heading is used. An agent with no active estimate holds its nominal position;
it does not switch back to the fixed goal.

The same target estimate is also converted into a moving `ObstacleProxy` for
the existing CBF. Its center and velocity are predicted to the current
control timestamp. Its radius is the physical target radius plus twice the
largest XY covariance standard deviation. Static obstacle equations and
distributed CBF solving are unchanged.

## Log and plots

Every JSONL cycle keeps controlled agents in `states`, while `target_truth`
and `targets.Target1` contain the target pose, velocity, command, figure-eight
phase/pattern, and Unreal collision status. `target_tracking.agents` contains
each agent's latest measurement, capture ID, visibility/source, age metadata,
estimate, covariance, active status, residual, ADMM iteration count, and
handoffs. `tracking_communication_links` is the all-type graph and
`safety_communication_links` is the same-type graph. The compatibility
`communication_links` field aliases the tracking graph.

The top-down MP4/GIF shows target truth in red, agent-colored target estimates
with covariance ellipses, tracking links as dashed lines, safety links as
dotted lines, UAV camera footprints, obstacle estimates, truth obstacles,
the goal, and collision markers. Bounds use robot/target truth and static
geometry rather than noisy estimates. Target truth is also included in chase
camera framing and receives a red screen-space ring in the chase video.

## Debugging camera observations

The target observation worker captures asynchronously at the configured rate.
For each agent, inspect `target_tracking.agents.<agent>.measurement` and check
`valid`, `visible`, `capture_id`, `sensor`, `metadata.position_frame`, and
`metadata.point_frame`. Repeated invalid camera measurements indicate either
that `Target1*` is not visible in the configured camera or that the named
tracking camera was not present at launch. The target actor's Unreal collision
record is diagnostic only; it is not fed back into tracking.
