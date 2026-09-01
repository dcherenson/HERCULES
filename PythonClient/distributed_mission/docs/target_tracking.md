# Distributed target tracking

The mission now spawns a separate `CPHusky` named `Target1`. It is not a
formation member and is not included in localization or safety-neighbor
messages. It follows a deterministic route-aligned Gerono figure-eight. The
target command is generated from startup path samples; the target does not
run the distributed CBF.

## Run modes

Target-centered tracking is the default. The controlled vehicles initially
use the common launch center with the established compact, non-overlapping
startup offsets (UAV and UGV body heights remain independent). `Target1` starts
one quarter of the way along the launch-to-goal route in RuralAustralia, offset
5 m to the chase-camera-right side, then shifted 10 m back toward the robot
launch point along the route, and at the resolved mission goal in FlyingCPP.
Literal physics-body
collocation would make the initial AirSim safety constraints infeasible and
can hang vehicle takeoff, so the target-centered formation is applied after
startup:

`Target1` is initialized at a fixed right-moving figure-eight phase with a
matching yaw, so its first visible motion is toward the right side of the
chase-camera frame. The route remains a fixed sampled figure-eight; no online
waypoints are created.

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

The previous fixed-goal formation remains available. In this mode target
tracking is completely disabled: `Target1` is not spawned, no target camera or
observation worker is started, the target is not added to CBF obstacles or
collision polling, and the target fields in JSONL are `null`/empty. The
controlled agents use the original Husky1-led goal formation controller:

```bash
/Users/dmrc/anaconda3/bin/python orchestrator.py --mission-objective fixed-goal --launch-mode headless
```

`--target-observation-source` and the target-motion/tracking tuning options
are ignored in `fixed-goal` mode. A managed launch starts a fresh world, so a
target from an earlier run is not carried into the mission. When attaching to
an already-running world, start a fresh Unreal world if that world was
previously used for target tracking; the fixed-goal controller will never
command or track a pre-existing actor named `Target1`.

The target motion and estimator parameters are command-line options. The
current simulator baseline is a 10 m longitudinal by 8 m lateral figure-eight
at 0.10 m/s, a 5-second rolling window, 4 Hz epochs, `rho=1`, 20 ADMM rounds,
and `1e-3` consensus tolerance. The slower target is intentional: the physical
CPHusky can otherwise reach a moving formation slot faster than its turn and
brake response can maintain the target clearance. The UGV triangle has a 5 m
circumradius by default; this keeps the forward vertex clear of the
pre-existing FlyingCPP map mesh while remaining close to the target. For
other layouts it remains tunable with `--target-ugv-circumradius`. For
RuralAustralia experiments, use
`--initial-heading-offset-deg 90`; the unoffset route is not the tested route
for this map. The existing UGV CBF control-point lookahead defaults to 0.1 m
and remains a tuning parameter. UGV target obstacle inflation is controlled
by `TARGET_CBF_SIGMA_MULTIPLIER = 4.0` near the top of
`modules/target_tracking.py`; it is the requested number of XY covariance
standard deviations. The default nominal tuning is 1.0 m/s speed, 0.5
position gain, and 1.0 rad/s UGV yaw-rate limit.

For a controlled target/formation baseline, use truth target observations,
`--tracking-measurement-std 0.25`, and a slower target while tuning the CBF:

```bash
/Users/dmrc/anaconda3/bin/python orchestrator.py --launch-mode headless \
  --map rural_australia --target-observation-source truth \
  --use-truth-obstacles --no-spawn-obstacles \
  --initial-heading-offset-deg 90 --target-ugv-circumradius 5 \
  --target-speed 0.10 --tracking-measurement-std 0.25 \
  --tracking-process-noise 0.20 --nominal-speed 1.0 \
  --leader-nominal-speed 1.0 --nominal-position-gain 0.5 \
  --ugv-heading-gain 1.0 --ugv-max-yaw-rate 1.0 \
  --ugv-lookahead-distance 0.10 --no-animation
```

The tested command uses the existing OSQP CBF solver, declared in
`PythonClient/requirements-herculesvenv.txt`. Install that dependency in the
same Python environment used to launch AirSim if the runtime reports the
SciPy SLSQP fallback. The conservative UGV settings and 2.5 m/s target-slot
heading gate are tuning values for the CPHusky dynamics; they do not change
the CBF equations. The simulator adapter also closes a bounded speed loop
because AirSim's car API accepts throttle rather than velocity.

When the FlyingCPP course is enabled, startup checks the target's preferred
figure-eight sample and all target-centered UAV/UGV slots against the known
spawned-box geometry. A blocked target sample advances to the nearest clear
sample on the same fixed route; a blocked vehicle slot is moved by the smallest
deterministic angular/radial adjustment that gives it clearance. These are
one-time initialization adjustments, not online waypoints or recovery resets.

`--use-truth-obstacles` only controls the obstacle set given to the CBF; it
does not remove Unreal's existing foliage or map meshes from physics. A
collision with `Target1` or another controlled vehicle is a target/formation
failure. A foliage or map-mesh collision is a separate perception/map issue.

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
valid handoff. A silent receiver may retain a connected estimate from current
consensus support; this updates only its support timestamp and communicated
covariance, not the local information matrix, so the rolling-window estimate
does not count the same information repeatedly.

## Target-centered formation

In `track-target` mode the UAV box is centered over the latest local target
estimate, with SimpleFlight at its center and the other four UAVs at the
corners. The three UGVs use an equilateral triangle centered on the target;
Husky1 is the vertex aligned with estimated target motion and Husky2/Husky3
are the ±120° vertices. At low estimated target speed the initial route
heading is used. An agent with no active estimate holds its nominal position;
it does not switch back to the fixed goal. This compact startup pose is only a
target-tracking initialization choice; fixed-goal retains its pre-target
RuralAustralia road placement.

The same target estimate is converted into a moving `ObstacleProxy` for the
existing CBF only for UGVs. UAVs do not spend CBF authority avoiding the
ground target. The proxy center and velocity are predicted to the current
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
`metadata.point_frame`. The estimator uses the fixed control-loop mission
clock. Camera captures retain their raw wall-clock time as
`metadata.capture_wall_timestamp`; the logged measurement timestamp is the
piecewise-linear mission-time conversion, so slow headless runs do not make a
5-second DRWT window expire merely because RPCs took longer in wall time.

The target camera back-projection uses no additional body-radius shift in the
AirSim mission because this fork's DepthPerspective ROI was empirically close
to the CPHusky actor center. The physical UGV radius is still used by the
target CBF proxy. Valid camera covariance has a floor set by the existing
`--tracking-measurement-std` parameter, preventing sub-pixel ROI spread from
being treated as centimeter-level certainty. Repeated invalid camera
measurements indicate either that `Target1*` is not visible in the configured
camera or that the named tracking camera was not present at launch. The
target actor's Unreal collision record is diagnostic only; it is not fed back
into tracking.

The latest short and full camera-mode checks used the launch-time UAV
front/side camera fan and completed without relevant obstacle or agent
contacts in both FlyingCPP and RuralAustralia. Normal perception still has a
large synchronous sensor-RPC cost and target-centered formation quality is
not yet uniform across all UAVs, so use `--use-truth-obstacles` to validate
target tracking and the CBF independently before changing detector
parameters.

The fixed target route advances monotonically through a short forward section
of the startup-generated figure-eight. This keeps a physical target that
overshoots one sample from getting stuck steering at an old point; it does
not generate online waypoints or alter the distributed controller.

The target route also uses a 0.75 minimum forward-alignment factor while
turning, and its AirSim throttle conversion has a 0.02 minimum nonzero
throttle. These values prevent a CPHusky target with a zero initial yaw from
stalling at a nearby fixed route sample; they do not change the target path
samples or the distributed target estimator.
