# MA-ICP AirSim case studies

This package runs the three paper cases independently through the ROS 2 AirSim
stack. The controlled team is `Drone1`, `Drone2`, `SimpleFlight`, `Husky1`,
`Husky2`, and `Husky3`: three UAVs and three UGVs. `Target1` is an observed
tracking actor, not a calibration agent. Algorithm and score definitions follow
`ma-cp-paper/method.tex`, `problem.tex`, and `simulation.tex`.

The current validation priority and default case is **target tracking**, followed
by cooperative localization. The collision CP study is deferred: its basic
controller and runner are present, but the Python reference's optimal-decay
QP still needs to be ported before accepting collision comparisons.

See [live validation results](../../validation/maicp/VALIDATION.md) for accepted
tracking/localization runs, checks, observed limitations, and superseded runs.

## Numerical defaults

`paper_config()` keeps the paper defaults: outer replicates 3, rounds 5,
calibration/deployment missions 200/200, `K_S=100`, `K_q=100`, and tracking
ADMM steps 50. Each mission has 100 samples at `control_dt=0.1` s (10 s).
Other defaults are `alpha=0.10`, `delta_cal=0.99`, initial margins 0.2,
`n=3`, `ell=1` for each class, mission rank `n-ell=2`, scale 1, weight 1,
`gamma_0=0.05`, exponent 0.75, proximal weight 0.25, and configured
`kappa=0.6` for UAV and UGV. The adaptive rank is 182 for 200 missions after
the finite calibration correction. Kappa is configured input, not computed and
not a formal sensitivity certificate.

Margins have a 100.0 ceiling in this AirSim adaptation, including collision.
This intentionally replaces the Python collision experiment's 4.0 operational
ceiling: measured AirSim acceleration residuals can exceed 4.0. A feasible
calibration update does not imply every subsequent controller QP is feasible;
controller failures and patrol completion are recorded separately.

The five callable methods are:

* `nominal`: zero margins.
* `fixed_margin`: one independent zero-margin batch, standard split-CP rank
  `ceil((M+1)(1-alpha))`, `+inf` when the rank exceeds `M`, then fixed margins.
* `centralized`: exact finite order statistic.
* `unshifted`: distributed finite pinball/flood/recovery with `kappa=0`.
* `maicp`: finite pinball, tagged flooding, finite upper recovery, and the
  coupled update `q_c + kappa_c*||(r-r_previous)/scale|| <= r_c`, minimizing
  `sum(weight_c*r_c) + proximal_weight/2*||(r-r_previous)/scale||^2`.

Ownership errors, disconnected graphs, non-finite data, exhausted recovery,
and infeasible coupled updates fail explicitly. Raw samples stay local to each
worker. Calibration and deployment are independent; deployment seeds are
paired across methods.

## Fixture and cases

Each case starts the six controlled agents in seeded lanes `y=-9, 0, +9` for
both classes. UAVs follow a 2.5 m radius circle; UGVs drive 2 m forward and
reverse back without turning around. The moving reference ends at 7.5 s,
leaving 2.5 s to settle at home. CPHusky's 15 degree/s turn limit makes a
full ground-vehicle circle infeasible within ten seconds. The current
fallback map is FlyingCPP, `/Game/FlyingCPP/Maps/FlyingExampleMap`. The reset
fixture adds collision columns at `(x,y)=(2.5,-9),(2.5,0),(2.5,9)` and walls at y=+/-15 using the one-metre `1M_Cube_Chamfer` asset.

* **collision:** Wang braking-distance CBF, Strategy B authority splitting;
  score is maximum sampled acceleration-model residual. UGV acceleration is
  converted to speed/steering at the actuator, so that mismatch is scored.
* **tracking:** distributed rolling-window estimator with exactly 50 ADMM
  rounds per complete epoch; score is target position error at the estimate's
  timestamp. Estimates are held between the 1 Hz epochs. The log scorer also
  reports current-time error and estimate age separately, so transport lag is
  visible. Incomplete or timed-out epochs invalidate the calibration mission.
* **localization:** recursive decentralized estimator; score is maximum planar
  own-position error. The calibrated radius inflates the relevant measurement
  covariance in tracking/localization: `R_nominal + gain * margin * I`.
  Gains default to 0.20 for UAVs and 0.30 for UGVs and are launch parameters.
  Process/prior covariance stays nominal. Estimator scoring uses the fixed
  2–10 s receipt window, allowing the camera and first distributed epoch to initialize
  without extending the mission. The boundary is fixed for every method.

`reset_scene` only resets AirSim, removes owned `maicp_*` objects, places the
six agents and `Target1` deterministically, zeros velocities, and optionally
writes a manifest. It does not fit dynamics or draw calibration samples.
The three cases use the same reset distribution for every method. Tracking
uses camera observations; localization uses camera relative observations.
The other components use simulator state so each case is studied independently.

## Build and simulator

From the repository root:

```sh
./docker/ros2/build_ws.sh --packages-up-to hercules_maicp
./docker/ros2/exec.sh bash -lc 'source /workspaces/hercules/docker/ros2/env.sh; ros2 run hercules_maicp experiment --help'
```

Launch the selected map on the native UE host. On macOS:

```sh
export UE_ROOT="/Users/Shared/Epic Games/UE_5.2"
export UNREAL_EDITOR="$UE_ROOT/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
export HERCULES_UNREAL_SETTINGS="$PWD/docker/ros2/settings.rural-nominal.json"
HERCULES_RPC_BIND_IP=0.0.0.0 ./docker/ros2/launch_sim.sh \
  '/Game/FlyingCPP/Maps/FlyingExampleMap' -ResX=640 -ResY=360
```

On Linux, set `UNREAL_EDITOR` to the native UE 5.2.1
`Engine/Binaries/Linux/UnrealEditor` and use the same command. Use
`host.docker.internal` with bridged Docker Desktop networking, or `127.0.0.1`
with host networking, consistently for every client.

## Reset and dynamics fitting

Reset accepts `--seed`, `--case`, `--host`, optional `--output`, and RPC ports
41451/41452:

```sh
./docker/ros2/exec.sh bash -lc 'source /workspaces/hercules/docker/ros2/env.sh; ros2 run hercules_maicp reset_scene --case tracking --seed 1001 --host host.docker.internal --output /workspaces/hercules/ros2/validation/maicp/tracking_1001.reset.json'
```

Collision calibration requires frozen models fitted from independent training
logs. `training:=true` collects a zero-model log; the model contains six
affine world-XY coefficients per class `[bias_x, x_x, y_x, bias_y, x_y, y_y]`:

```sh
./docker/ros2/exec.sh bash -lc 'source /workspaces/hercules/docker/ros2/env.sh; ros2 launch hercules_maicp case_study.launch.py case:=collision training:=true steps:=100 duration:=10.0 margin_uav:=0.0 margin_ugv:=0.0 host:=host.docker.internal log_path:=/workspaces/hercules/ros2/validation/maicp/training_1.jsonl'
./docker/ros2/exec.sh bash -lc 'source /workspaces/hercules/docker/ros2/env.sh; ros2 run hercules_maicp fit_dynamics fit --output /workspaces/hercules/ros2/validation/maicp/dynamics.json /workspaces/hercules/ros2/validation/maicp/training_1.jsonl /workspaces/hercules/ros2/validation/maicp/training_2.jsonl; ros2 run hercules_maicp fit_dynamics verify /workspaces/hercules/ros2/validation/maicp/dynamics.json'
```

Use distinct independent training log names. A collision batch must pass the
resulting `--dynamics-file`; missing or invalid artifacts fail before sampling.
Missions write `.done` or `.failed` sidecars. Stop a manual launch after the
marker before resetting. The live harness handles this cleanup automatically.
The runner writes a `.score.json` for each mission, including failures, and
stops calibration on its first invalid mission.

## Live harness and batch CLI

`ros2/validation/maicp/live_mission.py` performs reset, launch, completion
waiting, and summary. Its output path must not already exist:

```sh
./docker/ros2/exec.sh bash -lc 'source /workspaces/hercules/docker/ros2/env.sh; python3 ros2/validation/maicp/live_mission.py --case tracking --host host.docker.internal --seed 1001 --output /workspaces/hercules/ros2/validation/maicp/live/tracking_1001.jsonl'
```

The measured mission cycle is about 21–22 seconds including reset and ROS
startup/shutdown; the controlled mission itself is capped at 10 seconds.

The full runner resets before every mission and owns each launch process:

```sh
./docker/ros2/exec.sh bash -lc 'source /workspaces/hercules/docker/ros2/env.sh; ros2 run hercules_maicp experiment --cases tracking --methods nominal,fixed_margin,centralized,unshifted,maicp --repeats 3 --rounds 5 --calibration-missions 200 --deployment-missions 200 --steps 100 --kappa 0.6 --delta-cal 0.99 --host host.docker.internal --output-dir /workspaces/hercules/maicp_runs --report /workspaces/hercules/maicp_runs/report.json'
```

Use `--cases localization` for the second study. The default batch is serial.
One case with all five methods uses 24,600 missions/resets. Across 3 cases,
5 methods, 3 repeats, and 5 rounds this becomes 73,800: 9,000 nominal deployments, 10,800
fixed-margin missions, and 18,000 each for centralized, unshifted, and MA-ICP.
Six ROS calibration workers serve the distributed order-statistic callback;
subteam-parallel mission scheduling is not enabled or measured. Scores are
finite sampled-horizon diagnostics, not continuous-time certificates; report
coverage, collision-free telemetry, and completion separately.
