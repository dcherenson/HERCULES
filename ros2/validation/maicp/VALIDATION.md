# AirSim validation — 26 September 2026

Validated on native UE 5.2 FlyingCPP
(`/Game/FlyingCPP/Maps/FlyingExampleMap`) with the ROS 2 Humble container
`ros2-dev-1`. The controlled fleet is three UAVs and three UGVs; Target1 is
an additional observed actor.

## Target tracking: accepted results

The [accepted report](artifacts/tracking-validated/report.json) contains 12
live missions: three nominal deployments and, for MA-ICP, two calibration
missions plus one independent deployment in each of three rounds. Deployment
seeds are paired across methods. These small batches test integration; paper
defaults remain 200 calibration missions, 200 deployment missions, five rounds,
and three repeats, with kappa=0.6 and delta_cal=0.99.

| Round | UAV margin | UGV margin | UAV deployment score | UGV deployment score |
| --- | ---: | ---: | ---: | ---: |
| 1 | 6.348 | 6.346 | 1.554 | 1.546 |
| 2 | 3.782 | 3.780 | 1.722 | 1.721 |
| 3 | 2.615 | 2.612 | 1.477 | 1.475 |

All MA-ICP deployment scores were covered by their margins. Nominal raw
tracking errors were lower in this small run (UAV/UGV class scores approximately
0.909/0.914, 0.918/0.925, and 0.922/0.914). These results do **not** establish
MA-ICP superiority. The full five-method statistical comparison was not run.
All five numerical protocols were tested, including one-time fixed calibration,
centralized order statistics, and the zero-kappa unshifted ablation.

The [quality checks](artifacts/tracking-validated/quality_checks.json) verified:

- All 12 missions have 100 samples and finish within the 10-second horizon.
- All 5,760 scored tracker records are active, complete 50 ADMM iterations,
  and have no epoch timeout. Camera RPC errors are zero.
- All six robots returned in all six deployments, with maximum horizontal
  return error 0.294 m. Completion requires actual travel, so parked robots
  cannot pass.
- All three margin updates have six independently solved, agreeing ROS replicas.

Measurement inflation was checked directly in the running node:
`maicp_enabled=true`, margin=0.2, gain=0.2. The UAV initial covariance was
0.080497, matching the inflated measurement factor, versus nominal 0.053571.
At deployment margin 6.34759 it became 0.292620. The process/prior factors
remain nominal.

Scoring uses received outputs during the fixed 2–10 second window. Error is
measured against truth at the estimate's timestamp; current-time error and
estimate age are separate diagnostics. The maximum estimate age was 1.1 s.
The two-second startup boundary allows camera/first-epoch initialization;
incomplete or timed-out estimates after that boundary fail calibration.

Earlier tracking folders (`estimator-smoke`, `tracking-rounds-smoke`, and the
tracking portion of `final-ros-smoke`) are superseded: a duplicate launch key
had disabled covariance inflation. `tracking-accepted` caught initialization
at 1.2 s and motivated the fixed two-second warm-up. These artifacts are kept
for debugging, not accepted as method comparisons. The duplicate launch key
was fixed and covered by a regression test.

## Cooperative localization

The [accepted localization report](artifacts/localization-validated/report.json)
contains two calibration missions and one deployment with the same two-second
scoring window. All 1,440 scored estimates were initialized, valid, and fresh;
the margin update has six agreeing ROS replicas. Deployment class scores were
0.433 m for UAVs and 0.556 m for UGVs, below margins 8.222 and 9.108.

Five robots returned within 0.8 m. Husky1 missed home by 1.234 m, so the report
correctly records that the whole-team patrol did not complete, despite score
coverage. See the [localization quality checks](artifacts/localization-validated/quality_checks.json).

The wrapper's GPS origin calculation was corrected to use the fixed vehicle
spawn origin rather than its post-reset world pose. Relative camera updates
are intermittent and rejected-message/communication-age diagnostics remain
visible in the raw logs; own-state freshness does not imply a recent update
from every neighbor.

## Runtime, checks, and remaining scope

UAVs follow a 2.5 m radius circle; UGVs drive 2 m forward and reverse back.
References finish at 7.5 seconds to leave time to settle. Full UGV circles
are unsuitable for this horizon because CPHusky has a 15 degree/s turn limit.
A complete measured cycle takes roughly 21–22 seconds including reset and
ROS startup/shutdown; the controlled horizon itself is 10 seconds.

The main agent reviewed the implementation and corrections, ran 30 Python
MA-ICP tests, and checked the C++ tracking/localization, mission/reference,
actuation, and tracking adapter/protocol tests. All passed. Real ROS pinball
returned the expected order statistic 4.0 on a tied six-owner fixture, and
all six independent SOCP solutions agreed within 2e-6.

Multiple simultaneous subteams were not enabled because a speed benefit was
not proven. Collision CP was deferred at the user's request: its preliminary
controller/runner remain, but the Python reference's optimal-decay QP still
needs porting. The approved AirSim numerical margin ceiling is 100.0.

One case with all five methods and unchanged paper counts requires 24,600
missions; all three require 73,800. See the [run instructions](../../src/hercules_maicp/README.md).
