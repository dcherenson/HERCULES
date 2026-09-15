# C++/ROS 2 distributed CBF implementation plan

Planning baseline: `codex/ros2`, HEAD `9003d21e30c9628152581e84d21ee986573e092e` (`integrate distributed ROS 2 target tracking`), inspected 2026-09-12. The plan is being executed incrementally on this branch. Paths below are relative to the repository root.

The implementation target is **behavioral parity with the executable Python mission at this HEAD**. There are material differences between the requested mathematical description, Python execution, and the current ROS actuator boundary. These are resolved explicitly below rather than silently corrected. Conformal prediction, cooperative localization, tracking mathematics, formation redesign, and Vulkan/rendering repair are excluded.

## Verified baseline and source map

| Concern | Executable source and findings |
|---|---|
| Core CBF | `PythonClient/distributed_mission/modules/cbf.py`: request/configuration types, constraint builders, OSQP call, post-solve clipping, Mestres projection, failure paths. |
| Current CBF tests | `PythonClient/distributed_mission/tests/test_cbf.py`: 10 tests; useful starting coverage, but no row-level parity or moving-ego invalid-sensor check. |
| Mission inputs | `PythonClient/distributed_mission/agent.py`: `control_step()` forwards the current margin, local state, neighbors, obstacles, and validity to the filter. `orchestrator.py` builds adjacency every control step, overrides UGV method to Mestres, and builds nominal commands before safety filtering. |
| Perception | `modules/obstacle_detection.py` plus `orchestrator.py::_capture_obstacles`, detector configuration, successful-capture cache, age inflation, and body-proxy rejection. |
| Target coupling | `orchestrator.py::target_obstacle_proxy_for_agent` and `cbf_sensor_valid_for_target`; `modules/target_tracking.py::TargetTrack.predicted`; `TARGET_CBF_SIGMA_MULTIPLIER = 2.0`. |
| Native numerical patterns | `ros2/src/hercules_tracking`: Eigen core, deterministic tests, Python/C++ dump-and-compare tests. Existing `TargetTrack::predicted` already preserves the Python covariance prediction peculiarity. |
| Tracking transport | `ros2/src/hercules_tracking_ros`: per-agent trackers, epoch transport, neighbor graph, and a Python target-camera observer wrapping existing Python code. No CBF obstacle detector exists here. |
| Nominal mission | `hercules_mission_core/src/formation_controller.cpp`, `mission_config.cpp`, and associated headers/configuration: target-centered commands, five UAVs and three Huskies, control `dt=0.1`, tracking at 4 Hz, communication range 10 m. |
| Live boundary | `hercules_mission_ros/src/mission_node.cpp`: UAV nominal acceleration immediately enters `uavVelocityCommand`; UGV nominal `[speed,yaw_rate]` immediately enters `ugvCarCommand`. Target1 has its own figure-eight controller and publisher. State validity/calibration is already a mission-wide gate. |
| State and estimates | `hercules_interfaces/msg/GroundTruthState.msg`: canonical `airsim_world_ned` position, velocity, quaternion, yaw/yaw rate, validity; no acceleration. `TargetEstimate.msg`: XY position/velocity and row-major 4x4 covariance, active flag and timestamp. |
| Validation/media | `ros2/validation/rural_nominal/render_validation.py`, `compare_tracking_modes.py`, `docker/ros2/reproduce.sh`, and Python `modules/mission_plots.py`. Existing animation already draws static proxies, but explicitly skips proxies with source `target_tracking`. |

Verified during planning:

- Existing Docker image `hercules-ros2:humble` (local image ID `1f050821da94`) contains Ubuntu 22.04.5, CMake 3.22.1, Eigen 3.4.0, Python OSQP 1.0.4, NumPy 1.21.5, SciPy 1.8.0. It does not contain scikit-learn. The checked-out `herculesvenv` has NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.7.1, and OSQP 1.0.4; therefore perception currently depends on which interpreter/import path is used.
- All **134 Python distributed-mission tests passed** in the Docker image. They also passed with `herculesvenv`, with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to avoid an unrelated host ROS pytest plugin importing missing `lark`.
- `docker/ros2/test.sh` reran the existing built overlay: **115 tests, zero errors/failures/skips**, six selected packages. This was not a fresh workspace rebuild; Phase 1 must additionally prove a clean build.
- Pinned upstream OSQP C sources were built, installed to a temporary prefix, and their demo solved successfully in the existing Docker image. No implementation package or repository source was changed for this probe.
- No live simulator mission was run during this planning task. Historical validation reports are not evidence of present CBF behavior.

## Behavior contract and discrepancies Luna must preserve

### Solver, bounds, correction, and failures

The stated objective is `min 0.5*||u-u_nom||²`, subject to `A*u >= b` and `lower <= u <= upper`. **The OSQP branch does not actually put the box bounds in OSQP.** It supplies `P=I`, `q=-u_nom`, `A=barrier_rows`, `l=b`, `u=+infinity`, then clips the returned control to the box. The optional SciPy SLSQP branch does include bounds inside its optimization. These algorithms need not produce the same feasible result.

Decision for this port: reproduce the production Python OSQP path literally. Do not append identity/bound rows or replace the solve with a mathematically corrected bounded QP. Keep the objective and intended bounds documented, but name the implemented behavior `python_osqp_postclip` in run metadata. A corrected box-constrained formulation is a separate future behavior change, outside these phases. In particular, success does not prove final box-and-barrier feasibility.

Planning probe: `_solve_qp(nominal=[0], lower=[-1], upper=[1], rows=[[1]], rhs=[2])` returned `[1]`, `success=True`, `status='solved'`, although the final control violates the barrier. Make this a mandatory regression case alongside genuine infeasibility.

For Mestres, only after a successful constrained QP, iterate rows in original order:

```text
previous = u
for each (a, b):
    violation = b - a.dot(u)
    if violation > 0:
        u += violation / (a.dot(a) + 1e-9) * a
        u = clip(u, lower, upper)
stop if norm(u - previous) <= distributed_tolerance
otherwise stop at distributed_rounds
```

Defaults: 20 rounds and tolerance `1e-3`. Even an already feasible solution reports one round when rows exist and the round budget is positive. Zero rounds performs none. Projection is local; there is no additional ROS consensus algorithm to implement. Wang does not run these rounds. Python does not change success to failure when projection stalls with residual violation. Add a separate final residual/feasibility diagnostic without changing that status.

OSQP is set to `eps_abs=eps_rel=1e-5`, `max_iter=4000`, polishing off. A fresh solver is constructed for every call. Treat both solved and solved-inaccurate statuses as success, as Python's string test does. A no-row problem returns clipped nominal, successful `solved_no_constraints`, without loading OSQP. UAVs ordinarily still have the altitude row.

Failure behavior has two distinct paths:

| Trigger | Actual returned control | Required port behavior |
|---|---|---|
| Invalid/stale sensor | Zero vector, including a moving UAV | Preserve early-return behavior and `invalid_or_stale_sensor`, `success=false`, `fallback=true`. |
| Too-short or nonfinite nominal | Zero vector | Preserve `invalid_nominal_control`; extra nominal components are truncated in the Python interface. |
| Ordinary unsuccessful QP, unicycle | `[0,0]` | Preserve. |
| Ordinary unsuccessful QP, double integrator | `clip(-velocity, -uav_acceleration_limit, +uav_acceleration_limit)` | Preserve, including use of the UAV limit for a directly instantiated non-drone double integrator and ignoring custom request bounds on this fallback. |

The requested blanket “double-integrator fail-safe = clipped negative velocity” is true only for ordinary QP failure. A moving-ego probe confirmed invalid sensor returned `[0,0,0]` while genuine infeasibility returned `[-2,3,0]` for velocity `[2,-3,0]`. Do not change the early-return semantics in a parity port.

Other diagnostic details: active constraints are counted with strict `abs(a*u-b)<1e-3`; the count on solver failure is computed against the solver's temporary control (zero on OSQP failure), before replacing it with fallback. `minimum_barrier` includes the altitude barrier and therefore mixes metres with squared metres. `robust_terms` are the pre-Wang-split values. Python's direct `filter()` early return does not update `last_result`, but its mission-facing wrapper does; the native value-returning API should always publish the current result rather than reproduce an unused mutable-cache quirk. Invalid state/bounds and solver setup exceptions are not comprehensively handled in Python; do not invent parity expectations for malformed inputs. Reject malformed ROS/configuration inputs at the adapter boundary and identify those statuses separately.

### Exact constraint construction

Use ordered neighbor rows, then ordered obstacle rows, then the UAV altitude row. Preserve supplied order; in mission integration generate neighbors in the original eight-agent roster order, not lexicographic map order, because projection is order-dependent. `_as3` zero-pads/truncates Python vectors; the native production state uses fixed 3-vectors with optional acceleration/velocity defaults of zero.

For a double-integrator pair define:

```text
dp = p_i - p_j; dv = v_i - v_j
d = ego_radius + other_radius + obstacle_margin
h = dp.dot(dp) - d*d
h_dot = 2*dp.dot(dv)
psi = h_dot + k1*h
base = 2*dv.dot(dv) + k1*h_dot + k2*psi
robust = norm([2*dv + 2*k1*dp, 2*dp]) * max(0, uncertainty_radius)
A_row = 2*dp
b = -base + 2*dp.dot(a_j) + robust
```

Use full XYZ relative state. Same-type neighbor radius is the configured UAV/UGV radius; their optional acceleration enters literally. Obstacles supply center, radius and velocity; obstacle acceleration is zero. Only Wang same-type neighbor rows multiply the **whole computed b by 0.5**, after acceleration and robustness. Obstacles are never split. No UAV-vs-UGV pair rows. The double-integrator builder does not use an obstacle's `is_planar` flag to drop Z.

For a drone append `A=[0,0,-1]`, `h=floor_z-p_z`, `b=-alpha*h+uncertainty`, and robust term `uncertainty`. This is the executable altitude expression, without a vertical-velocity high-order term. Preserve it despite the double-integrator model. The RuralAustralia floor is **+1 NED**, from `map_ground_z_offset=2` minus 1; it is not the Target1 body-center Z of -1. Pair defaults `k1=k2=2`; altitude/unicycle `alpha=2`. Default acceleration bounds are ±6 for drones, ±3 for a direct UGV double-integrator request.

For the Mestres unicycle:

```text
q = p_xy + lookahead*[cos(yaw), sin(yaw)]
G = [[cos(yaw), -lookahead*sin(yaw)],
     [sin(yaw),  lookahead*cos(yaw)]]
delta = q - center_xy
d = ego_radius + other_radius + obstacle_margin
h = delta.dot(delta) - d*d
A_row = 2*delta.transpose()*G
robust = norm(2*delta)*max(0, uncertainty_radius)
b = -alpha*h + 2*delta.dot(other_velocity_xy) + robust
```

Neighbor centers are their body positions, not their lookahead points. Both neighbors and obstacles contribute XY velocity. Default bounds are `[0,-1.5] <= [speed,yaw_rate] <= [3,1.5]`. Preserve the configuration default lookahead 1.0 for standalone core tests; **mission UGV lookahead is 0.1**. The orchestrator's nominal yaw-rate limit is 1.0, but its CBF yaw-rate limit remains the separate default 1.5; do not conflate them.

### Heterogeneous mission, uncertainty, and target proxy

| Selected mission method | Five UAVs | Three physical Huskies | Target1 |
|---|---|---|---|
| Mestres | Double integrator + local projection | Mestres unicycle + local projection | Existing prescribed figure eight, no CBF request |
| Wang | Wang double integrator, neighbor b split | **Mestres unicycle + local projection** | Same |

`orchestrator.py` explicitly uses `replace(cbf_config, method='mestres', lookahead_distance=args.ugv_lookahead_distance)`. Its later “Wang UGV acceleration” branch is unreachable for the mission's `ugv` roster. Do not wire Huskies into it. The pure core may still test direct `method=wang, vehicle_type=ugv` as the Python class permits, without exposing a physical-Husky experimental mode.

Keep a scalar optional request `uncertainty_radius`; explicit request value overrides configuration, then clamps nonnegative. The Python mission currently forwards its placeholder conformal module's **0.0** on every step, overriding a nonzero `--uncertainty-radius` configuration. Reproduction therefore uses zero. Native explicit scalar injection is allowed and must be logged/tested, but nonzero native runs are sensitivity experiments, not parity with that unmodified CLI path. Do not implement conformal estimation.

Only a UGV with an active own local target estimate receives `target_Target1`:

- Center = `[estimate.x, estimate.y, target_ground_z]` and velocity = `[estimate.vx, estimate.vy,0]`.
- Symmetrize the 2x2 position covariance, then `sigma=sqrt(max(0,lambda_max))`; radius = target physical radius + `2.0*sigma`.
- Python passes `cbf_config.ugv_radius` (1.25 default) as Target1's physical radius. Keep that reproduction value explicit. Source=`target_tracking`, planar=true, timestamp from the predicted estimate.
- Use each agent's own estimate; never use another agent's estimate, a global fused substitute, or Target1 truth XY/velocity as a distributed-mode fallback.
- Between epochs Python predicts XY by constant velocity, and adds **only** `Q(dt)[:2,:2]` to reported position covariance. It does not compute full `F*P*Fᵀ+Q` here and leaves `state_covariance` unchanged. Apply this same presentation-time prediction once from the stored epoch estimate, using the existing tracking model helper; never accumulate it repeatedly in the cache.
- Current `mission_node.cpp::controlEstimate()` drops covariance. Add a CBF-side estimate context carrying covariance; do not change nominal/tracker mathematics. `TargetEstimate.msg` already provides the epoch 4x4 covariance; no tracking-message extension is necessary for current finalized epoch publications.
- Preserve the current ROS estimate freshness gate (active, age and receipt age no more than the 5 s tracking window, no future epoch). Document that this transport gate is stricter than Python's helper, which simply tests `active`. Use the same selected/predicted estimate for nominal control and proxy creation.
- Python takes target Z from the current target body state. Current ROS nominal geometry uses fixed configured ground Z. For CBF reproduction use the measured Target1 body Z when available, falling back only to the explicit ground-Z configuration in synthetic fixtures. This is the permitted ground-height input, not a truth XY/velocity path; label its source in logs. Keep existing nominal geometry unchanged and report any terrain-height difference.

Static and target freshness are independent: `cbf_sensor_valid = static_sensor_valid OR (UGV AND active_target_proxy)`. An active target proxy retains the cached static obstacles even when static perception is stale. See the exact cache rules in Phase 3.

## Phase 1 — pure C++ CBF core

**Deliverable:** a simulator-independent `ros2/src/hercules_cbf` C++17/Eigen library with literal Python OSQP-path behavior and directly compared fixtures. No ROS headers/runtime, AirSim, RPC, mission, or tracking dependency.

### Files and dependencies

Add:

```text
ros2/src/hercules_cbf/CMakeLists.txt
ros2/src/hercules_cbf/package.xml
ros2/src/hercules_cbf/cmake/hercules_cbfConfig.cmake.in
ros2/src/hercules_cbf/include/hercules_cbf/types.hpp
ros2/src/hercules_cbf/include/hercules_cbf/constraints.hpp
ros2/src/hercules_cbf/include/hercules_cbf/filter.hpp
ros2/src/hercules_cbf/src/constraints.cpp
ros2/src/hercules_cbf/src/filter.cpp
ros2/src/hercules_cbf/src/osqp_solver.hpp
ros2/src/hercules_cbf/src/osqp_solver.cpp
ros2/src/hercules_cbf/test/test_constraints.cpp
ros2/src/hercules_cbf/test/test_filter.cpp
ros2/src/hercules_cbf/test/cbf_parity_dump.cpp
ros2/src/hercules_cbf/test/test_python_cpp_cbf_parity.py
ros2/src/hercules_cbf/test/fixtures/cbf_cases.json
ros2/src/hercules_cbf/README.md
docker/ros2/install_osqp.sh
```

Modify `docker/ros2/Dockerfile`, `docker/ros2/test.sh`, `docker/ros2/README.md`, and `docker/ros2/reproduce.sh` only for dependency installation and adding new tests to existing gates. Do not modify the Python CBF oracle to make tests pass.

Use plain CMake with an exported `hercules_cbf::hercules_cbf` target and a colcon-discoverable `package.xml` with build type `cmake`; this supports a standalone build without sourcing ROS. Production dependencies: Eigen3 and native OSQP. Tests: GTest, Python3/pytest/NumPy/SciPy/OSQP. Use an explicit Docker `libgtest-dev` dependency if it is not already transitively installed. Avoid a JSON runtime dependency: the Python harness reads JSON inputs and supplies a small documented numeric/text protocol to the C++ dump executable; C++ emits JSON with sufficient precision. Mirror the established tracking parity harness pattern.

**Chosen solver:** the native OSQP C API directly, linked privately through a small RAII wrapper. Do not add OsqpEigen, Python embedding, a solver service, or a second production optimizer.

Reproducible source pins verified in the planning probe:

| Component | Release | Immutable commit |
|---|---|---|
| OSQP | v1.0.0 | `236713ce9a56c182ac3230d52108f952afce1523` |
| QDLDL | v0.1.8 | `138fdac58b9cd1c4137ff1b99152c8108a6cff5b` |
| Python oracle | OSQP wheel | existing `osqp==1.0.4` |

`install_osqp.sh` fetches and verifies those commits, sets `FETCHCONTENT_SOURCE_DIR_QDLDL` to the pinned source, builds Release with built-in algebra, double precision, 32-bit indices, derivatives/codegen off, and installs into `/usr/local`. Build static library and export CMake metadata. Do not let workspace builds fetch mutable Git branches. The probe used `OSQP_USE_FLOAT=OFF`, `OSQP_USE_LONG=OFF`, `OSQP_ENABLE_DERIVATIVES=OFF`, `OSQP_CODEGEN=OFF`; upstream demo was on for validation. Test a second offline configure/build using the populated sources and a CMake downstream consumer of the installed target. Record all effective OSQP settings and package versions in the fixture manifest. Python binding and C solver version numbers are different release streams, so do not assume wheel 1.0.4 means native tag v1.0.4.

Official support for the native CMake route is documented in [OSQP C integration](https://osqp.org/docs/get_started/C.html); the [OSQP release](https://github.com/osqp/osqp/releases/tag/v1.0.0) identifies the release and QDLDL update. The local successful build, rather than assumed availability of an Ubuntu/ROS apt package, is the installation evidence.

### API and bounded implementation steps

Use enum `VehicleType` and enum `Method`; keep method/model dispatch identical to Python. Define:

- `AgentState`: ID, `Vector3d` position/velocity, yaw, optional `Vector3d` acceleration, vehicle type, timestamp, yaw rate.
- `ObstacleProxy`: ID, `Vector3d` center, clamped-nonnegative radius, source, timestamp, point count, planar flag, optional velocity.
- `CBFConfig`: all Python configuration fields/defaults, including currently actuator-only limits. No silent parameter coupling.
- `CBFRequest`: ego, dimension-tagged nominal control (`Vector2d` unicycle or `Vector3d` acceleration), ordered neighbors/obstacles, optional scalar uncertainty override, optional dimension-matched bounds, sensor-valid flag.
- `ConstraintSet`: `MatrixXd A`, `VectorXd b`, barriers and robust terms, ordered row provenance (neighbor/obstacle/altitude and source ID), bounds. Expose deterministic construction for testing and diagnosis.
- `CBFResult`: dimension-tagged safe control, success/status/message/fallback, minimum barrier, counts, projected rounds, robust terms, total filter time, solver identity/raw status/iterations/residuals. Add final maximum row violation and bound violation with distinct names; do not substitute them for Python status or activity counts.

Implement in this order: types/default validation → ordered builders and bounds → OSQP wrapper/no-row fast path → exact clipping/projection/failure semantics → diagnostics → parity harness. Catch native API allocation/setup errors and return an explicit native adapter error/fallback rather than undefined behavior; these are additional diagnosed errors, not fabricated Python status parity. Make the ordinary solver-failure path independently injectable in unit tests.

### Test matrix and numerical contract

Every fixture compares dimensions, ordered row provenance, **every A coefficient and b**, lower/upper bounds, h values and robust terms before comparing output. Use `atol=1e-10, rtol=1e-10` for these deterministic algebra values on bounded fixtures; no solver tolerance may excuse a row mismatch.

Required cases:

| Cases | Specific discrimination |
|---|---|
| Safe nominal pass-through | Far UAV neighbor plus floor; no-obstacle Mestres UGV true no-row fast path; nominal outside custom bounds. |
| UAV neighbor | Nonzero XYZ relative position/velocity; separately nonzero optional neighbor acceleration; both methods. |
| UAV static/moving obstacle | Same geometry with changed obstacle velocity; obstacle RHS never split in Wang. |
| Altitude floor | Below/at/above configured floor; positive uncertainty; row count and sign; confirm no vertical-velocity term. |
| UGV neighbor/obstacle | Nonzero yaw; core lookahead 1.0 and mission 0.1; body center versus lookahead distinction. |
| Moving target obstacle | Explicit proxy XY velocity, covariance-inflated radius supplied from Phase 2 helper fixtures. |
| Robustness | Config fallback, request override, negative clamp, full mixed position/velocity gradient, pre-split diagnostic. |
| Wang responsibility | Same pair A in both methods, Wang b exactly half; both acceleration and robust terms inside split. |
| Custom bounds | 2D and 3D, asymmetric, nonnegative UGV default, saturation, and box/barrier-conflict success regression. |
| Infeasible QP | Zero row with positive RHS and nonzero ego velocity; ordinary fallback; forced iteration-limit failure. |
| Invalid input | Stale sensor with nonzero ego velocity; invalid nominal; exact early-zero semantics, sensor check precedence. |
| Fail-safe bounds | Nonzero velocity beyond limits; fallback ignores custom bounds; direct Wang UGV fallback uses UAV limit. |
| Projected rounds | Feasible first-round termination, multiple correcting rows, order sensitivity, tolerance boundary, zero rounds, exhausted/stalled corrections. Test projection directly from a fixed initial u. |
| Type exclusions | Both UAV→UGV and UGV→UAV excluded, same-type retained, supplied neighbor order preserved; Target1 omitted at mission adapter. |

For final successful controls, default `atol=1e-3, rtol=1e-5`; compare objective and final residual with scaled tolerance `1e-4*(1+abs(b_i)+norm(a_i)*norm(u))`. **Do not require final feasibility for the documented post-clip/projection-stall cases**: compare their residuals to the oracle and flag `final_feasible=false`. Zero/no-row and fallback controls compare at `1e-12`. Success/fallback, model dimension, constraint counts, and status category must match exactly on unambiguous fixtures. Test solved-inaccurate acceptance explicitly. Active count must match for fixtures away from the strict 1e-3 threshold; near-threshold cases test the counting function directly and retain raw slacks.

Exact OSQP iteration counts, elapsed times, and floating residual values are recorded, not equalized. Freeze settings/cold starts and avoid numerical boundary cases in required cross-solver status tests. Direct fixed-input projection tests require exact round count; integrated solves may start slightly differently, so compare final control and the bounded stopping rule, not arbitrary equality of integrated round counts. A tolerance exception requires an individually documented fixture and demonstrated equivalent residual/status; do not globally loosen tolerances. Also compare against the checked-out venv oracle to expose its different NumPy/SciPy stack.

### Acceptance, validation, and checkpoint

1. Standalone `cmake`, build and `ctest` succeed without ROS sourced; public includes/dependency graph contain no ROS/AirSim/RPC dependency.
2. Clean Humble image dependency build and clean colcon workspace build succeed, followed by all new unit/parity tests.
3. Existing Python 134-test suite and every existing test selected by `docker/ros2/test.sh` remain passing; preserve/extend the script's existing package list. No skipped mandatory parity tests when OSQP is missing: fail setup instead.
4. Fixture report includes row comparisons, final controls, explicit infeasibility/post-clipping cases and diagnostics. Save a small eight-agent repeated-request timing report; do not optimize away literal projection yet.
5. No live simulator validation is needed for this mathematical phase. A read-only replay of recorded states into fixtures is useful but must not claim closed-loop safety.

Checkpoint: commit the isolated core/dependency/test work once green. Existing mission has no CBF dependency yet and runs unchanged. Rollback is reverting that phase commit; debug from a single serialized request/constraint dump, not a live mission.

## Phase 2 — ROS and live mission integration

**Deliverable:** synchronous in-process safety filtering at the mission control rate with per-agent inputs and diagnostics. Add narrow **`hercules_cbf_ros`**, but do not create eight filter nodes or transport nominal/safe commands through separate topics. The package owns conversion and request assembly; `hercules_cbf` owns mathematics; the mission node owns scheduling and actuators.

### Files and dependencies

Add:

```text
ros2/src/hercules_cbf_ros/CMakeLists.txt
ros2/src/hercules_cbf_ros/package.xml
ros2/src/hercules_cbf_ros/README.md
ros2/src/hercules_cbf_ros/include/hercules_cbf_ros/cbf_adapter.hpp
ros2/src/hercules_cbf_ros/include/hercules_cbf_ros/obstacle_cache.hpp
ros2/src/hercules_cbf_ros/src/cbf_adapter.cpp
ros2/src/hercules_cbf_ros/src/obstacle_cache.cpp
ros2/src/hercules_cbf_ros/test/test_cbf_adapter.cpp
ros2/src/hercules_cbf_ros/test/test_obstacle_cache.cpp
ros2/src/hercules_cbf_ros/test/cbf_adapter_parity_dump.cpp
ros2/src/hercules_cbf_ros/test/test_python_cpp_coupling_parity.py
ros2/src/hercules_interfaces/msg/ObstacleProxy.msg
ros2/src/hercules_interfaces/msg/ObstacleProxyArray.msg
ros2/src/hercules_interfaces/msg/CBFDiagnostics.msg
ros2/src/hercules_interfaces/msg/CBFDiagnosticsArray.msg
ros2/src/hercules_mission_ros/config/rural_cbf.yaml
ros2/src/hercules_mission_ros/test/test_cbf_mission_transport.py
```

Modify `hercules_interfaces/CMakeLists.txt` (and manifest if necessary), `hercules_mission_ros/src/mission_node.cpp`, its `CMakeLists.txt`/`package.xml`, `launch/rural_nominal.launch.py`, `README.md`, and `test/test_live_mission_contract.cpp`. Extend `mission_actuation.hpp/.cpp` and `test/test_mission_actuation.cpp` for the explicitly separated reproduction adapter profile below. Extend `test/test_target_source.cpp` for shared selection/freshness behavior. Add packages to `docker/ros2/test.sh`.

The adapter library depends on `hercules_cbf`, `hercules_interfaces`, Eigen and the existing `hercules_tracking` model helper for prediction covariance, with ROS build tooling. It must not depend on `hercules_mission_ros` or actuators; the mission converts its nominal Eigen controls at the call site. Keep the existing tracking graph implementation untouched; the CBF adapter constructs a tiny roster-ordered current-state graph directly.

### Parameters and request flow

Expose launch and node parameters `cbf_enabled=false`, `cbf_method=mestres|wang`, `cbf_obstacle_source=none|truth|perception`, scalar `uncertainty_radius=0`, and the complete relevant CBF limits/gains from `rural_cbf.yaml`. Explicitly reject unknown methods and invalid limits at startup. Instantiate one filter configuration per controlled agent, forcing UGV effective method to Mestres regardless of selected method. Log selected and effective method separately.

At each 100 ms mission step:

1. Keep the existing nine-state calibration/freshness gate. Snapshot the freshest accepted canonical states once for all eight requests; validate message ID/type/frame and finite state values. A missing/stale canonical state retains the existing mission-wide safe stop, separate from obstacle-sensor fallback.
2. Rebuild same-type neighbor lists from current snapshot XYZ distance `<=10 m` (or configured communication range), at **10 Hz**, independently of the 4 Hz tracking epoch graph. Do not reuse `current_adjacency_`, which today only changes inside `publishTrackingEpoch()`. Preserve roster order. Do not infer neighbor acceleration from last command: Python mission neighbors do not carry it, so use absent/zero, while keeping the optional core field.
3. Select/predict each agent's own local target estimate once. Feed the same active/XY/velocity values to the existing nominal controller and the CBF target context. Carry the separately predicted 2x2 covariance as described above.
4. Assemble obstacles from that agent's independent static cache plus its UGV-only target proxy. Source `none` is an explicit valid empty static set for controlled testing; it is never the automatic interpretation of a missing perception message. `truth` uses a declared immutable per-agent fixture/proxy set, not invented RuralAustralia geometry.
5. Call the pure filter with nominal acceleration for UAVs, `[forward speed,yaw_rate]` for Huskies, current neighbor states, obstacles, scalar uncertainty and effective sensor validity. No Target1 filter instance or request.
6. When disabled, bypass filter calls and retain nominal/tracking-only behavior. When enabled, hand its safe model-level output, including fallback, to the actuator conversion. Preserve dry-run, enable-formation, target-only, shutdown and command-watchdog gates. Do not send commands from a diagnostics subscriber.

Message contract: `ObstacleProxyArray` has canonical frame, agent ID, capture ID, acquisition-success/validity, capture timestamp and explicitly identified clock domain, source, and ordered raw proxies (center/radius/optional velocity/source/timestamp/point count/planar). Last successful capture time must survive a failure message; retransmission cannot refresh capture age. No age inflation in the producer. The cache combines validated acquisition time and steady receipt age without comparing ROS epoch seconds to mission elapsed time. Preserve Python's completion-time capture-age definition for the reproduction RPC worker, also logging hardware timestamp when available. Derive increasing age between receptions from a monotonic clock; reset caches on mission/session change.

### Actuation differences: explicit compatibility profile

The current ROS adapter integrates/clips UAV acceleration, but **has no Python altitude-ceiling guard**. Its `ugvCarCommand` immediately stops at speed below 0.05, so it **has no Python zero-speed turn crawl**. It also uses the nominal maximum yaw rate (1.0) to normalize steering, while Python's mission uses the CBF yaw-rate limit (1.5). These must not be hidden by a CBF-only comparison.

Add `actuation_profile=current_ros|python_cbf`, default `current_ros`, independent of `cbf_enabled`. Keep existing tests/default behavior. All three matched reproduction runs select `python_cbf`, including the no-CBF comparison; additionally retain a separate untouched current-ROS baseline run. The new profile only ports existing boundary behavior and does not tune formation or CBF mathematics:

- UAV: `v_cmd=clip(v_current+dt*a_safe, ±uav_velocity_limit)` componentwise, then the Python ceiling safeguard `v_cmd.z=max(v_cmd.z,(ceiling_z-p_z)/dt)`, finally clip Z to its velocity limit. Rural ground reference 2 minus ceiling 10 gives `ceiling_z=-8`; floor is +1. Keep these distinct from target ground/body Z.
- UGV: model output stays `[speed,yaw_rate]`. If speed<0.05 and abs(yaw_rate)>0.05, actuator-only crawl is `min(0.3,ugv_speed_limit)`; otherwise low speed requests brake=1. Apply the existing speed adapter formula (0.02 feedforward, 0.10 positive speed-error gain, throttle cap 0.08, overspeed threshold 0.10 and brake scale 0.50), then steering `clip(yaw_rate/1.5,±1)` with the configured CBF yaw-rate limit. Preserve drive-command handbrake release and manual forward gear from Python `simulation/airsim_runtime.py::command_ugv`; startup/shutdown hard-stop handbrake remains separate. Log model and actuator output independently.
- The current ROS node forcibly stops controlled agents when their target estimate is inactive; Python supplies zero nominal but still runs the CBF/actuation path, so a UAV can retain velocity or respond to barriers. Preserve current default gating under `current_ros`; the reproduction profile runs the literal Python zero-nominal→CBF→actuator path when an estimate is inactive. State-invalid/dry-run/mission shutdown gates still dominate both profiles. Add explicit inactive-estimate tests; never revive a missing target from truth.
- Target1 remains on its existing figure-eight/target actuator path; do not apply controlled-Husky crawl or CBF changes to it. Record any remaining Python-versus-current-target actuator difference in the reproduction report rather than redesign the target controller.

### Diagnostics, tests, and acceptance

Publish one `CBFDiagnosticsArray` from the existing mission node at the control rate on `/hercules_mission/cbf_diagnostics`, containing eight entries. Each entry carries step ID/time/frame/agent, enabled flag, selected/effective method, control kind and dimension, nominal and safe model control, success/fallback/raw solver status, minimum barrier, constraint and active counts, projected rounds, total filter/solver timing, robust terms with row IDs, final residual/feasibility, static/target validity and ages, local target timestamp, intervention norm, and deadline-miss flag. Disabled entries must say `disabled`, not falsely claim a successful optimization. Record full data in JSONL each step; do not publish A/b per step by default—enable request/row dumps only for debugging. Emit nonfinite barrier values as JSON `null` with an availability flag, not zero.

Tests: ROS↔core field/dimension/frame conversion; per-agent distinct target covariance/velocity; covariance growth between epochs without repeated accumulation; target inactive/stale/absent; no truth XY substitution; UGV override under Wang; Target1 excluded; range crossings between tracking epochs; agent/body exclusions; none versus missing-perception validity; stale-static/active-target combination; bounds/config rejection; fallback through the actuator; disabled bit-for-bit model/actuator baseline; ceiling/crawl/yaw normalization and inactive-estimate profile differences. Direct Python/C++ coupling fixtures compare `target_obstacle_proxy_for_agent`, body rejection, age rules and target validity, not just hand-written native expected values.

Transport tests launch mission with synthetic canonical states/origins/estimates/proxies and subscribe to command/diagnostics topics; no AirSim required. Verify dry-run publishes no commands, target-only emits no controlled-agent actuation, stale canonical state invokes existing stop, and every emitted enabled command is derived from the logged safe output.

Live sequence: 5 s dry-run → 10 s target-only → 10 s enabled full team with valid empty static source → 30 s deterministic-proxy run, separately for both methods, with reset/calibration between runs. First use truth-observation tracking for isolation, then distributed camera once available. A rendering failure does not justify substituting truth and calling it camera validation. Phase 2 can complete its deterministic integration gate with camera validation deferred explicitly to Phase 3.

Acceptance: all existing/new tests pass; disabled/current-ROS fixture outputs are unchanged; both enabled modes execute eight filters per valid step, Huskies always use Mestres; current-control neighbor changes are visible before the next tracking epoch; target proxy always comes from own estimate; diagnostics agree with actual actuation and replayed core output. On deterministic feasible cases, no unexplained fallback/nonfinite command/collision, and zero CBF deadline misses in the 30 s gate. Define a 5 ms per-agent solve budget and 20 ms team CBF-stage budget as initial measured acceptance targets, within the 100 ms loop; report p50/p95/p99/max. These are measurement flags, not new OSQP time limits or silent command replacement. If missed, inspect allocations/settings and defer performance optimization until parity is retained.

Checkpoint: commit green adapter/mission work with `cbf_enabled=false` and `actuation_profile=current_ros` still default. Those two parameters restore the original baseline without removing packages. Keep per-request replay dumps for any failed live gate; revert the phase commit for a code rollback.

## Phase 3 — reproduce target-tracking + CBF experiments

**Deliverable:** explicit no-CBF/Mestres/Wang run configurations, reusable Python obstacle perception, matched experiment logs, metrics and extensions to the existing animation. Deliver configurations/perception first, metrics second, and the live comparison last, as specified in the execution-unit table.

### Files and dependencies

Add:

```text
ros2/src/hercules_cbf_ros/scripts/obstacle_observer_node.py
ros2/src/hercules_cbf_ros/scripts/mission_collision_observer_node.py
ros2/src/hercules_cbf_ros/config/rural_perception.yaml
ros2/src/hercules_cbf_ros/test/test_obstacle_observer.py
ros2/src/hercules_cbf_ros/test/test_obstacle_observer_transport.py
ros2/src/hercules_cbf_ros/test/test_collision_observer.py
ros2/src/hercules_interfaces/msg/MissionCollision.msg
ros2/src/hercules_mission_ros/config/rural_tracking_no_cbf.yaml
ros2/src/hercules_mission_ros/config/rural_tracking_mestres.yaml
ros2/src/hercules_mission_ros/config/rural_tracking_wang.yaml
ros2/validation/cbf/README.md
ros2/validation/cbf/REPORT.md
ros2/validation/cbf/compare_cbf_modes.py
ros2/validation/cbf/test_metrics.py
ros2/validation/cbf/fixtures/deterministic_proxies.json
docker/ros2/reproduce_cbf.sh
```

Modify the Phase 2 adapter cache/tests as necessary to complete perception transport, the new package's CMake/manifest, interface build, mission launch/node logging, `docker/ros2/Dockerfile`/README, `ros2/validation/rural_nominal/render_validation.py`, Python `modules/mission_plots.py` and `tests/test_mission_plots.py`, and the reproduction documentation. Store generated recordings/requests/media under a new Git-ignored `ros2/validation/cbf/artifacts/` directory using the repository's existing ignore policy. Do not commit large recordings.

The current ROS `airsim_ros_pkgs` already publishes DepthPerspective images, CameraInfo and LiDAR PointCloud2, and the repository has other depth/point-cloud utilities. It does **not** implement the Python proxy fitting, freshness/body masking, or complete capture-pose normalization pipeline. The wrapper applies ROS coordinate conventions and exposes a LiDAR static-transform path; blindly treating those clouds as canonical NED would be wrong.

Choose the least invasive reproduction route: a Python ROS observer around existing `_capture_obstacles`, `ObstacleDetector`, `estimate_ground_z`, pose/decode helpers, and `PerceptionTraceStore`, using read-only AirSim sensor RPC in worker threads as the existing target observer does. Construct an explicit read-only facade/state shim; never call the orchestrator's `main`, spawn/actuate functions, or share a mutable RPC client with the control loop. Use canonical ego state plus canonical orientation and calibrated origin to reproduce the capture helpers' actor-versus-kinematics pose contract; test translation is applied exactly once. Missing canonical capture pose means invalid acquisition. Camera fan requests retain response poses and front-only fallback. Each worker owns its client(s); publish bounded immutable snapshots into ROS. Keep all expensive work outside the C++ mission executor. A direct image/cloud subscriber is a later transport optimization after capture/frame parity, not a prerequisite rewrite.

Pin perception dependencies to the existing working venv versions in an explicit Docker installation step: NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.7.1 (existing joblib 1.5.1 is already pinned; also pin the resolved threadpoolctl), with Python OSQP remaining 1.0.4. First validate this stack in a disposable Humble image and rerun all tracking/CBF tests; record wheel hashes in a lock file added as `docker/ros2/requirements-perception.lock`. Do not silently use an arbitrary mounted venv. The detector's no-sklearn fallback is connected components and ignores DBSCAN's density/min-samples semantics; production reproduction must require the recorded DBSCAN backend. Keep fallback tests separately labeled.

### Explicit configurations and deterministic reproduction

Add launch `mission_config`/`cbf_config` parameter-file wiring so the files above are actually loaded; the existing core YAML is documentation of defaults, not currently a live YAML loader. Do not create decorative unused configurations. Keep numerical mission defaults in `ruralTargetTrackingConfig()` unchanged and log the fully resolved values plus any override. Configure these shared values for all three matched runs:

| Setting | Reproduction value |
|---|---|
| Agents | Drone1, Drone2, SimpleFlight, Drone4, Drone5; Husky1, Husky2, Husky3; separate Target1 |
| Mission inputs | `target_source=distributed_tracking`, `target_observation_source=camera` |
| Timing/network | `dt=0.1`, tracking 4 Hz, communication range 10 m |
| Target motion | Existing Gerono figure eight: 10x8 m, 64 samples, start index 5, speed 0.10 m/s, route heading from existing launch `-1.5083775167989393` |
| Formation | Existing gains/slots, UAV altitude -5, UGV circumradius 5, nominal speed 1, nominal max yaw rate 1 |
| CBF geometry/dynamics | UAV radius 1; UGV/target physical radius 1.25; obstacle margin 0; k1=k2=alpha=2; uncertainty 0 |
| Bounds/projection | UAV acceleration 6, velocity 3; UGV speed 3, CBF yaw rate 1.5, acceleration 3 for age calculation; lookahead 0.1; projection 20/1e-3 |
| Altitude | Floor +1 NED; ceiling -8 NED; target body/ground-Z source logged separately |
| Solver/profile | `python_osqp_postclip`; `actuation_profile=python_cbf`; OSQP 1e-5 tolerances, 4000 iterations, no polishing |
| Perception | Per-agent 2.5 Hz, top_n=5, stale threshold 1.3 s at these settings |

Mode files differ only in `cbf_enabled` and `cbf_method`: no-CBF=false/Mestres (method ignored), Mestres=true/Mestres, Wang=true/Wang. Run a fourth `cbf_enabled=false, actuation_profile=current_ros` capture as the historical ROS behavior checkpoint, outside the matched algorithm comparison.

Proposed commands, available after the Phase 2/3 launch wiring (these are **not current capabilities**):

```bash
./docker/ros2/reproduce_cbf.sh --mode no_cbf --obstacles none --observation camera
./docker/ros2/reproduce_cbf.sh --mode mestres --obstacles none --observation camera
./docker/ros2/reproduce_cbf.sh --mode wang --obstacles none --observation camera
./docker/ros2/reproduce_cbf.sh --mode mestres --obstacles truth --observation truth
./docker/ros2/reproduce_cbf.sh --mode wang --obstacles perception --observation camera
```

The driver resolves mode files, passes explicit launch values, resets using the existing Rural mission setup, waits for calibration/sensor warmup, records resolved config/HEAD/image/seed and artifacts, and invokes the existing renderer. Retain `--target-source truth` and `--observation truth` debug overrides; truth-observation seed 7 remains the current default. Each run gets a separate directory.

Start with valid-empty static proxies, then deterministic static/moving fixtures in offline/synthetic ROS tests. A live truth-obstacle run must use actual declared geometry that exists in the simulator. Python only spawns its course boxes in FlyingCPP; `--use-truth-obstacles` in unmodified RuralAustralia is normally an empty set. Do not label that run “validated obstacle avoidance.” No-static does not mean no barriers: same-type neighbors, UAV floor and UGV target proxies remain.

Collect same-step requests and replay them through both Python and C++ to compare rows/control/status under identical inputs. This isolates port correctness from asynchronous camera/vehicle trajectory divergence. Where practical run the existing Python orchestrator with the matching map, mission objective, camera observations, gains, geometry, seed and reset. Python has no ordinary no-CBF CLI baseline here; use the native disabled baseline and do not invent a Python CLI flag. Record nominal→safe intervention comparisons and Python/native trajectory/metric differences separately from exact request replay.

### Real Python perception and cache parity

Preserve the complete acquisition and proxy behavior:

- UAV fan: `front_center`, `front_left`, `front_right` DepthPerspective; on rejected optional fan RPC, front-only fallback; merge valid world point sets. Normalize perspective rays as Euclidean range, preserve increasing pixel-u→camera +Y and capture-pose origin correction. Missing/invalid image set is acquisition failure.
- Huskies: `Lidar1` on car RPC 41452; remove nonfinite/near-zero returns, truncate malformed trailing values as Python does, cap points and compose sensor pose with canonical vehicle pose. A successfully returned empty LiDAR cloud is **valid empty**, not failed perception. UAV RPC is 41451.
- Reuse finite/range filtering, ground estimation/rejection, voxelization, point caps, DBSCAN, recursive principal-axis patch splitting, proxy fit/padding/cap, ranking and top-N selection. Preserve point/proxy order, capture IDs and diagnostic counts.
- Shared defaults: range 0.4–30 m, voxel 0.2, max_points 5000, depth stride 2, top_n 5. Rural UAV: eps 0.65, cluster_min_samples 20, min_proxy_points 32, padding 0.25, max radius 1, ground band 0.25. Rural UGV: eps 0.85, cluster_min_samples 3, min_proxy_points 32, padding 0.35, max radius 1, ground band 0.15, surface offset 0, nearest-surface fitting/ranking enabled.
- Match the Python round-robin service demand `max(1,ceil(N*dt*sensor_rate))` captures per tick: at N=8, dt=.1, rate=2.5 this is two, cycling roster order. The ROS worker is asynchronous to avoid blocking control, so log actual per-agent rate/latency and retain exact capture-time/age semantics; do not silently turn the rate into a 2.5 Hz team-wide budget. Warm every sensor before the reproduction measurement interval.
- Only a successful acquisition replaces cached proxies and last-success time, including a valid empty acquisition. Failed captures retain the previous cache and continue aging it; no “failure refresh.” Python stamps completion with wall time rather than raw sensor timestamp. Log both plus mapping metadata; declare the asynchronous scheduling difference.
- Default stale limit is `max(0.25,3/sensor_rate+dt)`, which is **1.3 s** here, despite CLI help claiming one period plus 50 ms. Boundary valid when `age<=limit`.
- While static-valid, inflate each cached radius by `min(speed_limit*age+0.5*acceleration_limit*age²,0.25*max_proxy_radius)` in RuralAustralia, i.e. cap 0.25 m. Other-map cap is 0.5*max_proxy_radius. Static-invalid gets **zero inflation**, not capped inflation. Retain raw age (nullable before first success), applied inflation, and uninflated radius.
- Cached static proxies are retained even when stale. With an active UGV target proxy, the CBF still runs against these stale static proxies at their original radius; without either valid source it takes the early sensor-failure path. UAV target estimates never override static-invalid. This counterintuitive loss of inflation at the stale boundary is a required regression test, not a correction opportunity.
- Python reconstructs static proxies during age inflation without forwarding optional velocity, effectively setting it absent/zero. Preserve that cache behavior. The separately added moving target proxy retains estimated velocity; pure-core moving-obstacle fixtures still exercise explicit velocity inputs.
- Before target insertion, reject perception proxies whose centers lie within `vehicle_radius+0.5` in 3D of any current controlled-agent body. Exempt sources starting `truth` or exactly `target_tracking`; exclude Target1 from the controlled-body map. This avoids duplicate/cross-type body obstacles for the controlled roster, but Python does not explicitly remove a sensed Target1 cluster; do not silently add a new target-body mask.

Tests feed synthetic sensor responses through the original helper and wrapper, comparing world points, proxy lists/radii/order, source validity, capture time and diagnostic counts. Cover nonzero calibration translation/yaw, off-axis depth rays, fan fallback/partial fan, empty LiDAR, terrain-ground rejection, DBSCAN-versus-fallback distinction, top-N ties, zero returns, age cap, exact stale boundary, stale radius reset, dropped static velocity, controlled-body masks, failed capture retaining cache, delayed/out-of-order/duplicate snapshots, and stale static plus active target. ROS transport tests stop the producer and verify age continues increasing without new messages and no control executor stalls.

### Experiment metrics and existing visualization

Extend JSONL with versioned `cbf`, `obstacles`, `nominal_controls`, `safe_controls`, `actuation`, `safety_communication_links`, `timing`, and collision records while retaining existing fields consumed by the renderer. Do not keep the current placeholder empty collisions object and report it as zero collisions.

Add a read-only Python collision observer using the relevant vehicle RPC service for all nine vehicles; publish `MissionCollision` events on `/hercules_mission/collisions`, including vehicle, simulator event timestamp/object/contact details, validity and poll status. Deduplicate persistent `has_collided` events by event identity/time; distinguish unavailable from zero. This observer has no control authority. Log event resets/session boundaries and correlate to canonical state. It belongs alongside the new perception observer for transport reuse, not in the pure CBF library.

| Level | Required metrics |
|---|---|
| Per agent | Every nominal and safe model-control component with units; intervention `norm(safe-nominal)` and fraction of intervened steps; minimum barrier; active/total constraints; solver status, solved-inaccurate and final-feasibility counts; fallback events/duration; filter and solver time; desired-slot error; target position/velocity error against truth for evaluation only; target covariance, age and observation availability. |
| Global | Minimum same-type physical pairwise clearance across **all pairs**, including pairs outside the communication graph; minimum static/target proxy clearance; independent known-geometry clearance when available; collisions; loop period/work time/jitter/deadline misses; per-agent/team CBF budget misses; existing tracking error/availability/handoff/consensus metrics and camera capture/detection/rate/RPC-latency metrics. |

Use UAV 3D center distance minus radii, UGV XY center distance minus radii for physical pair clearance. Report UGV lookahead-point barrier clearance separately; it is not physical body clearance. Include physical target radius for measured UGV-target clearance and uncertainty-inflated proxy clearance as a separate quantity. Real RuralAustralia terrain has no complete obstacle ground-truth catalog: mark physical obstacle clearance unavailable where unknown, report perceived proxy clearance and collision evidence rather than asserting a scene-wide minimum. Report pair and altitude minimum barriers separately in addition to the literal mixed Python minimum. Mark slot/estimation errors unavailable when their reference estimate is inactive; do not score an inactive zero estimate as a real target.

Extend `render_validation.py` and add `compare_cbf_modes.py` to output metrics JSON, side-by-side matched-mode plots, per-agent command/intervention/status/timing series and clearance plots. Keep Python `plot_topdown_animation` as the animation backend: intervention halos/arrows keyed by magnitude; fallback/deadline markers; static proxy circles and velocity arrows; UGV target proxy circles using each agent's own covariance/center. Add a toggle/filter to avoid displaying three overlapping target proxies unreadably. Retain existing observer rings, distributed estimate/covariance overlays, links, trajectories, slots and figure eight. Add tests for old logs missing CBF fields, null barriers, stale proxies and rendering a small new log; no new animation pipeline.

Validation sequence per method/source: deterministic request parity → synthetic ROS replay → 5 s dry-run → target-only 10 s → full team 30 s → matched 120 s runs, at least three reset trials for no-CBF/Mestres/Wang with identical settings and capture/perception load. Camera noise/physics cannot be made identical merely by sharing a seed; match setup and additionally use recorded-request replay for exact parity. At Target1 speed 0.10 m/s, 120 s may not cover a full figure eight: report path fraction/sample advancement, and run longer only if full-route reproduction is required, without changing speed. Capture Python reference runs where practical and compare aggregate metrics with initialization/warmup differences reported.

Acceptance:

1. All existing/new tests pass after the perception dependency change; request replay maintains Phase 1 row/control tolerances and Phase 2 coupling parity.
2. The three executable reproduction configurations use distributed tracking and camera observations; both enabled configurations force Mestres on Huskies. Truth/debug labels are unambiguous.
3. Deterministic validation exercises a nonzero CBF intervention, a moving-target barrier, neighbor graph crossing, and forced failure/staleness. Feasible engineered scenarios have no unexplained collision/fallback/deadline miss; known post-clip residuals are visible rather than mislabeled safe.
4. Real-perception runs demonstrate correct capture frames, rate/age accounting, DBSCAN backend, controlled-body masking, target proxy provenance and stale behavior, with all metrics/media populated. Each normal 120 s acceptance trial targets zero CBF budget misses and no unexplained fallback. If physical collision/unsafe clearance persists while replay agrees with Python, classify it as reproduced source/model/actuator limitation, document it, and withhold any safety-performance claim; do not retune the algorithm to pass.
5. `REPORT.md` records actual run IDs/HEAD/config/image, methods, source and adapter profiles, initial geometry, Python/native request and trajectory comparisons, collision/freshness/timing results, missing observations/metrics, and visual inspection. A blocked camera/rendering prerequisite means Phase 3 camera reproduction is incomplete, not permission to repair rendering within this task or relabel truth results.

Checkpoint: commit configuration/replay/perception work, metrics/media work, and the live comparison report separately, following tasks 3A–3C below. `cbf_obstacle_source=none` or a saved fixture isolates perception, while `cbf_enabled=false, actuation_profile=current_ros` restores the original mission baseline. Preserve failed-run logs and sensor traces for offline replay. No control-law edits are a permitted workaround for a failed experiment.

## Recommended Luna execution units

Keep exactly these three architectural phases, but split delivery into **six bounded Luna tasks**:

| Task | Scope | Stop/acceptance boundary |
|---|---|---|
| 1 — Phase 1 | Native solver dependency, pure core, all constraint/control parity tests | One task is reasonable given the fixed equations/solver decision. Stop only when standalone/clean Docker builds and all regressions pass; commit the core. |
| 2A — Phase 2 | Messages, adapter/cache, target covariance/proxy coupling, synthetic integration tests | No live actuator wiring needed yet; adapter parity green and independent core intact. |
| 2B — Phase 2 | Mission insertion, diagnostics, explicit actuator profile and deterministic live gates | Disabled baseline unchanged; both enabled modes and rollback switches verified. |
| 3A — Phase 3 | Real Python perception wrapper, reproducible dependency lock, cache/capture parity | Validated raw capture→canonical proxy→CBF transport, outside the control thread. Include the configurations/replay driver early enough to exercise it. |
| 3B — Phase 3 | Collision capture, log/metrics extensions, existing animation extensions | Tested old/new log compatibility and populated comparison artifacts from fixtures/replays. |
| 3C — Phase 3 | Matched live camera experiments, Python reference/replay comparison, report | Actual three-mode results and limitations recorded; no claims based only on truth observations or old reports. |

Each task begins by verifying the branch/HEAD and reading its predecessor's committed acceptance record; the Python oracle stays tied to the inspected source revision unless a later user instruction deliberately changes it. Do not ask Luna to reselect the solver, redesign package boundaries, or reconcile these executable discrepancies from scratch. The default is the literal behavior described here.
