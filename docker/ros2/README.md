# Native HERCULES + Docker ROS 2 Humble

Unreal runs on the Linux host. This image contains ROS 2 Humble/Jammy, development
tools, and dependencies for the entire ROS workspace. The repository is mounted at
`/workspaces/hercules`; Unreal and repository source are not copied into the image.
Existing Python controllers are unchanged.

## Build and shell

Run these commands from the repository root:

```bash
./docker/ros2/build.sh
./docker/ros2/build_ws.sh
./docker/ros2/test.sh
./docker/ros2/shell.sh
```

The scripts match the container developer's UID/GID to the caller. Only Docker is
needed on the host; no host ROS installation is used. `shell.sh` opens the persistent
`dev` service and sources Humble plus the last successfully built overlay.

The default build is RelWithDebInfo, sequential packages, two compiler jobs:

```bash
BUILD_TYPE=Release BUILD_JOBS=2 ./docker/ros2/build_ws.sh
BUILD_TYPE=Release ./docker/ros2/shell.sh
./docker/ros2/deps.sh                     # after adding dependencies
./docker/ros2/exec.sh ros2 node list      # noninteractive container command
```

Inside the container, `source docker/ros2/env.sh` loads the environment. Outputs are
isolated under `ros2/.docker/humble/<BUILD_TYPE>/{build,install,log}`. The marker
`.build-complete` prevents sourcing a partially built workspace. `HERCULES_BUILD_ROOT`
can select a different **container** output directory. Open a new shell with
`BUILD_TYPE` set when switching configurations, so previous ROS overlays do not
remain in the shell environment. All packages are built;
rosdep failures are not skipped. Missing rpclib 2.3.0 and Eigen 3.4.0 source dependencies
are fetched by `bootstrap.sh`; existing directories are preserved. Native plugin
setup/build scripts are never invoked.

Image builds resolve dependencies from package manifests only. Rebuild the image
after changing manifests. Helpers deliberately do not restart a busy container:

```bash
# After stopping work in the dev container, adopt the rebuilt image:
docker compose -f docker/ros2/compose.yaml up -d --force-recreate dev
```

A completely fresh build, without an existing overlay in the environment:

```bash
docker compose -p hercules-clean -f docker/ros2/compose.yaml run --rm --no-deps \
  --entrypoint /workspaces/hercules/docker/ros2/build_ws.sh \
  -e HERCULES_BUILD_ROOT=/workspaces/hercules/ros2/.docker/humble/clean \
  -e BUILD_JOBS=1 dev
```

Use an empty output directory for each clean verification. The separate Compose
project avoids older Compose versions confusing a one-off build with `dev`.
Debuggers `gdb`, `gdbserver`, and `clangd` are installed; compile commands are exported.

## Start the native simulator

Use a built Blocks project with the AirSim plugin. The example settings select Hero
mode, one `Drone1` (SimpleFlight), and one `Husky1` (CPHusky), separated around the
map's PlayerStart with zero initial yaw. IMU and default non-image sensors are used.
The example does not overwrite `~/Documents/AirSim/settings.json`.

```bash
export UNREAL_EDITOR=/path/to/Unreal/Engine/Binaries/Linux/UnrealEditor
./docker/ros2/launch_sim.sh
```

Alternatively pass `-settings=<absolute-path-to-settings.hero-smoke.json>` to your
native simulator launcher. Choose an open, reasonably level PlayerStart area before
the motion test. `launch_sim.sh` also forwards Unreal command-line arguments.
`-nullrhi -unattended -NoSound` runs native physics without rendering; that mode
cannot validate camera/image output and its performance is not a rendered benchmark.

Hero's current native source uses drone RPC port 41451 and Husky RPC port 41452.
The host network makes `127.0.0.1` in the container reach those host servers. No GPU
passthrough or port mappings are needed. ROS discovery defaults to localhost and
domain 42; use the same domain for all participating terminals.

## Observe, then command

Open a development shell for each long-running command. Launch passive wrappers:

```bash
./docker/ros2/launch_wrapper.sh
```

In another development shell:

```bash
ros2 node list
ros2 topic list
ros2 service list
./docker/ros2/smoke.sh drone observe
./docker/ros2/smoke.sh ugv observe
```

Expected interfaces:

| Vehicle | Node | Odometry | Command suffix under node/vehicle |
|---|---|---|---|
| Drone1 | `/hercules_drone` | `/hercules_drone/Drone1/ground_truth/odom_local` | `vel_cmd_world_frame`, `takeoff`, `land` |
| Husky1 | `/hercules_ugv` | `/hercules_ugv/Husky1/ground_truth/odom_local` | `car_cmd` |

Stop the passive launch with Ctrl-C and restart explicitly enabling API control:

```bash
./docker/ros2/launch_wrapper.sh enable_api_control:=true
# In another shell, one test at a time:
./docker/ros2/smoke.sh drone motion
./docker/ros2/smoke.sh ugv motion
```

API control defaults **off**. Enabling it takes API control and arms vehicles selected
by the existing wrapper; the example contains just one vehicle per endpoint.
`vehicles:=drone`, `vehicles:=ugv`, or `vehicles:=both` selects wrappers. Launch
arguments include `host_ip`, `drone_port`, `ugv_port`, `benchmark_logging`,
`ugv_command_timeout_sec`, and the existing `update_*_every_n_sec` timer periods.
`use_sim_time` and `/clock` publication are disabled in this launch. Steady time
governs freshness and timeouts; odometry retains its simulator timestamp.

The smoke node requires two advancing, valid odometry samples and matching command
interfaces. Drone motion is takeoff, settle, +X at 0.2 m/s for at most one second,
stop, and land. Husky motion is 0.15 throttle/zero steering for at most one second,
then full brake; it stops earlier at 0.25 m displacement or 0.5 m/s. Both tests abort
at 0.5 m horizontal displacement or stale/invalid state; the drone also has a 5 m
relative altitude ceiling. No retries or command increases are automatic.

Successful motion requires at least 0.02 m observed movement and verified stopping.
`SMOKE_RESULT success=true` and exit code 0 indicate success. Ctrl-C requests cleanup
before exit, including landing after a takeoff attempt. A failed/unreachable RPC
server cannot guarantee a delivered stop. The UGV wrapper's test-profile 0.5-second
command lease brakes on the next functioning control tick; its upstream-compatible
default is disabled. Never use SIGKILL as a normal motion-test shutdown.

The smoke executable also accepts ROS parameters `vehicle_name`, `wrapper_prefix`,
`odom_topic`, `command_topic`, `takeoff_service`, `land_service`, `command_rate_hz`,
`duration_sec`, `warmup_sec`, and `wait_on_flight_task` (default false).
The default requests asynchronous takeoff/land submission; the smoke
sequence still requires measured altitude change, settling, stopping, and landing
within the same deadlines. This mode passed live validation with the
installed native plugin, whose blocking takeoff task returned false after climbing;
Setting `wait_on_flight_task:=true` retains strict blocking-result validation;
see the validation report for that remaining native limitation. Example:

```bash
./docker/ros2/smoke.sh drone observe -p duration_sec:=5.0
```

## Architecture and source corrections

`hercules_control_core` contains only plain C++/Eigen state types and a bounded test
sequence. `hercules_control_adapter` converts ROS messages; `smoke_node` owns ROS
subscriptions, publishing, and asynchronous service requests. No new bridge,
controller, estimator, or planner was introduced. Adapter command clamps are
smoke-test limits, not a general-purpose control API.

The current HERCULES wrapper flips Y/Z in odometry, subtracts its startup position,
and makes orientation startup-relative, while retaining fixed-axis velocity. This
is not ENU, and its frame labels should not be interpreted as conventional ROS
body-frame twist without further work. The smoke configuration uses zero initial
yaw and only world-X translation. Existing world-velocity commands are forwarded
as native NED components, so our adapter flips Y/Z and yaw-rate signs when sending
them. It does not flip incoming odometry again. A broader frame/TF correction is
outside this change; multi-vehicle odometry also retains the existing shared initial
reference behavior.

Build corrections are the first two items below; subscription/service fixes repair
the command path, and serialization/leases/empty-map checks are test-safety changes.

Necessary upstream corrections:

- Remove ROS 1 manifest entries; correct rosdep keys and missing declared dependencies.
- Honor build configuration, use C++17 in the wrapper, and preserve compiler flags.
- Restore per-vehicle world-velocity subscriptions with an explicit Humble-compatible
  callback signature, gated by API control.
- Return actual task results for blocking single-vehicle takeoff/land requests;
  asynchronous success means submission only. API-disabled requests return false.
  A separate persistent flight RPC client lets blocking tasks wait without holding
  the polling mutex or replacing the command client’s pending task.
- Serialize control timer invocations, protect command-cache consumption, add the
  optional car command lease, initialize command flags, and reject empty vehicle sets.
- Add optional aggregate boundary counters without changing message/service types.

## Benchmarks and evidence

Stop existing wrappers before running `./docker/ros2/benchmark.sh`. It owns its
wrapper processes, uses neutral commands, records 30 seconds of passive state per
vehicle, then 10/20/50/100 Hz command trials with 5-second warm-up and 15-second
measurement windows at 0.05- and 0.01-second wrapper periods. Avoid concurrent builds
or other simulator command publishers while benchmarking.

Logs, CPU samples, and `summary.csv` are saved beneath the active build root's
`validation` directory. The report separates local publication, ROS callbacks,
distinct state timestamps, wrapper command receipt, and RPC dispatch. Drone dispatch
counts asynchronous RPC submissions; UGV dispatch counts successful synchronous
returns. Neither is proof of an Unreal physics step or per-command acknowledgement.
Adapter timing is not controller computation timing. No controller exists yet, and
the documented roughly 50 Hz RPC ceiling is not assumed or modified.

See [VALIDATION.md](VALIDATION.md) for the actual build, test, simulator, and benchmark
results from this implementation.
