# Native HERCULES + Docker ROS 2 Humble

This guide applies to the macOS parity branch as well as the Linux-origin
`codex/ros2` branch; generated build and validation outputs stay outside Git.

Unreal runs natively on the simulator host, while this image contains ROS 2
Humble/Jammy, development tools, and dependencies for the entire ROS workspace.
The repository is mounted at `/workspaces/hercules`; Unreal and repository source
are not copied into the image. Existing Python controllers are unchanged. Ubuntu
22.04 x86-64 remains the reference end-to-end host. Docker Desktop on macOS can
run the deterministic image/workspace build, tests, and artifact comparison, and
can host a native UE 5.2.1 Metal run after the explicit Mac networking and
architecture preflight below.

## Build and shell

Run these commands from the repository root:

```bash
./docker/ros2/build.sh
./docker/ros2/build_ws.sh
./docker/ros2/test.sh
./docker/ros2/shell.sh
```

For a clean build plus the complete maintained ROS test gate, use the single
reproduction entry point:

```bash
./docker/ros2/reproduce.sh
```

Its logs and any optional rendered media are written only below ignored
`ros2/validation/**/artifacts/` directories. See
[ROS2_HANDOFF.md](../../ROS2_HANDOFF.md) for installation, Unreal Engine 5.2.1,
Rural Australia asset transfer, and optional live-video instructions.

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

Image builds resolve dependencies from package manifests only. The image also
installs the pinned Python perception stack from `requirements-perception.lock`
and builds the pinned native OSQP/QDLDL sources. Rebuild the image after changing
manifests or either dependency lock. Helpers deliberately do not restart a busy
container:

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

## macOS + Docker Desktop

On an Apple Silicon Mac, install and start Docker Desktop, then allow the
checkout's parent directory in Docker Desktop's File Sharing settings. Install
the native tools used by the AirSim/plugin build and the Python oracle:

```bash
brew install cmake ninja llvm@18 wget coreutils rsync unzip
brew install --cask docker
open -a Docker
docker version
docker compose version
```

Use Python 3.10 for the reference mission environment. The lockfile was
captured on 3.10.12; use a version manager when that exact patch is required.
Keep the environment outside the checkout so platform-specific wheels are
never mistaken for container dependencies:

```bash
python3.10 --version
mkdir -p ../.venvs
python3.10 -m venv ../.venvs/hercules-python310
source ../.venvs/hercules-python310/bin/activate
python -m pip install --upgrade pip
python -m pip install -r PythonClient/requirements-herculesvenv.txt
```

The Docker image itself is a Linux environment and may be arm64 or amd64
depending on Docker Desktop's selected architecture. The ROS/C++ build and RPC
protocol are architecture-independent, but the native UE editor and AirSim
plugin must match. The Darwin `build.sh` path now defaults to native `arm64` and
does not require Rosetta; use `HERCULES_MAC_ARCH=x86_64` only for an intentional
legacy Intel build. Record `uname -m`, editor architecture, image architecture,
and build target architecture in any comparison report.

The Compose file uses `network_mode: host`, which is the Linux assumption. On
Docker Desktop, first check whether the installed version supports optional host
networking and enable it if you want the Linux-like `127.0.0.1` path. Otherwise
use a bridge/forwarded setup and pass `host.docker.internal` to every AirSim
client (`host_ip` for ROS wrappers, `rpc_host` for the direct pose bridge, and
`--airsim-host` for Python). Mixing loopback and `host.docker.internal` clients
invalidates a comparison. Keep `ROS_DOMAIN_ID=42`; `ROS_LOCALHOST_ONLY=1` is
appropriate when all ROS nodes run in the same container, while host-side ROS
tools require a separate DDS/network configuration.

Record the selected network path and test the Mac host from the container before
starting ROS nodes:

```bash
docker compose -f docker/ros2/compose.yaml config
docker compose -f docker/ros2/compose.yaml up -d --build
docker compose -f docker/ros2/compose.yaml exec dev getent hosts host.docker.internal
docker compose -f docker/ros2/compose.yaml exec dev nc -vz host.docker.internal 41451
docker compose -f docker/ros2/compose.yaml exec dev nc -vz host.docker.internal 41452
```

The launch helpers select the Mac editor bundle on Darwin and skip the Linux
NVIDIA/Vulkan variables. If an older checkout is being used, launch the editor
directly as shown below. macOS uses Metal; `-nullrhi` is acceptable only for a
truth/physics check and cannot validate camera/depth output.

```bash
export UE_ROOT="/Users/Shared/Epic Games/UE_5.2"
export UNREAL_EDITOR="$UE_ROOT/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
"$UNREAL_EDITOR" "$PWD/Unreal/Environments/Blocks/Blocks.uproject" \
  /Game/RuralAustralia/Maps/RuralAustralia_Example_01 \
  -game -windowed -ResX=960 -ResY=540 -nosplash \
  "-settings=$PWD/docker/ros2/settings.rural-nominal.json"
```

Before a bridged live run, verify that *all* launch paths propagate the same
host value, including reset/benchmark helpers. If a node or helper has no RPC
host parameter, stop at the deterministic build/parity gate and treat that as a
Mac portability item rather than silently running against its own container
loopback.

## Optional CBF mode comparison

The default mission configuration keeps the CBF filter and obstacle observers
disabled. The separate harness records one resolved run directory per mode:

```bash
./docker/ros2/reproduce_cbf.sh --mode no_cbf --obstacles none --dry-run
./docker/ros2/reproduce_cbf.sh --mode mestres --obstacles none --dry-run
./docker/ros2/reproduce_cbf.sh --mode wang --obstacles perception --dry-run
```

Run these commands with a reachable AirSim/Unreal simulator already running.
`--dry-run` disables actuation but still requires live ROS state and origin
discovery; use `--live` (or `--dry-run false`) for an actuated comparison. On
macOS without a reachable simulator, use `./docker/ros2/reproduce.sh` and the
package parity tests; these mode commands are not an offline CBF comparison.

Use `no_cbf` as the baseline before enabling a filter. Perception-backed runs
require a rendered simulator and validated canonical camera/frame-origin data.
The observer reuses Python capture helpers, but a successful process launch does
not by itself establish Python/ROS perception parity. The mode driver currently
uses the launch's default 30-second duration; invoke the launch directly with
`duration_sec:=...` when another duration is required.

## Start the native simulator

The Linux helper path is the reference validation route:

Use a built Blocks project with the AirSim plugin. The example settings select Hero
mode, one `Drone1` (SimpleFlight), and one `Husky1` (CPHusky), separated around the
map's PlayerStart with zero initial yaw. IMU and default non-image sensors are used.
The example does not overwrite `~/Documents/AirSim/settings.json`.

```bash
export UNREAL_EDITOR=/path/to/Unreal/Engine/Binaries/Linux/UnrealEditor
./docker/ros2/launch_sim.sh
```

For a native macOS UE 5.2.1 editor, use the platform-aware helper after the Mac
preflight, or launch directly as a fallback. Do not export `VK_ICD_FILENAMES`,
PRIME, or GLX variables; macOS uses Metal:

```bash
export UE_ROOT="/Users/Shared/Epic Games/UE_5.2"
export UNREAL_EDITOR="$UE_ROOT/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
"$UNREAL_EDITOR" "$PWD/Unreal/Environments/Blocks/Blocks.uproject" \
  /Game/RuralAustralia/Maps/RuralAustralia_Example_01 \
  -game -windowed -ResX=960 -ResY=540 -nosplash \
  "-settings=$PWD/docker/ros2/settings.rural-nominal.json"
```

If Docker Desktop is using bridge networking, pass `host_ip:=host.docker.internal`
to the ROS AirSim wrapper and `rpc_host:=host.docker.internal` to the direct pose
bridge. Python runs use `--airsim-host host.docker.internal`. Use one host value
for every client and verify both RPC ports from inside the container before
starting the mission.

Alternatively pass `-settings=<absolute-path-to-settings.hero-smoke.json>` to your
native simulator launcher. Choose an open, reasonably level PlayerStart area before
the motion test. `launch_sim.sh` also forwards Unreal command-line arguments.
`-nullrhi -unattended -NoSound` runs native physics without rendering; that mode
cannot validate camera/image output and its performance is not a rendered benchmark.

Hero's current native source uses drone RPC port 41451 and Husky RPC port 41452.
On Linux with `network_mode: host`, `127.0.0.1` in the container reaches those
host servers. Docker Desktop/macOS has different host-network semantics: use
optional host networking only after verifying it, or use the bridge address
`host.docker.internal` consistently. ROS discovery defaults to localhost and
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
subscriptions, publishing, and asynchronous service requests. No new general
purpose controller, estimator, or planner was introduced. The separate
`hercules_cbf`/`hercules_cbf_ros` packages provide an optional native safety filter,
ROS adapter, obstacle cache, and read-only observers; these are disabled by the
default mission configuration. Adapter command clamps are smoke-test limits, not
a general-purpose control API.

The current HERCULES wrapper flips Y/Z in odometry, subtracts its startup position,
and makes orientation startup-relative, while retaining fixed-axis velocity. This
is not ENU, and its frame labels should not be interpreted as conventional ROS
body-frame twist without further work. The smoke configuration uses zero initial
yaw and only world-X translation. Existing world-velocity commands are forwarded
as native NED components, so our adapter flips Y/Z and yaw-rate signs when sending
them. It does not flip incoming odometry again. A broader frame/TF correction is
outside this change; multi-vehicle odometry also retains the existing shared initial
reference behavior.

The wrapper's `is_vulkan` parameter selects the expected RGB/BGR image encoding;
it is not a renderer selector. Linux Vulkan validation historically used `true`.
For macOS Metal, inspect one received image and set the encoding consistently
before enabling camera perception; do not infer it from the operating system.

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
Adapter timing is not controller computation timing. The mission formation and
optional CBF filters are not benchmarked by this wrapper smoke report, and the
documented roughly 50 Hz RPC ceiling is not assumed or modified.

See [VALIDATION.md](VALIDATION.md) for the **historical Linux** build, test,
simulator, and benchmark record. It predates the Mac workflow and is not a
current macOS/Metal acceptance result. CBF-specific offline/live evidence is in
[`ros2/validation/cbf/REPORT.md`](../../ros2/validation/cbf/REPORT.md) and is
likewise tied to the run metadata recorded there.
