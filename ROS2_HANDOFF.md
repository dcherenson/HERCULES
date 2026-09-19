# ROS 2 RuralAustralia mission handoff

This is the setup and reproduction guide for the `codex/ros2` branch and its
macOS parity worktree (`codex/ros2-macos-parity`). The
reference end-to-end simulator host is Ubuntu 22.04 x86-64; macOS can run the
ROS 2 Docker build and deterministic tests, and can host Unreal natively after
the Mac launch/network preflight below. Do not substitute `main`, and do not
upgrade Unreal Engine: this project is pinned to **Unreal Engine 5.2.1**.

The current branch snapshot is `0d72264b2ed8c3f1f39f1d781e4683e10c482832`
(`codex/ros2`, 2026-09-15). It contains the `hercules_cbf` numerical core,
`hercules_cbf_ros` adapter/observers, CBF interface messages and mode files,
and the `reproduce_cbf.sh` driver. The Linux validation reports below are
historical records; they do not certify a fresh macOS/Metal checkout.

The implemented stage runs Target1 on the ported Gerono figure eight and can
drive the nominal target-centered formation for Drone1, Drone2, SimpleFlight,
Drone4, Drone5, Husky1, Husky2, and Husky3 from either direct truth or each
agent's own distributed target estimate. The default live launch uses the
existing Python camera/depth observer and eight independent C++ tracker
processes. The branch also contains an optional CBF core/ROS adapter plus
read-only obstacle and collision observers; the default mission keeps CBF
disabled (`cbf_enabled=false`, `cbf_obstacle_source=none`). The numerical
Python/C++ replay gate is separate from live perception and actuation. The
perception-backed CBF path still requires calibrated capture-pose validation
before its Python-versus-ROS behavior should be treated as a parity result.
Conformal prediction, cooperative localization, and a normalized
safety-performance claim remain out of scope.

The Mac implementation changes are kept on `codex/ros2-macos-parity` until
review; no history on `codex/ros2` is rewritten by the setup.

## 1. Reference Linux host

Ubuntu 22.04 x86-64 is the supported path. Epic lists Ubuntu 22.04 and clang
15.0.1 for UE 5.2 development. A rendered RuralAustralia run also needs a
Vulkan-capable GPU and current vendor driver; Epic's general UE5 guidance is
32 GB RAM and at least 8 GB graphics memory. Allow substantial SSD space for
UE, its build products, and the Rural Australia asset pack.

Official references:

- [Unreal Engine for Linux download](https://www.unrealengine.com/en-US/linux)
- [Epic Linux quickstart](https://dev.epicgames.com/documentation/unreal-engine/linux-development-quickstart-for-unreal-engine)
- [Epic Linux requirements](https://dev.epicgames.com/documentation/unreal-engine/linux-development-requirements-for-unreal-engine)
- [Unreal Engine source access](https://www.unrealengine.com/ue-on-github)
- [Docker Engine for Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

Install basic host tools:

```bash
sudo apt update
sudo apt install -y ca-certificates curl git git-lfs unzip build-essential rsync
git lfs install
```

Install the NVIDIA or AMD driver appropriate for the workstation, reboot if
required, and verify Vulkan before opening Unreal:

```bash
sudo apt install -y vulkan-tools
vulkaninfo --summary
```

## 2. macOS host + Docker Desktop

This is the intended continuation path for an Apple Silicon Mac. Unreal runs
as a native macOS process and the ROS 2 Humble workspace runs in the Linux
container. Do not install ROS 2 on the Mac host. Docker Desktop must be running
and the checkout's parent directory must be allowed under Docker Desktop's
File Sharing settings.

Install the native build tools, Python tooling, and Docker Desktop:

```bash
brew install cmake ninja llvm@18 wget coreutils rsync unzip
brew install --cask docker
open -a Docker
docker version
docker compose version
```

The Python lockfile was captured with Python **3.10.12**. Homebrew's
`python@3.10` may have a newer patch release; use a version manager such as
`pyenv` when exact oracle parity is required. Keep this venv outside the
checkout so macOS wheels never enter the ROS container:

```bash
python3.10 --version
mkdir -p ../.venvs
python3.10 -m venv ../.venvs/hercules-python310
source ../.venvs/hercules-python310/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -r PythonClient/requirements-herculesvenv.txt
```

On Apple Silicon, check the architecture of the Unreal editor and native
AirSim plugin together. The Darwin `build.sh` path now defaults to native
`arm64` and no longer opts into Rosetta; use `HERCULES_MAC_ARCH=x86_64` only
when intentionally reproducing an Intel build. Do not silently mix an arm64
editor with an x86_64 plugin. The final Mac acceptance record must include
`uname -m`, `file "$UNREAL_EDITOR"`, and the CMake target architecture.

Docker Desktop networking needs an explicit choice. The checked-in Compose
file uses `network_mode: host`, which is the Linux assumption used by the
reference reports. If the installed Docker Desktop version has optional host
networking, enable it and verify that the container can reach the Mac's AirSim
RPC listeners on ports 41451 and 41452. Otherwise use a bridged/forwarded
configuration and pass `host.docker.internal` as the AirSim host to every ROS
and Python client; do not leave some clients on `127.0.0.1` and call the run
comparable. ROS nodes that all run in the same container may keep
`ROS_LOCALHOST_ONLY=1` and domain 42; host-side ROS tools require a separate
DDS/networking configuration.

Before starting a mission, record the selected path and check connectivity from
the container:

```bash
docker compose -f docker/ros2/compose.yaml config
docker compose -f docker/ros2/compose.yaml up -d --build
docker compose -f docker/ros2/compose.yaml exec dev getent hosts host.docker.internal
docker compose -f docker/ros2/compose.yaml exec dev nc -vz host.docker.internal 41451
docker compose -f docker/ros2/compose.yaml exec dev nc -vz host.docker.internal 41452
```

The launch helpers select platform-specific editor and renderer behavior when
invoked on a matching host. The Linux path remains the validated reference; the
Mac path is a portability path that still needs a fresh editor/plugin, Metal
camera, RPC, and image-format check. A passing image build or TCP probe does not
prove that Metal camera frames have the same format or pose contract as Linux
Vulkan.

## 3. Install exactly Unreal Engine 5.2.1

### Option A: Epic's macOS installed build

1. Install UE **5.2.1** through the Epic Games Launcher on the Mac, or use an
   exact `5.2.1-release` source build. Do not accept an editor upgrade or
   project conversion.
2. Set `UNREAL_EDITOR` to the executable inside the macOS application bundle:

```bash
export UE_ROOT="/Users/Shared/Epic Games/UE_5.2"
export UNREAL_EDITOR="$UE_ROOT/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
test -x "$UNREAL_EDITOR"
"$UNREAL_EDITOR" --version
```

The installed path can differ. Resolve it from the actual UE 5.2.1 app and
keep the engine outside the HERCULES checkout. Generate the Blocks project
files with the Mac script if Xcode files are needed:

```bash
./Unreal/Environments/Blocks/GenerateProjectFiles.sh "$UE_ROOT"
```

The `launch_sim.sh` and `launch_rural_mission_sim.sh` helpers select the Mac
editor bundle on Darwin and skip the Linux NVIDIA/Vulkan variables. If an older
checkout is being used, launch Unreal directly as shown in Section 9. macOS
uses Metal; do not copy `VK_ICD_FILENAMES`, PRIME, or GLX settings from the
Linux instructions.

### Option B: Epic's Linux installed build

1. Sign in on the [Unreal Engine for Linux page](https://www.unrealengine.com/en-US/linux).
2. Select and download the **5.2.1** Linux archive. Do not use a newer 5.2
   hotfix or another UE major/minor version.
3. Extract it to a user-owned location with ample free space. For example:

```bash
mkdir -p "$HOME/Unreal/UE_5.2.1"
unzip "$HOME/Downloads/Linux_Unreal_Engine_5.2.1.zip" \
  -d "$HOME/Unreal/UE_5.2.1"
export UNREAL_EDITOR="$HOME/Unreal/UE_5.2.1/Engine/Binaries/Linux/UnrealEditor"
test -x "$UNREAL_EDITOR"
```

The archive name can differ on Epic's site; adjust only the downloaded path,
not the selected engine version. Confirm **5.2.1** in the editor's About dialog
on the first launch.

### Option C: exact source release

Use this when the website no longer offers the 5.2.1 installed archive. Create
Epic and personal GitHub accounts, connect them on Epic's Apps & Accounts page,
accept the `@EpicGames` GitHub invitation, and then clone the exact release:

```bash
mkdir -p "$HOME/Unreal"
git clone --branch 5.2.1-release --single-branch \
  git@github.com:EpicGames/UnrealEngine.git "$HOME/Unreal/UE_5.2.1"
cd "$HOME/Unreal/UE_5.2.1"
./Setup.sh
./GenerateProjectFiles.sh
make -j"$(nproc)"
export UNREAL_EDITOR="$HOME/Unreal/UE_5.2.1/Engine/Binaries/Linux/UnrealEditor"
test -x "$UNREAL_EDITOR"
```

`Setup.sh` downloads Epic's matching native toolchain. Keep the engine outside
the HERCULES checkout; neither engine source nor binaries belong in this repo.

## 4. Clone this handoff branch

```bash
cd "$HOME"
git clone --branch codex/ros2 --single-branch \
  git@github.com:dcherenson/HERCULES.git
cd HERCULES
git status --short --branch
git log -1 --oneline
```

The working tree should be clean. Generated builds, logs, plots, screenshots,
JSONL, CSV, GIF, and MP4 validation outputs are ignored. Do not force-add files
from `ros2/validation/**/artifacts/`, `ros2/.docker/`, or
`PythonClient/distributed_mission/debug_runs/`.

## 5. Install Docker Engine and Compose

The ROS 2 build runs in Docker; do not install host ROS. On Ubuntu/Linux, these
commands follow Docker's official Ubuntu repository method:

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"
```

Log out and back in after adding the group, then verify:

```bash
docker run --rm hello-world
docker compose version
```

On macOS, install Docker Desktop instead of running the Ubuntu `apt` commands
above. Docker Desktop is sufficient for the image build, clean ROS workspace
build, unit tests, and Python-only/ROS artifact comparison. A native Mac UE
5.2.1 editor can also be used for live RPC/Metal experiments with the
platform-aware launch helpers, but that path remains unvalidated until the
Section 2 architecture, renderer, and networking preflight passes.

The ROS image definition and Compose configuration are colocated at
`docker/ros2/Dockerfile` and `docker/ros2/compose.yaml`. From the repository
root, lifecycle commands are:

```bash
LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)" \
  docker compose -f docker/ros2/compose.yaml up -d --build

docker compose -f docker/ros2/compose.yaml ps
docker compose -f docker/ros2/compose.yaml down
```

The source checkout is bind-mounted at `/workspaces/hercules`. ROS build,
install, log, and test products remain under the ignored `ros2/.docker/` tree.
Do not source a repository-root `install/` directory; it is generated output and
is intentionally not part of this workflow.

## 6. Build the native HERCULES plugin

The simulator runs on the host, not in the ROS container:

```bash
cd "$HOME/HERCULES"
./setup.sh
./build.sh
cd Unreal/Environments/Blocks
./update_from_git.sh
cd ../../..
```

`setup.sh` installs/downloads native dependencies and `build.sh` builds the
AirLib/plugin libraries. `update_from_git.sh` copies the built plugin into the
included Blocks project. Set the engine path in every new terminal:

```bash
export UNREAL_EDITOR="$HOME/Unreal/UE_5.2.1/Engine/Binaries/Linux/UnrealEditor"
```

On macOS, use the executable inside the UE application bundle instead:

```bash
export UE_ROOT="/Users/Shared/Epic Games/UE_5.2"
export UNREAL_EDITOR="$UE_ROOT/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
test -x "$UNREAL_EDITOR"
```

The native plugin must be built for the same architecture as this editor. On
Apple Silicon, check `file "$UNREAL_EDITOR"` and the build output before
launching; the supported default is native arm64 (with an explicit
`HERCULES_MAC_ARCH=x86_64` escape hatch for legacy Intel validation).

If Unreal requests a project rebuild on first open, accept it. Never accept an
engine-version conversion. If it proposes disabling AirSim, keep the plugin
enabled.

## 7. Download the four-folder Unreal content bundle

The large Unreal assets are intentionally not tracked by Git. The handoff bundle
is [UnrealAssets.zip on Google Drive](https://drive.google.com/file/d/1WseWuqjppBw3qaVExCpbEHDIf7ZTkruZ/view?usp=sharing).
Download it in a browser on the Linux or macOS machine; Google Drive may show a
large-file confirmation page.

The ZIP must contain these four directories at its root:

```text
RuralAustralia/
FlyingCPP/
Flying/
Geometry/
```

All four go directly under:

```text
Unreal/Environments/Blocks/Content/
```

The required map must then exist at:

```text
Unreal/Environments/Blocks/Content/RuralAustralia/Maps/RuralAustralia_Example_01.umap
```

**Rural Australia cannot be acquired directly through the Linux UE 5.2.1 setup.**
Epic Games Launcher is required for this marketplace workflow and is available
on Windows and macOS, while the newer in-editor Fab integration requires UE
5.3 or later. This bundle was made on macOS from platform-independent `.uasset`
and `.umap` content and also supplies `FlyingCPP`, `Flying`, and `Geometry`, which
are absent from the current Linux checkout.

Before extraction, confirm that the ZIP has the intended root layout:

```bash
cd "$HOME/Downloads"
for directory in RuralAustralia FlyingCPP Flying Geometry; do
  unzip -Z1 UnrealAssets.zip | grep -q "^${directory}/" || {
    echo "missing ${directory}/ in UnrealAssets.zip" >&2
    exit 1
  }
done
sha256sum UnrealAssets.zip
```

Record the printed SHA-256 alongside your local copy so later transfers can be
compared. Extract only after all four checks pass:

```bash
cd "$HOME/HERCULES/Unreal/Environments/Blocks/Content"
unzip -q "$HOME/Downloads/UnrealAssets.zip"
for directory in RuralAustralia FlyingCPP Flying Geometry; do
  test -d "$directory" || exit 1
done
test -f RuralAustralia/Maps/RuralAustralia_Example_01.umap || exit 1
```

The Drive link is intentionally external to Git. Do not put the ZIP or extracted
assets in a commit; the Blocks project ignores `Content/*`. Anyone redistributing
the bundle remains responsible for the licenses of its included third-party
assets. See [`docs/downloading_hercules_environments.md`](docs/downloading_hercules_environments.md)
for the original marketplace acquisition context.

## 8. Reproduce the clean build and all ROS tests

Run the checked-in reproduction script from the repository root:

```bash
./docker/ros2/reproduce.sh
```

It performs `docker compose up -d --build`, creates a new clean build root,
builds the entire ROS 2 workspace, and runs the full package test gate used by
this work. The gate includes the existing wrapper/control smoke tests,
`hercules_tracking`, Python/C++ tracking parity, `hercules_mission_core`,
Python/C++ mission parity, the deterministic eight-agent regression,
`hercules_tracking_ros` protocol/transport/truth-observer tests, and
`hercules_mission_ros` adapter/contract tests. The test script prints the
authoritative total after every run.

Console output is also retained under the ignored directory:

```text
ros2/validation/reproduction/artifacts/reproduction-<timestamp>.log
```

The earlier nominal-only handoff reported **95 tests, 0 failures, 0 errors,
0 skipped** and **133 passed** in the complete Python distributed-mission suite.
Those are historical figures rather than acceptance results for the new ROS
tracking transport. Treat the new machine's reproduction output as
authoritative.

The distributed-tracking handoff's older Linux record reported a clean
17-package workspace build, **115 ROS tests, 0 failures, 0 errors, 0 skipped**,
and **134 Python distributed-mission tests passed**. The later CBF report records
an offline/replay image run with **129 ROS test results**, **135 Python tests**,
and **2 CBF metrics tests**. These are historical counts from named images and
are not a Mac acceptance result. Re-run the clean gate and record the exact
`HEAD`, image digest, architecture, and test totals before comparing machines.

The optional CBF mode harness is separate from the default reproduction gate:

```bash
./docker/ros2/reproduce_cbf.sh --mode no_cbf --obstacles none --dry-run
```

The driver defaults to `dry_run=true`; this still requires a reachable AirSim
simulator for state/origin discovery. Use `--live` (or `--dry-run false`) only
after the simulator is running. On a Mac without a reachable simulator, use the
clean Docker build/test gate and the Python/C++ replay tests instead of treating
a live CBF run as available. The driver currently uses the launch's default
30-second duration; use `ros2 launch ... duration_sec:=...` directly when a
different duration is required.

The `mestres` and `wang` modes exercise the native filter and their selected
obstacle source. Run `no_cbf` first as the comparison baseline. Perception-backed
CBF runs require a rendered simulator and a separately validated canonical
camera/frame-origin contract; a successful launch alone is not evidence of
Python/ROS perception parity.

## 9. Reproduce the live mission, plots, and video

This is an optional hardware/simulator validation, not part of the deterministic
test gate. The Linux commands below reproduce the reference environment. A
macOS host can run the same mission only after the native editor, Metal image
encoding, RPC host, and Docker Desktop network path pass the Section 2 preflight.
The historical Linux reports and the 500-step comparison are not evidence that
the Mac camera path is equivalent.

First launch visible UE 5.2.1 in terminal 1:

```bash
cd "$HOME/HERCULES"
export UNREAL_EDITOR="$HOME/Unreal/UE_5.2.1/Engine/Binaries/Linux/UnrealEditor"
./docker/ros2/launch_rural_mission_sim.sh
```

On macOS, after the Section 2 preflight, use the platform-aware helper from the
native host. It selects the Mac editor and skips the Linux Vulkan/NVIDIA
variables. Set `UNREAL_EDITOR` and the settings file to absolute paths if the
checkout is not under `$HOME/HERCULES`:

```bash
cd "$HOME/HERCULES"
export UE_ROOT="/Users/Shared/Epic Games/UE_5.2"
export UNREAL_EDITOR="$UE_ROOT/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
export HERCULES_UNREAL_SETTINGS="$PWD/docker/ros2/settings.rural-nominal.json"
./docker/ros2/launch_rural_mission_sim.sh
```

For direct debugging instead of the helper:

```bash
"$UNREAL_EDITOR" "$PWD/Unreal/Environments/Blocks/Blocks.uproject" \
  /Game/RuralAustralia/Maps/RuralAustralia_Example_01 \
  -game -windowed -ResX=960 -ResY=540 -nosplash \
  "-settings=$PWD/docker/ros2/settings.rural-nominal.json"
```

For bridged Docker Desktop networking, set `HERCULES_RPC_BIND_IP=0.0.0.0`
before launching Unreal and pass `host_ip:=host.docker.internal` to the ROS
launch plus `rpc_host:=host.docker.internal` to the direct-pose bridge. Keep one
RPC host value across every ROS and Python client in a comparison run.
The Python orchestrator accepts `--airsim-host host.docker.internal`; the
complete launch must use one host value for every client.

Wait until RuralAustralia is visibly running and AirSim has opened RPC ports
41451 and 41452. In terminal 2 run:

```bash
cd "$HOME/HERCULES"
./docker/ros2/reproduce.sh --with-live-video --duration 30
```

The script reruns the clean deterministic gate, performs the required no-motion
dry-run calibration, resets the simulator between modes, and executes three
live runs: the direct-truth nominal baseline, distributed tracking with seeded
truth observations, and distributed tracking with the production camera/depth
observer. It invokes the existing Python media renderer for every mode and
builds a truth-versus-camera comparison. Outputs are written only beneath:

```text
ros2/validation/reproduction/artifacts/
```

Expected mode directories are `truth_nominal/`, `distributed_truth/`, and
`distributed_camera/`. Each contains `mission.jsonl`, `metrics.json`, trajectory
and error PNGs, plus `topdown.mp4` and `topdown.gif`. The common directory also
contains `dry_run.jsonl`, `mode_comparison.json`, `mode_comparison.png`, and the
timestamped reproduction log. This entire output tree is ignored by Git. The
MP4/GIF are top-down diagnostic animations; the Unreal window is the visible
physics/rendering view.

Camera mode requires a rendered RHI because it calls AirSim
`DepthPerspective`. Do not add `-nullrhi`: in this environment that mode could
run physics-only truth validation but crashed when the first depth frame was
requested. If a noninteractive rendered session is needed, use:

```bash
./docker/ros2/launch_rural_mission_sim.sh \
  -RenderOffscreen -unattended -NoSound -stdout
```

The checked-in Rural settings include the exact executable-source
`target_bottom` camera definition for all five UAVs. The three CPHusky trackers
use their existing `front_center` cameras. See
`ros2/validation/reproduction/README.md` for the latest bounded live metrics.

For a short manual distributed gate without rerunning the clean build, use:

```bash
./docker/ros2/exec.sh ros2 launch hercules_mission_ros rural_nominal.launch.py \
  target_source:=distributed_tracking \
  target_observation_source:=truth duration_sec:=10
```

Change only `target_observation_source:=camera` for the real camera/depth path.
Distributed camera mode intentionally commands zero/stop for missing, inactive,
future, or stale local estimates; there is no Target1 truth fallback.

Stop Unreal with Ctrl-C in terminal 1. Stop the development container when
finished:

```bash
docker compose -f docker/ros2/compose.yaml down
```

For a truth-only headless RPC/physics check, append
`-nullrhi -unattended -NoSound` to the simulator launch. It cannot produce a
visible Unreal screenshot and must not be used for camera observation. The
logged top-down animation can still be generated from a truth-only run.

## 10. Useful manual commands

```bash
./docker/ros2/build.sh
./docker/ros2/build_ws.sh
./docker/ros2/test.sh
./docker/ros2/shell.sh
./docker/ros2/reproduce_cbf.sh --mode no_cbf --obstacles none --observation truth
./docker/ros2/reproduce_cbf.sh --mode mestres --obstacles none --observation truth --live
./docker/ros2/reproduce_cbf.sh --mode wang --obstacles perception --observation camera --live
```

The CBF driver defaults to a dry-run and a 30-second mission. `--live` (or
`--dry-run false`) enables actuation; `--obstacles perception` requires the
rendered sensor path. Its logs and metadata belong under the ignored
`ros2/validation/cbf/artifacts/` directory. For the exact Python/C++ numerical
contract, run the native parity test in the clean container; for old-versus-new
mission behavior, compare separately reset JSONL runs and retain the map,
settings, architecture, branch HEAD, and actuation profile in the report.

Mission-specific implementation and validation details are in:

- [`ros2/src/hercules_mission_core/README.md`](ros2/src/hercules_mission_core/README.md)
- [`ros2/src/hercules_mission_ros/README.md`](ros2/src/hercules_mission_ros/README.md)
- [`ros2/src/hercules_tracking_ros/README.md`](ros2/src/hercules_tracking_ros/README.md)
- [`ros2/validation/reproduction/README.md`](ros2/validation/reproduction/README.md)
- [`ros2/validation/rural_nominal/REPORT.md`](ros2/validation/rural_nominal/REPORT.md)
- [`ros2/docs/mission_artifact_compatibility.md`](ros2/docs/mission_artifact_compatibility.md)

The nominal mission fixture is
`ros2/src/hercules_mission_core/config/rural_target_tracking.yaml`. The frozen
Python reference is commit
`2ef27d8ddc7f019f4b1af0196f7fbb4d91ce595f`. Where old prose disagreed with
that executable Python mission, this branch deliberately uses executable-source
behavior: the Rural target translation is 5 m toward the robot launch point,
target covariance visualization uses a 2-sigma multiplier, and the tested Rural
target speed is explicitly 0.10 m/s even though the generic parser default is
0.5 m/s.

## 11. Continuation boundary

Do not infer that the current stage implements the eventual complete tracking
experiment. Distributed estimation, the existing camera/depth observation path,
and the optional CBF core/ROS integration are present. The concrete CBF files
are under `ros2/src/hercules_cbf`, `ros2/src/hercules_cbf_ros`, and
`ros2/src/hercules_interfaces/msg`; the native OSQP wrapper is currently
implemented in `hercules_cbf/src/filter.cpp`, not as a separate `osqp_solver.*`
file. CBF is disabled by default, and its perception-backed path is not yet a
validated macOS/Metal Python/ROS parity experiment because capture pose/origin,
RPC networking, and image encoding still require a fresh platform check.
Conformal prediction, cooperative localization, and a normalized
safety-performance claim remain out of scope. Preserve the pure C++ numerical
behavior and its parity tests. Do not move plotting/video code into production
ROS packages; preserve the documented JSONL compatibility boundary.
