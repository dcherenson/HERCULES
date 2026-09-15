# ROS 2 RuralAustralia mission handoff

This is the setup and reproduction guide for continuing the work on a second
Linux workstation. Use the `codex/ros2` branch. Do not substitute `main`, and do
not upgrade Unreal Engine: this project is pinned to **Unreal Engine 5.2.1**.

The implemented stage runs Target1 on the ported Gerono figure eight and can
drive the nominal target-centered formation for Drone1, Drone2, SimpleFlight,
Drone4, Drone5, Husky1, Husky2, and Husky3 from either direct truth or each
agent's own distributed target estimate. The default live launch uses the
existing Python camera/depth observer and eight independent C++ tracker
processes. CBF, conformal prediction, and live safety filtering remain
intentionally out of scope.

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

## 2. Install exactly Unreal Engine 5.2.1

### Option A: Epic's Linux installed build

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

### Option B: exact source release

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

## 3. Clone this handoff branch

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

## 4. Install Docker Engine and Compose

The ROS 2 build runs in Docker; do not install host ROS. These commands follow
Docker's official Ubuntu repository method:

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

## 5. Build the native HERCULES plugin

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

If Unreal requests a project rebuild on first open, accept it. Never accept an
engine-version conversion. If it proposes disabling AirSim, keep the plugin
enabled.

## 6. Download the four-folder Unreal content bundle

The large Unreal assets are intentionally not tracked by Git. The handoff bundle
is [UnrealAssets.zip on Google Drive](https://drive.google.com/file/d/1WseWuqjppBw3qaVExCpbEHDIf7ZTkruZ/view?usp=sharing).
Download it in a browser on the Linux machine; Google Drive may show a large-file
confirmation page.

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

## 7. Reproduce the clean build and all ROS tests

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

The distributed-tracking handoff was last verified with a clean 17-package
workspace build, **115 ROS tests, 0 failures, 0 errors, 0 skipped**, and
**134 Python distributed-mission tests passed**. Re-run the script rather than
assuming these counts remain valid after future edits.

## 8. Reproduce the live mission, plots, and video

This is an optional hardware/simulator validation, not part of the deterministic
test gate. First launch visible UE 5.2.1 in terminal 1:

```bash
cd "$HOME/HERCULES"
export UNREAL_EDITOR="$HOME/Unreal/UE_5.2.1/Engine/Binaries/Linux/UnrealEditor"
./docker/ros2/launch_rural_mission_sim.sh
```

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

## 9. Useful manual commands

```bash
./docker/ros2/build.sh
./docker/ros2/build_ws.sh
./docker/ros2/test.sh
./docker/ros2/shell.sh
```

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

## 10. Continuation boundary

Do not infer that the current stage implements the eventual complete tracking
experiment. Distributed estimation and the existing camera/depth observation
path are integrated, but CBF, conformal prediction, obstacle perception,
cooperative localization, and the final safety-filtered mission are not.
Preserve the pure C++ numerical behavior and its parity tests. Do not move
plotting/video code into production ROS packages; preserve the documented JSONL
compatibility boundary.
