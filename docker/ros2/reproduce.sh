#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/compose.yaml"
ARTIFACT_DIR="$REPO_ROOT/ros2/validation/reproduction/artifacts"
WITH_LIVE_VIDEO=false
MISSION_DURATION=30

usage() {
  cat <<'EOF'
Usage: ./docker/ros2/reproduce.sh [--with-live-video] [--duration SECONDS]

Builds the Docker image, performs a clean ROS 2 workspace build, and runs the
complete ROS test gate. With --with-live-video, a UE 5.2.1 RuralAustralia
simulator must already be running; the script then performs a dry-run preflight,
runs the truth-target baseline, distributed truth-observation gate, and final
distributed camera-observation mission. It renders PNG, MP4, GIF, JSON, JSONL,
and text output under the Git-ignored
ros2/validation/reproduction/artifacts directory.

For chase and FPV streams, start Unreal with a settings file produced by
`prepare_rural_video_settings.py` and set `HERCULES_UNREAL_SETTINGS` before
running this script.

Camera mode requires a rendered Unreal session. Do not launch with -nullrhi;
use the ordinary visible launch or -RenderOffscreen.
EOF
}

while (($#)); do
  case "$1" in
    --with-live-video)
      WITH_LIVE_VIDEO=true
      shift
      ;;
    --duration)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      MISSION_DURATION="$2"
      [[ "$MISSION_DURATION" =~ ^[1-9][0-9]*$ ]] || {
        echo "--duration must be a positive integer" >&2
        exit 2
      }
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$ARTIFACT_DIR"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"
export BUILD_TYPE="${BUILD_TYPE:-RelWithDebInfo}"
export BUILD_JOBS="${BUILD_JOBS:-2}"
export HERCULES_BUILD_ROOT="/workspaces/hercules/ros2/.docker/humble/reproduction/$RUN_ID"
RESULT_LOG="$ARTIFACT_DIR/reproduction-$RUN_ID.log"

run_live_stage() {
  local timeout_seconds="$1"
  shift
  local status=0
  timeout --foreground --signal=INT --kill-after=15 "${timeout_seconds}s" \
    "$SCRIPT_DIR/exec.sh" ros2 launch hercules_mission_ros rural_nominal.launch.py "$@" || status=$?
  # Signals delivered through `docker exec` do not reliably reap every child
  # launched by ROS 2. Recreate the disposable development service after each
  # live stage so wrappers, observers, and trackers cannot leak into the next
  # stage and contend for the AirSim RPC server.
  docker compose -f "$COMPOSE_FILE" restart dev >/dev/null
  # The mission node completes and safely stops, but the included wrapper nodes
  # remain alive until timeout interrupts the enclosing launch.
  if [[ $status -ne 0 && $status -ne 124 && $status -ne 130 ]]; then
    return "$status"
  fi
}

reset_simulator() {
  "$SCRIPT_DIR/exec.sh" env PYTHONPATH=/workspaces/hercules/PythonClient \
    python3 -c 'import hercules_cosysairsim as airsim; client = airsim.MultirotorClient(port=41451); client.confirmConnection(); client.reset()'
  sleep 2
}

main() {
  cd "$REPO_ROOT"
  echo "Reproduction run: $RUN_ID"
  echo "Clean build root: $HERCULES_BUILD_ROOT"
  docker compose -f "$COMPOSE_FILE" up -d --build dev
  "$SCRIPT_DIR/build_ws.sh"
  "$SCRIPT_DIR/test.sh"
  "$SCRIPT_DIR/exec.sh" env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    python3 -m pytest /workspaces/hercules/PythonClient/distributed_mission/tests

  if ! $WITH_LIVE_VIDEO; then
    echo "Deterministic build and test reproduction complete."
    echo "For live mission media, start UE 5.2.1 and rerun with --with-live-video."
    return
  fi

  for port in 41451 41452; do
    if ! timeout 2 bash -c "</dev/tcp/127.0.0.1/$port" 2>/dev/null; then
      echo "AirSim RPC port $port is unavailable." >&2
      echo "Start the RuralAustralia simulator first; see ROS2_HANDOFF.md." >&2
      return 1
    fi
  done

  local container_artifacts=/workspaces/hercules/ros2/validation/reproduction/artifacts
  mkdir -p "$ARTIFACT_DIR/truth_nominal" \
    "$ARTIFACT_DIR/distributed_truth" "$ARTIFACT_DIR/distributed_camera"
  run_live_stage 50 dry_run:=true duration_sec:=5 \
    target_source:=truth \
    "log_path:=$container_artifacts/dry_run.jsonl"
  [[ "$(wc -l < "$ARTIFACT_DIR/dry_run.jsonl")" -ge 40 ]] || {
    echo "dry-run preflight ended before 40 records" >&2
    return 1
  }
  reset_simulator
  run_live_stage "$((MISSION_DURATION + 45))" dry_run:=false \
    enable_target:=true enable_formation:=true \
    target_source:=truth \
    record_video:=true \
    "video_output_dir:=$container_artifacts/truth_nominal/media" \
    "video_staging_dir:=$container_artifacts/truth_nominal/recording_frames" \
    "duration_sec:=$MISSION_DURATION" \
    "log_path:=$container_artifacts/truth_nominal/mission.jsonl"
  reset_simulator
  run_live_stage "$((MISSION_DURATION + 45))" dry_run:=false \
    enable_target:=true enable_formation:=true \
    target_source:=distributed_tracking target_observation_source:=truth \
    record_video:=true \
    "video_output_dir:=$container_artifacts/distributed_truth/media" \
    "video_staging_dir:=$container_artifacts/distributed_truth/recording_frames" \
    "duration_sec:=$MISSION_DURATION" \
    "log_path:=$container_artifacts/distributed_truth/mission.jsonl"
  reset_simulator
  run_live_stage "$((MISSION_DURATION + 45))" dry_run:=false \
    enable_target:=true enable_formation:=true \
    target_source:=distributed_tracking target_observation_source:=camera \
    record_video:=true \
    "video_output_dir:=$container_artifacts/distributed_camera/media" \
    "video_staging_dir:=$container_artifacts/distributed_camera/recording_frames" \
    "duration_sec:=$MISSION_DURATION" \
    "log_path:=$container_artifacts/distributed_camera/mission.jsonl"

  local mode
  local minimum_records=$((MISSION_DURATION * 8))
  for mode in truth_nominal distributed_truth distributed_camera; do
    [[ -s "$ARTIFACT_DIR/$mode/mission.jsonl" ]] || {
      echo "$mode did not produce a non-empty mission.jsonl" >&2
      return 1
    }
    local records
    records="$(wc -l < "$ARTIFACT_DIR/$mode/mission.jsonl")"
    [[ "$records" -ge "$minimum_records" ]] || {
      echo "$mode ended early: $records records, expected at least $minimum_records" >&2
      return 1
    }
    "$SCRIPT_DIR/exec.sh" python3 \
      /workspaces/hercules/ros2/validation/rural_nominal/render_validation.py \
      "$container_artifacts/$mode/mission.jsonl" \
      --output-dir "$container_artifacts/$mode"
  done
  "$SCRIPT_DIR/exec.sh" python3 \
    /workspaces/hercules/ros2/validation/rural_nominal/compare_tracking_modes.py \
    "$container_artifacts/truth_nominal/mission.jsonl" \
    "$container_artifacts/distributed_camera/mission.jsonl" \
    --output-dir "$container_artifacts"
  echo "Live media written to $ARTIFACT_DIR"
}

main 2>&1 | tee "$RESULT_LOG"
