#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$TOOL_DIR/../.." && pwd)"
MODE=""; OBSTACLES="none"; OBSERVATION="camera"; OUTPUT_DIR="$REPO_ROOT/ros2/validation/cbf/artifacts"; DRY_RUN="true"; RECORD_VIDEO="false"; DURATION="30"; RUN_ID="${HERCULES_RUN_ID:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --obstacles) OBSTACLES="$2"; shift 2;;
    --observation) OBSERVATION="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --duration) DURATION="$2"; shift 2;;
    --record-video) RECORD_VIDEO="true"; shift;;
    --dry-run) DRY_RUN="true"; if [[ $# -gt 1 && ( "$2" == "true" || "$2" == "false" ) ]]; then DRY_RUN="$2"; shift 2; else shift; fi;;
    --live) DRY_RUN="false"; shift;;
    -h|--help) echo "usage: $0 --mode no_cbf|mestres|wang [--obstacles none|perception] [--observation truth|camera] [--duration SEC] [--record-video] [--dry-run [true|false]|--live]"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
case "$MODE" in no_cbf|mestres|wang) ;; *) echo "--mode is required" >&2; exit 2;; esac
case "$OBSTACLES" in none|perception) ;; truth) echo "truth obstacles require an explicit fixture; use --obstacles none or perception" >&2; exit 2;; *) echo "invalid --obstacles" >&2; exit 2;; esac
case "$OBSERVATION" in truth|camera) ;; *) echo "invalid --observation" >&2; exit 2;; esac
[[ "$DURATION" =~ ^[1-9][0-9]*([.][0-9]+)?$ ]] || { echo "--duration must be positive" >&2; exit 2; }
case "$MODE" in
  no_cbf) CONFIG="$REPO_ROOT/ros2/src/hercules_mission_ros/config/rural_tracking_no_cbf.yaml";;
  mestres) CONFIG="$REPO_ROOT/ros2/src/hercules_mission_ros/config/rural_tracking_mestres.yaml";;
  wang) CONFIG="$REPO_ROOT/ros2/src/hercules_mission_ros/config/rural_tracking_wang.yaml";;
esac
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_DIR="$OUTPUT_DIR/$MODE-$RUN_ID"
mkdir -p "$RUN_DIR"
if [[ ! -f "$RUN_DIR/config.yaml" ]]; then cp "$CONFIG" "$RUN_DIR/config.yaml"; fi
if [[ ! -f "$RUN_DIR/git_head.txt" ]]; then git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_head.txt"; fi
if [[ ! -f "$RUN_DIR/metadata.json" ]]; then cat > "$RUN_DIR/metadata.json" <<EOF
{
  "mode": "$MODE",
  "obstacles": "$OBSTACLES",
  "observation": "$OBSERVATION",
  "dry_run": $DRY_RUN,
  "record_video": $RECORD_VIDEO,
  "duration_sec": $DURATION,
  "run_id": "$RUN_ID",
  "config": "$CONFIG"
}
EOF
fi
source "$TOOL_DIR/common.sh"
if [[ ! -f /.dockerenv ]]; then
  ensure_dev
  export HERCULES_RUN_ID="$RUN_ID"
  # Translate a repository-local host output path to the bind-mounted
  # container path before re-executing.  Relative paths already resolve from
  # the Compose working directory and are left untouched.
  forwarded_args=("$@")
  for ((index = 0; index < ${#forwarded_args[@]}; index++)); do
    if [[ "${forwarded_args[index]}" == "--output-dir" && $((index + 1)) -lt ${#forwarded_args[@]} ]]; then
      value="${forwarded_args[index + 1]}"
      case "$value" in
        "$REPO_ROOT"/*)
          forwarded_args[index + 1]="/workspaces/hercules/${value#"$REPO_ROOT/"}"
          ;;
      esac
    fi
  done
  exec dev_exec "$TOOL_DIR_IN_CONTAINER/reproduce_cbf.sh" "${forwarded_args[@]}"
fi
source "$TOOL_DIR/env.sh"
if [[ "$DRY_RUN" == "false" ]]; then
  # Reset only the simulator owned by this run, then stage the ROS mission
  # from the same settings/ports.  Reset failures are fatal instead of
  # silently comparing different initial states.
  python3 - "$AIRSIM_HOST" "$AIRSIM_MULTIROTOR_PORT" <<'PY'
import sys
import socket
import hercules_cosysairsim as airsim
host, port = sys.argv[1], int(sys.argv[2])
if host in {"host.docker.internal", "docker.for.mac.host.internal"}:
    try:
        addresses = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
        if addresses:
            host = addresses[0][4][0]
    except OSError:
        pass
client = airsim.MultirotorClient(ip=host, port=port)
client.confirmConnection()
client.reset()
PY
fi
LAUNCH_ARGS=(
  dry_run:="$DRY_RUN" target_source:=distributed_tracking \
  target_observation_source:="$OBSERVATION" cbf_obstacle_source:="$OBSTACLES" \
  truth_rng_mode:=python_global \
  cbf_enabled:="$([[ "$MODE" == no_cbf ]] && echo false || echo true)" \
  cbf_method:="$([[ "$MODE" == wang ]] && echo wang || echo mestres)" \
  actuation_profile:=python_cbf mission_config:="$CONFIG" \
  enable_collision_observer:=true airsim_host:="${AIRSIM_HOST:-127.0.0.1}" \
  drone_port:="${AIRSIM_MULTIROTOR_PORT:-41451}" ugv_port:="${AIRSIM_CAR_PORT:-41452}" \
  duration_sec:="$DURATION" log_path:="$RUN_DIR/mission.jsonl"
)
if [[ "$RECORD_VIDEO" == "true" ]]; then
  LAUNCH_ARGS+=(record_video:=true video_output_dir:="$RUN_DIR/media" video_staging_dir:="$RUN_DIR/recording_frames")
fi
ros2 launch hercules_mission_ros rural_nominal.launch.py "${LAUNCH_ARGS[@]}"

# Postprocess in the same run directory.  Rendering is best-effort for dry
# runs without media but a requested recording must produce a manifest.
if [[ -s "$RUN_DIR/mission.jsonl" ]]; then
  render_args=(--log "$RUN_DIR/mission.jsonl" --output-dir "$RUN_DIR/rendered")
  [[ -d "$RUN_DIR/recording_frames" ]] && render_args+=(--staging-dir "$RUN_DIR/recording_frames")
  python3 /workspaces/hercules/ros2/src/hercules_mission_ros/scripts/render_ros_media.py "${render_args[@]}" || {
    if [[ "$RECORD_VIDEO" == "true" ]]; then exit 1; fi
  }
fi
