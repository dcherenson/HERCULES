#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$TOOL_DIR/../.." && pwd)"
MODE=""; OBSTACLES="none"; OBSERVATION="camera"; OUTPUT_DIR="$REPO_ROOT/ros2/validation/cbf/artifacts"; DRY_RUN="true"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --obstacles) OBSTACLES="$2"; shift 2;;
    --observation) OBSERVATION="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --dry-run) DRY_RUN="true"; if [[ $# -gt 1 && ( "$2" == "true" || "$2" == "false" ) ]]; then DRY_RUN="$2"; shift 2; else shift; fi;;
    --live) DRY_RUN="false"; shift;;
    -h|--help) echo "usage: $0 --mode no_cbf|mestres|wang [--obstacles none|truth|perception] [--observation truth|camera] [--dry-run [true|false]|--live]"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
case "$MODE" in no_cbf|mestres|wang) ;; *) echo "--mode is required" >&2; exit 2;; esac
case "$OBSTACLES" in none|truth|perception) ;; *) echo "invalid --obstacles" >&2; exit 2;; esac
case "$OBSERVATION" in truth|camera) ;; *) echo "invalid --observation" >&2; exit 2;; esac
case "$MODE" in
  no_cbf) CONFIG="$REPO_ROOT/ros2/src/hercules_mission_ros/config/rural_tracking_no_cbf.yaml";;
  mestres) CONFIG="$REPO_ROOT/ros2/src/hercules_mission_ros/config/rural_tracking_mestres.yaml";;
  wang) CONFIG="$REPO_ROOT/ros2/src/hercules_mission_ros/config/rural_tracking_wang.yaml";;
esac
RUN_DIR="$OUTPUT_DIR/$MODE-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
cp "$CONFIG" "$RUN_DIR/config.yaml"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_head.txt"
cat > "$RUN_DIR/metadata.json" <<EOF
{
  "mode": "$MODE",
  "obstacles": "$OBSTACLES",
  "observation": "$OBSERVATION",
  "dry_run": $DRY_RUN,
  "config": "$CONFIG"
}
EOF
source "$TOOL_DIR/common.sh"
if [[ ! -f /.dockerenv ]]; then
  ensure_dev
  exec dev_exec "$TOOL_DIR_IN_CONTAINER/reproduce_cbf.sh" "$@"
fi
source "$TOOL_DIR/env.sh"
ros2 launch hercules_mission_ros rural_nominal.launch.py \
  dry_run:="$DRY_RUN" target_source:=distributed_tracking \
  target_observation_source:="$OBSERVATION" cbf_obstacle_source:="$OBSTACLES" \
  cbf_enabled:="$([[ "$MODE" == no_cbf ]] && echo false || echo true)" \
  cbf_method:="$([[ "$MODE" == wang ]] && echo wang || echo mestres)" \
  actuation_profile:=python_cbf mission_config:="$CONFIG" \
  enable_collision_observer:=true log_path:="$RUN_DIR/mission.jsonl"
