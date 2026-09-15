#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
inside_or_exec "$@"
source "$TOOL_DIR/env.sh"
vehicle="${1:-drone}"
mode="${2:-observe}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
exec ros2 run hercules_control smoke_node --ros-args -p "vehicle_type:=$vehicle" -p "mode:=$mode" "$@"
