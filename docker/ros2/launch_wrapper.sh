#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
inside_or_exec "$@"
source "$TOOL_DIR/env.sh"
exec ros2 launch airsim_ros_pkgs hercules_host.launch.py "$@"
