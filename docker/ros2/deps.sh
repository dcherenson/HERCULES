#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
inside_or_exec "$@"
source "$TOOL_DIR/env.sh"
rosdep update --rosdistro humble
sudo apt-get update
rosdep install --from-paths "$REPO_ROOT/ros2/src" --ignore-src --rosdistro humble -y
"$TOOL_DIR/bootstrap.sh"
