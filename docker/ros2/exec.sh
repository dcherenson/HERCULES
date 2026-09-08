#!/usr/bin/env bash
# Run a noninteractive command with the container's ROS overlay sourced.
source "$(dirname -- "$0")/common.sh"
ensure_dev
dev_exec bash -c 'source /workspaces/hercules/docker/ros2/env.sh; exec "$@"' bash "$@"
