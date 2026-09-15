#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
inside_or_exec "$@"
source "$TOOL_DIR/env.sh"
colcon --log-base "$HERCULES_BUILD_ROOT/log" test \
  --base-paths "$REPO_ROOT/ros2/src" --build-base "$HERCULES_BUILD_ROOT/build" \
  --install-base "$HERCULES_BUILD_ROOT/install" --executor sequential \
  --packages-select airsim_ros_pkgs hercules_cbf hercules_cbf_ros hercules_control hercules_tracking hercules_tracking_ros hercules_mission_core hercules_mission_ros --event-handlers console_direct+ "$@"
colcon --log-base "$HERCULES_BUILD_ROOT/log" test-result --test-result-base "$HERCULES_BUILD_ROOT/build" --verbose
