#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
inside_or_exec "$@"
HERCULES_SOURCE_OVERLAY=0 source "$TOOL_DIR/env.sh"
"$TOOL_DIR/bootstrap.sh"
# The checkout is bind-mounted from macOS and may appear with a different
# numeric owner inside the container.  Allow setuptools_scm/ament's git
# introspection to inspect this explicitly trusted workspace.
git config --global --add safe.directory "$REPO_ROOT" 2>/dev/null || true
export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-2}"
export MAKEFLAGS="-j$CMAKE_BUILD_PARALLEL_LEVEL"
mkdir -p "$HERCULES_BUILD_ROOT"
rm -f "$HERCULES_BUILD_ROOT/.build-complete"
colcon --log-base "$HERCULES_BUILD_ROOT/log" build \
  --base-paths "$REPO_ROOT/ros2/src" --build-base "$HERCULES_BUILD_ROOT/build" \
  --install-base "$HERCULES_BUILD_ROOT/install" --executor sequential \
  --event-handlers console_direct+ --cmake-args \
  "-DCMAKE_BUILD_TYPE=$BUILD_TYPE" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "$@"
touch "$HERCULES_BUILD_ROOT/.build-complete"
