#!/usr/bin/env bash
# Source this file; it does not alter the caller's shell options.
HERCULES_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export BUILD_TYPE="${BUILD_TYPE:-RelWithDebInfo}"
case "$BUILD_TYPE" in Release|RelWithDebInfo|Debug) ;; *) echo "Invalid BUILD_TYPE: $BUILD_TYPE" >&2; return 2 ;; esac
export HERCULES_BUILD_ROOT="${HERCULES_BUILD_ROOT:-$HERCULES_ROOT/ros2/.docker/humble/$BUILD_TYPE}"
# ROS setup scripts are not nounset-safe.
_hercules_nounset=false
[[ $- == *u* ]] && _hercules_nounset=true && set +u
source /opt/ros/humble/setup.bash
if [[ ${HERCULES_SOURCE_OVERLAY:-1} == 1 && -f "$HERCULES_BUILD_ROOT/.build-complete" ]]; then
  source "$HERCULES_BUILD_ROOT/install/setup.bash"
fi
if $_hercules_nounset; then set -u; fi
unset _hercules_nounset
