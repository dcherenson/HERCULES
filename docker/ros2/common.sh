#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$TOOL_DIR/../.." && pwd)"
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
compose() { docker compose -f "$TOOL_DIR/compose.yaml" "$@"; }
ensure_dev() {
  # Never recreate a busy container just because an image was rebuilt.
  if [[ -z "$(compose ps -q dev)" ]]; then compose up -d dev; fi
}
dev_exec() {
  local options=(-i -e "BUILD_TYPE=${BUILD_TYPE:-RelWithDebInfo}" -e "BUILD_JOBS=${BUILD_JOBS:-2}")
  if [[ -t 0 && -t 1 ]]; then options+=(-t); fi
  if [[ -n ${HERCULES_BUILD_ROOT:-} ]]; then options+=(-e "HERCULES_BUILD_ROOT=$HERCULES_BUILD_ROOT"); fi
  options+=(-e "HERCULES_SOURCE_OVERLAY=${HERCULES_SOURCE_OVERLAY:-1}")
  # Some Compose versions select a one-off build container for `exec dev`.
  # `ps -q` excludes one-off containers and identifies the persistent service.
  docker exec "${options[@]}" "$(compose ps -q dev)" "$@"
}
inside_or_exec() {
  if [[ ! -f /.dockerenv ]]; then
    ensure_dev
    dev_exec "$TOOL_DIR_IN_CONTAINER/$(basename "$0")" "$@"
    exit $?
  fi
}
TOOL_DIR_IN_CONTAINER=/workspaces/hercules/docker/ros2
