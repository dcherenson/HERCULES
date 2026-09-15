#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
inside_or_exec "$@"
source "$TOOL_DIR/env.sh"
exec python3 "$TOOL_DIR/benchmark.py" "$@"
