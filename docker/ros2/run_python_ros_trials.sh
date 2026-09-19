#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${HERCULES_COMPARE_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/../.venvs/hercules-python310/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/../.venvs/hercules-python310/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || true)"
  fi
fi
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
  echo "No usable Python 3 interpreter found; set HERCULES_COMPARE_PYTHON." >&2
  exit 1
}
exec "$PYTHON_BIN" "$REPO_ROOT/ros2/validation/python_ros/run_trials.py" "$@"
