#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
COMPARATOR="${REPO_ROOT}/ros2/validation/python_ros/compare_python_ros.py"

if [[ ! -f "${COMPARATOR}" ]]; then
  echo "Comparator not found: ${COMPARATOR}" >&2
  exit 1
fi

if [[ -n "${HERCULES_COMPARE_PYTHON:-}" ]]; then
  PYTHON_BIN="${HERCULES_COMPARE_PYTHON}"
elif [[ -x "${REPO_ROOT}/../.venvs/hercules-python310/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/../.venvs/hercules-python310/bin/python"
elif [[ -x "${REPO_ROOT}/herculesvenv/bin/python3" ]]; then
  PYTHON_BIN="${REPO_ROOT}/herculesvenv/bin/python3"
else
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "No executable Python 3 interpreter found. Set HERCULES_COMPARE_PYTHON." >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${COMPARATOR}" "$@"
