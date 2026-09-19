#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_UNREAL_EDITOR="/Users/Shared/Epic Games/UE_5.2/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"

CHECK_RPC=0
CHECK_DOCKER=1
FAILURES=0

usage() {
  cat <<'EOF'
Usage: ./docker/ros2/preflight_macos.sh [--check-rpc] [--no-docker]

Check the native Apple Silicon host before starting Unreal plus the ROS 2
Humble Docker workspace. The default checks do not require Unreal to be
running; --check-rpc additionally probes ports 41451 and 41452.

Environment:
  UNREAL_EDITOR          Native UnrealEditor executable override.
  HERCULES_UNREAL_EDITOR Same override when UNREAL_EDITOR is unset.
  HERCULES_UNREAL_SETTINGS  AirSim settings JSON to validate.
  HERCULES_PYTHON_VENV   Python 3.10 venv directory override (default:
                         ../.venvs/hercules-python310 next to this checkout).
  HERCULES_RPC_HOST      Hostname/IP used by the optional RPC probe
                         (default: 127.0.0.1).
  HERCULES_RPC_BIND_IP   AirSim bind address expected by the launch scripts
                         (default: 127.0.0.1).
  AIRSIM_MULTIROTOR_PORT / AIRSIM_CAR_PORT
                         RPC ports used by the optional probe (defaults:
                         41451 / 41452).
EOF
}

ok() {
  printf 'OK   %s\n' "$*"
}

warn() {
  printf 'WARN %s\n' "$*"
}

fail() {
  printf 'FAIL %s\n' "$*" >&2
  FAILURES=$((FAILURES + 1))
}

require_command() {
  local command_name="$1"
  if command -v "$command_name" >/dev/null 2>&1; then
    ok "$command_name: $(command -v "$command_name")"
  else
    fail "missing command: $command_name"
  fi
}

require_formula() {
  local formula="$1"
  if brew list --formula "$formula" >/dev/null 2>&1; then
    ok "Homebrew formula: $formula"
  else
    fail "missing Homebrew formula: $formula"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-rpc)
      CHECK_RPC=1
      shift
      ;;
    --no-docker)
      CHECK_DOCKER=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $(uname -s) != Darwin ]]; then
  fail "this preflight is for macOS; detected $(uname -s)"
fi
if [[ $(uname -m) == arm64 ]]; then
  ok "native Apple Silicon process: $(uname -m)"
else
  fail "expected native arm64; detected $(uname -m) (do not run the ROS/Unreal comparison under Rosetta)"
fi

if command -v brew >/dev/null 2>&1; then
  ok "Homebrew: $(brew --prefix)"
  require_formula cmake
  require_formula coreutils
  require_formula ffmpeg
  require_formula python@3.10
  if brew list --formula llvm@18 >/dev/null 2>&1; then
    ok 'Homebrew formula: llvm@18'
  elif brew list --formula llvm >/dev/null 2>&1; then
    warn 'Homebrew formula llvm@18 is absent; the build will fall back to unversioned llvm'
  else
    fail 'missing Homebrew formula: llvm@18 (or llvm)'
  fi
else
  fail 'Homebrew is not installed or is not on PATH'
fi

for command_name in cmake ffmpeg ffprobe greadlink nproc rsync python3; do
  require_command "$command_name"
done

UNREAL_EDITOR="${UNREAL_EDITOR:-${HERCULES_UNREAL_EDITOR:-$DEFAULT_UNREAL_EDITOR}}"
UPROJECT="$REPO_ROOT/Unreal/Environments/Blocks/Blocks.uproject"
UNREAL_SETTINGS="${HERCULES_UNREAL_SETTINGS:-$SCRIPT_DIR/settings.rural-nominal.json}"
HERO_SETTINGS="$SCRIPT_DIR/settings.hero-smoke.json"
AIRSIM_MULTIROTOR_PORT="${AIRSIM_MULTIROTOR_PORT:-41451}"
AIRSIM_CAR_PORT="${AIRSIM_CAR_PORT:-41452}"

if [[ -x "$UNREAL_EDITOR" ]]; then
  ok "UnrealEditor: $UNREAL_EDITOR"
  if command -v file >/dev/null 2>&1; then
    editor_file_info="$(file "$UNREAL_EDITOR")"
    if [[ "$editor_file_info" == *arm64* ]]; then
      ok "UnrealEditor contains arm64 code"
    else
      fail "UnrealEditor is not arm64: $editor_file_info"
    fi
  fi
else
  fail "UnrealEditor not found or not executable: $UNREAL_EDITOR"
fi
[[ -f "$UPROJECT" ]] && ok "Blocks project: $UPROJECT" || fail "missing Blocks project: $UPROJECT"
if [[ -x "$UNREAL_EDITOR" ]]; then
  unreal_engine_dir="$(cd -- "$(dirname -- "$(dirname -- "$(dirname -- "$(dirname -- "$(dirname -- "$(dirname -- "$UNREAL_EDITOR")")")")")")" && pwd)"
  build_version="$unreal_engine_dir/Build/Build.version"
  if [[ -f "$build_version" ]]; then
    if python3 - "$build_version" <<'PY'
import json
import pathlib
import sys

with pathlib.Path(sys.argv[1]).open(encoding="utf-8") as handle:
    version = json.load(handle)
major = int(version.get("MajorVersion", -1))
minor = int(version.get("MinorVersion", -1))
patch = int(version.get("PatchVersion", -1))
if (major, minor, patch) != (5, 2, 1):
    raise SystemExit(f"expected Unreal 5.2.1, found {major}.{minor}.{patch}")
PY
    then
      ok 'Unreal Engine version: 5.2.1'
    else
      fail "Unreal Engine version is not 5.2.1: $build_version"
    fi
  else
    fail "Unreal Engine Build.version not found: $build_version"
  fi
fi
PLUGIN_UPROJECT="$REPO_ROOT/Unreal/Environments/Blocks/Plugins/AirSim/AirSim.uplugin"
RURAL_MAP="$REPO_ROOT/Unreal/Environments/Blocks/Content/RuralAustralia/Maps/RuralAustralia_Example_01.umap"
[[ -f "$PLUGIN_UPROJECT" ]] && ok "AirSim plugin descriptor: $PLUGIN_UPROJECT" || fail "missing AirSim plugin descriptor: $PLUGIN_UPROJECT"
[[ -f "$RURAL_MAP" ]] && ok "Rural Australia map: $RURAL_MAP" || fail "missing Rural Australia map: $RURAL_MAP"
PLUGIN_BINARY="$REPO_ROOT/Unreal/Environments/Blocks/Plugins/AirSim/Binaries/Mac/UnrealEditor-AirSim.dylib"
if [[ -f "$PLUGIN_BINARY" ]]; then
  if command -v file >/dev/null 2>&1 && [[ "$(file "$PLUGIN_BINARY")" == *arm64* ]]; then
    ok 'native arm64 AirSim plugin binary is present'
  elif command -v file >/dev/null 2>&1; then
    fail "AirSim plugin binary is not arm64: $(file "$PLUGIN_BINARY")"
  else
    warn "AirSim plugin binary is present but file(1) is unavailable: $PLUGIN_BINARY"
  fi
else
  warn "AirSim plugin binary is absent; run ./build.sh and rebuild the Blocks project: $PLUGIN_BINARY"
fi

validate_settings() {
  local settings_path="$1"
  if [[ ! -f "$settings_path" ]]; then
    fail "AirSim settings file not found: $settings_path"
    return
  fi
  if python3 - "$settings_path" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
with path.open(encoding="utf-8") as handle:
    settings = json.load(handle)
if not isinstance(settings, dict):
    raise SystemExit("settings JSON must contain an object")
if not settings.get("RpcEnabled", settings.get("EnableRpc", False)):
    raise SystemExit("AirSim RPC is disabled in settings JSON")
PY
  then
    ok "AirSim settings: $settings_path"
  else
    fail "invalid settings JSON or AirSim RPC disabled: $settings_path"
  fi
}

# Validate both standard launch profiles. A custom HERCULES_UNREAL_SETTINGS
# replaces only the rural profile; launch_sim still defaults to hero-smoke.
validate_settings "$UNREAL_SETTINGS"
if [[ "$UNREAL_SETTINGS" != "$HERO_SETTINGS" ]]; then
  validate_settings "$HERO_SETTINGS"
fi

WORKSPACE_PARENT="$(cd -- "$REPO_ROOT/.." && pwd)"
if [[ -n "${HERCULES_PYTHON_VENV:-}" ]]; then
  PYTHON_VENV="${HERCULES_PYTHON_VENV}"
elif [[ -d "$WORKSPACE_PARENT/.venvs/hercules-python310" ]]; then
  PYTHON_VENV="$WORKSPACE_PARENT/.venvs/hercules-python310"
else
  # Keep compatibility with older checkouts that still have the in-repo venv.
  PYTHON_VENV="$REPO_ROOT/herculesvenv"
fi
VENV_PYTHON="$PYTHON_VENV/bin/python"
if [[ -x "$VENV_PYTHON" ]]; then
  venv_version="$("$VENV_PYTHON" -c 'import platform; print(platform.python_version())' 2>/dev/null || true)"
  if [[ "$venv_version" == 3.10.* ]]; then
    ok "Python venv uses Python $venv_version"
  else
    fail "Python venv must use Python 3.10; detected ${venv_version:-unreadable}"
  fi
  if [[ -f "$PYTHON_VENV/pyvenv.cfg" ]]; then
    cfg_version="$(sed -n 's/^version = //p' "$PYTHON_VENV/pyvenv.cfg" | head -n 1)"
    if [[ -n "$cfg_version" && "$cfg_version" != "$venv_version" ]]; then
      fail "$PYTHON_VENV/pyvenv.cfg says Python $cfg_version but its interpreter is $venv_version; recreate the venv"
    fi
  fi
  if "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
    ok 'Python venv pip is callable through python -m pip'
  else
    fail 'Python venv pip is not callable; recreate the venv'
  fi
else
  fail "Python venv missing: $VENV_PYTHON"
fi

RPC_HOST="${HERCULES_RPC_HOST:-127.0.0.1}"
RPC_BIND_IP="${HERCULES_RPC_BIND_IP:-127.0.0.1}"
if [[ "$RPC_HOST" != 127.0.0.1 && "$RPC_HOST" != localhost && "$RPC_BIND_IP" == 127.0.0.1 ]]; then
  warn "RPC clients target $RPC_HOST but AirSim is configured for loopback; use HERCULES_RPC_BIND_IP=0.0.0.0 (or a host interface) for bridged Docker networking"
fi

# Keep a safety margin for the Docker image, ROS build outputs, and trial
# artifacts.  The threshold is intentionally configurable for small CI disks.
MIN_FREE_GB="${HERCULES_PREFLIGHT_MIN_FREE_GB:-20}"
if [[ "$MIN_FREE_GB" =~ ^[0-9]+$ ]]; then
  free_kb="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')"
  required_kb=$((MIN_FREE_GB * 1024 * 1024))
  if [[ "$free_kb" =~ ^[0-9]+$ && "$free_kb" -ge "$required_kb" ]]; then
    ok "free disk space: $((free_kb / 1024 / 1024)) GiB (minimum ${MIN_FREE_GB} GiB)"
  elif [[ "$free_kb" =~ ^[0-9]+$ ]]; then
    fail "free disk space is only $((free_kb / 1024 / 1024)) GiB (minimum ${MIN_FREE_GB} GiB)"
  else
    fail "could not determine free disk space for $REPO_ROOT"
  fi
else
  fail "HERCULES_PREFLIGHT_MIN_FREE_GB must be a nonnegative integer"
fi

if [[ "$CHECK_DOCKER" == 1 ]]; then
  if command -v docker >/dev/null 2>&1; then
    ok "Docker CLI: $(command -v docker)"
    if docker compose version >/dev/null 2>&1; then
      ok 'Docker Compose plugin is available'
    else
      fail 'Docker Compose plugin is unavailable'
    fi
    if docker info >/dev/null 2>&1; then
      ok 'Docker daemon is reachable'
      docker_arch="$(docker info --format '{{.Architecture}}' 2>/dev/null || true)"
      case "$docker_arch" in
        arm64|aarch64) ok "Docker server architecture: $docker_arch" ;;
        '') warn 'Docker server architecture could not be read' ;;
        *) fail "Docker server is not arm64/aarch64: $docker_arch" ;;
      esac
    else
      fail 'Docker daemon is not reachable; start Docker Desktop before building the ROS image'
    fi
    compose_file="$REPO_ROOT/docker/ros2/compose.yaml"
    if compose_config="$(docker compose -f "$compose_file" config 2>/dev/null)"; then
      ok "Compose configuration parses: $compose_file"
      if [[ "$compose_config" == *$'network_mode: host'* ]]; then
        warn 'Compose uses host networking; Docker Desktop host networking must be explicitly enabled and tested, or use host.docker.internal with a non-loopback AirSim bind address'
      fi
    else
      fail "Compose configuration does not parse: $compose_file"
    fi
  else
    fail 'Docker CLI is missing; install ARM64 Docker Desktop'
  fi
else
  warn 'Docker checks skipped by --no-docker'
fi

check_rpc_port() {
  local port="$1"
  python3 - "$RPC_HOST" "$port" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=2):
        pass
except OSError as exc:
    print(f"{host}:{port}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

if [[ "$CHECK_RPC" == 1 || "${HERCULES_PREFLIGHT_REQUIRE_RPC:-0}" == 1 ]]; then
  for port in "$AIRSIM_MULTIROTOR_PORT" "$AIRSIM_CAR_PORT"; do
    if check_rpc_port "$port"; then
      ok "AirSim RPC reachable at $RPC_HOST:$port"
    else
      fail "AirSim RPC is not reachable at $RPC_HOST:$port (start the native Unreal simulator first)"
    fi
  done
else
  warn 'RPC probes skipped; use --check-rpc after starting Unreal'
fi

if [[ "$FAILURES" -eq 0 ]]; then
  printf 'Preflight passed.\n'
else
  printf 'Preflight found %d blocking issue(s).\n' "$FAILURES" >&2
fi
exit "$FAILURES"
