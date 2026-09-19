#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
[[ ! -f /.dockerenv ]] || { echo 'Launch Unreal on the native host, not inside Docker.' >&2; exit 2; }

if [[ -z ${UNREAL_EDITOR:-} ]]; then
  if [[ $(uname -s) == Darwin ]]; then
    UNREAL_EDITOR="${HERCULES_UNREAL_EDITOR:-/Users/Shared/Epic Games/UE_5.2/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor}"
  else
    UNREAL_EDITOR=/home/dasc-lab/Desktop/Unreal/Engine/Binaries/Linux/UnrealEditor
  fi
fi
[[ -x "$UNREAL_EDITOR" ]] || {
  echo "UnrealEditor not found or not executable: $UNREAL_EDITOR" >&2
  if [[ $(uname -s) == Darwin ]]; then
    echo 'Set UNREAL_EDITOR to the native Mac UnrealEditor executable.' >&2
  else
    echo 'Set UNREAL_EDITOR to your native Engine/Binaries/Linux/UnrealEditor executable.' >&2
  fi
  exit 2
}
UNREAL_SETTINGS="${HERCULES_UNREAL_SETTINGS:-$TOOL_DIR/settings.rural-nominal.json}"
[[ -f "$UNREAL_SETTINGS" ]] || {
  echo "AirSim settings file not found: $UNREAL_SETTINGS" >&2
  exit 2
}

# AirSim binds its RPC server using LocalHostIp from the settings JSON. The
# default remains loopback; set HERCULES_RPC_BIND_IP (usually 0.0.0.0 when
# ROS runs in a bridged Docker Desktop network) to create a temporary,
# non-mutating settings override for the native simulator.
RPC_BIND_IP="${HERCULES_RPC_BIND_IP:-127.0.0.1}"
SETTINGS_OVERRIDE=''
cleanup_settings() {
  [[ -z "$SETTINGS_OVERRIDE" ]] || rm -f -- "$SETTINGS_OVERRIDE"
}
trap cleanup_settings EXIT
if [[ "$RPC_BIND_IP" != 127.0.0.1 ]]; then
  command -v python3 >/dev/null 2>&1 || {
    echo 'python3 is required when HERCULES_RPC_BIND_IP is not 127.0.0.1.' >&2
    exit 2
  }
  SETTINGS_OVERRIDE="$(mktemp "${TMPDIR:-/tmp}/hercules-airsim-settings.XXXXXX")"
  python3 - "$UNREAL_SETTINGS" "$SETTINGS_OVERRIDE" "$RPC_BIND_IP" <<'PY'
import json
import pathlib
import sys

source, destination, bind_ip = sys.argv[1:]
with pathlib.Path(source).open(encoding="utf-8") as handle:
    settings = json.load(handle)
if not isinstance(settings, dict):
    raise SystemExit("AirSim settings must contain a JSON object")
settings["LocalHostIp"] = bind_ip
with pathlib.Path(destination).open("w", encoding="utf-8") as handle:
    json.dump(settings, handle, indent=2)
    handle.write("\n")
PY
  UNREAL_SETTINGS="$SETTINGS_OVERRIDE"
fi

if [[ $(uname -s) != Darwin ]]; then
  # The laptop's iGPU is the boot/display adapter. Force Unreal's Vulkan
  # loader to use the discrete NVIDIA adapter for rendered AirSim sessions.
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
  export __NV_PRIME_RENDER_OFFLOAD=1
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
fi
: "${HERCULES_RESX:=960}"
: "${HERCULES_RESY:=540}"
UNREAL_ARGS=(
  "$REPO_ROOT/Unreal/Environments/Blocks/Blocks.uproject"
  /Game/RuralAustralia/Maps/RuralAustralia_Example_01
  -game -windowed -ResX="$HERCULES_RESX" -ResY="$HERCULES_RESY" -nosplash
  "-settings=$UNREAL_SETTINGS"
  "$@"
)

if [[ -n "$SETTINGS_OVERRIDE" ]]; then
  if "$UNREAL_EDITOR" "${UNREAL_ARGS[@]}"; then
    status=0
  else
    status=$?
  fi
  # The EXIT trap removes the temporary settings file after the simulator
  # process returns; keep the real settings JSON untouched.
  exit "$status"
fi
exec "$UNREAL_EDITOR" "${UNREAL_ARGS[@]}"
