#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
[[ ! -f /.dockerenv ]] || { echo 'Launch Unreal on the native host, not inside Docker.' >&2; exit 2; }
: "${UNREAL_EDITOR:=/home/dasc-lab/Desktop/Unreal/Engine/Binaries/Linux/UnrealEditor}"
[[ -x "$UNREAL_EDITOR" ]] || {
  echo "UnrealEditor not found or not executable: $UNREAL_EDITOR" >&2
  echo 'Set UNREAL_EDITOR to your native Engine/Binaries/Linux/UnrealEditor executable.' >&2
  exit 2
}
UNREAL_SETTINGS="${HERCULES_UNREAL_SETTINGS:-$TOOL_DIR/settings.rural-nominal.json}"
[[ -f "$UNREAL_SETTINGS" ]] || {
  echo "AirSim settings file not found: $UNREAL_SETTINGS" >&2
  exit 2
}
# The laptop's iGPU is the boot/display adapter. Force Unreal's Vulkan loader
# to use the discrete NVIDIA adapter for rendered AirSim sessions.
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
: "${HERCULES_RESX:=960}"
: "${HERCULES_RESY:=540}"
exec "$UNREAL_EDITOR" "$REPO_ROOT/Unreal/Environments/Blocks/Blocks.uproject" \
  /Game/RuralAustralia/Maps/RuralAustralia_Example_01 \
  -game -windowed -ResX="$HERCULES_RESX" -ResY="$HERCULES_RESY" -nosplash \
  "-settings=$UNREAL_SETTINGS" "$@"
