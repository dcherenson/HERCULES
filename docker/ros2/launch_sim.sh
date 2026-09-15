#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
[[ ! -f /.dockerenv ]] || { echo 'Launch Unreal on the native host, not inside Docker.' >&2; exit 2; }
: "${UNREAL_EDITOR:?Set UNREAL_EDITOR to the native Engine/Binaries/Linux/UnrealEditor executable}"
# The laptop's iGPU is the boot/display adapter. Force Unreal's Vulkan loader
# to use the discrete NVIDIA adapter for rendered AirSim sessions.
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
exec "$UNREAL_EDITOR" "$REPO_ROOT/Unreal/Environments/Blocks/Blocks.uproject" \
  -game -windowed -ResX=960 -ResY=540 -nosplash \
  "-settings=$TOOL_DIR/settings.hero-smoke.json" "$@"
