#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
[[ ! -f /.dockerenv ]] || { echo 'Launch Unreal on the native host, not inside Docker.' >&2; exit 2; }
: "${UNREAL_EDITOR:=/home/dasc-lab/Desktop/Unreal/Engine/Binaries/Linux/UnrealEditor}"
[[ -x "$UNREAL_EDITOR" ]] || {
  echo "UnrealEditor not found or not executable: $UNREAL_EDITOR" >&2
  echo 'Set UNREAL_EDITOR to your native Engine/Binaries/Linux/UnrealEditor executable.' >&2
  exit 2
}
exec "$UNREAL_EDITOR" "$REPO_ROOT/Unreal/Environments/Blocks/Blocks.uproject" \
  /Game/RuralAustralia/Maps/RuralAustralia_Example_01 \
  -game -windowed -ResX=1280 -ResY=720 -nosplash \
  "-settings=$TOOL_DIR/settings.rural-nominal.json" "$@"
