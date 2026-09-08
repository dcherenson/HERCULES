#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
[[ ! -f /.dockerenv ]] || { echo 'Launch Unreal on the native host, not inside Docker.' >&2; exit 2; }
: "${UNREAL_EDITOR:?Set UNREAL_EDITOR to the native Engine/Binaries/Linux/UnrealEditor executable}"
exec "$UNREAL_EDITOR" "$REPO_ROOT/Unreal/Environments/Blocks/Blocks.uproject" \
  -game -windowed -ResX=960 -ResY=540 -nosplash \
  "-settings=$TOOL_DIR/settings.hero-smoke.json" "$@"
