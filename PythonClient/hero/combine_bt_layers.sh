#!/bin/bash
set -euo pipefail

BT_DIR="/home/sgarimella34/multi-robot-coordination/data_binvox_octomap/ausenv_semanticrag_1mcubed/"
MERGE_TOOL="$HOME/octomap/bin/merge_octomaps"
OUTPUT="$BT_DIR/combined.bt"

usage() {
  cat <<EOF
Usage: $0 [layer_indices_or_ranges...]

If no args are given, merges every layer_*.bt in BT_DIR.
Examples:
  $0              # merge all
  $0 0 2 5        # merge only 0,2,5
  $0 1-4 8-9      # merge ranges
EOF
}

declare -a LAYERS=()

if (( $# > 0 )); then
  for spec in "$@"; do
    if [[ $spec =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start=${BASH_REMATCH[1]}
      end=${BASH_REMATCH[2]}
      if (( end < start )); then
        echo "Invalid range: $spec" >&2
        exit 1
      fi
      for ((i=start; i<=end; i++)); do
        f="$BT_DIR/layer_${i}.bt"
        [[ -f $f ]] && LAYERS+=( "$f" ) || echo "Warning: $f not found, skipping." >&2
      done
    elif [[ $spec =~ ^[0-9]+$ ]]; then
      f="$BT_DIR/layer_${spec}.bt"
      [[ -f $f ]] && LAYERS+=( "$f" ) || echo "Warning: $f not found, skipping." >&2
    else
      echo "Bad layer spec: $spec" >&2
      usage
      exit 1
    fi
  done
else
  mapfile -t LAYERS < <(ls "$BT_DIR"/layer_*.bt 2>/dev/null | sort -V || true)
fi

if (( ${#LAYERS[@]} == 0 )); then
  echo "No layer bt files to merge." >&2
  exit 1
fi

echo "Merging ${#LAYERS[@]} layers:"
printf '  %s\n' "${LAYERS[@]}"

"$MERGE_TOOL" "$OUTPUT" "${LAYERS[@]}"

echo "Done. Merged octree at: $OUTPUT"
