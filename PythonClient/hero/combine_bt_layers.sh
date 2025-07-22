#!/bin/bash
set -euo pipefail

BT_DIR="/home/sgarimella34/multi-robot-coordination/data_binvox_octomap/customcity_0p5mcubed"
MERGE_TOOL="$HOME/octomap/bin/merge_octomaps"
OUTPUT="$BT_DIR/combined.bt"

# Build the list of layers to merge:
# - If you pass args, each arg can be a single index (e.g. 2)
#   or a range (e.g. 1-4).  
# - Otherwise, merge every layer_*.bt in sorted order.
mapfile -t LAYERS < <(true)  # initialize empty array

if (( $# > 0 )); then
  for spec in "$@"; do
    if [[ $spec =~ ^([0-9]+)-([0-9]+)$ ]]; then
      # Range: start-end
      start=${BASH_REMATCH[1]}  # first capture group :contentReference[oaicite:0]{index=0}
      end  =${BASH_REMATCH[2]}  # second capture group :contentReference[oaicite:1]{index=1}
      for ((i=start; i<=end; i++)); do
        file="$BT_DIR/layer_${i}.bt"
        [[ -f $file ]] && LAYERS+=( "$file" ) || \
          echo "Warning: $file not found, skipping." >&2
      done
    elif [[ $spec =~ ^[0-9]+$ ]]; then
      # Single index
      file="$BT_DIR/layer_${spec}.bt"
      [[ -f $file ]] && LAYERS+=( "$file" ) || \
        echo "Warning: $file not found, skipping." >&2
    else
      echo "Invalid layer spec: $spec" >&2
      exit 1
    fi
  done
else
  # No args: grab all layers sorted by filename
  mapfile -t LAYERS < <(ls "$BT_DIR"/layer_*.bt | sort -V)  # mapfile reads lines into an array :contentReference[oaicite:2]{index=2}
fi

if (( ${#LAYERS[@]} == 0 )); then
  echo "No layers found to merge." >&2
  exit 1
fi

echo "Merging ${#LAYERS[@]} layers:"
printf '  %s\n' "${LAYERS[@]}"  # array expansion to list each file :contentReference[oaicite:3]{index=3}

# Invoke your merger
"$MERGE_TOOL" "$OUTPUT" "${LAYERS[@]}"

echo "Done. Merged octree at: $OUTPUT"
