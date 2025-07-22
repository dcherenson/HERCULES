#!/bin/bash
set -euo pipefail

# --- config ---
BINVOX_DIR="/home/sgarimella34/multi-robot-coordination/data_binvox_octomap/customcity_0p5mcubed"
BINVOX2BT="/home/sgarimella34/octomap/bin/binvox2bt_unique_offsets"
OUTPUT_FILE="$BINVOX_DIR/customcity_0p5mcubed.bt"
PATCH_SIZE=100            # must match the Python script
BBOX="-500 -500 -600 500 500 600"
# ----------------

CMD=( "$BINVOX2BT" --mark-free --bb $BBOX )

shopt -s nullglob
for f in "$BINVOX_DIR"/patch_*_layer*.binvox; do
    base=$(basename "$f" .binvox)
    # pattern: patch_<cx>_<cy>_layer<iz>.binvox
    if [[ $base =~ ^patch_([^_]*)_([^_]*)_layer([0-9]+)$ ]]; then
        cx="${BASH_REMATCH[1]}"
        cy="${BASH_REMATCH[2]}"
        iz="${BASH_REMATCH[3]}"

        # same XY transform you already used
        ox=$(echo "$cy" | bc -l)
        oy=$(echo "$cx" | bc -l)

        # Python saved layer centers at cz = 0, -100, -200 ...
        # To stack "up" in OctoMap (positive Z), flip the sign here:
        oz=$(echo "$iz * $PATCH_SIZE" | bc -l)

        CMD+=( --offset "$ox" "$oy" "$oz" "$f" )
    else
        echo "Skipping unexpected filename: $base" >&2
    fi
done

CMD+=( -o "$OUTPUT_FILE" )
echo "Running command:"
printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
