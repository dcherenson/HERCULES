#!/bin/bash

# === Configuration ===
BINVOX_DIR="/home/sgarimella34/multi-robot-coordination/data_binvox_octomap/customcity_0p5mcubed"
BINVOX2BT="/home/sgarimella34/octomap/bin/binvox2bt_unique_offsets"
PATCH_SIZE=100.0
OUTPUT_DIR="$BINVOX_DIR"
# bounding-box: xmin ymin zmin xmax ymax zmax
BBOX="-500 -500 -3 500 500 500"

# find all distinct layer indices
mapfile -t LAYERS < <(
  ls "$BINVOX_DIR"/patch_*_layer*.binvox \
    | sed -n 's/.*_layer\([0-9]\+\)\.binvox$/\1/p' \
    | sort -n \
    | uniq
)

for iz in "${LAYERS[@]}"; do
  LAYER_BT="$OUTPUT_DIR/layer_${iz}.bt"
  cmd=( "$BINVOX2BT" --mark-free --bb $BBOX )

  # collect this layer’s binvox files
  for f in "$BINVOX_DIR"/patch_*_layer${iz}.binvox; do
    base=$(basename "$f" .binvox)
    # strip "patch_" prefix and "_layerN" suffix → "cx_cy"
    offsets=${base#patch_}
    offsets=${offsets%_layer${iz}}
    IFS='_' read -r cx cy <<< "$offsets"
    cz=$(echo "$iz * $PATCH_SIZE" | bc -l)
    cmd+=( --offset "$cx" "$cy" "$cz" "$f" )
  done

  cmd+=( -o "$LAYER_BT" )
  echo "Merging layer $iz → $LAYER_BT"
  "${cmd[@]}"
done
