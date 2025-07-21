#!/bin/bash

# Directory containing your binvox files
BINVOX_DIR="/home/sgarimella34/multi-robot-coordination/data_binvox_octomap/customcity_0p5mcubed"
# Path to your modified binvox2bt executable
BINVOX2BT="/home/sgarimella34/octomap/bin/binvox2bt_unique_offsets"
# Output filename for the combined octree
OUTPUT_FILE="/home/sgarimella34/multi-robot-coordination/data_binvox_octomap/customcity_0p5mcubed/test1_0p5mcubed_customcity.bt"
# Patch size in meters (each binvox covers a 100×100×100 m cube)
PATCH_SIZE=100.0

# Base command with global options
CMD="$BINVOX2BT --mark-free --bb -500 -500 -3 500 500 500"

# Loop through all patch_*.binvox files
for file in "$BINVOX_DIR"/patch_*.binvox; do
    base=$(basename "$file" .binvox)
    # strip "patch_" prefix → e.g. "-50.0_-50.0_layer0"
    offsets=${base#patch_}
    # split into cx, cy, layerTag
    IFS='_' read -r cx cy layerTag <<< "$offsets"
    # extract numeric layer index from "layer0", "layer1", etc.
    iz=${layerTag#layer}
    # compute vertical offset: layer index × patch size
    cz=$(echo "$iz * $PATCH_SIZE" | bc -l)
    # compute horizontal offsets (apply same transform you used before)
    new_offset_x=$(echo "$cy" | bc -l)
    new_offset_y=$(echo "$cx" | bc -l)
    # append this file’s offsets and filename
    CMD+=" --offset $new_offset_x $new_offset_y $cz $file"
done

# append output option, run
CMD+=" -o $OUTPUT_FILE"
echo "Running command:"
echo "$CMD"
eval $CMD
