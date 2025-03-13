#!/bin/bash

# Usage: ./binvox2bt_unique_offsets [OPTIONS] <binvox filenames>
# 	OPTIONS:
# 	 -o <file>        Output filename (default: first input filename + .bt)
# 	 --mark-free      Mark not occupied cells as 'free' (default: unknown)
# 	 --rotate         Rotate left by 90 deg. to fix the coordinate system when exported from Webots
# 	 --bb <minx> <miny> <minz> <maxx> <maxy> <maxz>: force bounding box for OcTree
# 	 --offset <x> <y> <z>: add an offset to the final coordinates for the following file only
# If more than one binvox file is given, the models are composed into a single octree.

# for 1m cubed
# Set the directory containing your binvox files
# BINVOX_DIR="/home/sgarimella34/Downloads/data_binvox_octomap/tesselation_1mcubed"
# # Set the path to your modified binvox2bt executable
# BINVOX2BT="/home/sgarimella34/octomap/bin/binvox2bt_unique_offsets"
# # Define the output file for the combined octree
# OUTPUT_FILE="/home/sgarimella34/Downloads/data_binvox_octomap/tesselation_1mcubed/test1_1mcubed_ausenv.bt"

# for 0.5m cubed
BINVOX_DIR="/home/sgarimella34/Downloads/data_binvox_octomap/tesselation_0p5mcubed"
BINVOX2BT="/home/sgarimella34/octomap/bin/binvox2bt_unique_offsets"
OUTPUT_FILE="/home/sgarimella34/Downloads/data_binvox_octomap/tesselation_0p5mcubed/test1_0p5mcubed_ausenv.bt"

# Initialize command with any global options (e.g., --mark-free)
CMD="$BINVOX2BT --mark-free --bb -500 -500 -3 500 500 500"

# Loop through all patch_*.binvox files in the directory
for file in "$BINVOX_DIR"/patch_*.binvox; do
    # Get the basename without the directory and extension, e.g., "patch_-50.0_-50.0"
    base=$(basename "$file" .binvox)
    # Remove the "patch_" prefix to get the offset string, e.g., "-50.0_-50.0"
    offsets=${base#patch_}
    # Split the string on underscore to extract cx and cy
    IFS='_' read -r cx cy <<< "$offsets"
    # Set the z offset (assumed to be 0)
    cz=0.0
    # Compute the composite offset:
    # Previous transform for y-flip and 90 deg clockwise was: (cx,cy) -> (-cy, -cx)
    # To apply an extra 180deg rotation (i.e. multiply by -1), we get:
    # (-cy, -cx) * -1 = (cy, cx)
    new_offset_x=$(echo "scale=6; $cy" | bc -l)
    new_offset_y=$(echo "scale=6; $cx" | bc -l)
    # Append the per-file offset and filename to the command
    CMD+=" --offset $new_offset_x $new_offset_y $cz $file"
done

# Append the output filename option
CMD+=" -o $OUTPUT_FILE"

# Print and run the command
echo "Running command: $CMD"
eval $CMD
