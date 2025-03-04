#!/bin/bash
# Set the directory containing your binvox files
BINVOX_DIR="/home/sgarimella34/Downloads/data_binvox_octomap/tesselation_test"
# Set the path to your modified binvox2bt executable
BINVOX2BT="/home/sgarimella34/octomap/bin/binvox2bt_unique_offsets"
# Define the output file for the combined octree
OUTPUT_FILE="/home/sgarimella34/Downloads/data_binvox_octomap/tesselation_test/test1_4tiles.bt"

# Initialize command with any global options (e.g., --mark-free)
CMD="$BINVOX2BT --mark-free"

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
    # Append the per-file offset and filename to the command
    CMD+=" --offset $cx $cy $cz $file"
done

# Append the output filename option
CMD+=" -o $OUTPUT_FILE"

# Print and run the command
echo "Running command: $CMD"
eval $CMD
