#!/bin/bash

# Absolute path to your UGV executable
EXECUTABLE_PATH=/home/sgarimella34/multi-robot-coordination/Cosys-AirSim/build_debug/output/bin/UGVWaypointControl

# Base path for waypoint files
WAYPOINT_DIR="/home/sgarimella34/multi-robot-coordination/trajectory_data"

# Default number of UGVs if none specified
DEFAULT_NUM_UGVS=2

# Default linear speed (in m/s) to be used when no individual speed is provided
DEFAULT_SPEED=1.75

# Array to hold process IDs for launched instances
PIDS=()

# Usage help function
usage() {
    echo "Usage:"
    echo "  $0                           # Run UGV1 to UGV${DEFAULT_NUM_UGVS} with default speed (${DEFAULT_SPEED} m/s)"
    echo "  $0 <num_ugvs>                # Run UGV1 to UGV<num_ugvs> with default speed (${DEFAULT_SPEED} m/s)"
    echo "  $0 UGV3                     # Run only UGV3 with default speed (${DEFAULT_SPEED} m/s)"
    echo "  $0 UGV3 <speed>             # Run only UGV3 with the specified speed"
    echo "  $0 <num_ugvs> <speed>         # Run UGV1 to UGV<num_ugvs> with the specified speed for all"
    exit 1
}

# Determine the launch mode based on the number and type of arguments.
if [[ $# -eq 0 ]]; then
    # Default: Run UGV1 to UGV${DEFAULT_NUM_UGVS} with default speed.
    for i in $(seq 1 $DEFAULT_NUM_UGVS); do
        UGV_NAME="Husky$i"
        WAYPOINT_FILE="$WAYPOINT_DIR/${UGV_NAME}_trajectory.txt"
        echo "Launching $UGV_NAME with speed ${DEFAULT_SPEED} m/s and waypoints from $WAYPOINT_FILE"
        $EXECUTABLE_PATH "$UGV_NAME" "$DEFAULT_SPEED" "$WAYPOINT_FILE" &
        PIDS+=($!)
    done
elif [[ $# -eq 1 ]]; then
    if [[ $1 =~ ^UGV[0-9]+$ ]]; then
        # Run only a specific UGV (e.g., UGV3) with default speed.
        UGV_NAME="$1"
        WAYPOINT_FILE="$WAYPOINT_DIR/${UGV_NAME}_trajectory.txt"
        echo "Launching $UGV_NAME with default speed ${DEFAULT_SPEED} m/s and waypoints from $WAYPOINT_FILE"
        $EXECUTABLE_PATH "$UGV_NAME" "$DEFAULT_SPEED" "$WAYPOINT_FILE" &
        PIDS+=($!)
    elif [[ $1 =~ ^[0-9]+$ ]]; then
        # Run UGV1 to UGV<num_ugvs> with default speed.
        NUM_UGVS=$1
        for i in $(seq 1 $NUM_UGVS); do
            UGV_NAME="Husky$i"
            WAYPOINT_FILE="$WAYPOINT_DIR/${UGV_NAME}_trajectory.txt"
            echo "Launching $UGV_NAME with default speed ${DEFAULT_SPEED} m/s and waypoints from $WAYPOINT_FILE"
            $EXECUTABLE_PATH "$UGV_NAME" "$DEFAULT_SPEED" "$WAYPOINT_FILE" &
            PIDS+=($!)
        done
    else
        usage
    fi
elif [[ $# -eq 2 ]]; then
    # Two arguments provided.
    if [[ $1 =~ ^UGV[0-9]+$ ]]; then
        # Mode: Run a single UGV (e.g., UGV3) with the specified speed.
        UGV_NAME="$1"
        SPEED="$2"
        WAYPOINT_FILE="$WAYPOINT_DIR/${UGV_NAME}_trajectory.txt"
        echo "Launching $UGV_NAME with specified speed ${SPEED} m/s and waypoints from $WAYPOINT_FILE"
        $EXECUTABLE_PATH "$UGV_NAME" "$SPEED" "$WAYPOINT_FILE" &
        PIDS+=($!)
    elif [[ $1 =~ ^[0-9]+$ ]]; then
        # Mode: Run UGV1 to UGV<num_ugvs> with the specified speed for all.
        NUM_UGVS=$1
        SPEED="$2"
        for i in $(seq 1 $NUM_UGVS); do
            UGV_NAME="Husky$i"
            WAYPOINT_FILE="$WAYPOINT_DIR/${UGV_NAME}_trajectory.txt"
            echo "Launching $UGV_NAME with specified speed ${SPEED} m/s and waypoints from $WAYPOINT_FILE"
            $EXECUTABLE_PATH "$UGV_NAME" "$SPEED" "$WAYPOINT_FILE" &
            PIDS+=($!)
        done
    else
        usage
    fi
else
    usage
fi

# Wait for all launched UGV processes to complete.
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "Completed all UGV waypoint missions."
