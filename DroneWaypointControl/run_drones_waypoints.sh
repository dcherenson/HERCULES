#!/bin/bash

# Absolute path to your executable
EXECUTABLE_PATH=/home/sgarimella34/multi-robot-coordination/Cosys-AirSim/build_debug/output/bin/DroneWaypointControl

# Base path for waypoint files
WAYPOINT_DIR="/home/sgarimella34/multi-robot-coordination/trajectory_data"

# Number of drones to run
NUM_DRONES=2  # Change this to however many drones you have

# Store PIDs for waiting later
PIDS=()

for i in $(seq 1 $NUM_DRONES); do
    DRONE_NAME="Drone$i"
    WAYPOINT_FILE="$WAYPOINT_DIR/${DRONE_NAME}_trajectory.txt"

    echo "Launching $DRONE_NAME with waypoints from $WAYPOINT_FILE"
    $EXECUTABLE_PATH "$DRONE_NAME" "$WAYPOINT_FILE" &
    PIDS+=($!)
done

# Wait for all drones to complete
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "All drone flights complete."
