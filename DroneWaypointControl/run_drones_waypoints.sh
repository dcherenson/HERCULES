#!/bin/bash

# Absolute path to your executable
# EXECUTABLE_PATH=/home/sgarimella34/multi-robot-coordination/Cosys-AirSim/build_debug/output/bin/DroneWaypointControl
EXECUTABLE_PATH=/home/sgarimella34/multi-robot-coordination/Cosys-AirSim/build_release/output/bin/DroneWaypointControl

# Base path for waypoint files
WAYPOINT_DIR="/home/sgarimella34/multi-robot-coordination/trajectory_data"

# Default number of drones if none specified
DEFAULT_NUM_DRONES=1

# Default velocity
VELOCITY=1.5

# Default fly altitude
FLY_ALTITUDE=-5.0

# Check if a velocity argument was provided via environment variable
if [[ -n "$WAYPOINT_VELOCITY" ]]; then
    VELOCITY=$WAYPOINT_VELOCITY
fi

# Check if altitude argument was provided via environment variable
if [[ -n "$FLY_ALTITUDE" ]]; then
    ALTITUDE=$FLY_ALTITUDE
else
    ALTITUDE=$FLY_ALTITUDE
fi

# Store PIDs to wait on
PIDS=()

# Usage help
usage() {
    echo "Usage:"
    echo "  $0                     # Run all drones from Drone1 to Drone$DEFAULT_NUM_DRONES"
    echo "  $0 <num_drones>        # Run all drones from Drone1 to Drone<num_drones>"
    echo "  $0 Drone3              # Run only Drone3"
    echo ""
    echo "To set flight velocity, use:"
    echo "  WAYPOINT_VELOCITY=3.5 $0 Drone1"
    echo "To also set flight altitude, use:"
    echo "  WAYPOINT_VELOCITY=3.5 FLY_ALTITUDE=-50.0 $0 Drone1"
    exit 1
}

# Determine what mode we're running
if [[ $# -eq 0 ]]; then
    # Default: run from 1 to DEFAULT_NUM_DRONES
    for i in $(seq 1 $DEFAULT_NUM_DRONES); do
        DRONE_NAME="Drone$i"
        WAYPOINT_FILE="$WAYPOINT_DIR/${DRONE_NAME}_trajectory.txt"

        echo "Launching $DRONE_NAME with waypoints from $WAYPOINT_FILE at velocity ${VELOCITY} m/s and altitude ${ALTITUDE} m"
        $EXECUTABLE_PATH "$DRONE_NAME" "$WAYPOINT_FILE" "$VELOCITY" "$ALTITUDE" &
        PIDS+=($!)
    done
elif [[ $# -eq 1 ]]; then
    if [[ $1 =~ ^Drone[0-9]+$ ]]; then
        # Run only a specific drone like Drone3
        DRONE_NAME="$1"
        WAYPOINT_FILE="$WAYPOINT_DIR/${DRONE_NAME}_trajectory.txt"

        echo "Launching $DRONE_NAME with waypoints from $WAYPOINT_FILE at velocity ${VELOCITY} m/s and altitude ${ALTITUDE} m"
        $EXECUTABLE_PATH "$DRONE_NAME" "$WAYPOINT_FILE" "$VELOCITY" "$ALTITUDE" &
        PIDS+=($!)
    elif [[ $1 =~ ^[0-9]+$ ]]; then
        # Run from Drone1 to DroneN
        for i in $(seq 1 $1); do
            DRONE_NAME="Drone$i"
            WAYPOINT_FILE="$WAYPOINT_DIR/${DRONE_NAME}_trajectory.txt"

            echo "Launching $DRONE_NAME with waypoints from $WAYPOINT_FILE at velocity ${VELOCITY} m/s and altitude ${ALTITUDE} m"
            $EXECUTABLE_PATH "$DRONE_NAME" "$WAYPOINT_FILE" "$VELOCITY" "$ALTITUDE" &
            PIDS+=($!)
        done
    else
        usage
    fi
else
    usage
fi

# Wait for all launched drones to finish
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "Completed all requested drone flights."


# EXAMPLE USAGE
# Run Drone1 and Drone2 (default)
# ./run_drones.sh

# Run Drone1 through Drone5
# ./run_drones.sh 5

# Run only Drone3
# ./run_drones.sh Drone3

# Run Drone3 with 4.0 m/s velocity and -50.0 m altitude
# WAYPOINT_VELOCITY=4.0 FLY_ALTITUDE=-50.0 ./run_drones.sh Drone3
