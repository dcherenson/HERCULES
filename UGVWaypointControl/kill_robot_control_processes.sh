#!/bin/bash

# Define the list of exact process names to kill
process_names=("UGVWaypointCont" "DroneWaypointCo")

for name in "${process_names[@]}"; do
    # Find matching PIDs using pgrep that match the exact process name
    pids=$(pgrep -x "$name")
    if [ -n "$pids" ]; then
        echo "Killing $name (PIDs: $pids)"
        kill -9 $pids
    else
        echo "No process found for $name"
    fi
done
