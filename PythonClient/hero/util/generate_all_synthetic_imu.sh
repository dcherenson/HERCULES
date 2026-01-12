#!/usr/bin/env bash

# -----------------------------
# CONFIGURATION
# -----------------------------

# Root directory containing Drone1, Drone2, Husky1, Husky2
DATA_ROOT="/home/sgarimella34/Documents/raw_data_hercules"

# Path to the IMU generation script
IMU_SCRIPT="/home/sgarimella34/multi-robot-coordination/Cosys-AirSim/PythonClient/hero/data_collection/imu_from_GTodometry.py"

# IMU rate (Hz)
IMU_RATE=200.0

# Subdirectories to process
AGENTS=("Drone1" "Drone2" "Husky1" "Husky2")

# -----------------------------
# PROCESS
# -----------------------------

for agent in "${AGENTS[@]}"; do
    AGENT_DIR="${DATA_ROOT}/${agent}"
    ODOM_FILE="${AGENT_DIR}/odom.txt"
    IMU_OUT="${AGENT_DIR}/synthetic_imu.txt"

    if [[ ! -f "${ODOM_FILE}" ]]; then
        echo "[WARNING] Missing odom file: ${ODOM_FILE}"
        continue
    fi

    echo "[INFO] Generating IMU for ${agent}"

    python3 "${IMU_SCRIPT}" \
        "${ODOM_FILE}" \
        "${IMU_OUT}" \
        --imu-rate "${IMU_RATE}"

    if [[ $? -eq 0 ]]; then
        echo "[OK] Wrote ${IMU_OUT}"
    else
        echo "[ERROR] Failed for ${agent}"
    fi

    echo
done

echo "All done."
