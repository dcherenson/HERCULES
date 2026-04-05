#!/usr/bin/env bash

# -----------------------------
# CONFIGURATION
# -----------------------------

# Root directory containing Drone1, Drone2, Husky1, Husky2
DATA_ROOT="/home/sgarimella34/Documents/raw_data_hercules"

# Path to the IMU generation script
IMU_SCRIPT="/home/sgarimella34/multi-robot-coordination/Cosys-AirSim/PythonClient/hero/data_collection/imu_from_GTodometry.py"

# IMU rate (Hz)
IMU_RATE=500.0

# Subdirectories to process
AGENTS=("Drone1" "Drone2" "Husky1" "Husky2")

# -----------------------------
# DERIVE OUTPUT NAME
# -----------------------------

# Turn "200.0" -> "200" (if it's essentially an integer), otherwise keep as-is (e.g., "200.5")
IMU_RATE_STR="$(python3 - <<PY
r = float("${IMU_RATE}")
if abs(r - round(r)) < 1e-9:
    print(int(round(r)))
else:
    s = ("%.10g" % r).rstrip("0").rstrip(".")
    print(s)
PY
)"

# Naming convention: synthetic_imu_9axis_200Hz.txt / synthetic_imu_9axis_500Hz.txt
IMU_BASENAME="synthetic_imu_9axis_${IMU_RATE_STR}Hz"
IMU_FILENAME="${IMU_BASENAME}.txt"

# -----------------------------
# PROCESS
# -----------------------------

for agent in "${AGENTS[@]}"; do
    AGENT_DIR="${DATA_ROOT}/${agent}"
    ODOM_FILE="${AGENT_DIR}/odom.txt"
    IMU_OUT="${AGENT_DIR}/${IMU_FILENAME}"

    if [[ ! -f "${ODOM_FILE}" ]]; then
        echo "[WARNING] Missing odom file: ${ODOM_FILE}"
        continue
    fi

    echo "[INFO] Generating IMU for ${agent} @ ${IMU_RATE_STR}Hz"

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
