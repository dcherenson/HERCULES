# HERCULES Collaborative SLAM Dataset Collection

Two-phase workflow: **(1)** manually teleop each robot to record its waypoint
trajectory, **(2)** replay all trajectories simultaneously while the data
collector records the dataset.

All paths below assume the Cosys-AirSim checkout at
`~/multi-robot-coordination/Cosys-AirSim` and the UE environment already
running with the multi-vehicle `settings.json`.

See also: `DATA_COLLECTOR_README.txt` in this folder for collector details.

---

## Phase 1 — Record trajectories (manual teleop)

Scripts: `record_ugv_waypoints_teleop.py` (Husky) and
`record_drone_waypoints_teleop.py` (drone), both in this folder.

For **each vehicle**, one at a time:

1. Open the script and edit the constants at the top:
   - `OUTFILE_PATH` — where the trajectory is written, e.g.
     `~/multi-robot-coordination/trajectory_data/<dataset>/<run>/Husky2_trajectory.txt`.
     One file per vehicle; keep the `Husky<N>_` / `Drone<N>_` naming so the
     replay scripts pick them up. **Check you are not overwriting a
     trajectory you want to keep.**
   - Vehicle name / `RPC_PORT` if driving a different vehicle than last time.
2. Run it: `python3 record_<ugv|drone>_waypoints_teleop.py`
3. Toggle recording ON (**V** on the UGV script, **C** on the drone script),
   drive the route, toggle recording off, then **Q** to quit.

Key teleop controls (full list in each script's docstring):

| | UGV (Husky) | Drone |
|---|---|---|
| Move | W/S throttle, A/D steer | W/S/A/D body-frame velocity |
| Up/down | — | R/F (altitude hold target) |
| Yaw | — | J/L |
| Stop | SPACE (brake), C handbrake | SPACE/0 stop, H hover |
| Takeoff / land | — | T / G |
| Pause sim | P | P |
| **Record toggle** | **V** | **C** |
| Quit | Q | Q |

Waypoints are saved as `X Y Z T` lines. UGV logs force `z=0` (2D); drone
logs flip NED z so Z is up-positive.

---

## Phase 2 — Replay trajectories + collect dataset

Start these in order:

1. **Data collector + calibration**
   ```bash
   cd ~/multi-robot-coordination/Cosys-AirSim/PythonClient/hero/data_collection
   python3 hercules_multi_vehicle_data_collector.py   # CHECK: OUTDIR at top of the
                                                      # script must NOT overwrite an
                                                      # existing dataset on the SSD
   python3 calibration_camimu_UAV.py
   python3 calibration_camimu_UGV.py                  # wait for calibration to fully
                                                      # complete, then wait 10 more sec
   ```

2. **Replay UGV trajectories**
   ```bash
   cd ~/multi-robot-coordination/Cosys-AirSim/UGVWaypointControl
   ./run_UGVs_waypoints.sh    # CHECK: WAYPOINT_DIR inside the script points at the
                              # trajectory folder recorded in Phase 1
   ```

3. **Replay drone trajectories**
   ```bash
   cd ~/multi-robot-coordination/Cosys-AirSim/DroneWaypointControl
   ./run_drones_waypoints.sh  # CHECK: WAYPOINT_DIR here too — both scripts keep a
                              # history of old dirs commented out; make sure the
                              # single uncommented one is your current run
   ```

Both replay scripts use the `build_release` executables
(`build_release/output/bin/{UGV,Drone}WaypointControl`) — rebuild Cosys-AirSim
first if the C++ side changed.
