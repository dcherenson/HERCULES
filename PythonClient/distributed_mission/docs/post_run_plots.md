# Post-run mission plots

After every orchestrator run, the completed JSONL diagnostic log is converted
into two PNG files and two top-down animation files in the same `--debug-dir`
directory:

- `<method>_<timestamp>_trajectories_3d.png` shows the logged 3D path of every
  UAV and UGV. Circles mark starting positions and `X` markers mark final
  positions. The display converts AirSim NED Z to physical altitude (`Z-up =
  -NED-Z`), so vehicles with more negative AirSim Z appear higher. Red `X`
  markers show the vehicle position when Unreal reported a collision.
- `<method>_<timestamp>_collision_clearance.png` puts every vehicle on one
  plot. Each line is that vehicle's minimum estimated clearance to another
  vehicle or a perceived obstacle. Clearance is distance minus the relevant
  vehicle and obstacle radii; zero is contact and negative values indicate
  overlap in the logged geometry. The dashed horizontal line marks zero
  clearance. Red `X` markers are authoritative AirSim collision reports.
- `<method>_<timestamp>_topdown.mp4` and `.gif` are route-up square animations
  showing colored trajectories, UAV camera footprints, agent-local perceived
  obstacle circles, deterministic truth obstacles, actual same-type dashed
  communication links, and authoritative collision markers. UGV LiDAR FOV
  volumes remain omitted.
- `<method>_<timestamp>_perception_points.npz` stores bounded, compressed
  sensor-frame and world-frame point samples for each fresh capture.
- `<method>_<timestamp>_perception_report.{json,md}` and
  `<method>_<timestamp>_perception_timeline.png` summarize sensor age, mount
  offsets, proxy counts, gated proxy jumps, and coordinate-frame anomalies.
  Per-agent worst-capture overlays are also generated when the NPZ sidecar is
  available.

The orchestrator also queries AirSim's collision service every control cycle.
When a new collision is detected, it prints a warning containing the vehicle,
object name, mission time, wall-clock time, and penetration depth, and stores
the event in the JSONL record's `collisions` field. Ground/landscape/floor
contact is marked `ignored_ground` for UGVs because wheel contact with the
ground is expected. Collision reports from the initial `t=0` control cycle are
marked `ignored_initial` to suppress spawn/initialization contact noise; the
raw Unreal report is still retained. Negative clearance against a sensor
obstacle proxy is a possible/conservative perception overlap; it is not by
itself proof that the simulator registered a physical collision.

The plots are generated after AirSim cleanup, so matplotlib does not add work
to the real-time control cycle. The log also stores `dt`, `vehicle_types`, and
`vehicle_radii`, allowing the plotting module to be used independently on an
older run:

```bash
PYTHONPATH=PythonClient/distributed_mission ./herculesvenv/bin/python -c \
  "from modules.mission_plots import generate_mission_plots; \
   print(generate_mission_plots('PythonClient/distributed_mission/debug_runs/mestres_<timestamp>.jsonl'))"
```

The plotting module uses a non-interactive Matplotlib backend and saves all
artifacts after AirSim cleanup, so rendering cannot affect control-loop timing.
Use `--animation-fps N` to override the source mission sampling rate. Outputs
play at `--playback-speed` times that rate (default 2x); use
`--playback-speed 1` for real time. Use `--no-animation` to generate only the
two PNGs. Each obstacle record now stores
a cached `sensor_view`: UAV depth-response pose, horizontal/vertical FOV,
range, capture time, and age; or UGV LiDAR pose, 360-degree horizontal FOV,
plus/minus 10-degree vertical FOV, range, capture time, and age. Successful
`BLOCK_COURSE` spawns are stored in `true_obstacles` with box centers and
dimensions. RuralAustralia and `--no-spawn-obstacles` explicitly store an
empty truth list.

Older logs remain usable: trajectories and estimates are rendered, while
missing FOV or true-obstacle layers are skipped. Current logs additionally
store `communication_links`, containing same-type neighbor pairs that exchange
localization messages.

To analyze a log independently:

```bash
PYTHONPATH=PythonClient/distributed_mission ./herculesvenv/bin/python -m \
  modules.perception_diagnostics PythonClient/distributed_mission/debug_runs/mestres_<timestamp>.jsonl \
  --sidecar PythonClient/distributed_mission/debug_runs/mestres_<timestamp>_perception_points.npz
```
