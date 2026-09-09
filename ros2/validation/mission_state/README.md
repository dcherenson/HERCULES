# Live mission-state validation

This directory is validation tooling, not part of the
`hercules_mission_ros` production package. The recorder subscribes to one
canonical state and, from that callback, immediately samples
`simGetObjectPose(vehicle, True)` and `simGetGroundTruthKinematics`. It invokes
the existing `hercules_control` smoke executable for motion and never publishes
a command itself.

Start the visible native simulator, then the wrapper with existing API control,
and then the read-only state adapter in separate terminals:

```bash
export UNREAL_EDITOR=/home/dasc-lab/Desktop/Unreal/Engine/Binaries/Linux/UnrealEditor
./docker/ros2/launch_sim.sh

./docker/ros2/launch_wrapper.sh enable_api_control:=true

./docker/ros2/exec.sh ros2 run hercules_mission_ros state_node --ros-args \
  --params-file /workspaces/hercules/ros2/validation/mission_state/artifacts/hero_smoke_observed_origins.yaml
```

Run each validation separately from the repository root. The extra Python path
provides the repository AirSim client and validation-only matplotlib from the
local virtual environment; neither is a package dependency:

```bash
./docker/ros2/exec.sh bash -c '
  export PYTHONPATH=/workspaces/hercules/PythonClient:/workspaces/hercules/herculesvenv/lib/python3.10/site-packages:$PYTHONPATH
  python3 /workspaces/hercules/ros2/validation/mission_state/live_state_validation.py \
    --vehicle Drone1 --vehicle-type drone --port 41451 \
    --topic /hercules_mission/ground_truth/Drone1 \
    --origin 0 -3 0.8238999247550964 --spawn-origin 0 -3 -0.2'

./docker/ros2/exec.sh bash -c '
  export PYTHONPATH=/workspaces/hercules/PythonClient:/workspaces/hercules/herculesvenv/lib/python3.10/site-packages:$PYTHONPATH
  python3 /workspaces/hercules/ros2/validation/mission_state/live_state_validation.py \
    --vehicle Husky1 --vehicle-type ugv --port 41452 \
    --topic /hercules_mission/ground_truth/Husky1 \
    --origin 0.004974978044629097 3.001892566680908 0.7162424921989441 \
    --spawn-origin 0 3 -0.2'
```

Each run writes a small timestamped CSV, metrics JSON, and top-down PNG under
the Git-ignored `artifacts/` directory. Once both CSVs exist, the second run
also creates `combined_state_comparison.png` there.
The plots show equal-scaled X/Y axes, both trajectories, configured origins,
and start/end points. They are a visual sanity check only; the metrics JSON and
deterministic tests are the acceptance evidence.

`--origin` is the observed world location of the wrapper odometry's local zero.
`--spawn-origin` is the requested settings pose and is used only to verify the
separate `actor - simGetGroundTruthKinematics().position` relationship. They
are deliberately distinct because the live vehicles settle before the wrapper
establishes its local odometry origin; see the package README for the measured
discrepancy and the future runtime-origin requirement.

After each smoke run, a visible desktop screenshot can be captured on the host:

```bash
herculesvenv/bin/python ros2/validation/mission_state/capture_unreal_screenshot.py \
  ros2/validation/mission_state/artifacts/drone1_unreal_after_smoke.png
```

Use the analogous `artifacts/husky1_unreal_after_smoke.png` path after the
Husky run.
