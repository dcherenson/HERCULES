# Python-only vs ROS mission comparison

`compare_python_ros.py` is a host-side comparison driver for two mission logs.
It accepts either a run directory containing `mission.jsonl` or a direct JSONL
path. The Python logger's per-agent CBF dictionary and the ROS logger's
`cbf.entries` array are normalized into the same semantic fields.

The core tool uses only the Python standard library. Matplotlib is optional
and is used for one diagnostic PNG; missing Matplotlib does not prevent CSV,
JSON, or Markdown output.

## Usage

```bash
python3 ros2/validation/python_ros/compare_python_ros.py \
  /path/to/python_run \
  /path/to/ros_run \
  --output-dir /path/to/comparison \
  --dt 0.1
```

The repository wrapper is equivalent when invoked from any directory:

```bash
docker/ros2/compare_python_ros.sh \
  /path/to/python_run /path/to/ros_run \
  --output-dir /path/to/comparison
```

The wrapper uses `HERCULES_COMPARE_PYTHON` when set, then the external
`../.venvs/hercules-python310/bin/python` created by the Mac setup, then the
legacy in-repository `herculesvenv/bin/python3` if present, and finally
`python3` from `PATH`.

## Paired trial runner

`run_trials.py` owns only an Unreal process started by its `--unreal-command`,
restarts it between implementations, and runs the four requested rows three
times each by default. Each implementation command is run in its own process
group with a bounded timeout; a ROS launch that leaves passive wrappers alive
is interrupted after the trial window and its completed log is still checked.
The Python command receives `output_dir` set to the
trial's `python/` directory; the ROS command receives its `ros/` directory.
Both commands also receive `{trial_dir}`, `{python_output_dir}`,
`{ros_output_dir}`, `{settings}`, `{duration}`, `{row}`, `{method}`, `{observation}`,
`{obstacles}`, `{airsim_host}`, `{drone_port}`, and `{ugv_port}`. `{python}` expands to the interpreter running the trial
runner. The Python command additionally receives
`{python_obstacle_args}`, which expands to `--no-spawn-obstacles` for the
empty-obstacle truth rows and to an empty string for camera/perception rows.
After each command exits, the runner preserves the original files and copies
the single primary JSONL log to `python/mission.jsonl` or `ros/mission.jsonl`
for the comparison contract.
Each completed pair automatically gets a
`comparison/` directory containing the normalized logs and Markdown report.
After the runner finishes, `comparison_summary.json` and
`comparison_summary.md` index every discovered pair and link to its detailed
comparison directory.
The runner also records the canonical-settings SHA-256, resolved Git SHAs,
local Docker image digest/architecture, and host Python/package versions in
`runner_manifest.json` and each trial's `metadata.json`; if the ROS checkout is
dirty, it records a binary diff SHA-256 as well.

For example, after the native simulator launcher and the container command are
known, run:

```bash
./docker/ros2/run_python_ros_trials.sh \
  --settings docker/ros2/settings.rural-nominal.json \
  --output-root ros2/validation/python_ros/artifacts/run-$(date -u +%Y%m%dT%H%M%SZ) \
  --unreal-command 'env HERCULES_UNREAL_SETTINGS={settings} ./docker/ros2/launch_rural_mission_sim.sh -RenderOffscreen' \
  --python-command '{python} PythonClient/distributed_mission/orchestrator.py --launch-mode existing --target-camera-preconfigured --map rural_australia --cbf-method {method} --target-observation-source {observation} --airsim-host {airsim_host} --multirotor-port {drone_port} --car-port {ugv_port} --steps 300 --dt 0.1 --target-speed 0.10 --target-pattern-length 10 --target-pattern-width 8 {python_obstacle_args} --debug-dir {output_dir}' \
  --ros-command './docker/ros2/exec.sh ros2 launch hercules_mission_ros rural_nominal.launch.py duration_sec:={duration} cbf_method:={method} target_observation_source:={observation} cbf_obstacle_source:={obstacles} airsim_host:={airsim_host} drone_port:={drone_port} ugv_port:={ugv_port} truth_rng_mode:=python_global log_path:={output_dir}/mission.jsonl' \
  --prepare-reset-command './docker/ros2/exec.sh {python} ros2/validation/cbf/prepare_perception_parity_reset.py --python-log {python_log} --airsim-host {airsim_host} --rpc-port {drone_port} --car-port {ugv_port}' \
  --repetitions 3 --duration 30
```

Use `--dry-run` first to inspect all twelve pairs without starting Unreal.

## Output artifacts

The output directory contains:

- `normalized_python.csv` and `normalized_ros.csv`: long-form rows keyed by
  canonical mission time and agent.
- `normalized_python.json` and `normalized_ros.json`: the same resampled
  records plus source metadata and the original time samples.
- `comparison.csv`: one row per common agent and 0.1-second grid point with
  absolute position, velocity, slot, command, target-truth, and CBF scalar
  differences.
- `comparison.json`: machine-readable metrics, mismatch counts, warnings, and
  comparison rows.
- `comparison.md`: a concise human-readable report with implementation-local
  metrics and aligned differences.
- `comparison_plot.png`: optional plot of slot error, target path, position
  difference, and CBF intervention.
- `comparison_summary.json` / `comparison_summary.md`: consolidated index for
  the full trial matrix, including completion counts, warnings, global
  differences, and implementation summaries.
- `ros/media/`: ROS top-down MP4/GIF rendered with bounds shared from the paired
  Python log; camera MP4/GIF files are added when a command supplies a staging
  directory.

Time origins are aligned independently to each run's first valid timestamp.
The comparison grid spans the overlapping duration, so it does not extrapolate
one implementation beyond the other. Numeric values are linearly
interpolated; flags and labels use the latest value at or before each grid
point. Missing fields remain empty/`null`.

## Interpreting the baseline

The Python mission normally runs its CBF filter, while the ROS mission can be
launched with CBF disabled. Use the comparator for both of these intentional
baselines, but compare CBF behavior only when both runs use the same CBF
configuration. The report is descriptive and does not impose scenario-specific
tolerances; add those after establishing the first matched replay.

The comparator does not make two runs deterministic. For a meaningful
apples-to-apples result, use the same mission route, target trajectory,
vehicle assignments, controller settings, sensor/observation mode, and random
seed or recorded inputs where available.
