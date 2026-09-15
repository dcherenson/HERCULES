# CBF reproduction artifacts

`docker/ros2/reproduce_cbf.sh` records resolved mode metadata and starts the
existing mission launch with one of the three executable Phase 3 configurations.
Use `--obstacles none` for a deterministic valid-empty static source, or
`--obstacles perception` when the asynchronous observer is available. Generated
logs, plots and media belong under `artifacts/` and are intentionally ignored.

The no-CBF, Mestres and Wang logs are compared with `compare_cbf_modes.py`.
Trajectory differences are interpreted separately from same-request Python/C++
replay because camera timing and vehicle physics are not identical across runs.
