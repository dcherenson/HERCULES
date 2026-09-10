# Full nominal RuralAustralia validation

`render_validation.py` consumes the JSONL written by
`rural_nominal_mission_node`. It produces an equal-axis top-down trajectory
plot (actual vehicles, desired slots, Target1, and sampled figure eight), UAV
and UGV slot-error time series, per-agent estimator error/consensus plots,
tracking availability counts, `metrics.json`, and the existing Python top-down
MP4/GIF animation. Direct observers are ringed in green; every agent's local
estimate and covariance are rendered independently. Generated inputs and outputs belong under the
Git-ignored `artifacts/` directory.

```bash
herculesvenv/bin/python ros2/validation/rural_nominal/render_validation.py \
  ros2/validation/rural_nominal/artifacts/mission.jsonl
```

Use `--skip-animation` only for a fast dependency check. These are empirical
visual and implementation diagnostics, not stability or safety evidence.

`compare_tracking_modes.py` compares separately reset truth-target and
distributed-camera runs. `docker/ros2/reproduce.sh --with-live-video` performs
the truth baseline, distributed truth-observation gate, and distributed camera
gate, renders each under its own ignored directory, and writes
`mode_comparison.json` plus `mode_comparison.png`.
