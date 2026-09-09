# Full nominal RuralAustralia validation

`render_validation.py` consumes the JSONL written by
`rural_nominal_mission_node`. It produces an equal-axis top-down trajectory
plot (actual vehicles, desired slots, Target1, and sampled figure eight), UAV
and UGV slot-error time series, `metrics.json`, and the existing Python
top-down MP4/GIF animation. Generated inputs and outputs belong under the
Git-ignored `artifacts/` directory.

```bash
herculesvenv/bin/python ros2/validation/rural_nominal/render_validation.py \
  ros2/validation/rural_nominal/artifacts/mission.jsonl
```

Use `--skip-animation` only for a fast dependency check. These are empirical
visual and implementation diagnostics, not stability or safety evidence.
