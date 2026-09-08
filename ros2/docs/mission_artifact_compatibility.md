# Mission artifact JSONL compatibility

The existing Python plotting, top-down animation, and chase-video overlay code
remains the media implementation. A future ROS mission logger can reuse it by
writing one JSON object per control cycle with the following contract. Numeric
vectors use the existing three-element AirSim/NED boundary convention; the
plotter performs display conversion itself.

## Common record fields

- `step`, `dt`, and `timestamp` establish mission ordering and animation time.
  `wall_timestamp` is additionally required to align asynchronously captured
  videos with mission records.
- `states.<agent>.position` is required for trajectories, communication links,
  collision markers, and chase framing. `velocity`, `yaw`, and `yaw_rate`
  preserve the richer diagnostics and overlays. `vehicle_types` identifies
  UAV/UGV rendering.
- `goal` is used to derive the route-up display orientation and is rendered in
  fixed-goal runs. `vehicle_radii`, `obstacles.<agent>.proxies`, and
  `true_obstacles` support bounds and clearance plots.

## Target truth and estimates

- Write target truth at `target_truth` and, for current-schema compatibility,
  alias the same object at `target` and `targets.<target_id>`. It needs
  `name`, `position`, `velocity`, `yaw`, `command`, `phase`, and `collision`.
- `target_truth.pattern` needs `center`, `route_heading`,
  `longitudinal_span`, and `lateral_span` for stable top-down bounds. Current
  diagnostics also carry `type`, `speed`, `direction`, `sample_count`,
  `phase`, and `index`.
- Each `target_tracking.agents.<agent>.estimate` needs `target_id`, `position`,
  `velocity`, `covariance` (row-shaped 2x2), `timestamp`, and `active` for
  estimate markers and covariance ellipses. Preserve each agent's
  `measurement`, top-level tracking `iterations`, `consensus_residual`, and
  `handoffs` for diagnostics even though the animation does not require all of
  them.

## Links, formation, commands, and collisions

- `tracking_communication_links` is a list of two-agent ID pairs. Older
  consumers fall back to `communication_links`, so keep it as an alias.
- `safety_communication_links` is the separate same-type link list.
- `formation` carries `formation_rms_error`, `formation_max_error`,
  `formation_xy_rms_error`, `formation_xy_max_error`,
  `leader_goal_xy_distance`, `converged_2m_xy`, and `convergence_time`.
- `commands.<agent>` stores the final command vector applied for that cycle.
- `collisions.<agent>` and `target_truth.collision` need `has_collided` or the
  preferred `relevant` flag; `object_name`, penetration depth, and ignored
  ground/initial flags preserve current diagnostics.

## Chase-camera overlays

Video matching requires `timestamp`, `wall_timestamp`, agent
`states.*.position`, and target truth under `targets`, `target_truth`, or
`target`. Reproducing the existing camera analysis also requires
`recording.chase_camera.world_position` and the current recording timestamps
and camera-pose metadata written by `video_recording.py`. The future logger
must not reinterpret coordinates before handing records to that module.
