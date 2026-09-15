# hercules_tracking_ros

ROS 2 transport for the validated `hercules_tracking` numerical core. Each
controlled vehicle runs one `tracker_node` and owns one `TargetTracker`.
Numerical fusion remains in `hercules_tracking`; this package only converts
messages, freezes the per-epoch communication graph, synchronizes rounds, and
publishes local estimates and diagnostics.

The production camera observer is a thin ROS wrapper around
`PythonClient/distributed_mission/modules/target_observation.py`. Camera RPCs
remain on that module's worker thread and never execute on the C++ mission
control timer.

## Topics

- `/hercules_tracking/epoch`: frozen agent ordering and adjacency for an epoch.
- `/hercules_tracking/consensus`: epoch- and round-qualified tracker messages.
- `/hercules_tracking/round_status`: decentralized all-agent convergence barrier.
- `/hercules_tracking/handoff`: typed information-form handoff messages.
- `/hercules_tracking/<agent>/Target1/measurement`: direct local observation.
- `/hercules_tracking/<agent>/Target1/estimate`: that agent's local estimate.
- `/hercules_tracking/<agent>/Target1/diagnostics`: local protocol diagnostics.
- `/hercules_tracking/observation_diagnostics`: asynchronous observer timing.

At each epoch adjacency is frozen. A tracker waits briefly for its measurement,
uses a bounded seed-announcement interval, and then accepts only consensus
messages matching its target, epoch, round, and frozen neighbors. Each round
waits for every expected neighbor until `round_timeout_sec`. On timeout it uses
the matching messages that arrived, marks the epoch timed out, and never reuses
older messages. All trackers exchange round status; a round ends when the
global maximum residual meets tolerance, the iteration limit is reached, or a
status barrier times out. A status timeout publishes the best local estimate
available and records the timeout.

`target_observation_source:=truth` is deterministic validation input: truth is
range-gated, seeded, and enters only as a `TargetMeasurement`. The default
`camera` mode uses every camera configured by the executable Python mission:
`target_bottom` for all five UAVs and `front_center` for all three controlled
UGVs. Camera-generated covariance is retained. `tracking_measurement_std` is
used by truth observations and as the perception covariance floor/invalid
message covariance, not as a replacement for valid camera covariance.
