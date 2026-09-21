# hercules_localization_ros

ROS 2 transport and estimator integration for cooperative localization. The package keeps
ROS conversion, algorithm selection, peer-message retention, lifecycle
services, and diagnostics separate from estimator mathematics. The adapter
dispatches to the separately built `hercules_localization` core for either
`recursive_decentralized` or `gs_ci`; callers can also register an estimator
callback for experiments. Before the first valid GPS fix, the node forwards a
clearly marked invalid `waiting_for_first_gps` observation.

## Node

`localization_node` accepts `agent_id`, `anchor_agent`, optional `vehicle_type`
(`drone` or `ugv` for AirSim topic namespaces), and `algorithm` parameters. It subscribes
to the following configurable inputs:

* `odom_local_topic` (`/<agent_id>/ground_truth/odom_local` by default),
  `nav_msgs/msg/Odometry`;
* `global_gps_topic` (`/<agent_id>/global_gps` by default),
  `sensor_msgs/msg/NavSatFix`;
* `gps_origin_topic` and `gps_origin_topic_secondary` (the drone and UGV
  AirSim wrapper home-origin topics by default), `airsim_interfaces/msg/GPSYaw`;
* `measurement_topic` (`/hercules_localization/<agent_id>/measurement`),
  `hercules_interfaces/msg/LocalizationMeasurement`;
* `peer_topic` (`/hercules_localization/peer_estimate`),
  `hercules_interfaces/msg/LocalizationPeerEstimate`; and
* `relative_topic` (`/hercules_localization/relative`),
  `hercules_interfaces/msg/PlanarRelativeMeasurement`.

GS-CI also broadcasts and consumes typed `GlobalCiBelief` packets on
`global_ci_topic` (default `/hercules_localization/gs_ci`), using the ordered
`[x1,y1,...,xn,yaw_sender]` state and deterministic self/peer weights. Launch exposes the stale and
transaction timeouts, camera range/rate, process/measurement covariance floors,
`dcl_lambda`, and `ci_self_weight` parameters.

It publishes `LocalizationEstimate` and `LocalizationDiagnostics` on
per-agent `estimate_topic` and `diagnostics_topic` topics, and broadcasts a
`LocalizationPeerEstimate` on `peer_output_topic`. `~/set_algorithm` validates
the two algorithm names; `~/reset` clears transport state; and `~/recursive_pair` retains a typed relative
observation for a recursive core. GPS is converted from AirSim
latitude/longitude degrees to local NED around one fixed, configurable origin
(or the first valid fix), and the numerical core performs the covariance and
recursive update. The relative observer supports a deterministic truth mode
for dry-run/reproduction and a camera mode that uses AirSim detection
metadata, depth ROI back-projection, and camera extrinsics. Both modes publish
the same range/bearing and covariance contract and exclude `Target1`.

The mission launch defaults to estimator-driven control. Set
`control_source:=truth` only for an explicit legacy/troubleshooting run; stale or
invalid estimator samples stop/hover the corresponding vehicle when the normal
`control_source:=estimate` mode is active.
