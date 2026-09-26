# hercules_cbf_ros

This package is the ROS 2 boundary for the simulator-independent `hercules_cbf`
library. It converts canonical `airsim_world_ned` state messages, validates and
ages per-agent obstacle captures using a steady receipt clock, adds an active
UGV local-target proxy with covariance-inflated radius, and returns the core
filter result with diagnostics. The mission node calls this adapter in process
at its 100 ms control step; no filter nodes or command topics are created.

`ObstacleCache` accepts only successful, valid, frame-qualified captures and
retains the last successful capture when an acquisition fails. Repeated capture
IDs are rejected and freshness is measured from monotonic receipt time. The
cache does not inflate proxy geometry or compare ROS epoch timestamps to the
mission clock.

Phase 3 adds `obstacle_observer_node` and `mission_collision_observer_node`.
The former runs the checked-out Python capture/detection helpers in bounded
worker threads and publishes immutable per-agent snapshots; the latter polls
authoritative AirSim collision state and de-duplicates persistent events. Both
observers are read-only and have no actuator authority.

The default controlled fleet is three UAVs (`Drone1`, `Drone2`, and
`SimpleFlight`) and three UGVs (`Husky1`, `Husky2`, and `Husky3`). The obstacle
observer publishes only those six agents. The collision observer additionally
polls `Target1` as a separate target vehicle. These defaults are explicit in
`config/rural_perception.yaml` and can be overridden with the corresponding
ROS parameters without changing the generic formation slot map.

The mission defaults remain `cbf_enabled=false`, `cbf_method=mestres`,
`cbf_obstacle_source=none`, and `actuation_profile=current_ros`. Enabling CBF
selects Wang for UAVs when requested and always forces physical UGVs through
the Mestres unicycle model, matching the executable Python orchestrator.
