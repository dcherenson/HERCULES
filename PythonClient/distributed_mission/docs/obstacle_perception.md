# Local obstacle perception

UAVs use `front_center` `DepthPerspective` images. Huskies use `Lidar1`.
Measurements are transformed into world coordinates, range filtered,
downsampled, ground filtered, and clustered with DBSCAN. Large clusters are
split along their principal axis so each proxy has bounded size. UAV depth
responses use the vehicle starting-point frame in this Hero configuration, so
the runtime adds the measured actor-to-kinematics frame origin before
back-projecting into world NED. Husky LiDAR points are sensor-local, so the
runtime composes the vehicle world pose with the LiDAR mount pose before
clustering. Unreal's zero-filled out-of-range LiDAR returns are removed before
this transform.

Ground rejection is local rather than hard-coded to NED `z=0`: the detector
estimates the dominant in-range horizontal return near each UGV, while an
airborne estimate must be several metres below the UAV before it is accepted.
This is important in FlyingCPP, whose visible ground return is around `z=2`.
The round-robin capture scheduler budgets `--sensor-rate` per vehicle across
the full team, so the default rate is not diluted by the number of agents.

Each cluster becomes a conservative sphere (UAV) or planar circle (Husky).
UAV depth patches use 0.75 m fit padding because the camera sees only a partial
surface. UGV proxies use the XY bounding-box footprint with zero extra fit
padding and shift the center 0.75 m away from the LiDAR along the measured
line of sight to approximate the occupied volume behind the returned surface.
These are fixed detector parameters; there is no persistent map, identity
tracker, or online waypoint generator. An empty fresh sensor frame means no
locally detected obstacle; stale or malformed data invokes the controller
fail-safe.

The forward-facing UAV sensor creates a field-of-view limitation: occluded or
behind-camera obstacles are outside the v1 perception guarantee. Every fresh
capture receives a stable ID and stores bounded intermediate point-cloud
samples in a compressed NPZ sidecar. The offline perception report compares
only fresh captures; cached control cycles do not appear as artificial
zero-motion detections.

The CBF equations are unchanged by these perception repairs. The UGV
Mestres adapter uses the existing 0.5 m unicycle lookahead tuned for the
surface-based LiDAR proxy; UAVs retain the existing 1 m model. FlyingCPP uses
one fixed course waypoint `(14, -14, -1)` (AirSim NED); UAVs use it as a
temporary virtual formation center and no online waypoint generation is used.
The orchestrator contains only the minimal AirSim actuation adaptations needed
to realize the requested model commands: a turn-dominated Wang command retains
throttle, and a zero-speed Mestres turn uses a small crawl speed because a car
cannot rotate in place.

For controlled CBF experiments, `--use-truth-obstacles` replaces sensor
proxies with fixed proxies built from the successfully spawned course boxes.
UGVs receive one horizontal bounding circle per box; UAVs receive a bounded
vertical stack of horizontal spheres so mixed-height geometry is represented
without one overly conservative full-box sphere. This mode skips sensor captures for control and labels the
JSONL obstacle view as `truth_obstacle_geometry`; it is not a substitute for
the normal perception path. On maps or runs without spawned orchestrator
boxes, the option explicitly produces an empty truth set and prints a warning.
