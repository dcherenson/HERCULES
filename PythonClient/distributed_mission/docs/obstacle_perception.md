# Local obstacle perception

UAVs use `front_center` `DepthPerspective` images. Huskies use `Lidar1`.
Measurements are transformed into world coordinates, range filtered,
downsampled, ground filtered, and clustered with DBSCAN. Large clusters are
split along their principal axis so each proxy has bounded size. UAV depth
responses already contain a world camera pose. Husky LiDAR points are
sensor-local, so the runtime composes the vehicle world pose with the LiDAR
mount pose before clustering. Unreal's zero-filled out-of-range LiDAR returns
are removed before this transform.

Each cluster becomes a conservative sphere (UAV) or planar circle (Husky).
Proxies are ranked by surface clearance and capped at the configurable default
of five. There is no persistent map or object identity. An empty fresh sensor
frame means no locally detected obstacle; stale or malformed data invokes the
controller fail-safe.

The forward-facing UAV sensor creates a field-of-view limitation: occluded or
behind-camera obstacles are outside the v1 perception guarantee. Every fresh
capture receives a stable ID and stores bounded intermediate point-cloud
samples in a compressed NPZ sidecar. The offline perception report compares
only fresh captures; cached control cycles do not appear as artificial
zero-motion detections.
