# Testing

Run offline tests with:

```bash
./herculesvenv/bin/python -m pytest PythonClient/distributed_mission/tests
```

The suite covers formation geometry, heading wrapping, barrier behavior,
solver failure, same-type filtering, depth back-projection, clustering, nearest
five selection, launcher modes, invalid configuration, NED-to-Z-up conversion,
camera/LiDAR wireframe geometry, cached sensor-age propagation, and post-run
artifact generation. The post-run test creates a synthetic MP4 and checks its
file size, frame count, and duration with `ffprobe`; ensure `ffmpeg` and
`ffprobe` are on `PATH`. Perception diagnostics tests also cover bounded NPZ
capture samples, cached-capture deduplication, proxy association, and corrupted
frame invariants. AirSim smoke tests should be added/run separately with a
`--airsim` marker once Unreal is available.

The acceptance criterion is collision-free behavior for represented
same-type agents and local obstacle proxies. Goal completion and deadlock-free
routing are not v1 requirements.
