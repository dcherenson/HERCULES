# Mission recording and presentation animation

Video capture is opt-in because Unreal scene capture adds GPU and disk load to
the simulation:

```bash
cd /Users/dmrc/git/HERCULES/PythonClient/distributed_mission
python3 orchestrator.py --launch-mode headless --record-video
```

The defaults capture a 1280x720, 20 fps chase stream, Drone1 front-camera
stream, and Husky1 front-camera stream. Outputs are played at 2x mission time
by default, so their output rate is 40 fps. To capture at 10 fps and play at
20 fps, use:

```bash
python3 orchestrator.py --launch-mode headless --record-video \
  --video-fps 10 --playback-speed 2
```

Each stream produces an MP4 and a 540-pixel-high GIF in the configured debug
directory. Select different vehicles with `--record-uav NAME` and
`--record-ugv NAME`. Set `--playback-speed 1` for real-time playback.

Every normal run also produces the route-up top-down animation unless
`--no-animation` is supplied. It uses the same 2x playback speed by default;
its MP4 is 720x720 and its GIF is 540x540. It contains agent paths, UAV camera
footprints, truth obstacles, local obstacle beliefs, actual same-type
communication links, and collision markers. The
existing static 3D trajectory and collision-clearance PNGs remain available;
the old animated 3D perception output has been replaced.

The chase view is route-aligned and automatically backs away when the team
spreads out. The reserved `mission_follow` camera is detached from Drone1 and
commanded explicitly in Unreal-world NED; it is not treated as a vehicle-local
camera. UAVs have green screen-space rings and UGVs have blue rings. The rings
are added after Unreal capture, so foliage cannot hide them. The live
ExternalCamera follows the same pose in visible mode. Camera roll is forced to
zero. Each cycle logs the commanded pose and calibration offset; each captured
image logs the returned AirSim pose in the staging metadata.

The mission starts a dedicated background AirSim `simGetImages` worker after
startup and takeoff stabilization. The worker captures Unreal SceneCapture
images for the three selected cameras into timestamped PNG staging frames
beneath a run-specific folder; this is used because the current AirSim fork's
native `startRecording()` can report active without writing files in headless
mode. The worker is off the control thread, and its capture errors are written
to the manifest.

The post-run encoder converts and verifies all requested MP4s before deleting
that folder. Use `--keep-recording-frames` to retain the source images, or to
preserve them for diagnosing a failed encode. A JSON recording manifest lists
camera names, source counts, output paths, timing, duplicate/dropped frame
counts, and capture-worker errors. Recording runs also produce
`<run>_camera_alignment.json` and `.md`, which compare each captured chase
image's returned pose with the logged commanded pose.

## Chase-camera diagnosis

When debugging camera placement, retain the staging frames:

```bash
python3 orchestrator.py --launch-mode headless --record-video \
  --video-fps 5 --playback-speed 1 --keep-recording-frames \
  --debug-dir /tmp/hercules-camera-debug
```

Inspect the run's `*_recording_frames/capture_metadata.jsonl`. Each chase
image contains the pose returned by the same `simGetImages` request that
produced the PNG. Compare `camera_position` with the cycle's
`recording.chase_camera.world_position`; the difference should stay below a
small rendering tolerance. `camera_base_offset` records the fixed Unreal
camera spawn offset that is compensated before each command. This is the
authoritative check—do not use `simGetCameraInfo` for `mission_follow` on this
fork because its detached-camera implementation can crash Unreal. The
post-run chase overlay also uses the per-frame image-response pose, so a
projection mismatch is visible independently of the commanded pose.

Recording requires `--launch-mode visible` or `--launch-mode headless` because
the additional external camera and native recording settings must be supplied
before Unreal starts. `--launch-mode existing --record-video` fails early and
does not alter the running simulator.

The recording pipeline does not change CBF, formation, localization, or
obstacle-detection code. It only adds a recording camera, logging metadata,
and post-run media processing.
