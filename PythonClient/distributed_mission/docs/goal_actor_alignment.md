# RuralAustralia goal actor and startup alignment

## Automatic goal selection

On RuralAustralia, the orchestrator first keeps the course goal as a fallback,
then asks Unreal for generic user-created actor names matching `Actor` or
`Actor_<number>`. A candidate is accepted when it has a valid world-NED pose,
is away from the launch point, and lies substantially opposite the old course
direction. This prevents map infrastructure from becoming the goal by
accident. In the current map the placed actor is `Actor_0`, at approximately
`[1.875, 36.309, 2.040]` NED.

Use `--goal-actor Actor_0` to select it explicitly. If the actor is not found,
the mission prints a warning and uses the course goal. On other maps, actor
selection is only attempted when `--goal-actor` is supplied.

## Formation and camera

The detected goal direction determines the initial vehicle heading. The UAV
box and UGV triangle offsets are rotated into that heading; the CBF and
formation equations are unchanged. The external `ExternalCamera` actor is
moved behind the goal direction and pointed along the route. `CameraDirector`
is a compatibility fallback for AirSim builds that do not expose the child
camera name. Vehicle-mounted camera poses are never changed.

For `--top-down-camera`, the camera keeps the requested top-down height and
uses the opposite yaw convention needed to make forward travel appear away
from the camera. Without that option, the existing pitched CameraDirector
view is retained in height/distance and is reoriented toward the goal.

The goal actor pose is already in the RuralAustralia world frame, so it is not
subjected to the course-frame rotation or the Rural ground offset that apply
to the fallback course geometry.
