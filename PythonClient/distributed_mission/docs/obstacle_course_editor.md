# Interactive obstacle-course editor

`tools/obstacle_course_editor.py` is a small top-down Matplotlib editor for
placing explicit obstacle offsets in AirSim's world NED frame. It is a layout
tool, not an online planner: it does not modify the CBF equations, generate
mission waypoints, or change controller gains.

## Start the editor

From `PythonClient/distributed_mission`:

```bash
../../herculesvenv/bin/python tools/obstacle_course_editor.py \
  --output debug/edited_obstacle_course.json
```

This opens with the current built-in FlyingCPP course. To start empty:

```bash
../../herculesvenv/bin/python tools/obstacle_course_editor.py \
  --empty --output debug/my_obstacle_course.json
```

To edit a previous layout, pass `--input path/to/course.json`.

## Controls

- Drag an existing obstacle to move its X/Y center.
- Press `b`, then click to add a box; press `s`, then click to add a sphere.
- Arrow keys nudge the selected obstacle by 0.25 m.
- `+`/`-` resize the selected obstacle in X/Y. For spheres this changes the
  radius.
- `u`/`n` move the selected obstacle physically up/down by 0.5 m (`u`
  decreases AirSim NED Z).
- `delete` removes the selected obstacle.
- `w` or `enter` saves; `q` saves and exits.

The default dimensions and radius can be changed at startup with
`--box-dimensions X Y Z` and `--sphere-radius R`. The goal can be set with
`--goal X Y Z`.

## Use the generated offsets

The JSON contains explicit `center`, `dimensions`, and/or `radius` values:

```bash
../../herculesvenv/bin/python orchestrator.py \
  --launch-mode headless --map flyingcpp --cbf-method mestres \
  --obstacle-course debug/my_obstacle_course.json \
  --use-truth-obstacles --steps 800 --no-animation \
  --debug-dir debug/edited_course_run
```

Omit `--use-truth-obstacles` to evaluate the same layout with live perception.
Boxes use the existing `1M_Cube_Chamfer` asset. Spheres request the AirSim
`SM_Sphere` asset. The orchestrator checks the asset registry first; if that
mesh is unavailable, it safely falls back to a cube with the sphere's
diameter and records `requested_shape: "sphere"` in the truth log. The JSON
remains the source of truth for the requested offsets, while the truth log
records the geometry that actually spawned.

The editor's optional `waypoints` list is displayed and logged as persistent
Unreal markers, but it is not consumed as an online route by formation
control. The current mission goal is always plotted in Unreal as well.
