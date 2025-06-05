#!/usr/bin/env python3
"""
trajectory_editor.py

A Python3 tool to load multiple robot trajectory TXT files and an occupancy grid map image (PNG/PGM),
plot the trajectories on top of the map, and allow interactive dragging of trajectory points
to modify trajectories. Modified trajectories can be saved back to their respective files.

If a YAML file with the same base name as the map (for example, map.pgm and map.yaml) exists,
it will automatically parse and apply 'resolution' and 'origin' from that YAML.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def _parse_yaml_for_map_params(map_img_path):
    """
    Given a map image path (e.g. ".../Ausenv_ground_OGM_0p5m.pgm"),
    look for a YAML file with the same base name (.../Ausenv_ground_OGM_0p5m.yaml).
    If found, extract 'resolution' (float) and 'origin' ([ox, oy, oz]) → return (origin, resolution).
    Otherwise, return (None, None).
    """
    base, _ = os.path.splitext(map_img_path)
    yaml_path = base + '.yaml'
    if not os.path.isfile(yaml_path):
        return None, None

    resolution = None
    origin = None

    with open(yaml_path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith('resolution:'):
                # e.g. "resolution: 0.5"
                try:
                    resolution = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('origin:'):
                # e.g. "origin: [-500, -500, 0]"
                try:
                    bracket_start = line.find('[')
                    bracket_end = line.find(']')
                    if bracket_start != -1 and bracket_end != -1:
                        inside = line[bracket_start+1 : bracket_end]
                        parts = [p.strip() for p in inside.split(',')]
                        if len(parts) >= 2:
                            ox = float(parts[0])
                            oy = float(parts[1])
                            origin = (ox, oy)
                except ValueError:
                    pass

    return origin, resolution


class Trajectory:
    def __init__(self, filename, color):
        """
        Represents a single trajectory loaded from filename.
        Stores points as Nx4 numpy array: [x, y, z, timestamp].
        """
        self.filename = filename
        self.color = color
        self.points = self._load_from_file(filename)  # Nx4 array
        self.line = None
        self.scatter = None

    def _load_from_file(self, filename):
        """
        Load trajectory from a text file. Each line: "X Y Z timestamp".
        Returns an Nx4 numpy array.
        """
        data = []
        with open(filename, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(f"Line {idx+1} in {filename} does not have 4 elements: {line}")
                x, y, z, t = map(float, parts)
                data.append([x, y, z, t])
        return np.array(data)  # shape (N, 4)

    def save_to_file(self):
        """
        Save current trajectory points back to the original file.
        Keeps Z and timestamp unchanged, writes updated X and Y.
        """
        with open(self.filename, 'w') as f:
            for pt in self.points:
                x, y, z, t = pt
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {t:.6f}\n")


class TrajectoryEditor:
    def __init__(self, map_img_path, traj_files, cli_origin, cli_resolution):
        self.map_img_path = map_img_path
        self.traj_files = traj_files

        # First, attempt to parse YAML for origin & resolution
        yaml_origin, yaml_resolution = _parse_yaml_for_map_params(self.map_img_path)

        # Use YAML values if available; otherwise fall back to CLI inputs
        if yaml_origin is not None:
            self.origin = yaml_origin
            print(f"Loaded origin from YAML: {self.origin}")
        else:
            self.origin = cli_origin

        if yaml_resolution is not None:
            self.resolution = yaml_resolution
            print(f"Loaded resolution from YAML: {self.resolution}")
        else:
            self.resolution = cli_resolution

        # Load map image (PNG or PGM)
        self.map_img = mpimg.imread(self.map_img_path)
        if self.map_img.dtype == np.float32 or self.map_img.dtype == np.float64:
            # If image was loaded as normalized floats, convert to 0-255 uint8
            self.map_img = (self.map_img * 255).astype(np.uint8)

        # Compute image extents in world coordinates
        h, w = self.map_img.shape[:2]
        ox, oy = self.origin
        res = self.resolution
        self.extent = [ox, ox + w * res, oy, oy + h * res]

        # Prepare trajectory objects
        colors = plt.cm.get_cmap('tab10', len(self.traj_files))
        self.trajectories = []
        for idx, tf in enumerate(self.traj_files):
            traj = Trajectory(tf, color=colors(idx))
            self.trajectories.append(traj)

        # State for interactive dragging
        self.selected_traj = None       # index of selected trajectory
        self.selected_pt_idx = None     # index of selected point within trajectory
        self.dragging = False
        self.offset = (0, 0)            # offset between click and point

        # Initialize figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        self._draw_map_and_trajectories()

        # Connect event handlers
        self.cid_pick = self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Instructions
        print("=== Trajectory Editor ===")
        print(" - Click and drag trajectory points to move them.")
        print(" - Press 's' to save all modified trajectories.")
        print(" - Press 'q' to quit without saving.")
        print("=========================")

    def _draw_map_and_trajectories(self):
        """Draw the occupancy grid map and all trajectories."""
        # Draw the occupancy grid background
        self.ax.imshow(self.map_img, origin='lower', extent=self.extent)
        self.ax.set_xlim(self.extent[0], self.extent[1])
        self.ax.set_ylim(self.extent[2], self.extent[3])
        self.ax.set_aspect('equal')
        self.ax.set_title("Occupancy Grid with Trajectories")

        # Overlay each trajectory (line + scatter)
        for traj in self.trajectories:
            xs = traj.points[:, 0]
            ys = traj.points[:, 1]
            traj.line, = self.ax.plot(xs, ys, '-', color=traj.color, linewidth=1.5, alpha=0.8)
            traj.scatter = self.ax.scatter(xs, ys, s=50, color=traj.color, edgecolors='black', picker=5)

        # Legend shows the file names
        labels = [os.path.basename(traj.filename) for traj in self.trajectories]
        self.ax.legend(labels)

    def on_pick(self, event):
        """
        Handle pick event when user clicks near a trajectory point.
        """
        for idx, traj in enumerate(self.trajectories):
            if event.artist == traj.scatter:
                ind = event.ind
                if len(ind) == 0:
                    return
                pt_idx = ind[0]
                self.selected_traj = idx
                self.selected_pt_idx = pt_idx
                self.dragging = True
                # Calculate offset between click and actual point location
                x_click, y_click = event.mouseevent.xdata, event.mouseevent.ydata
                x_pt, y_pt = traj.points[pt_idx, 0], traj.points[pt_idx, 1]
                self.offset = (x_pt - x_click, y_pt - y_click)
                return

    def on_motion(self, event):
        """
        Handle mouse movement: update the selected point's position as the mouse drags.
        """
        if not self.dragging or self.selected_traj is None or self.selected_pt_idx is None:
            return
        if event.xdata is None or event.ydata is None:
            return  # out of bounds

        # Compute new world‐coordinates for this point
        new_x = event.xdata + self.offset[0]
        new_y = event.ydata + self.offset[1]

        traj = self.trajectories[self.selected_traj]
        traj.points[self.selected_pt_idx, 0] = new_x
        traj.points[self.selected_pt_idx, 1] = new_y

        # Update line + scatter on the plot
        xs = traj.points[:, 0]
        ys = traj.points[:, 1]
        traj.scatter.set_offsets(np.c_[xs, ys])
        traj.line.set_data(xs, ys)

        self.fig.canvas.draw_idle()

    def on_release(self, event):
        """
        Handle mouse button release: end dragging.
        """
        if self.dragging:
            self.dragging = False
            self.selected_traj = None
            self.selected_pt_idx = None
            self.offset = (0, 0)

    def on_key(self, event):
        """
        Handle key press: 's' saves all trajectories, 'q' quits without more saving.
        """
        if event.key == 's':
            for traj in self.trajectories:
                traj.save_to_file()
                print(f"Saved: {traj.filename}")
        elif event.key == 'q':
            print("Quitting without additional saves.")
            plt.close(self.fig)

    def run(self):
        """Start the interactive Matplotlib loop."""
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive Trajectory Editor")
    parser.add_argument('--map', required=True,
                        help="Path to occupancy grid map (PNG or PGM).")
    parser.add_argument('--traj', nargs='+', required=True,
                        help="One or more trajectory text files to load and edit.")
    parser.add_argument('--origin', nargs=2, type=float, default=[0.0, 0.0],
                        metavar=('OX', 'OY'),
                        help="Map origin (world coordinates) if no YAML is found [default: 0 0].")
    parser.add_argument('--resolution', type=float, default=1.0,
                        help="Map resolution (world units/pixel) if no YAML is found [default: 1.0].")
    args = parser.parse_args()

    # Validate that the map image exists
    if not os.path.isfile(args.map):
        print(f"Error: Map image file not found: {args.map}")
        sys.exit(1)

    # Validate each trajectory file exists
    for tf in args.traj:
        if not os.path.isfile(tf):
            print(f"Error: Trajectory file not found: {tf}")
            sys.exit(1)

    cli_origin = (args.origin[0], args.origin[1])
    cli_resolution = args.resolution

    editor = TrajectoryEditor(args.map, args.traj, cli_origin, cli_resolution)
    editor.run()


if __name__ == "__main__":
    main()
