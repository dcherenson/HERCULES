#!/usr/bin/env python3
"""
trajectory_editor.py

A Python3 tool to load multiple robot trajectory TXT files and an occupancy grid map image (PNG/PGM),
plot the trajectories on top of the map (with the map flipped vertically so that its horizontal axis is mirrored),
apply a 90° clockwise rotation followed by a horizontal reflection to the displayed waypoints,
and allow interactive dragging of trajectory points to modify trajectories, as well as adding new points by clicking.
Modified trajectories can be saved back to their respective files.

If a YAML file with the same base name as the map (for example, map.pgm and map.yaml) exists,
it will automatically parse and apply 'resolution' and 'origin' from that YAML.

New Features:
 - Each trajectory can be toggled on/off individually via checkboxes labeled by robot name.
 - Drones use a triangular marker (different colors), Huskies use a square marker (different colors).
 - Press 'a' to toggle add-point mode: when enabled, left-clicking on the map will append a new point to the selected trajectory.
 - Press number keys 1–N to choose which trajectory to append points to.

Usage example:
python3 /path/to/trajectory_editor_with_add_point.py \
    --map /path/to/map.pgm \
    --traj Drone1_trajectory.txt Drone2_trajectory.txt
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import CheckButtons


def _parse_yaml_for_map_params(map_img_path):
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
                try:
                    resolution = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('origin:'):
                try:
                    bracket_start = line.find('[')
                    bracket_end = line.find(']')
                    if bracket_start != -1 and bracket_end != -1:
                        inside = line[bracket_start+1:bracket_end]
                        parts = [p.strip() for p in inside.split(',')]
                        if len(parts) >= 2:
                            ox = float(parts[0]); oy = float(parts[1])
                            origin = (ox, oy)
                except ValueError:
                    pass
    return origin, resolution


class Trajectory:
    def __init__(self, filename, color, marker):
        self.filename = filename
        self.color = color
        self.marker = marker
        self.points = self._load_from_file(filename)  # Nx4 array
        self.line = None
        self.scatter = None

    def _load_from_file(self, filename):
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
        return np.array(data)

    def save_to_file(self):
        with open(self.filename, 'w') as f:
            for x, y, z, t in self.points:
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {t:.6f}\n")


class TrajectoryEditor:
    def __init__(self, map_img_path, traj_files, cli_origin, cli_resolution):
        self.map_img_path = map_img_path
        self.traj_files = traj_files
        # Parse YAML or use CLI origin/resolution
        yaml_origin, yaml_resolution = _parse_yaml_for_map_params(self.map_img_path)
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

        # Load and flip map image
        self.map_img = mpimg.imread(self.map_img_path)
        if self.map_img.dtype in (np.float32, np.float64):
            self.map_img = (self.map_img * 255).astype(np.uint8)
        self.map_img = np.flipud(self.map_img)

        # Compute map extents
        h, w = self.map_img.shape[:2]
        ox, oy = self.origin
        res = self.resolution
        self.extent = [ox, ox + w * res, oy, oy + h * res]

        # Prepare trajectories
        cmap = plt.cm.get_cmap('tab10', len(self.traj_files))
        self.trajectories = []
        for idx, tf in enumerate(self.traj_files):
            base = os.path.basename(tf)
            marker = '^' if base.lower().startswith('drone') else ('s' if base.lower().startswith('husky') else 'o')
            self.trajectories.append(Trajectory(tf, color=cmap(idx), marker=marker))

        # Interactive state
        self.selected_traj = None
        self.selected_pt_idx = None
        self.dragging = False
        self.offset = (0, 0)
        self.add_mode = False
        self.current_traj_idx = 0

        # Set up figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)
        self._draw_map_and_trajectories()
        self._create_toggle_buttons()

        # Connect event handlers
        self.fig.canvas.mpl_connect('pick_event',           self.on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event',self.on_release)
        self.fig.canvas.mpl_connect('button_press_event',  self.on_click)
        self.fig.canvas.mpl_connect('key_press_event',     self.on_key)

        # Instructions
        print("=== Trajectory Editor ===")
        print(" - Use the checkboxes on the right to show/hide individual trajectories.")
        print(" - Click and drag trajectory points to move them.")
        print(" - Press 'a' to toggle add-point mode.")
        print(f" - Press 1–{len(self.trajectories)} to select trajectory for new points. (Currently 1)")
        print(" - Press 's' to save all modified trajectories.")
        print(" - Press 'q' to quit without saving.")
        print("=========================")

    def _draw_map_and_trajectories(self):
        self.ax.clear()
        self.ax.imshow(self.map_img, origin='lower', extent=self.extent)
        self.ax.set_xlim(self.extent[0], self.extent[1])
        self.ax.set_ylim(self.extent[2], self.extent[3])
        self.ax.set_aspect('equal')
        self.ax.set_title("Occupancy Grid (Flipped) with Transformed Trajectories")

        for traj in self.trajectories:
            pts = traj.points[:, :2]
            rot = np.column_stack((pts[:,1], -pts[:,0]))
            trans = np.column_stack((rot[:,0], -rot[:,1]))
            xs, ys = trans[:,0], trans[:,1]
            traj.line = self.ax.plot(xs, ys, '-', color=traj.color, linewidth=1.5, alpha=0.8)[0]
            traj.scatter = self.ax.scatter(xs, ys, s=50, color=traj.color,
                                           marker=traj.marker, edgecolors='black', picker=5)
        labels = [os.path.basename(t.filename).replace('_trajectory.txt','') for t in self.trajectories]
        self.checkable = [t.scatter for t in self.trajectories]
        self.ax.legend(self.checkable, labels, loc='upper left', bbox_to_anchor=(0.01,0.99))

    def _create_toggle_buttons(self):
        axbox = self.fig.add_axes([0.80, 0.1, 0.15, 0.8])
        labels = [os.path.basename(t.filename).replace('_trajectory.txt','') for t in self.trajectories]
        visibility = [True] * len(labels)
        self.check = CheckButtons(axbox, labels, visibility)
        self.check.on_clicked(self._toggle_visibility)

    def _toggle_visibility(self, label):
        for traj in self.trajectories:
            name = os.path.basename(traj.filename).replace('_trajectory.txt','')
            if name == label:
                vis = not traj.line.get_visible()
                traj.line.set_visible(vis)
                traj.scatter.set_visible(vis)
                break
        self.fig.canvas.draw_idle()

    def on_pick(self, event):
        for i, traj in enumerate(self.trajectories):
            if event.artist == traj.scatter:
                ind = event.ind
                if not len(ind): return
                self.selected_traj   = i
                self.selected_pt_idx = ind[0]
                self.dragging = True
                x_c, y_c = event.mouseevent.xdata, event.mouseevent.ydata
                x_o, y_o = traj.points[self.selected_pt_idx,0], traj.points[self.selected_pt_idx,1]
                disp_x, disp_y = y_o, x_o
                self.offset = (disp_x - x_c, disp_y - y_c)
                return

    def on_motion(self, event):
        if not self.dragging or self.selected_traj is None: return
        if event.xdata is None or event.ydata is None: return
        new_disp_x = event.xdata + self.offset[0]
        new_disp_y = event.ydata + self.offset[1]
        x_new = new_disp_y; y_new = new_disp_x
        traj = self.trajectories[self.selected_traj]
        traj.points[self.selected_pt_idx,0] = x_new
        traj.points[self.selected_pt_idx,1] = y_new
        self._update_plot(traj)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.dragging:
            self.dragging = False
            self.selected_traj = None
            self.selected_pt_idx = None
            self.offset = (0,0)

    def on_key(self, event):
        if event.key == 'a':
            self.add_mode = not self.add_mode
            print(f"Add-point mode {'ON' if self.add_mode else 'OFF'}")
        elif event.key in [str(i+1) for i in range(len(self.trajectories))]:
            idx = int(event.key)-1
            self.current_traj_idx = idx
            name = os.path.basename(self.traj_files[idx])
            print(f"Selected trajectory for adding: {event.key} ({name})")
        elif event.key == 's':
            for traj in self.trajectories:
                traj.save_to_file()
                print(f"Saved: {traj.filename}")
        elif event.key == 'q':
            print("Quitting without additional saves.")
            plt.close(self.fig)

    def on_click(self, event):
        if not self.add_mode or event.button != 1 or event.inaxes != self.ax:
            return
        x_orig = event.ydata
        y_orig = event.xdata
        traj = self.trajectories[self.current_traj_idx]
        last_z = traj.points[-1,2]
        last_t = traj.points[-1,3]
        new_point = [x_orig, y_orig, last_z, last_t+1.0]
        traj.points = np.vstack([traj.points, new_point])
        self._update_plot(traj)
        self.fig.canvas.draw_idle()

    def _update_plot(self, traj):
        pts = traj.points[:,:2]
        rot = np.column_stack((pts[:,1], -pts[:,0]))
        trans = np.column_stack((rot[:,0], -rot[:,1]))
        xs, ys = trans[:,0], trans[:,1]
        traj.line.set_data(xs, ys)
        traj.scatter.set_offsets(np.c_[xs, ys])

    def run(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive Trajectory Editor with Add-Point Mode and Selection")
    parser.add_argument('--map', required=True, help="Path to occupancy grid map (PNG or PGM).")
    parser.add_argument('--traj', nargs='+', required=True, help="One or more trajectory text files.")
    parser.add_argument('--origin', nargs=2, type=float, default=[0.0, 0.0], metavar=('OX','OY'),
                        help="Map origin if no YAML found [default: 0 0].")
    parser.add_argument('--resolution', type=float, default=1.0,
                        help="Map resolution if no YAML found [default: 1.0].")
    args = parser.parse_args()
    if not os.path.isfile(args.map):
        print(f"Error: Map image file not found: {args.map}")
        sys.exit(1)
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
