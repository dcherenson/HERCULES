#!/usr/bin/env python3
"""
Real-time ROS2 Humble Topic Frequency Monitor

Features:
 1. Single Robot View: per-robot tabs + TF
 2. Control Room View: grid of all robots
 3. Measurement Mode: arrival-count vs. header-stamp-based

Accurately computes frequency by counting messages in 1s windows (count mode) or
by header timestamp intervals (header mode). Displays sliding 15s timeseries for count,
and sample-index timeseries for header, with dynamic xticks in count mode.
"""
import time
import threading
import re
from collections import defaultdict, deque
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu, NavSatFix, MagneticField, FluidPressure
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage

# Configuration
WINDOW_DURATION = 15.0    # seconds shown on X-axis for count mode
SAMPLE_INTERVAL_MS = 1000 # GUI update interval
ROBOT_PREFIX = '/hercules_node/'
PERCEPTION_KEYS = ['Scene/image', 'DepthPlanar/image', 'Segmentation/image', '/lidar/points', '/lidar/labels']
OTHER_KEYS = ['imu', 'global_gps', 'barometer', 'magnetometer', 'odom_local']
MAX_HEADER_SAMPLES = 50   # number of header-based samples to track

# Friendly labels
def get_topic_label(topic: str) -> str:
    if 'Scene/image' in topic: return 'rgb_image'
    if 'DepthPlanar/image' in topic: return 'depth_image'
    if 'Segmentation/image' in topic: return 'segmentation_image'
    if '/lidar/points' in topic: return 'lidar_points'
    if '/lidar/labels' in topic: return 'lidar_labels'
    return topic.strip('/')

class FrequencyMonitor(Node):
    def __init__(self):
        super().__init__('frequency_monitor')
        self.lock = threading.Lock()
        # arrival timestamps per topic
        self.arrival_times = defaultdict(deque)
        # instantaneous header-based frequencies per topic
        self.header_freqs = defaultdict(lambda: deque(maxlen=MAX_HEADER_SAMPLES))
        self.last_header = {}
        # topics by robot
        self.robot_topics = defaultdict(lambda: {'perception': [], 'other': []})
        qos = QoSProfile(depth=10)
        # subscribe to robot topics
        for topic, _ in self.get_topic_names_and_types():
            if not topic.startswith(ROBOT_PREFIX): continue
            m = re.match(rf'{ROBOT_PREFIX}([^/]+)/(.+)', topic)
            if not m: continue
            robot, _ = m.groups()
            msg_type = None; category = None
            if any(k in topic for k in PERCEPTION_KEYS):
                category = 'perception'
                if topic.endswith('/image'): msg_type = Image
                elif '/camera_info' in topic: msg_type = CameraInfo
                elif '/lidar/' in topic: msg_type = PointCloud2
            elif any(k in topic for k in OTHER_KEYS):
                category = 'other'
                if topic.endswith('imu'): msg_type = Imu
                elif 'global_gps' in topic: msg_type = NavSatFix
                elif 'barometer' in topic: msg_type = FluidPressure
                elif 'magnetometer' in topic: msg_type = MagneticField
                elif 'odom_local' in topic: msg_type = Odometry
            if category and msg_type:
                self.create_subscription(msg_type, topic, self._cb(topic), qos)
                self.robot_topics[robot][category].append(topic)
        # always include /tf
        self.create_subscription(TFMessage, '/tf', self._cb('/tf'), qos)
        self.robot_topics['TF']['perception'].append('/tf')
        self.robot_topics['TF']['other'].append('/tf')
        self.get_logger().info('Initialized robots: ' + ', '.join(self.robot_topics.keys()))

    def _cb(self, topic_name):
        def callback(msg):
            now = time.monotonic()
            with self.lock:
                # arrival-based
                self.arrival_times[topic_name].append(now)
                # header-based instantaneous
                stamp = None
                if hasattr(msg, 'header'):
                    s = msg.header.stamp
                    stamp = s.sec + s.nanosec * 1e-9
                elif isinstance(msg, TFMessage) and msg.transforms:
                    s = msg.transforms[0].header.stamp
                    stamp = s.sec + s.nanosec * 1e-9
                if stamp is not None:
                    prev = self.last_header.get(topic_name)
                    if prev is not None:
                        dt = stamp - prev
                        if dt > 0:
                            self.header_freqs[topic_name].append(1.0/dt)
                    self.last_header[topic_name] = stamp
        return callback

class FrequencyGUI:
    def __init__(self, monitor: FrequencyMonitor):
        self.monitor = monitor
        self.t0 = time.monotonic()
        self.count_history = defaultdict(lambda: deque(maxlen=int(WINDOW_DURATION)))
        # build GUI
        self.root = tk.Tk(); self.root.title("ROS2 Topic Frequency Monitor")
        # controls
        ctrl = ttk.Frame(self.root); ctrl.pack(side='top', fill='x')
        self.view_mode = tk.StringVar(value='single')
        ttk.Radiobutton(ctrl, text='Single View',      variable=self.view_mode, value='single',  command=self._refresh).pack(side='left')
        ttk.Radiobutton(ctrl, text='Control Room',     variable=self.view_mode, value='control', command=self._refresh).pack(side='left')
        self.measure_mode = tk.StringVar(value='count')
        ttk.Radiobutton(ctrl, text='Count Mode',       variable=self.measure_mode, value='count').pack(side='left')
        ttk.Radiobutton(ctrl, text='Header Mode',      variable=self.measure_mode, value='header').pack(side='left')
        # single-view tabs
        self.tabs = ttk.Notebook(self.root)
        self.selected = {}; self.frames = {}; self.figures = {}; self.canvases = {}
        # control-room frame
        self.control = ttk.Frame(self.root)
        self.control_cat = tk.StringVar(value='perception')
        self.ctrl_figs = {}; self.ctrl_cans = {}
        # build single tabs
        for robot in sorted(self.monitor.robot_topics.keys()):
            f = ttk.Frame(self.tabs); self.tabs.add(f, text=robot)
            self.frames[robot] = f
            self.selected[robot] = tk.StringVar(value='perception')
            bf = ttk.Frame(f); bf.pack(side='top', fill='x')
            ttk.Button(bf, text='Perception', command=lambda r=robot: self.selected[r].set('perception')).pack(side='left')
            ttk.Button(bf, text='Other',      command=lambda r=robot: self.selected[r].set('other')).pack(side='left')
            fig = plt.Figure(figsize=(6,3), dpi=100); self.figures[robot] = fig
            c = FigureCanvasTkAgg(fig, master=f); c.get_tk_widget().pack(fill='both', expand=True)
            self.canvases[robot] = c
        # build control-room grid
        cols = 3
        bf = ttk.Frame(self.control); bf.grid(row=0, column=0, columnspan=cols, sticky='ew', pady=5)
        ttk.Button(bf, text='Perception All', command=lambda:self.control_cat.set('perception')).pack(side='left')
        ttk.Button(bf, text='Other All',      command=lambda:self.control_cat.set('other')).pack(side='left')
        for c in range(cols): self.control.columnconfigure(c, weight=1)
        robots = sorted(self.monitor.robot_topics.keys())
        for i,robot in enumerate(robots):
            r = i//cols; c = i%cols
            self.control.rowconfigure(r+1, weight=1)
            lf = ttk.LabelFrame(self.control, text=robot)
            lf.grid(row=r+1, column=c, sticky='nsew', padx=4, pady=4)
            fig = plt.Figure(figsize=(4,2), dpi=80); self.ctrl_figs[robot] = fig
            c = FigureCanvasTkAgg(fig, master=lf); c.get_tk_widget().pack(fill='both', expand=True)
            self.ctrl_cans[robot] = c
        # start loop
        self._refresh(); self._update()

    def _refresh(self):
        if self.view_mode.get() == 'single':
            self.control.pack_forget(); self.tabs.pack(fill='both', expand=True)
        else:
            self.tabs.pack_forget(); self.control.pack(fill='both', expand=True)

    def _update(self):
        now = time.monotonic()
        # update count-history
        with self.monitor.lock:
            for topic, dq in self.monitor.arrival_times.items():
                while dq and dq[0] < now - WINDOW_DURATION:
                    dq.popleft()
                cnt = sum(1 for t in dq if t > now - 1.0)
                self.count_history[topic].append((now, cnt))
        # for count-mode, compute xticks
        end_rel = now - self.t0; start_rel = end_rel - WINDOW_DURATION
        ticks_rel = [0, WINDOW_DURATION/3, 2*WINDOW_DURATION/3, WINDOW_DURATION]
        labels = [f"{start_rel + tr:.0f}" for tr in ticks_rel]
        count_mode = (self.measure_mode.get() == 'count')
        # plot single or control
        if self.view_mode.get() == 'single':
            for robot in self.frames:
                cat = self.selected[robot].get(); fig = self.figures[robot]; fig.clear()
                ax = fig.add_subplot(111)
                ax.set_title(f"{robot} - {cat}")
                ax.set_xlabel('Time (s)' if count_mode else 'Sample'); ax.set_ylabel('Hz')
                ax.grid(True); ax.minorticks_on()
                for topic in self.monitor.robot_topics[robot][cat]:
                    if count_mode:
                        hist = self.count_history.get(topic, ())
                        if not hist: continue
                        xs = [t - self.t0 - start_rel for t,_ in hist]
                        ys = [f for _,f in hist]
                    else:
                        hist = list(self.monitor.header_freqs.get(topic, ()))
                        if not hist: continue
                        xs = list(range(len(hist)))
                        ys = hist
                    ax.plot(xs, ys, label=get_topic_label(topic))
                if count_mode:
                    ax.set_xlim(0, WINDOW_DURATION); ax.set_xticks(ticks_rel); ax.set_xticklabels(labels)
                else:
                    ax.set_xlim(0, max(xs) if xs else MAX_HEADER_SAMPLES)
                ax.set_ylim(0, None); ax.legend(fontsize=6)
                self.canvases[robot].draw()
        else:
            for robot, fig in self.ctrl_figs.items():
                fig.clear(); ax = fig.add_subplot(111)
                cat = self.control_cat.get()
                ax.set_title(f"{robot} - {cat}")
                ax.set_xlabel('Time (s)' if count_mode else 'Sample'); ax.set_ylabel('Hz')
                ax.grid(True); ax.minorticks_on()
                for topic in self.monitor.robot_topics[robot][cat]:
                    if count_mode:
                        hist = self.count_history.get(topic, ())
                        if not hist: continue
                        xs = [t - self.t0 - start_rel for t,_ in hist]
                        ys = [f for _,f in hist]
                    else:
                        hist = list(self.monitor.header_freqs.get(topic, ()))
                        if not hist: continue
                        xs = list(range(len(hist)))
                        ys = hist
                    ax.plot(xs, ys, label=get_topic_label(topic))
                if count_mode:
                    ax.set_xlim(0, WINDOW_DURATION); ax.set_xticks(ticks_rel); ax.set_xticklabels(labels)
                else:
                    ax.set_xlim(0, max(xs) if xs else MAX_HEADER_SAMPLES)
                ax.set_ylim(0, None); ax.legend(fontsize=5)
                self.ctrl_cans[robot].draw()
        self.root.after(SAMPLE_INTERVAL_MS, self._update)

    def run(self): self.root.mainloop()


def main():
    rclpy.init()
    mon = FrequencyMonitor()
    gui = FrequencyGUI(mon)
    t = threading.Thread(target=rclpy.spin, args=(mon,), daemon=True)
    t.start()
    gui.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()