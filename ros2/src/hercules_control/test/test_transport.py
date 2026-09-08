"""Exercise the real C++ executable against synthetic ROS transport, never Unreal."""
import os
import signal
import subprocess
import time

import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from airsim_interfaces.msg import CarControls, VelCmd
from airsim_interfaces.srv import Takeoff, Land
from nav_msgs.msg import Odometry


@pytest.mark.parametrize('vehicle,mode,scenario', [
    ('ugv', 'observe', 'normal'),
    ('ugv', 'motion', 'normal'),
    ('drone', 'motion', 'normal'),
    ('drone', 'motion', 'normal_async'),
    ('drone', 'motion', 'interrupt'),
    ('ugv', 'motion', 'interrupt'),
    ('ugv', 'motion', 'stale'),
    ('drone', 'motion', 'service_failure'),
    ('ugv', 'benchmark', 'normal'),
])
def test_transport(vehicle, mode, scenario, tmp_path):
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node('synthetic_smoke_transport', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    prefix = '/synthetic_smoke'
    publisher = node.create_publisher(Odometry, prefix + '/odom', 5)
    commands = []
    state = dict(x=0.0, z=0.0, vx=0.0, landed=False, after_land=0)

    def command(msg):
        if vehicle == 'ugv':
            state['vx'] = msg.throttle if msg.brake == 0 else 0.0
            commands.append((time.monotonic(), msg.throttle, msg.brake))
        else:
            state['vx'] = msg.twist.linear.x
            commands.append((time.monotonic(), msg.twist.linear.x, 0))
            if state['landed']:
                state['after_land'] += 1

    sub = node.create_subscription(CarControls if vehicle == 'ugv' else VelCmd,
                                   prefix + '/command', command, 10)

    def takeoff(request, response):
        assert request.wait_on_last_task == (scenario != 'normal_async')
        response.success = scenario != 'service_failure'
        if response.success:
            state['z'] = 3.0
        return response

    def land(request, response):
        assert request.wait_on_last_task == (scenario != 'normal_async')
        state['z'] = 0.0
        state['landed'] = True
        response.success = True
        return response

    services = [node.create_service(Takeoff, prefix + '/takeoff', takeoff),
                node.create_service(Land, prefix + '/land', land)]
    logfile = tmp_path / 'smoke.log'
    with logfile.open('w') as output:
        process = subprocess.Popen([
            os.environ['SMOKE_EXE'], '--ros-args',
            '-p', f'vehicle_type:={vehicle}', '-p', f'mode:={mode}',
            '-p', f'odom_topic:={prefix}/odom', '-p', f'command_topic:={prefix}/command',
            '-p', f'takeoff_service:={prefix}/takeoff', '-p', f'land_service:={prefix}/land',
            '-p', f"wait_on_flight_task:={str(scenario != 'normal_async').lower()}",
            '-p', 'duration_sec:=0.4', '-p', 'warmup_sec:=0.1',
        ], stdout=output, stderr=subprocess.STDOUT)
        started = last = time.monotonic()
        sent_interrupt = False
        try:
            while process.poll() is None and time.monotonic() - started < 16:
                now = time.monotonic()
                state['x'] += state['vx'] * (now - last)
                last = now
                has_motion = any(c[1] > 0 for c in commands)
                if not (scenario == 'stale' and has_motion):
                    msg = Odometry()
                    stamp = int((now - started + 1) * 1e9)
                    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1000000000)
                    msg.pose.pose.orientation.w = 1.0
                    msg.pose.pose.position.x = state['x']
                    msg.pose.pose.position.z = state['z']
                    msg.twist.twist.linear.x = state['vx']
                    publisher.publish(msg)
                if scenario == 'interrupt' and has_motion and not sent_interrupt:
                    process.send_signal(signal.SIGINT)
                    sent_interrupt = True
                executor.spin_once(timeout_sec=0.02)
            assert process.poll() is not None, 'smoke executable failed to terminate'
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            # Drain final DDS samples before checking cleanup.
            for _ in range(10):
                executor.spin_once(timeout_sec=0.01)
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown(context=context)
    log = logfile.read_text()
    assert process.returncode == (0 if scenario.startswith('normal') else 1), log
    assert 'SMOKE_RESULT' in log
    if mode == 'observe':
        assert not commands
    elif mode == 'motion':
        if scenario != 'service_failure':
            assert any(c[1] > 0 for c in commands), log
        assert commands[-1][1] == 0, 'final command must stop'
        if vehicle == 'ugv':
            assert commands[-1][2] == 1
            assert max(c[1] for c in commands) <= 0.150001
        else:
            assert state['landed'] and state['after_land'] == 0
    else:
        assert commands and all(c[1] == 0 for c in commands)
        assert 'BENCHMARK_RESULT' in log
