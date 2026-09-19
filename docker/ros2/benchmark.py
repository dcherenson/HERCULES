"""Measure neutral commands at the ROS adapter and existing wrapper boundaries."""
import csv
import datetime
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time


def fields(line):
    return dict(re.findall(r'(\w+)=([^\s]+)', line))


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)


def main():
    host = os.environ.get('AIRSIM_HOST', '127.0.0.1')
    drone_port = int(os.environ.get('AIRSIM_MULTIROTOR_PORT', '41451'))
    car_port = int(os.environ.get('AIRSIM_CAR_PORT', '41452'))
    for port in (drone_port, car_port):
        with socket.create_connection((host, port), timeout=2):
            pass
    nodes = subprocess.check_output(['ros2', 'node', 'list'], text=True)
    if '/hercules_drone' in nodes or '/hercules_ugv' in nodes:
        raise RuntimeError('Stop your wrappers first; the benchmark manages its own wrappers.')
    output = Path(os.environ['HERCULES_BUILD_ROOT']) / 'validation' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output.mkdir(parents=True)
    print(f'Benchmark logs: {output}', flush=True)
    rows = []
    for period, active in [(0.05, False), (0.05, True), (0.01, True)]:
        label = f'{period}-' + ('command' if active else 'passive')
        wrapper_log = output / (label + '-wrapper.log')
        wrapper = cpu = None
        with wrapper_log.open('w') as wrapper_file, (output / (label + '-cpu.log')).open('w') as cpu_file:
            try:
                wrapper = subprocess.Popen([
                    'ros2', 'launch', 'airsim_ros_pkgs', 'hercules_host.launch.py',
                    f'host_ip:={host}', f'drone_port:={drone_port}', f'ugv_port:={car_port}',
                    f'enable_api_control:={str(active).lower()}', 'benchmark_logging:=true',
                    f'update_airsim_control_every_n_sec:={period}',
                ], stdout=wrapper_file, stderr=subprocess.STDOUT, start_new_session=True)
                cpu = subprocess.Popen(['pidstat', '-h', '-u', '-p', 'ALL', '1'], stdout=cpu_file,
                                       stderr=subprocess.STDOUT, start_new_session=True)
                for vehicle in ('drone', 'ugv'):
                    for rate in ([10, 20, 50, 100] if active else [20]):
                        mode = 'benchmark' if active else 'observe'
                        log = output / f'{label}-{vehicle}-{rate}.log'
                        with log.open('w') as stream:
                            subprocess.run([
                                'ros2', 'run', 'hercules_control', 'smoke_node', '--ros-args',
                                '-p', f'vehicle_type:={vehicle}', '-p', f'mode:={mode}',
                                '-p', f'command_rate_hz:={float(rate)}',
                                '-p', f'warmup_sec:={5.0 if active else 0.0}',
                                '-p', f'duration_sec:={15.0 if active else 30.0}',
                            ], stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=45)
                        result = next(fields(line) for line in log.read_text().splitlines() if 'BENCHMARK_RESULT' in line)
                        # Select only wrapper samples lying within the node's steady-time window.
                        name = 'Drone1' if vehicle == 'drone' else 'Husky1'
                        samples = [fields(line) for line in wrapper_log.read_text().splitlines() if 'HERCULES_METRICS' in line]
                        samples = [s for s in samples if s['vehicle'] == name and
                                   float(result['start_s']) <= float(s['steady_s']) <= float(result['end_s'])]
                        if len(samples) < 2:
                            raise RuntimeError('Insufficient wrapper metrics; refusing to infer RPC rate from publication rate')
                        first, last = samples[0], samples[-1]
                        elapsed = float(last['steady_s']) - float(first['steady_s'])
                        row = {'vehicle': vehicle, 'mode': mode, 'wrapper_period_s': period,
                               'requested_hz': rate if active else 0, **result}
                        for key in ('state_reads', 'unique_stamps', 'odom_published', 'commands_received', 'rpc_dispatched'):
                            row[key + '_hz'] = (int(last[key]) - int(first[key])) / elapsed
                        rows.append(row)
                        print(row, flush=True)
            finally:
                stop(wrapper)
                stop(cpu)
        # Discovery needs time to remove the just-stopped participants.
        time.sleep(2)
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('RPC dispatch is async submission for drones, synchronous return for UGVs; neither is a physics-step counter.')


if __name__ == '__main__':
    main()
