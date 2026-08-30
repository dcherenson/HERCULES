#!/usr/bin/env python3

"""
Distributed Multi-Agent Mission Orchestrator

This script coordinates the synchronous/asynchronous execution loop of multiple
decentralized agents performing target tracking, cooperative localization,
conformal prediction uncertainty estimation, and distributed CBF collision avoidance.
"""

import os
import sys
import time
import numpy as np

# Use below in settings.json with Blocks environment
"""
{
    "SettingsVersion": 1.2,
    "SimMode": "Hero",

    "Vehicles": {}
}

"""

# ---------------------------------------------------------
# Global Configuration & Toggles
# ---------------------------------------------------------
USE_SYNCHRONOUS_PHYSICS = True     # True: Steppable Clock (simPause + simContinueForTime), False: Real-time Wall Clock
COMMUNICATION_RANGE_METERS = 30.0 # Maximum radius for inter-agent message passing
CONTROL_DT = 0.1                  # Loop update interval in seconds (10 Hz)
SIMULATION_STEPS = 100            # Total iterations to run (or run indefinitely)

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, ".."))
import hercules_cosysairsim as airsim
from agent import Agent


def build_adjacency_matrix(positions: dict, comm_range: float) -> dict:
    """
    Computes a distance-dependent adjacency map based on true ground truth positions.

    Returns:
        neighbors_map: {agent_id: [list of in-range neighbor agent_ids]}
    """
    agent_ids = list(positions.keys())
    neighbors_map = {a_id: [] for a_id in agent_ids}

    for i in range(len(agent_ids)):
        id_i = agent_ids[i]
        pos_i = positions[id_i]
        for j in range(i + 1, len(agent_ids)):
            id_j = agent_ids[j]
            pos_j = positions[id_j]
            dist = np.linalg.norm(pos_i - pos_j)
            if dist <= comm_range:
                neighbors_map[id_i].append(id_j)
                neighbors_map[id_j].append(id_i)

    return neighbors_map


def main():
    print("=" * 60)
    print("Starting Distributed Multi-Agent Mission Orchestrator")
    print(f"Physics Synchronization: {'SYNCHRONOUS (Steppable)' if USE_SYNCHRONOUS_PHYSICS else 'ASYNCHRONOUS (Real-time)'}")
    print(f"Communication Range:     {COMMUNICATION_RANGE_METERS} meters")
    print(f"Control Rate (dt):       {CONTROL_DT} seconds ({1.0/CONTROL_DT:.1f} Hz)")
    print("=" * 60)

    # 1. Connect to AirSim
    client = airsim.MultirotorClient(port=41451)
    print("Connecting to AirSim simulator...")
    try:
        client.confirmConnection()
        print("Connected to AirSim successfully.")
    except Exception as e:
        print(f"Could not connect to AirSim: {e}")
        print("Please ensure Unreal Engine is running with AirSim in Hero mode.")
        return

    # 2. Define Team Configuration & Initialize Agents
    team_configs = [
        {"id": "Drone1", "type": "drone", "initial_pose": airsim.Pose(airsim.Vector3r(0, -4, -5), airsim.to_quaternion(0, 0, 0))},
        {"id": "Drone2", "type": "drone", "initial_pose": airsim.Pose(airsim.Vector3r(0, -2, -5), airsim.to_quaternion(0, 0, 0))},
        {"id": "Drone3", "type": "drone", "initial_pose": airsim.Pose(airsim.Vector3r(0, 0, -5), airsim.to_quaternion(0, 0, 0))},
        {"id": "Drone4", "type": "drone", "initial_pose": airsim.Pose(airsim.Vector3r(0, 2, -5), airsim.to_quaternion(0, 0, 0))},
        {"id": "Drone5", "type": "drone", "initial_pose": airsim.Pose(airsim.Vector3r(0, 4, -5), airsim.to_quaternion(0, 0, 0))},
    ]

    agents = {}
    for cfg in team_configs:
        agent_id = cfg["id"]
        # Dynamically spawn vehicle if not already present
        try:
            client.simAddVehicle(agent_id, "simpleflight", cfg["initial_pose"])
        except Exception as e:
            print(f"Note on spawning {agent_id}: {e}")

        client.enableApiControl(True, agent_id)
        client.armDisarm(True, agent_id)
        agents[agent_id] = Agent(agent_id=agent_id, vehicle_type=cfg["type"])

    # Arm and initial takeoff for drones
    takeoff_futures = [client.takeoffAsync(vehicle_name=a_id) for a_id in agents.keys()]
    for f in takeoff_futures:
        f.join()
    print("All agents armed, spawned, and hovering.")

    # 3. Setup Physics Stepping Mode
    if USE_SYNCHRONOUS_PHYSICS:
        print("Pausing AirSim physics for steppable synchronous execution...")
        client.simPause(True)

    # Inbound message mailbox for the network: {agent_id: {msg_type: [messages]}}
    network_mailboxes = {a_id: {"localization": {}, "tracking": {}} for a_id in agents.keys()}

    try:
        for step in range(SIMULATION_STEPS):
            step_start_time = time.time()

            # --- PHASE 1: SENSE & GRAPH ---
            # Retrieve ground truth poses to evaluate the true distance-dependent communication graph
            ground_truth_positions = {}
            sensor_data_dict = {}

            for a_id in agents.keys():
                kinematics = client.simGetGroundTruthKinematics(vehicle_name=a_id)
                pos = np.array([kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val])
                vel = np.array([kinematics.linear_velocity.x_val, kinematics.linear_velocity.y_val, kinematics.linear_velocity.z_val])
                ground_truth_positions[a_id] = pos
                # Gather new sensors based on vehicle type
                try:
                    imu_data = client.getImuData(vehicle_name=a_id)
                    thermal_req = airsim.ImageRequest("front_center", airsim.ImageType.ThermalIR, False, False)
                    thermal_res = client.simGetImages([thermal_req], vehicle_name=a_id)
                    
                    dist_data = None
                    lidar_data = None
                    
                    if agent.vehicle_type == "drone":
                        dist_data = client.getDistanceSensorData(distance_sensor_name="Distance", vehicle_name=a_id)
                    elif agent.vehicle_type == "ugv":
                        lidar_data = client.getLidarData(lidar_name="Lidar1", vehicle_name=a_id)
                except Exception as e:
                    print(f"Sensor error on {a_id}: {e}")
                    imu_data = None
                    thermal_res = None

                sensor_data_dict[a_id] = {
                    "position": pos,
                    "velocity": vel,
                    "kinematics": kinematics,
                    "imu": imu_data,
                    "thermal_image": thermal_res,
                    "distance": dist_data,
                    "lidar": lidar_data
                }

            # Evaluate distance-dependent communication topology
            adjacency = build_adjacency_matrix(ground_truth_positions, COMMUNICATION_RANGE_METERS)

            # --- PHASE 2: COMPUTE (Local Estimation & Margins) ---
            outbound_messages = {}
            for a_id, agent in agents.items():
                inbound_for_agent = network_mailboxes[a_id]
                out_msg = agent.compute_step(
                    sensor_data=sensor_data_dict[a_id],
                    inbound_msgs=inbound_for_agent
                )
                outbound_messages[a_id] = out_msg

            # --- PHASE 3: COMMUNICATE (Distance-Gated Routing) ---
            # Reset mailboxes for the next round
            network_mailboxes = {a_id: {"localization": {}, "tracking": {}} for a_id in agents.keys()}

            for sender_id, out_msg in outbound_messages.items():
                for receiver_id in adjacency[sender_id]:
                    # Deliver message only if in communication range
                    network_mailboxes[receiver_id]["localization"][sender_id] = out_msg["localization"]
                    network_mailboxes[receiver_id]["tracking"][sender_id] = out_msg["tracking"]

            # --- PHASE 4: CONTROL (Distributed CBF Safety Filtering) ---
            control_commands = {}
            for a_id, agent in agents.items():
                # Collect localized neighbor states from in-range neighbors
                neighbor_states = [
                    network_mailboxes[a_id]["localization"][n_id]
                    for n_id in adjacency[a_id]
                    if n_id in network_mailboxes[a_id]["localization"]
                ]
                safe_vel = agent.control_step(in_range_neighbor_states=neighbor_states)
                control_commands[a_id] = safe_vel

            # --- PHASE 5: ACTUATION & SIMULATION STEPPING ---
            # Dispatch commands to AirSim actuators
            for a_id, cmd_vel in control_commands.items():
                client.moveByVelocityAsync(
                    vx=float(cmd_vel[0]),
                    vy=float(cmd_vel[1]),
                    vz=float(cmd_vel[2]),
                    duration=CONTROL_DT,
                    vehicle_name=a_id
                )

            # Advance simulation clock
            if USE_SYNCHRONOUS_PHYSICS:
                client.simContinueForTime(CONTROL_DT)
            else:
                elapsed = time.time() - step_start_time
                sleep_time = max(0.0, CONTROL_DT - elapsed)
                time.sleep(sleep_time)

            if step % 10 == 0:
                print(f"[Step {step:03d}/{SIMULATION_STEPS}] Agents: {len(agents)} | "
                      f"Sample Margin (Drone1): {agents['Drone1'].current_margin:.2f}m | "
                      f"Sample NomVel (Drone1): {np.round(agents['Drone1'].nominal_velocity, 2)}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        print("Cleaning up and disarming agents...")
        if USE_SYNCHRONOUS_PHYSICS:
            client.simPause(False)
        for a_id in agents.keys():
            try:
                client.enableApiControl(False, a_id)
            except Exception:
                pass
        print("Orchestration finished.")


if __name__ == "__main__":
    main()
