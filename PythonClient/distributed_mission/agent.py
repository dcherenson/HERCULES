"""
Agent Class

Represents a single decentralized robotic agent (Drone or UGV) running its local
algorithmic pipeline:
1. Conformal Prediction (Robustness Margin)
2. Cooperative Localization (Self-State Estimation)
3. Target Tracking (Target State Estimation + Nominal Guidance)
4. Distributed CBF (Safety-Critical Control Filtering)
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from modules.conformal_prediction import ConformalPredictionModule
from modules.coop_localization import CooperativeLocalizationModule
from modules.target_tracking import TargetTrackingModule
from modules.cbf import CBFConfig, DistributedCBFModule, ObstacleProxy

class Agent:
    def __init__(self, agent_id: str, vehicle_type: str = "drone", cbf_config: Optional[CBFConfig] = None):
        """
        Initialize a decentralized agent.

        Args:
            agent_id: Unique string identifier (e.g. 'Drone1', 'Husky1').
            vehicle_type: 'drone' or 'ugv'.
        """
        self.agent_id = agent_id
        self.vehicle_type = vehicle_type

        # Initialize internal algorithmic submodules
        self.cp_module = ConformalPredictionModule(agent_id)
        self.loc_module = CooperativeLocalizationModule(agent_id)
        self.tracking_module = TargetTrackingModule(agent_id)
        self.cbf_module = DistributedCBFModule(agent_id, vehicle_type, cbf_config)

        # Internal agent state storage
        self.current_margin = 0.0
        self.local_state_estimate = {}
        self.target_estimate = {}
        self.nominal_velocity = np.zeros(3)
        method = cbf_config.method if cbf_config is not None else "mestres"
        self.nominal_control = np.zeros(2 if vehicle_type == "ugv" and method == "mestres" else 3)
        self.last_cbf_result = None

    def compute_step(self, 
                     sensor_data: Dict[str, Any], 
                     inbound_msgs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the local estimation and computation phase for this round.

        Args:
            sensor_data: Local measurements (IMU, visual odometry, camera detections).
            inbound_msgs: Filtered messages received from in-range neighbors.

        Returns:
            outbound_msgs: Bundle of messages to broadcast to the network.
        """
        # 1. Compute dynamic robustness margin via Conformal Prediction
        self.current_margin = self.cp_module.compute_robustness_margin(
            sensor_data=sensor_data,
            inbound_msgs=inbound_msgs
        )

        # 2. Update Cooperative Localization using local sensors + neighbor info + margin
        self.local_state_estimate, out_loc_msg = self.loc_module.update_and_estimate(
            sensor_data=sensor_data,
            inbound_msgs=inbound_msgs.get("localization", {}),
            robustness_margin=self.current_margin
        )

        # 3. Update Distributed Target Tracking and produce nominal velocity
        self.target_estimate, out_track_msg, self.nominal_velocity = self.tracking_module.update_and_track(
            sensor_data=sensor_data,
            inbound_msgs=inbound_msgs.get("tracking", {}),
            robustness_margin=self.current_margin,
            local_state_estimate=self.local_state_estimate
        )

        outbound_msgs = {
            "localization": out_loc_msg,
            "tracking": out_track_msg,
            "sender_id": self.agent_id
        }

        self.local_state_estimate["agent_id"] = self.agent_id
        self.local_state_estimate["vehicle_type"] = self.vehicle_type
        self.local_state_estimate["yaw"] = float(sensor_data.get("yaw", 0.0))
        self.local_state_estimate["yaw_rate"] = float(sensor_data.get("yaw_rate", 0.0))
        out_loc_msg["vehicle_type"] = self.vehicle_type
        out_loc_msg["yaw"] = self.local_state_estimate["yaw"]
        out_loc_msg["yaw_rate"] = self.local_state_estimate["yaw_rate"]

        return outbound_msgs

    def set_nominal_control(self, nominal_control: np.ndarray) -> None:
        self.nominal_control = np.asarray(nominal_control, dtype=float).reshape(-1)

    def control_step(
        self,
        in_range_neighbor_states: List[Dict[str, Any]],
        obstacles: Optional[List[ObstacleProxy]] = None,
        sensor_valid: bool = True,
    ) -> np.ndarray:
        """
        Runs the safety control phase using Distributed CBFs.

        Args:
            in_range_neighbor_states: Localized states of neighboring agents within safety range.

        Returns:
            safe model-level command ready for actuator dispatch.
        """
        safe_velocity = self.cbf_module.compute_safe_control(
            nominal_velocity=self.nominal_velocity,
            nominal_control=self.nominal_control,
            local_state_estimate=self.local_state_estimate,
            neighbor_states=in_range_neighbor_states,
            robustness_margin=self.current_margin,
            obstacles=obstacles or [],
            sensor_valid=sensor_valid,
        )
        self.last_cbf_result = self.cbf_module.last_result
        return safe_velocity
