#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
import json
import os
from pathlib import Path

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..teleoperator import Teleoperator
from .config_kuka_leader import KUKALeaderConfig

# from pydrake.all import MultibodyPlant, Parser, DiagramBuilder, JacobianWrtVariable
import pinocchio as pin
from pinocchio.utils import rotate  # Now available with proper pinocchio
from pinocchio.visualize import MeshcatVisualizer
import meshcat
from meshcat.geometry import Sphere, MeshLambertMaterial, Cylinder
import scipy.spatial.transform

import numpy as np

logger = logging.getLogger(__name__)


class KukaLeader(Teleoperator):
    """
    Kuka Leader Arm for teleoperation.
    """

    config_class = KUKALeaderConfig
    name = "kuka_leader"

    def __init__(self, config: KUKALeaderConfig):
        # Ensure calibration_dir is a Path before calling super().__init__
        if config.calibration_dir is None:
            config.calibration_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        else:
            config.calibration_dir = Path(config.calibration_dir)
        super().__init__(config)
        urdf_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        urdf_path = urdf_dir / "Kuka_leader_description/urdf/Kuka_leader_links.urdf"
        # self.builder = DiagramBuilder()
        # self.plant = MultibodyPlant(time_step=0.0)
        # self.model_instance = Parser.AddModels("lerobot/common/teleoperators/kuka_leader/description/kuka_description.urdf")
        # self.plant.Finalize()
        # self.context = self.plant.CreateDefaultContext()
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        # Alternative: self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.config = config

        # Set up Meshcat viewer
        self.vis = None
        # self.vis = MeshcatVisualizer(self.model, pin.GeometryModel(), pin.GeometryModel(), meshcat.Visualizer())
        # self.vis.initViewer(open=True)
        # self.vis.loadViewerModel()

        self.t_static = time.perf_counter()
        self.if_gripping = False
        self.prev_trigger_tau = 0
        self.gripper_effort_limit = 2

        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "joint1": Motor(1, "sts3215", MotorNormMode.RADIANS),
                "joint2": Motor(2, "sts3215", MotorNormMode.RADIANS),
                "joint3": Motor(3, "sts3215", MotorNormMode.RADIANS),
                "joint4": Motor(4, "sts3215", MotorNormMode.RADIANS),
                "joint5": Motor(5, "sts3215", MotorNormMode.RADIANS),
                "joint6": Motor(6, "sts3215", MotorNormMode.RADIANS),
                "joint7": Motor(7, "sts3215", MotorNormMode.RADIANS),
            },
            calibration=self.calibration,
        )
        # if self.bus.is_calibrated:
        self.trigger_limits = (-1, 1)
        
        
        self.joint_limits = {
            "joint1": (-2.6878, 2.6878),  # limits in radians
            "joint2": (-1.701, 1.701),
            "joint3": (-1.482, 1.482), # -170*1000
            "joint4": (-1.74, 1.74), # [+- 100*1000]
            "joint5": (-1.22, 1.22), # [+-70*1000]
            "joint6": (-1.745, 1.745),
            "joint7": (0.0, 0.08), 
        }

        self.home_positions = {
            "joint1": -0.12,
            "joint2": -0.2,
            "joint3": 0.2,
            "joint4": 0,
            "joint5": 1,
            "joint6": -0.08,
            "joint7": 0.0735, 
        }

        self.rest_positions = {
            "joint1": 0,
            "joint2": -1.69,
            "joint3": 1.61,
            "joint4": 0,
            "joint5": 0.55,
            "joint6": -0.1,
            "joint7": 0.0, 
        }
        
        self.is_homed = True
        self.is_going_to_rest = False
        self.position_reached = False


        self.if_synced = False
        self.joint_integrals = np.zeros(6)
        
        # Delta limiting parameters for smooth robot following
        self.max_joint_delta = {
            "joint1": 0.03,  # rad per timestep (conservative starting values)
            "joint2": 0.02,
            "joint3": 0.02,
            "joint4": 0.05,
            "joint5": 0.05,
            "joint6": 0.08,
            "joint7": 0.01,
        }
        
        # Robot following force parameters
        self.robot_following_gains = {
            "joint1": 3.0,  # Force gain when robot lags behind leader
            "joint2": 4.0,
            "joint3": 4.0,
            "joint4": 2.0,
            "joint5": 2.0,
            "joint6": 1.5,
            "joint7": 2.0,
        }
        
        self.last_robot_position = None
        self.position_lag_threshold = 0.05  # rad - threshold for applying drag-back forces

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()

        # Check if calibration file exists at the same path as used in calibrate (self.calibration_fpath)
        calibration_file = self.calibration_dir / f"{self.id}.json"

        if not self.is_calibrated and calibrate:
            # if calibration_file.exists():
            #     logger.info(f"Calibration file exists at {calibration_file}.")
            #     self.load_calibration()
            # else:
            #     logger.info(f"Calibration file does not exist at {calibration_file}.")
            self.calibrate()
            

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    
    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        # Save calibration file in the KukaLeader directory
        self.bus.write_calibration(self.calibration)
        # calibration_fpath = self.calibration_dir / f"{self.id}_kuka_calibration.json"
        # self._save_calibration(fpath=calibration_fpath)
        # logger.info(f"Calibration saved to {calibration_fpath}")
        self._save_calibration()
        logger.info("Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        # self.bus.enable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            # if motor == "joint7":
            #     self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            # else:
            self.bus.write("Operating_Mode", motor, OperatingMode.PWM.value)
        logger.info("motors configed")

    def setup_motors(self) -> None:
        # Original implementation:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

        # Updated implementation to skip the last two motors:
        # motors_to_setup = list(self.bus.motors)[:-2]  # Skip the last two motors
        # for motor in reversed(motors_to_setup):
        #     input(f"Connect the controller board to the '{motor}' motor only and press enter.")
        #     self.bus.setup_motor(motor)
        #     print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")
    
    def _get__trigger_limits(self):
        trigger_min = self.bus.calibration["joint7"].range_min
        trigger_max = self.bus.calibration["joint7"].range_max
        trigger_mid = (trigger_min + trigger_max) / 2
        max_res = self.bus.model_resolution_table[self.bus._id_to_model(7)] - 1
        trigger_limits = ((trigger_min - trigger_mid)/max_res * 2*np.pi, \
                            (trigger_max - trigger_mid)/max_res * 2*np.pi)
        return trigger_limits

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        action["joint7.pos"] = (action["joint7.pos"] - self._get__trigger_limits()[0]) \
            / (self._get__trigger_limits()[1] - self._get__trigger_limits()[0]) * 0.08
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        valid_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        q = np.array([action[f"{joint}.pos"] for joint in valid_joints])

        if self.vis is not None:
            self._visualize_joint_origins(q)
        # self._print_joint_model_info()
        return action
    
    def get_limited_action(self, current_robot_pos: dict[str, float]) -> dict[str, float]:
        """
        Get leader action with delta limiting applied to prevent robot from lagging too far behind
        """
        raw_action = self.get_action()
        
        if current_robot_pos is None:
            return raw_action
            
        # Apply delta limiting based on what the robot can actually achieve
        limited_action = self._limit_position_delta(current_robot_pos, raw_action)
        
        return limited_action
    
    def get_velocity(self) -> dict[str, float]:
        start = time.perf_counter()
        velocity = self.bus.sync_read("Present_Velocity")
        velocity = {f"{motor}.vel": val for motor, val in velocity.items()}
        # velocity["joint7.vel"] = 0
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return velocity

    
    def get_load(self) -> dict[str, float]:
        start = time.perf_counter()
        load = self.bus.sync_read("Present_Load")
        load = {f"{motor}.load": val for motor, val in load.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read load: {dt_ms:.1f}ms")
        return load
    
    # def sync_leader_position(self, action, velocity, observation):
    #     local_sync_flag = True
    #     for joint, leader_pos in action.items():
    #         follower_pos = observation[joint]
    #         leader_vel = velocity[joint.replace(".pos", ".vel")]
    #         diff = leader_pos - follower_pos
    #         if abs(diff) > 0.1 and leader_vel > 0.05:
    #             # print(f"Leader {joint} position {leader_pos:.2f} out of sync with follower {follower_pos:.2f}, adjusting...")
    #             local_sync_flag = False
    #     self.if_synced = local_sync_flag
    #     return local_sync_flag

    def if_arm_ready(self):
        return self.if_synced
    
    def _check_if_synced(self, q_leader, q_follower, q_dot):
        for i in range(len(q_leader)):
            joint_diff = q_leader[i] - q_follower[i]
            if np.abs(joint_diff) > 0.2 or (np.abs(q_dot[i]) > 0.1):
                return False
        return True

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Sync write to Goal_Position using feedback dict
        # feedback keys may be like 'shoulder_pan.pos', so strip '.pos' if present
        #TODO: replace implmentation with force loop
        print(f"Sending feedback: {feedback}")
        for motor, value in feedback.items():
            motor = motor.split(".")[0]
            self.bus.write("Goal_Position", motor, value)
        # logger.info(f"{self} sent feedback: {feedback}")

    def send_force_feedback(self, observation, effort_feedback):
        tau, effort_to_send = self._compute_force_output(observation, effort_feedback)
        # tau = np.append(tau, 0)
        tau_dict = {f"{joint}.tau": t for joint, t in zip(self.bus.motors.keys(), tau)}
        effort_dict = {f"{joint}.effort": t for joint, t in zip(self.bus.motors.keys(), effort_to_send)}
        # print("Tau table:", tau_dict)
        for motor, value in tau_dict.items():
            motor = motor.split(".")[0]
            pwm_int = np.round(value * 100).astype(int)
            
            # VOLTAGE PROTECTION: Limit PWM to safe range for STS3215 servos
            # STS3215 safe PWM range is typically ±100 (adjust if needed)
            max_pwm = 165  # Conservative limit to prevent voltage errors
            pwm_int = np.clip(pwm_int, -max_pwm, max_pwm)
            
            self.bus.write("Goal_Time", motor, pwm_int)
        # logger.info(f"{self} test sent feedback: {feedback}")
        # print("CoM:", self.model.inertias[2].lever)
        return effort_dict

    def _print_joint_model_info(self):
        for jname in self.model.names[1:]:
            j_id = self.model.getJointId(jname)
            joint = self.model.joints[j_id]
            print(f"{jname}: axis = {joint.axis}, placement translation = {joint.placement.translation}")

    def _compute_force_output(self, observation, effort):
        action = self.get_action()
        velocity = self.get_velocity()

        # Build q and q_dot in correct order
        valid_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        q = np.array([action[f"{joint}.pos"] for joint in valid_joints])
        q_dot = np.array([velocity[f"{joint}.vel"] for joint in valid_joints])
        q_dot_ee = np.array([velocity[f"{joint}.vel"] for joint in self.bus.motors.keys()])
        q_follower = np.array([observation[f"{joint}.pos"] for joint in valid_joints])
        trigger_pos = action["joint7.pos"]
        trigger_vel = velocity["joint7.vel"]
        gripper_pos = observation["joint7.pos"]
        gripper_effort = effort["joint7.effort"]

        # Store robot position for delta limiting and following forces
        self.last_robot_position = observation

        # pin.forwardKinematics(self.model, self.data, q)
        # joint2_id = self.model.getJointId("joint2")
        tau_g = self._compute_gravity_compensation(q, q_dot)
        tau_ss = self._compute_static_friction_compensation(q_dot_ee, freq=500)
        tau_vf = self._compute_viscous_friction_compensation(q_dot_ee)
        
        # Robot following forces - drag leader back when robot lags
        #tau_robot_following = self._compute_robot_following_forces(action, observation)
        
        if self.is_homed:
            tau_joint = self._compute_joint_diff_compensation(q, q_dot, q_follower, valid_joints)
        else:
            tau_joint = self._lead_to_home(action, velocity)
        if self.is_going_to_rest:
            tau_joint = self._lead_to_position(action, velocity, self.rest_positions)
            self.is_going_to_rest = not self.position_reached
        tau_trigger, gripper_effort_to_send = self._compute_gripper_force(trigger_pos, trigger_vel, gripper_pos, gripper_effort)
        tau_trigger = 0

        tau = tau_vf + tau_g + tau_joint + tau_trigger + tau_ss# + tau_robot_following
        # tau = tau_trigger
        tau = self._safe_guard_torque(tau)
        effort_to_send = np.zeros(6)
        effort_to_send = np.append(effort_to_send, gripper_effort_to_send)
        return tau, effort_to_send

    # Gravity compensation
    def _compute_gravity_compensation(self, q, q_dot):
        tau_g_full = pin.rnea(self.model, self.data, q, q_dot, q_dot*0)

        tau_g = np.zeros_like(tau_g_full)
        tau_g[1] = -tau_g_full[1]
        # tau_g[1] = tau_g[1]
        tau_g[2] = -tau_g_full[2]
        tau_g[4] = -tau_g_full[4]
        tau_g = np.append(tau_g, 0)
        return tau_g

    def _compute_static_friction_compensation(self, q_dot, freq=500):
        tau_ss = np.zeros_like(q_dot)
        if_increment_time = False
        dt = 0.000001
        q_threshold = 0.09
        us = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4]
        if self.if_gripping:
            us = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3]
        for i, qd in enumerate(q_dot):
            if abs(qd) < q_threshold:
                if_increment_time = True
                tau_ss[i] = us[i] * np.cos(np.pi * (time.perf_counter() - self.t_static) * freq)
            else:
                tau_ss[i] = 0
        if if_increment_time:
            self.t_static += dt
        else:
            self.t_static = time.perf_counter()
        # print(time.perf_counter() - self.t_static)
        return tau_ss
    
    def _compute_viscous_friction_compensation(self, q_dot):
        tau_vf = np.zeros_like(q_dot)
        q_threshold = 0.1
        uc = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.75]
        uv = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.75]
        # uc = [0.33, 0.3, 0.33, 0.33, 0.33, 0.33, 0.45]
        # uv = [0.33, 0.3, 0.33, 0.33, 0.33, 0.33, 0.55]
        # if self.if_gripping:
        #     uc = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3]
        #     uv = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
        for i, qd in enumerate(q_dot):
            if abs(qd) > q_threshold:
                tau_vf[i] = uc[i] * np.sign(qd) + uv[i] * qd
        return tau_vf * -1
    
    def _compute_joint_diff_compensation(self, q_leader, q_dot, q_follower, valid_joints):
        Kp = [1, 5, 7, 3, 3, 3]
        Kd = 0.03 * np.ones_like(Kp)
        Ki = 0 * np.ones_like(Kp)
        tau_joint = np.zeros_like(q_leader)
        if self._check_if_synced(q_leader, q_follower, q_dot):
            self.if_synced = True
        for i, q_leader_val in enumerate(q_leader):
            joint_diff = q_leader_val - q_follower[i]
            # if np.abs(joint_diff) > 0.3 and (np.abs(q_leader_val) < self.joint_limits[valid_joints[i]][1]):
            #     if_synced_local_flag = False
            # print(f"Joint {valid_joints[i]} diff: {joint_diff:.3f}, integral: {self.joint_integrals[i]:.3f}, tau_joint: {tau_joint[i]:.3f}")
            if np.abs(joint_diff) > 0.01 and (np.abs(q_leader_val) > self.joint_limits[valid_joints[i]][1] or not self.if_synced):
                tau_joint[i] = Kp[i] * joint_diff - Kd[i] * q_dot[i] + Ki[i] * self.joint_integrals[i]
                self.joint_integrals[i] += joint_diff * 0.2
                if self.joint_integrals[i] > 0.5:
                    self.joint_integrals[i] = 0.5
                elif self.joint_integrals[i] < -0.5:
                    self.joint_integrals[i] = -0.5
            else:
                self.joint_integrals[i] = 0
                tau_joint[i] = 0
        
        # self.if_synced = if_synced_local_flag
        # print("if_synced:", self.if_synced, "tau_joint:", tau_joint)
        tau_joint = np.append(tau_joint, 0)
        return tau_joint
    
    def _compute_robot_following_forces(self, q_leader, robot_actual_pos):
        """
        Compute forces to drag the leader arm back toward the robot's actual position
        when the robot is lagging behind due to speed/acceleration limits
        """
        tau_following = np.zeros(7)
        
        # if robot_actual_pos is None:
        #     return tau_following
        
        # # Convert observation dict to array (remove .pos suffix)
        # robot_pos_array = np.array([
        #     robot_actual_pos.get(f"joint{i+1}.pos", 0.0) for i in range(6)
        # ])
        # robot_pos_array = np.append(robot_pos_array, robot_actual_pos.get("joint7.pos", 0.0))
        
        # q_leader_full = np.array([q_leader[f"joint{i+1}.pos"] for i in range(7)])
        
        # for i, joint_name in enumerate([f"joint{j+1}" for j in range(7)]):
        #     position_lag = q_leader_full[i] - robot_pos_array[i]
            
        #     # Only apply forces if the lag is significant
        #     if abs(position_lag) > self.position_lag_threshold:
        #         # Force proportional to lag, pulling leader back toward robot
        #         force_magnitude = self.robot_following_gains[joint_name] * position_lag
        #         tau_following[i] = -force_magnitude  # Negative to pull back
                
        return tau_following
    
    def _limit_position_delta(self, current_action, target_action):
        """
        Limit the change in leader position to prevent too-rapid movements
        that the robot cannot follow
        """
        if self.last_robot_position is None:
            return target_action
        
        limited_action = target_action.copy()
        
        for joint in target_action:
            if joint.endswith('.pos'):
                joint_name = joint.replace('.pos', '')
                
                if joint_name in self.max_joint_delta:
                    current_val = current_action.get(joint, 0.0)
                    target_val = target_action[joint]
                    delta = target_val - current_val
                    
                    max_delta = self.max_joint_delta[joint_name]
                    
                    if abs(delta) > max_delta:
                        limited_delta = np.sign(delta) * max_delta
                        limited_action[joint] = current_val + limited_delta
                        
        return limited_action
        
    def _compute_gripper_force(self, trigger_pos, trigger_vel, gripper_pos, gripper_effort):
        trigger_tau = 0
        tau = np.zeros(6)
        gripper_effort_to_send = 0
        trigger_diff  = gripper_pos - trigger_pos
        if trigger_diff > 0.001 and gripper_effort < -0.25:
            # trigger_tau = -(50 * (gripper_pos - trigger_pos))# - 0.01 * trigger_vel + 0.5 * gripper_effort
            trigger_tau = -(25 * trigger_diff) - 0.2#0.1 * (-50*(trigger_diff) + (1 - 0.1) * self.prev_trigger_tau)
            gripper_effort_delta =  trigger_diff * 40
            gripper_effort_to_send = 0.5 + gripper_effort_delta
            self.prev_trigger_tau = trigger_tau
            self.if_gripping = True
        else:
            self.if_gripping = False
        if trigger_pos > 0.08:
            trigger_tau = -50 * trigger_diff
        tau = np.append(tau, trigger_tau)
        if gripper_effort_to_send > self.gripper_effort_limit:
            gripper_effort_to_send = self.gripper_effort_limit
        elif gripper_effort_to_send < -self.gripper_effort_limit:
            gripper_effort_to_send = -self.gripper_effort_limit
        print(f"Gripper force: {gripper_effort_to_send}, trigger_pos: {trigger_pos}, gripper_pos: {gripper_pos}, trigger_tau: {trigger_tau}")
        return tau, gripper_effort_to_send
    
    def lead_to_home(self):
        self.is_homed = False
        print("Moving leader arm to home position...")

    def _lead_to_home(self, action, velocity):
        Kp = [1, 5, 7, 3, 3, 3, 1]
        Kd = 0.03 * np.ones_like(Kp)
        Ki = 0 * np.ones_like(Kp)
        tau_joint = np.zeros(7)
        q = np.array([action[f"{joint}.pos"] for joint in self.bus.motors.keys()])
        q_dot = np.array([velocity[f"{joint}.vel"] for joint in self.bus.motors.keys()])
        q_home = np.array([self.home_positions[f"{joint}"] for joint in self.bus.motors.keys()])
        if self._check_if_synced(q, q_home, q_dot):
            self.is_homed = True
        for i, q_leader_val in enumerate(q):
            joint_diff = q_leader_val - q_home[i]
            if np.abs(joint_diff) > 0.01:
                tau_joint[i] = Kp[i] * joint_diff - Kd[i] * q_dot[i] 
                # self.joint_integrals[i] += joint_diff * 0.2
                # if self.joint_integrals[i] > 0.5:
                #     self.joint_integrals[i] = 0.5
                # elif self.joint_integrals[i] < -0.5:
                #     self.joint_integrals[i] = -0.5
            else:
                tau_joint[i] = 0

        return tau_joint

    def lead_to_rest(self):
        self.is_going_to_rest = True

    def _lead_to_position(self, action, velocity, target):
        Kp = [1, 5, 7, 3, 3, 3, 1]
        Kd = 0.03 * np.ones_like(Kp)
        Ki = 0 * np.ones_like(Kp)
        tau_joint = np.zeros(7)
        q = np.array([action[f"{joint}.pos"] for joint in self.bus.motors.keys()])
        q_dot = np.array([velocity[f"{joint}.vel"] for joint in self.bus.motors.keys()])
        target_q = np.array([target[f"{joint}"] for joint in self.bus.motors.keys()])
        self.position_reached = self._check_if_synced(q, target_q, q_dot)
        for i, q_leader_val in enumerate(q):
            joint_diff = q_leader_val - target_q[i]
            if np.abs(joint_diff) > 0.01:
                tau_joint[i] = Kp[i] * joint_diff - Kd[i] * q_dot[i] 
                # self.joint_integrals[i] += joint_diff * 0.2
                # if self.joint_integrals[i] > 0.5:
                #     self.joint_integrals[i] = 0.5
                # elif self.joint_integrals[i] < -0.5:
                #     self.joint_integrals[i] = -0.5
            else:
                tau_joint[i] = 0

        return tau_joint

    def _safe_guard_torque(self, tau):
        """
        Ensure the torque does not exceed the limits.
        """
        for i, joint in enumerate(self.bus.motors.keys()):
            if tau[i] > 2.5:
                tau[i] = 2.5
            elif tau[i] < -2.5:
                tau[i] = -2.5
        return tau
    
    def update_force_feedback_params(self, **kwargs):
        """
        Update force feedback parameters for real-time tuning
        
        Usage:
        leader.update_force_feedback_params(
            robot_following_gains={"joint1": 2.0, "joint2": 3.0},
            max_joint_delta={"joint1": 0.05},
            position_lag_threshold=0.08
        )
        """
        if 'robot_following_gains' in kwargs:
            self.robot_following_gains.update(kwargs['robot_following_gains'])
        if 'max_joint_delta' in kwargs:
            self.max_joint_delta.update(kwargs['max_joint_delta'])
        if 'position_lag_threshold' in kwargs:
            self.position_lag_threshold = kwargs['position_lag_threshold']
            
        logger.info(f"Updated force feedback parameters: {kwargs}")

    def _visualize_joint_origins(self, q=None):
        """
        Visualize joint origins and frames using Meshcat.
        If q is None, use zero configuration.
        """
        if q is None:
            q = np.zeros(self.model.nq)

        # Compute FK
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Display frames and markers
        for jname in self.model.names[1:]:
            j_id = self.model.getJointId(jname)
            placement = self.data.oMi[j_id]
            joint = self.model.joints[j_id]
            
            # Set transform for the frame
            self.vis.viewer[jname].set_transform(placement.homogeneous)
            # self.draw_axis(jname + "/axes")

            # Add marker at joint origin
            self.vis.viewer[f"{jname}_marker"].set_object(Sphere(0.01), MeshLambertMaterial(color=0xff0000))
            self.vis.viewer[f"{jname}_marker"].set_transform(placement.homogeneous)

            # Compute CoM position in world frame
            inertia = self.model.inertias[j_id]
            com_world = placement.act(inertia.lever)

            # Add CoM marker
            self.vis.viewer[f"com/{jname}"].set_object(
                Sphere(0.008), MeshLambertMaterial(color=0x00ffff)  # cyan CoM
            )
            self.vis.viewer[f"com/{jname}"].set_transform(
                pin.SE3(np.eye(3), com_world).homogeneous
            )

            self.draw_z_axis(jname, placement)

                    
        # print("Visualization running. Open Meshcat in your browser to inspect.")
    
    def draw_z_axis(self, name, placement, length=0.05, radius=0.001):
        # Draw Z axis (blue cylinder pointing along local Z)
        self.vis.viewer[f"z_axis/{name}"].set_object(
            Cylinder(length, radius),
            MeshLambertMaterial(color=0x0000ff)
        )
        # Transform so it starts at origin and points along Z
        R = rotate('x', -np.pi/2)
        t = np.zeros(3)
        T_comp = pin.SE3(R, t)
        T = placement * T_comp#* pin.SE3(np.eye(3), np.array([0, 0, length / 2]))
        self.vis.viewer[f"z_axis/{name}"].set_transform(T.homogeneous)

    @staticmethod
    def _compute_tau_null(J, q, q_dot, q_rest, Knp, Knd):
        J_pinv = np.linalg.pinv(J)
        N = np.eye(J.shape[1]) - J_pinv @ J
        u = -Knp * (q - q_rest) - Knd * q_dot
        return N @ u


    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")

    def load_calibration(self) -> None:
        """
        Load calibration from the JSON file, set self.calibration, and write to motors.
        """
        with open(self.calibration_fpath, "r") as f:
            calib_dict = json.load(f)
        self.calibration = {
            motor: MotorCalibration(**params)
            for motor, params in calib_dict.items()
        }
        self.bus.write_calibration(self.calibration)
        logger.info(f"Loaded calibration from {self.calibration_fpath} and wrote to motors.")

    # def _save_calibration(self) -> None:
    #     # Ensure the calibration file is saved inside the teleoperator directory
    #     calibration_dir = self.config.teleoperator_dir  # Assuming teleoperator_dir is defined in the config
    #     calibration_path = calibration_dir / "calibration.json"

    #     with open(calibration_path, "w") as f:
    #         json.dump(self.calibration, f, indent=4)

    #     logger.info(f"Calibration saved to {calibration_path}")

        # def _calculate_force_output(self):
    #     action = self.get_action()
    #     velocity = self.get_velocity()
    #     # q = np.array(list(action.values()))
    #     # q_dot = np.array(list(velocity.values()))
    #     keys = list(self.bus.motors.keys())[:-1]  # All except last
    #     # q = np.array([action[f"{joint}.pos"] for joint in keys])
    #     # q_dot = np.array([velocity[f"{joint}.vel"] for joint in keys])
    #     q = np.array([0.1, 0, 0, 0, 0, 0])
    #     q_dot = np.zeros(6)


    #     # self.plant.SetPositions(self.context, q)
    #     # self.plant.SetVelocities(self.context, q_dot)
    #     # frame = self.plant.GetFrameByName("link6")
    #     # J = self.plant.CalcJacobianSpatialVelocity(
    #     # self.context,
    #     # JacobianWrtVariable.kV,
    #     # frame,
    #     # [0, 0, 0],
    #     # self.plant.world_frame(),
    #     # self.plant.world_frame()
    #     # )
    #     # q_rest = np.zeros_like(q)
    #     # Knp = np.ones_like(q)
    #     # Knd = 0.1 * np.ones_like(q)
    #     # tau_null = self._compute_tau_null(J, q, q_dot, q_rest, Knp, Knd)
    #     # return tau_null

    #     self.model.gravity = pin.Motion.Zero()
    #     pin.forwardKinematics(self.model, self.data, q, q_dot)
    #     pin.computeJointJacobians(self.model, self.data, q)

    #     frame_id = self.model.getFrameId("link6")
    #     J = pin.getFrameJacobian(self.model, self.data, frame_id, pin.LOCAL_WORLD_ALIGNED)
    #     print("Jacobian rank:", np.linalg.matrix_rank(J))

    #     q_rest = np.zeros_like(q)
    #     Knp = np.ones_like(q)
    #     Knd = 0.1 * np.ones_like(q)

    #     tau_null = self._compute_tau_null(J, q, q_dot, q_rest, Knp, Knd)
    #     return tau_null
