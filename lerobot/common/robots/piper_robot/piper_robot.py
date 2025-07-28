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
from functools import cached_property
from typing import Any
from dataclasses import dataclass, field, replace

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger  # Common for simple enable/reset/stop services

from ..robot import Robot
# from ..utils import ensure_safe_goal_position
from .config_piper_robot import PiperRobotConfig
import math
import numpy as np

logger = logging.getLogger(__name__)

class PiperRobot(Robot):

    config_class = PiperRobotConfig
    name = "piper_robot"

    def __init__(self, config: PiperRobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.node_name = f"{self.name}_piper"
        self.is_connected = False
        self.logs = {}

        self.joints = {
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
        }

        self.joint_limits = {
            "joint1": (-2.6878, 2.6878),  # Example limits in radians
            "joint2": (0, 3.403),
            "joint3": (-2.965, 0), # -170*1000
            "joint4": (-1.74, 1.74), # [+- 100*1000]
            "joint5": (-1.22, 1.22), # [+-70*1000]
            "joint6": (-1.745, 1.745),
            "joint7": (0.0, 0.08),  # Example limits for gripper
        }

        # self.home_position = {
        #     "joint1": 0.0,
        #     "joint2": 0,
        #     "joint3": 0,
        #     "joint4": 0.0,
        #     "joint5": 1,
        #     "joint6": 0.0,
        #     "joint7": 0.0, 
        # } # In leader arm frame !!!
        # self.home_position = {
        #     "joint1": -0.71,
        #     "joint2": 0.01,
        #     "joint3": 0.33,
        #     "joint4": -0.01,
        #     "joint5": 1.23,
        #     "joint6": 0.34,
        #     "joint7": 0.07, 
        # } 
        
        self.home_position = {
            "joint1": -0.12,   # -0.12 rad
            "joint2": -0.2,
            "joint3": 0.2,
            "joint4": 0,
            "joint5": 1,
            "joint6": -0.08,
            "joint7": 0.0735, 
        } 

        #-0.46570246800000004, 1.823107328, -1.58531072, -0.08441151600000002, 1.2499149320000003, -0.17218972400000002, 0.0735
        self.is_homed = False
        self.zero_velocity = {f"{joint}.vel": 0.0 for joint in self.joints}

        # self.joint_limits = {
        #     joint: (math.radians(lim[0]), math.radians(lim[1]))
        #     for joint, lim in self.joint_limits_degrees.items() if joint != "joint7"
        # }

        self.transform = {
            "joint1": (-1, 0),
            "joint2": (1, 1.55),
            "joint3": (1, -1.63),
            "joint4": (-1, 0),
            "joint5": (1, 0),
            "joint6": (-1, 0),
            "joint7": (1, 0),  
        }


        # Initialize ROS node (anonymous=True to allow multiple launches)
        rospy.init_node(self.node_name, anonymous=True)

        # Publishers
        self.enable_pub = rospy.Publisher('/right_arm/enable_flag', Bool, queue_size=10)
        self.joint_pub = rospy.Publisher('/right_arm/joint_ctrl_single', JointState, queue_size=10)

        # Subscribers (add callbacks as needed)
        self.arm_status_sub = rospy.Subscriber('/right_arm/arm_status', rospy.AnyMsg, self._arm_status_callback)
        self.end_pose_sub = rospy.Subscriber('/right_arm/end_pose', PoseStamped, self._end_pose_callback)
        self.end_pose_euler_sub = rospy.Subscriber('/right_arm/end_pose_euler', rospy.AnyMsg, self._end_pose_euler_callback)
        self.joint_states_single_sub = rospy.Subscriber('/right_arm/joint_states_single', JointState, self._joint_states_single_callback)
        self.pos_cmd_sub = rospy.Subscriber('/right_arm/pos_cmd', JointState, self._pos_cmd_callback)

        # Service proxies
        self.enable_srv = rospy.ServiceProxy('/enable_srv', Trigger)
        self.go_zero_srv = rospy.ServiceProxy('/go_zero_srv', Trigger)
        # self.reset_srv = rospy.ServiceProxy('/reset_srv', Trigger)
        self.stop_srv = rospy.ServiceProxy('/stop_srv', Trigger)
        # self.gripper_srv = rospy.ServiceProxy('/gripper_srv', GripperSrv)  # Replace with actual type

        # State holders
        self.current_joint_states = None
        self.current_arm_status = None
        self.current_end_pose = None
        self.current_end_pose_euler = None
        self.current_joint_states_single = None
        self.current_pos_cmd = None

    def _arm_status_callback(self, msg):
        self.current_arm_status = msg

    def _end_pose_callback(self, msg):
        self.current_end_pose = msg

    def _end_pose_euler_callback(self, msg):
        self.current_end_pose_euler = msg

    def _joint_states_single_callback(self, msg):
        self.current_joint_states_single = msg

    def _pos_cmd_callback(self, msg):
        self.current_pos_cmd = msg

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        # self.enable_pub.publish(Bool(data=True))
        time.sleep(1)
        self.is_connected = True

    def is_connected(self) -> bool:
        return self.is_connected
    
    @property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.joints}

    @property
    def observation_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.joints}

    def calibrate(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        return True

    def configure(self):
        pass

    def setup_motors(self):
        pass

    def get_observation(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        observation = {}
        effort = {}
        if self.current_joint_states_single:
            for name, position, effort_val in zip(
                self.current_joint_states_single.name, 
                self.current_joint_states_single.position, 
                self.current_joint_states_single.effort
                ):
                #Increment joint index
                if name.startswith("joint") and name[5:].isdigit():
                    idx = int(name[5:]) + 1
                    name = f"joint{idx}"
                
                converted_position = self._convert_observation(name, position)
                observation[f"{name}.pos"]  = converted_position
                effort[f"{name}.effort"] = effort_val

        return observation, effort

    # Convert so102 action to piper action
    # input action should have .pos removed from the keys
    def _convert_action(self, action: dict[str, float]) -> dict[str, float]:
        converted_action = {}
        for joint, value in action.items():
            # print(self.joint_limits)
            if joint in self.joint_limits:
                converted_value = value* self.transform[joint][0] + self.transform[joint][1]
                # so102 joint angles [rad] converted to piper joint angles [rad]
                min_limit, max_limit = self.joint_limits[joint]
                # Ensure the value is within the joint limits
                safe_value = self._ensure_safe_goal_position(converted_value, min_limit, max_limit)
                converted_action[joint] = safe_value
            else:
                raise ValueError(f"Joint {joint} not recognized in limits.")
        return converted_action
    
    def _convert_observation(self, name, value):
        converted_value = value * self.transform[name][0] - self.transform[name][1]
        return converted_value

    def _ensure_safe_goal_position(self, value: float, min_limit: float, max_limit: float) -> float:
        """
        Ensure the goal position is within the joint limits.
        If the value is outside the limits, it will be clamped to the nearest limit.
        """
        if value < min_limit:
            return min_limit
        elif value > max_limit:
            return max_limit
        return value

    def send_action(self, action: dict[str, float], effort: dict[str, float], velocity: dict[str, float] = None) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # if self.teleop is None:
        #     raise ValueError("Teleoperator not set for the robot")

        # goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items()}
        converted_action = self._convert_action(goal_pos)
        joint_state_msg = JointState()
        joint_state_msg.name = list(converted_action.keys())      # Compliant with ROS JointState
        joint_state_msg.position = list(converted_action.values())
        if effort["joint7.effort"] >= 0.5:
            joint_state_msg.effort = list(effort.values())
        if velocity:
            joint_state_msg.velocity = list(velocity.values())
        # else:
        #     joint_state_msg.velocity = [20, 20, 20, 20, 20, 20, 20]
        self.joint_pub.publish(joint_state_msg)
        # print(f"Sending action: {joint_state_msg}")

        return {f"{motor}.pos": val for motor, val in converted_action.items()}

    def home(self) -> bool:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Ensure the robot is homed
        # if self.is_homed:
            # logger.info(f"{self} is already homed.")
            # return True
        home_action = {f"{joint}.pos": pos for joint, pos in self.home_position.items()}
        effort = {f"{joint}.effort": 0.0 for joint in self.joints}
        velocity = {f"{joint}.vel": 20.0 for joint in self.joints}
        if self._check_if_homed():
            self.is_homed = True
        # Send home position
        if not self.is_homed:
            self.send_action(home_action, effort, velocity)
            
        return self.is_homed

    def _check_if_homed(self) -> bool:
        observation, effort = self.get_observation()
        observation = {key.removesuffix(".pos"): val for key, val in observation.items()}
        for joint, value in observation.items():
            error = value - self.home_position[joint]
            if abs(error) > 0.15:
                return False
        return True

    
    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # self.enable_pub.publish(Bool(data=False))
        time.sleep(1)
        self.is_connected = False
        # Shutdown ROS cleanly
        rospy.signal_shutdown("User requested shutdown")
