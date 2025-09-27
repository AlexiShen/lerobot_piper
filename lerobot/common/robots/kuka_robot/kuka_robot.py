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
from .config_kuka_robot import KukaRobotConfig
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
            "joint_a1": 0.0,
            "joint_a2": 0.0,
            "joint_a3": 0.0,
            "joint_a4": 0.0,
            "joint_a5": 0.0,
            "joint_a6": 0.0,
            "linear_axis_joint_e1": 0.0,
        }

        self.joint_limits = {
            "joint_a1": (-3.2, 3.2),
            "joint_a2": (-2.4, -0.0873),
            "joint_a3": (-2.1, 2.9),
            "joint_a4": (-6.1, 6.1),
            "joint_a5": (-2.15, 2.15),
            "joint_a6": (-6.1, 6.1),
            "linear_axis_joint_e1": (0.0, 0.08),
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
            "joint_a1": -0.12,
            "joint_a2": -0.2,
            "joint_a3": 0.2,
            "joint_a4": 0,
            "joint_a5": 1,
            "joint_a6": -0.08,
            "linear_axis_joint_e1": 0.0,
        }

        #-0.46570246800000004, 1.823107328, -1.58531072, -0.08441151600000002, 1.2499149320000003, -0.17218972400000002, 0.0735
        self.is_homed = False
        self.zero_velocity = {f"{joint}.vel": 0.0 for joint in self.joints}

        # self.joint_limits = {
        #     joint: (math.radians(lim[0]), math.radians(lim[1]))
        #     for joint, lim in self.joint_limits_degrees.items() if joint != "joint7"
        # }

        self.transform = {
            "joint_a1": (-1, 0),
            "joint_a2": (1, 1.55),
            "joint_a3": (1, -1.63),
            "joint_a4": (-1, 0),
            "joint_a5": (1, 0),
            "joint_a6": (-1, 0),
            "linear_axis_joint_e1": (1, 0),
        }


        # Initialize ROS node (anonymous=True to allow multiple launches)
        rospy.init_node(self.node_name, anonymous=True)

        # Publishers for each joint command topic
        self.enable_pub = rospy.Publisher('/arm_controller/state', Bool, queue_size=10)
        from std_msgs.msg import Float64
        self.joint_pubs = {
            "joint_a1": rospy.Publisher('/link_1_controller/command', Float64, queue_size=10),
            "joint_a2": rospy.Publisher('/link_2_controller/command', Float64, queue_size=10),
            "joint_a3": rospy.Publisher('/link_3_controller/command', Float64, queue_size=10),
            "joint_a4": rospy.Publisher('/link_4_controller/command', Float64, queue_size=10),
            "joint_a5": rospy.Publisher('/link_5_controller/command', Float64, queue_size=10),
            "joint_a6": rospy.Publisher('/link_6_controller/command', Float64, queue_size=10),
            "linear_axis_joint_e1": rospy.Publisher('/link_e_controller/command', Float64, queue_size=10),
        }

        # Subscribers (update to match available topics)
        self.joint_states_sub = rospy.Subscriber('/joint_states', JointState, self._joint_states_callback)
        # The following subscribers are commented out because their topics do not exist in your topic list:
        # self.arm_status_sub = rospy.Subscriber('/right_arm/arm_status', rospy.AnyMsg, self._arm_status_callback)
        # self.end_pose_sub = rospy.Subscriber('/right_arm/end_pose', PoseStamped, self._end_pose_callback)
        # self.end_pose_euler_sub = rospy.Subscriber('/right_arm/end_pose_euler', rospy.AnyMsg, self._end_pose_euler_callback)
        # self.joint_states_single_sub = rospy.Subscriber('/right_arm/joint_states_single', JointState, self._joint_states_callback)
        # self.pos_cmd_sub = rospy.Subscriber('/right_arm/pos_cmd', JointState, self._pos_cmd_callback)

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
        self.current_joint_states = None
        self.current_pos_cmd = None

    def _arm_status_callback(self, msg):
        self.current_arm_status = msg

    def _end_pose_callback(self, msg):
        self.current_end_pose = msg

    def _end_pose_euler_callback(self, msg):
        self.current_end_pose_euler = msg

    def _joint_states_callback(self, msg):
        self.current_joint_states = msg

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
        if self.current_joint_states:
            # Use the topic joint names directly
            for name, position, effort_val in zip(
                self.current_joint_states.name,
                self.current_joint_states.position,
                self.current_joint_states.effort if hasattr(self.current_joint_states, 'effort') else [0.0]*len(self.current_joint_states.name)
            ):
                converted_position = self._convert_observation(name, position)
                observation[f"{name}.pos"] = converted_position
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

        # Use topic joint names directly
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items()}
        converted_action = self._convert_action(goal_pos)
        from std_msgs.msg import Float64
        for joint, value in converted_action.items():
            if joint in self.joint_pubs:
                self.joint_pubs[joint].publish(Float64(value))
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
