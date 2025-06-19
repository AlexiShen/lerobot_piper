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
from ..utils import ensure_safe_goal_position
from .config_piper_robot import PiperRobotConfig

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
            "joint0": 0.0,
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
        }

        # Initialize ROS node (anonymous=True to allow multiple launches)
        rospy.init_node(self.node_name, anonymous=True)

        # Publishers
        self.enable_pub = rospy.Publisher('/enable_flag', Bool, queue_size=10)
        self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)

        # Subscribers (add callbacks as needed)
        self.arm_status_sub = rospy.Subscriber('/arm_status', rospy.AnyMsg, self._arm_status_callback)
        self.end_pose_sub = rospy.Subscriber('/end_pose', PoseStamped, self._end_pose_callback)
        self.end_pose_euler_sub = rospy.Subscriber('/end_pose_euler', rospy.AnyMsg, self._end_pose_euler_callback)
        self.joint_states_single_sub = rospy.Subscriber('/joint_states_single', JointState, self._joint_states_single_callback)
        self.pos_cmd_sub = rospy.Subscriber('/pos_cmd', JointState, self._pos_cmd_callback)

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
        if self.current_joint_states_single:
            for name, position in zip(self.current_joint_states_single.name, self.current_joint_states_single.position):
                observation[f"{name}.pos"] = position
        # TODO: Add effort observations later

        return observation

    def send_action(self, action):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # if self.teleop is None:
        #     raise ValueError("Teleoperator not set for the robot")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        
        joint_state_msg = JointState()
        joint_state_msg.name = list(goal_pos.keys())      # Compliant with ROS JointState
        joint_state_msg.position = list(goal_pos.values())
        # joint_state_msg.velocity = [0.0] * len(goal_pos)
        # joint_state_msg.effort = [0.0] * len(goal_pos)
        self.joint_pub.publish(joint_state_msg)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}
    
    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # self.enable_pub.publish(Bool(data=False))
        time.sleep(1)
        self.is_connected = False
        # Shutdown ROS cleanly
        rospy.signal_shutdown("User requested shutdown")
