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

class KukaRobot(Robot):

    config_class = KukaRobotConfig
    name = "kuka_robot"

    def __init__(self, config: KukaRobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.node_name = f"{self.name}_kuka"
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
            "joint_a1": (1, 0),
            "joint_a2": (1, -1.64),
            "joint_a3": (1, 1.51),
            "joint_a4": (1, 0),
            "joint_a5": (1, 0),
            "joint_a6": (1, 0),
            "linear_axis_joint_e1": (1, 0),
        }


        # Initialize ROS node (anonymous=True to allow multiple launches)
        rospy.init_node(self.node_name, anonymous=True)

        # Publishers for joint commands
        self.enable_pub = rospy.Publisher('/arm_controller/state', Bool, queue_size=10)
        from std_msgs.msg import Float64MultiArray
        self.position_commands_pub = rospy.Publisher('/position_commands', Float64MultiArray, queue_size=10)

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
        velocity = {}
        effort = {}
        if self.current_joint_states:
            # Use the topic joint names directly
            for name, position, velocity_val, effort_val in zip(
                self.current_joint_states.name,
                self.current_joint_states.position,
                self.current_joint_states.velocity if hasattr(self.current_joint_states, 'velocity') else [0.0]*len(self.current_joint_states.name),
                self.current_joint_states.effort if hasattr(self.current_joint_states, 'effort') else [0.0]*len(self.current_joint_states.name)
            ):
                converted_position = self._convert_observation(name, position)
                kuka_joint_name = self._convert_joint_name_to_leader_style(name)
                observation[f"{kuka_joint_name}.pos"] = converted_position
                effort[f"{kuka_joint_name}.effort"] = effort_val
                velocity[f"{kuka_joint_name}.vel"] = velocity_val

        return observation, effort, velocity

    # Convert kuka_leader action to piper action
    # input action should have .pos removed from the keys
    def _convert_action(self, action: dict[str, float]) -> dict[str, float]:
        converted_action = {}
        for joint, value in action.items():
            # Convert from kuka_leader style to kuka_robot style if needed
            robot_joint_name = self._convert_joint_name_to_robot_style(joint)
            # print(self.joint_limits)
            if robot_joint_name in self.joint_limits:
                converted_value = value* self.transform[robot_joint_name][0] + self.transform[robot_joint_name][1]
                # kuka_leader joint angles [rad] converted to piper joint angles [rad]
                min_limit, max_limit = self.joint_limits[robot_joint_name]
                # Ensure the value is within the joint limits
                safe_value = self._ensure_safe_goal_position(converted_value, min_limit, max_limit)
                converted_action[robot_joint_name] = safe_value
            else:
                raise ValueError(f"Joint {joint} (robot style: {robot_joint_name}) not recognized in limits.")
        return converted_action
    
    def _convert_observation(self, name, value):
        converted_value = value * self.transform[name][0] - self.transform[name][1]
        return converted_value
    
    def _convert_joint_name_to_leader_style(self, ros_joint_name):
        """Convert ROS joint names to kuka_leader style"""
        joint_name_mapping = {
            "joint_a1": "joint1",
            "joint_a2": "joint2", 
            "joint_a3": "joint3",
            "joint_a4": "joint4",
            "joint_a5": "joint5",
            "joint_a6": "joint6",
            "linear_axis_joint_e1": "joint7"
        }
        return joint_name_mapping.get(ros_joint_name, ros_joint_name)
    
    def _convert_joint_name_to_robot_style(self, leader_joint_name):
        """Convert kuka_leader joint names back to ROS/kuka_robot style"""
        reverse_joint_name_mapping = {
            "joint1": "joint_a1",
            "joint2": "joint_a2",
            "joint3": "joint_a3", 
            "joint4": "joint_a4",
            "joint5": "joint_a5",
            "joint6": "joint_a6",
            "joint7": "linear_axis_joint_e1"
        }
        return reverse_joint_name_mapping.get(leader_joint_name, leader_joint_name)

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

        # Convert action from kuka_leader style to robot style
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items()}

        ######################################################################
        # Temporarily hold linear axis position steady
        ######################################################################
        
        # Filter out joint7 (gripper/trigger) to prevent it from controlling linear axis
        filtered_goal_pos = {joint: val for joint, val in goal_pos.items() if joint != "joint7"}
        
        converted_action = self._convert_action(filtered_goal_pos)
        
        # Get current linear axis position to hold it steady
        observation, _, velocity_obs = self.get_observation()
        current_linear_pos = None
        for obs_key, obs_val in observation.items():
            if obs_key == "joint7.pos":  # This is the linear axis in leader style
                # Convert back to robot style (linear_axis_joint_e1)
                current_linear_pos = obs_val * self.transform["linear_axis_joint_e1"][0] + self.transform["linear_axis_joint_e1"][1]
                break
        
        # If we couldn't get current position, use 0.0 as safe default
        if current_linear_pos is None:
            current_linear_pos = 0.0
        
        # Always set linear axis to current position to prevent movement
        converted_action["linear_axis_joint_e1"] = current_linear_pos
        ######################################################################
        #######################################################################
        
        # Create ordered joint arrays [a1, a2, a3, a4, a5, a6, e1]
        joint_order = ["joint_a1", "joint_a2", "joint_a3", "joint_a4", "joint_a5", "joint_a6", "linear_axis_joint_e1"]
        position_array = []
        velocity_array = []
        
        # Build position array
        for joint in joint_order:
            if joint in converted_action:
                position_array.append(converted_action[joint])
            else:
                # If joint not in action, use 0.0 as default
                position_array.append(0.0)
        
        # Build velocity array (default to 0.0 if not provided)
        for joint in joint_order:
            # Convert robot joint name back to leader style for velocity lookup
            leader_joint_name = self._convert_joint_name_to_leader_style(joint)
            velocity_key = f"{leader_joint_name}.vel"
            
            if velocity and velocity_key in velocity:
                velocity_array.append(velocity[velocity_key])
            else:
                # Default velocity (you can adjust this as needed)
                velocity_array.append(0.0)
        
        # Combine arrays: [pos1, pos2, ..., vel1, vel2, ...]
        combined_array = position_array + velocity_array
        
        # Publish to /position_commands topic
        from std_msgs.msg import Float64MultiArray
        msg = Float64MultiArray()
        msg.data = combined_array
        self.position_commands_pub.publish(msg)
        
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
            # Convert kuka_leader joint name to robot style for home_position lookup
            robot_joint = self._convert_joint_name_to_robot_style(joint)
            if robot_joint in self.home_position:
                error = value - self.home_position[robot_joint]
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
