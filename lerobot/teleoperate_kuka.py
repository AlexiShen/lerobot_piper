# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Simple script to control a robot from teleoperation.

Example:

```shell
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/t.ty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

python -m lerobot.teleoperate --robot.type=piper_robot --teleop.type=so102_leader --teleop.port=/dev/ttyACM0 --teleop.id=right --display_data=false
python -m lerobot.teleoperate_kuka --robot.type=kuka_robot --teleop.type=kuka_leader --teleop.port=/dev/ttyACM0 --teleop.id=kuka00 --display_data=false
"""

import logging
import time
import sys
import select
import termios
import os
import tty
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import numpy as np
import rerun as rr
import rospy

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    piper_robot,
    kuka_robot,
)
from lerobot.common.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up
from lerobot.common.utils.visualization_utils import _init_rerun

from .common.teleoperators import koch_leader, so100_leader, so101_leader, so102_leader, kuka_leader # noqa: F401


def setup_terminal():
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    return old_settings

def restore_terminal(old_settings):
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def check_key_press():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None

@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


def teleop_loop(
    teleop: Teleoperator, robot: Robot, fps: int, display_data: bool = False, duration: float | None = None
):
    # display_len = max(len(key) for key in robot.action_features)
    display_len = max(len(key) for key in teleop.action_features)
    start = time.perf_counter()
    
    # Initialize controlled joint
    CONTROLLED_JOINT = "joint1.pos"  # Default to controlling joint1
    
    while True:
        if rospy.is_shutdown():
            break
        loop_start = time.perf_counter()
        
        # Check for 'h' key press
        key = check_key_press()
        if key == 'h':
            teleop.lead_to_home()
        elif key == 'r':
            teleop.lead_to_rest()
        elif key in ['1', '2', '3', '4', '5', '6']:
            # Allow user to switch which joint to control
            CONTROLLED_JOINT = f"joint{key}.pos"
            print(f"\nNow controlling: {CONTROLLED_JOINT}")
        
        action = teleop.get_action()
        load = teleop.get_load()
        velocity = teleop.get_velocity()
        observation, effort = robot.get_observation()
        
        # DEBUG: Print joint names once to see what we're working with
        if loop_start - start < 1.0:  # Only print for first second
            print(f"DEBUG: Action keys: {list(action.keys())}")
            print(f"DEBUG: Observation keys: {list(observation.keys())}")
            # print(f"DEBUG: Controlling joint: {CONTROLLED_JOINT}")
        
        # Send force feedback to teleoperator
        effort_to_send = teleop.send_force_feedback(observation, effort)
        
        # Check if arm is ready (synced) before allowing movement
        if_arm_ready = teleop.if_arm_ready()

        # Create single hybrid action: controlled joint from teleoperator, others from observation
        hybrid_action = {}
        for joint_key in action.keys():
            if joint_key.endswith('.pos'):
                if joint_key == CONTROLLED_JOINT and if_arm_ready:
                # if if_arm_ready:
                    # Use teleoperator action for the controlled joint (only when synced)
                    hybrid_action[joint_key] = action[joint_key]
                else:
                    # Use current observation for all other joints (hold position)
                    hybrid_action[joint_key] = observation.get(joint_key, 0.0)
        
        # Always keep linear axis at current position regardless of which joint is controlled
        if "joint7.pos" in hybrid_action:
            # Find the corresponding observation key for linear axis
            for obs_key in observation.keys():
                if "joint7" in obs_key:  # This will match joint7.pos from observation
                    hybrid_action["joint7.pos"] = observation[obs_key]
                    break
        
        action_sent = robot.send_action(hybrid_action, effort_to_send)

        # effort= {
        #     "joint1.effort": 0,
        #     "joint2.effort": -90,
        #     "joint3.effort": 90,
        #     "joint4.effort": 0,
        #     "joint5.effort": 100,
        #     "joint6.effort": 0,
        #     # "joint7.effort": 0,
        #             }
        
        # zero_pos = [0.2,0.3,-0.2,0.3,-0.2,0.5,0.01]
        # joint_names = [key.removesuffix(".pos") for key in robot.action_features]
        # zero_action = {key: zero_pos[i] for i, key in enumerate(robot.action_features)}
        # action_sent = robot.send_action(zero_action)


        # if display_data:
        #     observation = robot.get_observation()
        #     for obs, val in observation.items():
        #         if isinstance(val, float):
        #             rr.log(f"observation_{obs}", rr.Scalars(val))
        #         elif isinstance(val, np.ndarray):
        #             rr.log(f"observation_{obs}", rr.Image(val), static=True)
        #     for act, val in action.items():
        #         if isinstance(val, float):
        #             rr.log(f"action_{act}", rr.Scalars(val))
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start

        # print(observation.__sizeof__())
        # os.system('clear')
        # # print("\n" + "-" * (display_len + 10))
        # # print(f"{'NAME':<{display_len}} | {'ACTION':>7} | {'EFFORT':>7} | {'VELOCITY':>7} | {'OBSERVATION':>7} | {'EFFORT':>7}")
        # col_width = 12
        # header = f"{'NAME':<{col_width}} | {'ACTION':>7} | {'EFFORT':>7} | {'VELOCITY':>9} | {'OBSERV':>9} | {'EFFORT':>7}"
        # print('-' * len(header))
        # print(header)
        # for (motor, load_val), (motor2, action_val), (motor3, velocity_val), (joint, obs_val) , (joint2, eff_val)\
        #     in zip(load.items(), action.items(), velocity.items(), observation.items(), effort.items()):
        #     print(f"{motor:<{display_len}} | {action_val:>7.2f} | {load_val:>7.2f} | {velocity_val:>7.2f} | {obs_val:>7.2f} | {eff_val:>7.2f}")
        
        # print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        # os.system('clear')
        col_widths = [16, 10, 10, 10, 10, 10]
        header_fields = ["NAME", "ACTION", "LOAD", "VELOCITY", "OBSERV", "EFFORT"]
        header = " | ".join(f"{name:<{w}}" for name, w in zip(header_fields, col_widths))
        print('-' * len(header))
        print(header)
        print('-' * len(header))
        for (motor, load_val), (motor2, action_val), (motor3, velocity_val), (joint, obs_val), (joint2, eff_val) in zip(
            load.items(), action.items(), velocity.items(), observation.items(), effort.items()
        ):
            # Highlight the controlled joint
            # marker = ">>> " if f"{motor2}" == CONTROLLED_JOINT else "    "
            print(
                # f"{marker}{motor:<{col_widths[0]-4}} | "
                f"{action_val:>{col_widths[1]}.2f} | "
                f"{load_val:>{col_widths[2]}.2f} | "
                f"{velocity_val:>{col_widths[3]}.2f} | "
                f"{obs_val:>{col_widths[4]}.2f} | "
                f"{eff_val:>{col_widths[5]}.2f}"
            )
        print('-' * len(header))
        
        # Display sync status and controlled joint
        sync_status = "✅ SYNCED" if if_arm_ready else "⚠️  WAITING FOR SYNC"
        print(f"\nStatus: {sync_status}")
        # print(f"Controlling: {CONTROLLED_JOINT}")
        print(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        print("Press 'h' to home, 'r' to rest, '1'-'6' to switch controlled joint")

        if duration is not None and time.perf_counter() - start >= duration:
            break

        # move_cursor_up(len(action)+ 8)

# def shutdown_loop(
#         teleop: Teleoperator, robot: Robot, fps: int, display_data: bool = False, duration: float | None = None
# ):  
#     while True:
#         loop_start = time.perf_counter()
#         print("Shutting down teleoperation loop...")
#         action = teleop.get_action()
#         load = teleop.get_load()
#         velocity = teleop.get_velocity()
#         observation, effort = robot.get_observation()
#         if teleop.lead_to_rest(action, velocity):
#             break
#         effort_to_send = teleop.send_force_feedback(observation, effort)
#         action_sent = robot.send_action(action, effort_to_send)

#         dt_s = time.perf_counter() - loop_start
#         busy_wait(1 / fps - dt_s)

#         loop_s = time.perf_counter() - loop_start

#         col_widths = [16, 10, 10, 10, 10, 10]
#         header_fields = ["NAME", "ACTION", "LOAD", "VELOCITY", "OBSERV", "EFFORT"]
#         header = " | ".join(f"{name:<{w}}" for name, w in zip(header_fields, col_widths))
#         print('-' * len(header))
#         print(header)
#         print('-' * len(header))
#         for (motor, load_val), (motor2, action_val), (motor3, velocity_val), (joint, obs_val), (joint2, eff_val) in zip(
#             load.items(), action.items(), velocity.items(), observation.items(), effort.items()
#         ):
#             print(
#                 f"{motor:<{col_widths[0]}} | "
#                 f"{action_val:>{col_widths[1]}.2f} | "
#                 f"{load_val:>{col_widths[2]}.2f} | "
#                 f"{velocity_val:>{col_widths[3]}.2f} | "
#                 f"{obs_val:>{col_widths[4]}.2f} | "
#                 f"{eff_val:>{col_widths[5]}.2f}"
#             )
#         print('-' * len(header))
#         print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

            
        


@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect()
    old_settings = setup_terminal()
    try:
        
        teleop_loop(teleop, robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        print("\nTeleoperation interrupted by user")
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()
        rospy.signal_shutdown("User requested shutdown")
        restore_terminal(old_settings)


if __name__ == "__main__":
    teleoperate()
