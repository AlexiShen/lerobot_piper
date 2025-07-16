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
"""

import logging
import time
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
)
from lerobot.common.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up
from lerobot.common.utils.visualization_utils import _init_rerun

from .common.teleoperators import koch_leader, so100_leader, so101_leader, so102_leader # noqa: F401


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
    while True:
        if rospy.is_shutdown():
            break
        loop_start = time.perf_counter()
        action = teleop.get_action()
        load = teleop.get_load()
        velocity = teleop.get_velocity()
        observation, effort = robot.get_observation()
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

        if_arms_synced = teleop.sync_leader_position(action, observation)
        # print(f"if_arms_synced: {if_arms_synced}")
        # if if_arms_synced:
        action_sent = robot.send_action(action)
        # effort= {
        #     "joint1.effort": 0,
        #     "joint2.effort": -90,
        #     "joint3.effort": 90,
        #     "joint4.effort": 0,
        #     "joint5.effort": 100,
        #     "joint6.effort": 0,
        #     # "joint7.effort": 0,
        #             }
        teleop.send_force_feedback(observation, effort)
        # zero_pos = [0.2,0.3,-0.2,0.3,-0.2,0.5,0.01]
        # joint_names = [key.removesuffix(".pos") for key in robot.action_features]
        # zero_action = {key: zero_pos[i] for i, key in enumerate(robot.action_features)}
        # action_sent = robot.send_action(zero_action)
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start

        # print(observation.__sizeof__())
        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'ACTION':>7} | {'EFFORT':>7} | {'VELOCITY':>7} | {'OBSERVATION':>7} | {'EFFORT':>7}")
        for (motor, load_val), (motor2, action_val), (motor3, velocity_val), (joint, obs_val) , (joint2, eff_val)\
            in zip(load.items(), action.items(), velocity.items(), observation.items(), effort.items()):
            print(f"{motor:<{display_len}} | {action_val:>7.2f} | {load_val:>7.2f} | {velocity_val:>7.2f} | {obs_val:>7.2f} | {eff_val:>7.2f}")
        
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        # move_cursor_up(len(action)+ 8)


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

    try:
        teleop_loop(teleop, robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()
        rospy.signal_shutdown("User requested shutdown")


if __name__ == "__main__":
    teleoperate()
