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
Simple script to interface with a teleoperator.

Example:

```shell
python -m lerobot.teleoperate_only \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

python -m lerobot.teleoperate_only --teleop.port=/dev/ttyACM0 --teleop.id=right --display_data=false
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import rerun as rr



from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    piper_robot,
)

from lerobot.common.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so102_leader,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up
from lerobot.common.utils.visualization_utils import _init_rerun


@dataclass
class TeleoperateOnlyConfig:
    #teleop: TeleoperatorConfig
    teleop: so102_leader.SO102LeaderConfig
    fps: int = 60
    teleop_time_s: float | None = None
    display_data: bool = False


def teleop_only_loop(
    teleop: Teleoperator, fps: int, display_data: bool = False, duration: float | None = None
):
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        action = teleop.get_action()
        load = teleop.get_load()

        # Send feedback: set each joint to 0
        feedback = {joint: 0 for joint in action.keys()}
        effort= {
            "joint1.effort": 0,
            "joint2.effort": 0, #-120,
            "joint3.effort": 100,
            "joint4.effort": 0,
            "joint5.effort": 0,
            "joint6.effort": 0,
            # "joint7.effort": 0,
                    }
        teleop.send_feedback_test(effort)
        # teleop.send_feedback(feedback)

        if display_data:
            for act, val in action.items():
                rr.log(f"action_{act}", rr.Scalars(val))

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start

        print("\n" + "-" * 20)
        print(f"{'NAME':<10} | {'VALUE':>7}")
        # for motor, value in action.items():
        for motor, value in load.items():
            print(f"{motor:<10} | {value:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        move_cursor_up(len(action) + 5)


@draccus.wrap()
def teleoperate_only(cfg: TeleoperateOnlyConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation_only")

    teleop = make_teleoperator_from_config(cfg.teleop)

    teleop.connect()

    try:
        teleop_only_loop(teleop, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()


if __name__ == "__main__":
    teleoperate_only()
