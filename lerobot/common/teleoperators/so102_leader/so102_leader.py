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
from .config_so102_leader import SO102LeaderConfig

logger = logging.getLogger(__name__)


class SO102Leader(Teleoperator):
    """
    SO-102 Leader Arm inspired by SO-101 designed by TheRobotStudio and Hugging Face.
    """

    config_class = SO102LeaderConfig
    name = "so102_leader"

    def __init__(self, config: SO102LeaderConfig):
        # Ensure calibration_dir is a Path before calling super().__init__
        if config.calibration_dir is None:
            config.calibration_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        else:
            config.calibration_dir = Path(config.calibration_dir)
        super().__init__(config)
        self.config = config
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "joint0": Motor(1, "sts3215", MotorNormMode.RADIANS),
                "joint1": Motor(2, "sts3215", MotorNormMode.RADIANS),
                "joint2": Motor(3, "sts3215", MotorNormMode.RADIANS),
                "joint3": Motor(4, "sts3215", MotorNormMode.RADIANS),
                "joint4": Motor(5, "sts3215", MotorNormMode.RADIANS),
                "joint5": Motor(6, "sts3215", MotorNormMode.RADIANS),
                # "joint6": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

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

        # Save calibration file in the SO102Leader directory
        self.bus.write_calibration(self.calibration)
        # calibration_fpath = self.calibration_dir / f"{self.id}_so102_calibration.json"
        # self._save_calibration(fpath=calibration_fpath)
        # logger.info(f"Calibration saved to {calibration_fpath}")
        self._save_calibration()
        logger.info("Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

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

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Sync write to Goal_Position using feedback dict
        # feedback keys may be like 'shoulder_pan.pos', so strip '.pos' if present
        print(f"Sending feedback: {feedback}")
        for motor, value in feedback.items():
            motor = motor.split(".")[0]
            self.bus.write("Goal_Position", motor, value)
        logger.info(f"{self} sent feedback: {feedback}")


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
