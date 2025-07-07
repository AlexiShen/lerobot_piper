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

# from pydrake.all import MultibodyPlant, Parser, DiagramBuilder, JacobianWrtVariable
import pinocchio as pin
from pinocchio.utils import rotate
from pinocchio.visualize import MeshcatVisualizer
import meshcat
from meshcat.geometry import Sphere, MeshLambertMaterial, Cylinder
import scipy.spatial.transform

import numpy as np

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
        urdf_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        urdf_path = urdf_dir / "Leader_arm_links.SLDASM/urdf"#"so102_description.urdf"
        # self.builder = DiagramBuilder()
        # self.plant = MultibodyPlant(time_step=0.0)
        # self.model_instance = Parser.AddModels("lerobot/common/teleoperators/so102_leader/Leader_arm_links.SLDASM/urdf/Leader_arm_links.SLDASM_old_convention.urdf")
        # self.plant.Finalize()
        # self.context = self.plant.CreateDefaultContext()
        self.model = pin.buildModelFromUrdf("lerobot/common/teleoperators/so102_leader/Leader_arm_links.SLDASM/urdf/Leader_arm_links.SLDASM.urdf")
        # self.model = pin.buildModelFromUrdf("lerobot/common/teleoperators/so102_leader/so102_description.urdf")
        self.data = self.model.createData()
        self.config = config

        # Set up Meshcat viewer
        self.vis = None
        # self.vis = MeshcatVisualizer(self.model, pin.GeometryModel(), pin.GeometryModel(), meshcat.Visualizer())
        # self.vis.initViewer(open=True)
        # self.vis.loadViewerModel()

        self.t_static = time.perf_counter()

        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "joint1": Motor(1, "sts3215", MotorNormMode.RADIANS),
                "joint2": Motor(2, "sts3215", MotorNormMode.RADIANS),
                "joint3": Motor(3, "sts3215", MotorNormMode.RADIANS),
                "joint4": Motor(4, "sts3215", MotorNormMode.RADIANS),
                "joint5": Motor(5, "sts3215", MotorNormMode.RADIANS),
                "joint6": Motor(6, "sts3215", MotorNormMode.RADIANS),
                "joint7": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
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
        # self.bus.enable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            if motor == "joint7":
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            else:
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

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        action["joint7.pos"] = action["joint7.pos"]*0.06/100
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        valid_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        q = np.array([action[f"{joint}.pos"] for joint in valid_joints])

        if self.vis is not None:
            self._visualize_joint_origins(q)
        # self._print_joint_model_info()
        return action
    
    def get_velocity(self) -> dict[str, float]:
        start = time.perf_counter()
        velocity = self.bus.sync_read("Present_Velocity")
        velocity = {f"{motor}.vel": val for motor, val in velocity.items()}
        velocity["joint7.vel"] = 0
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

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Sync write to Goal_Position using feedback dict
        # feedback keys may be like 'shoulder_pan.pos', so strip '.pos' if present
        #TODO: replace implmentation with force loop
        print(f"Sending feedback: {feedback}")
        for motor, value in feedback.items():
            motor = motor.split(".")[0]
            self.bus.write("Goal_Position", motor, value)
        # logger.info(f"{self} sent feedback: {feedback}")

    # def send_feedback_test(self, feedback: dict[str, float]) -> None:
    #     print(f"Sending feedback: {feedback}")
    #     for motor, value in feedback.items():
    #         motor = motor.split(".")[0]
    #         self.bus.write("Goal_Time", motor, value)
    #     # logger.info(f"{self} test sent feedback: {feedback}")

    def send_feedback_test(self, feedback):
        tau = self._calculate_force_output()
        tau = np.append(tau, 0)
        tau_dict = {f"{joint}.tau": t for joint, t in zip(self.bus.motors.keys(), tau)}
        # print("Tau table:", tau_dict)
        for motor, value in tau_dict.items():
            motor = motor.split(".")[0]
            pwm_int = np.round(value * 100).astype(int)
            self.bus.write("Goal_Time", motor, pwm_int)
        # logger.info(f"{self} test sent feedback: {feedback}")
        # print("CoM:", self.model.inertias[2].lever)

    def _print_joint_model_info(self):
        for jname in self.model.names[1:]:
            j_id = self.model.getJointId(jname)
            joint = self.model.joints[j_id]
            print(f"{jname}: axis = {joint.axis}, placement translation = {joint.placement.translation}")

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
    
    def _calculate_force_output(self):
        action = self.get_action()
        velocity = self.get_velocity()

        # Build q and q_dot in correct order
        valid_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        q = np.array([action[f"{joint}.pos"] for joint in valid_joints])
        q_dot = np.array([velocity[f"{joint}.vel"] for joint in valid_joints])

        # pin.forwardKinematics(self.model, self.data, q)
        # joint2_id = self.model.getJointId("joint2")

        # Gravity compensation
        tau_g_full = pin.rnea(self.model, self.data, q, q_dot, q_dot*0)

        tau_g = np.zeros_like(tau_g_full)
        tau_g[1] = -tau_g_full[1]
        # tau_g[1] = tau_g[1]
        tau_g[2] = -tau_g_full[2]
        tau_g[4] = -tau_g_full[4]
        return tau_g
    
    # def _compute_static_friction_compensation(self, q_dot, freq = 500):
    #     tau_ss = np.zeros_like(q_dot)
    #     if_increment_time = False
    #     dt = 0.001
    #     q_threshold = 
    #     for i, qd in enumerate(q_dot):
    #         if abs(qd) < 1e-3:
    #             tau_ss[i] = 0.2 * 


    

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
