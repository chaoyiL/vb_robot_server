import rb_python
import time
import hblog
import numpy as np


class RobotWrapper:
    def __init__(self, vel_max: float = 0.1):
        print(f"new a robot")

        # config = {
        # "refresh_rate": "30 seconds",
        # "appenders": {
        #     "stderr": {
        #         "kind": "console",
        #         "target": "stderr",
        #         "encoder": {
        #             "pattern": "{h({d(%Y-%m-%d %H:%M:%S.%3f)} [{t}] {l} {m})}{n}"
        #         },
        #     },
        #     "file": {
        #         "kind": "file",
        #         "path": "log/file.log",
        #         "encoder": {
        #             "pattern": "{h({d(%Y-%m-%d %H:%M:%S.%3f)} [{t}] {l} {m})}{n}"
        #         },
        #     },
        # },
        # "root": {"level": "error", "appenders": ["stderr"]},
        # "loggers": {},
        # }
        # hblog.start(config)
        
        cfg_str = {
            "hardware": {

                "left_arm": {
                    "type": "eyou",
                    "ids": [10, 11, 12, 13, 14, 15, 16],
                    "length_per_radian": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "invert_directions": [False, True, False, True, False, True, False],
                    "control_freq": 250,
                    "interpolation_points": 13,
                    "max_velocity": vel_max,  # 驱动限制每个关节旋转最大速度rad/s
                    "gravity_compensation_tolerance": 0.0,
                    "friction_compensation_scale": 0.0,
                    "friction_compensation_stiffness": 10,
                    "external_protections": [],
                    "offset_at_hardware_zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "joint_names": [
                        "left-joint_arm_1",
                        "left-joint_arm_2",
                        "left-joint_arm_3",
                        "left-joint_arm_4",
                        "left-joint_arm_5",
                        "left-joint_arm_6",
                        "left-joint_arm_7",
                    ],
                    "max_torque": [1800, 1800, 2400, 2400, 2000, 2000, 2000],
                    "protection_rebound": 0.0
                },

                "right_arm": {
                        "type": "eyou",
                        "ids": [20, 21, 22, 23, 24, 25, 26],
                        "length_per_radian": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        "invert_directions": [True, False, True, False, True, False, True],
                        "control_freq": 250,
                        "interpolation_points": 13,
                        "max_velocity":vel_max,  # 驱动限制每个关节旋转最大速度rad/s
                        "gravity_compensation_tolerance": 0.0,
                        "friction_compensation_scale": 0.0,
                        "friction_compensation_stiffness": 10,
                        "external_protections": [],
                        "offset_at_hardware_zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "joint_names": [
                            "right-joint_arm_1",
                            "right-joint_arm_2",
                            "right-joint_arm_3",
                            "right-joint_arm_4",
                            "right-joint_arm_5",
                            "right-joint_arm_6",
                            "right-joint_arm_7",
                        ],
                        "max_torque": [1800, 1800, 2400, 2400, 2000, 2000, 2000],
                        "protection_rebound": 0.0
                        },

                "left_gripper": {
                        "type": "eyou",
                        "ids": [30],
                        "length_per_radian": [0.0099],
                        "invert_directions": [True],
                        "control_freq": 250,
                        "interpolation_points": 13,
                        "max_velocity": 0.05,
                        "gravity_compensation_tolerance": 0.0,
                        "friction_compensation_scale": 0.0,
                        "friction_compensation_stiffness": 0.0,
                        "external_protections": [],
                        "offset_at_hardware_zero": [0.0],
                        "joint_names": ["left-joint_gripper_finger_1"],
                        "max_torque": [1000],
                        "protection_rebound": 0.0,
                    },
                "right_gripper": {
                        "type": "eyou",
                        "ids": [50],
                        "length_per_radian": [0.0099],
                        "invert_directions": [True],
                        "control_freq": 250,
                        "interpolation_points": 13,
                        "max_velocity": 0.05,
                        "gravity_compensation_tolerance": 0.0,
                        "friction_compensation_scale": 0.0,
                        "friction_compensation_stiffness": 0.0,
                        "external_protections": [],
                        "offset_at_hardware_zero": [0.0],
                        "joint_names": ["right-joint_gripper_finger_1"],
                        "max_torque": [1000],
                        "protection_rebound": 0.0,
                    },
            },
            "planner": None,
            "robot_model": "",
        }

        self._robot = rb_python.robot.Robot(cfg_str)
        time.sleep(1)
    
    def set_joint_angle(self, arm: str, position: np.ndarray) -> None:
        # print("func: set_joint_angle", action)
        action = {arm: {"type": "position", 
                        "position": list(position)}}
        self._robot.set_actions(action)

    def get_joint_angle(self, arm: str)->np.ndarray:
        state = self._robot.get_states([arm])
        # print("func: get_joint_angle", state)
        position = np.array(state[arm]['position'])

        return position
    
        
    def get_joint_velo(self, arm: str)->np.ndarray:
        state = self._robot.get_states([arm])
        velocity = np.array(state[arm]['velocity'])

        return velocity
    