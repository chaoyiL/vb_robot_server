import sys
import os
import time


from sympy.geometry.plane import x
dir = os.getcwd()
sys.path.append(dir)

save_dir = "manual_snapshots"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"saved_poses_{time.strftime('%Y.%m.%d_%H.%M.%S')}.npy")

import signal
import numpy as np
from real_world.robot_api.arm.RobotControl_pykin import RobotControl
import scipy.spatial.transform as t3d
from utils.pose_util import pose_to_mat, mat_to_pose

controller = 0
control_running = False


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    global control_running
    control_running = False
    controller.robot._robot.shutdown()
    print("Robot shutdown finish")
    exit()

def pos_orn_to_mat(position : list[float, float, float], quaternion : list[float, float, float, float]) -> np.ndarray:
    # Create rotation matrix from quaternion

    rotation_matrix = t3d.Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
    
    # Create 4x4 transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = position
    
    return matrix

def mat_to_pos_orn(matrix : np.ndarray) -> tuple[list[float, float, float], list[float, float, float, float]]:
    # Extract position
    position = matrix[:3, 3]
    
    # Extract rotation matrix and convert to quaternion
    rotation_matrix = matrix[:3, :3]
    quaternion = t3d.Rotation.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    return np.concatenate([position, quaternion])

# 注册信号处理函数
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    # control_frenq 如果是

    config = {
        "refresh_rate": "30 seconds",
        "appenders": {
            "stderr": {
                "kind": "console",
                "target": "stderr",
                "encoder": {
                    "pattern": "{h({d(%Y-%m-%d %H:%M:%S.%3f)} [{t}] {l} {m})}{n}"
                },
            },
            "file": {
                "kind": "file",
                "path": "log/file.log",
                "encoder": {
                    "pattern": "{h({d(%Y-%m-%d %H:%M:%S.%3f)} [{t}] {l} {m})}{n}"
                },
            },
        },
        "root": {"level": "debug", "appenders": ["stderr"]},
        "loggers": {},
    }
    # hblog.start(config)

    controller = RobotControl(vel_max=0.3)
    print(f"Server up")

    # 步长设置
    move_step = 0.02

    # 控制变量
    arm = 'l'
    joint = 0
    direction = 'w'
    err = 0
    control_running = True
    gripper_l = [0.0]

    controller.execute()


    while control_running:

        '''获取当前关节状态'''
        curr_joint_state = controller.get_robot_joints()
        position_l = curr_joint_state["left_arm"]
        position_r = curr_joint_state["right_arm"]
        gripper_l = curr_joint_state["left_gripper"]
        gripper_r = curr_joint_state["right_gripper"]
        print("left joint:")
        print(position_l)
        print("right joint:")
        print(position_r)
        print("left gripper:")
        print(gripper_l)
        print("right gripper:")
        print(gripper_r)

        '''遥控信号输入'''
        control_signal = input(f"请输入要控制的臂(l,r)/要控制的关节(0-6)/要控制的方向(w,s)，按q退出. 当前为{arm}/{joint}/{direction}:")
        
        if control_signal == 'l' or control_signal == 'r':
            arm = control_signal
        elif control_signal == 'w' or control_signal == 's':
            direction = control_signal
        elif control_signal == '':
            pass
        elif control_signal == 'q':
            print("退出...")
            control_running = False
        elif control_signal == '0' or control_signal == '1' or control_signal == '2' or control_signal == '3' or control_signal == '4' or control_signal == '5' or control_signal == '6':
            joint = int(control_signal)
        elif control_signal == 'lo':
            gripper_l[0] = 0.002
        elif control_signal == 'lc':
            gripper_l[0] = -0.02
        elif control_signal == 'ro':
            gripper_r[0] = 0.002
        elif control_signal == 'rc':
            gripper_r[0] = -0.02

        elif control_signal == 'save':

            ee_pose = controller.get_ee_pose()
            ee2rb_left = ee_pose["left_arm_ee2rb"]
            ee2rb_right = ee_pose["right_arm_ee2rb"]
            gripper_l = ee_pose["left_gripper"]
            gripper_r = ee_pose["right_gripper"]
            pose_data = {
                "ee2rb_left": ee2rb_left,
                "ee2rb_right": ee2rb_right,
                "gripper_l": gripper_l,
                "gripper_r": gripper_r
            }
            print("saved poses:")
            print(pose_data)

            # If file exists, append to list; else, create new list
            if os.path.exists(save_path):
                existing_data = np.load(save_path, allow_pickle=True)
                if isinstance(existing_data, np.ndarray) and len(existing_data.shape) == 0:
                    saved_list = [existing_data.item()]
                else:
                    saved_list = list(existing_data)
                saved_list.append(pose_data)
            else:
                saved_list = [pose_data]

            np.save(save_path, saved_list)
            print(f"Saved current pose data to {save_path}")
        
        elif control_signal == 'b':
            joint_l = [-0.281008, 0.80009343, -0.29597571, 1.60130698, 2.43974423, -0.4297301, -0.4303623]
            joint_r = [-1.38536501, 0.46157349, 0.55121075, 1.54583022, 2.07328705, 0.06058749, 0.4031778 ]
            gripper_l = [0.01]
            gripper_r = [0.01]
            controller.set_target_JP(joint_l, joint_r, gripper_l, gripper_r)
            controller.execute()
            time.sleep(0.3)
        
        elif control_signal == 'sr':
            l2r_pose = np.array([-4.65774357e-01,-2.38045557e-01,4.98711966e-04,-2.85670998e+00,1.09655970e+00,1.99338444e-01])
            l2r_pose_mat = pose_to_mat(l2r_pose)

            ee2ab_left = controller.get_ee_pose()["left_arm_ee2rb"]
            ee2ab_left_mat = pos_orn_to_mat(ee2ab_left[0:3], ee2ab_left[3:7])
            ee2ab_right =  mat_to_pos_orn(ee2ab_left_mat @ np.linalg.inv(l2r_pose_mat))
            # print(ee2ab_right)

            gripper_l = controller.get_robot_joints()["left_gripper"]
            gripper_r = controller.get_robot_joints()["right_gripper"]

            target_pose ={
                "left_arm_ee2rb": ee2ab_left,
                "right_arm_ee2rb": ee2ab_right,
                "left_gripper": gripper_l,
                "right_gripper": gripper_r
            }

            controller.set_target_CP(target_pose)
        
        else:
            print("输入有误，请重新输入")
            err = 1

        if (control_signal in ['w', 's', '']) and err == 0:

            if direction == 'w':
                move_step_dir = move_step
            elif direction == 's':
                move_step_dir = -1 * move_step

            if arm == 'l':
                position_l[joint] += move_step_dir
            elif arm == 'r':
                position_r[joint] += move_step_dir
            
            controller.set_target_JP(position_l, position_r, gripper_l, gripper_r)

        if (control_signal in ['lo', 'lc', 'ro', 'rc']) and err == 0:
            for i in range(100):
                controller.set_target_JP(position_l, position_r, gripper_l, gripper_r)
                time.sleep(0.01)

        err = 0
        controller.execute()
        time.sleep(0.3)

    controller.robot._robot.shutdown()
    exit()