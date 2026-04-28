import sys
import os
import time


from sympy.geometry.plane import x
dir = os.getcwd()
sys.path.append(dir)

save_dir = "/home/rvsa/codehub/VB-vla/real_world/robot_api"
save_path = os.path.join(save_dir, f"saved_poses_{time.time()}.npy")

import signal
import numpy as np
from real_world.robot_api.arm.RobotControl_pybullet import RobotControl
from utils.pose_util import pose_to_mat

controller = 0
control_running = False


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    global control_running
    control_running = False
    controller.robot._robot.shutdown()
    print("Robot shutdown finish")
    exit()


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

    controller = RobotControl(GUI = False, vel_max=0.5)
    print(f"Server up")

    # 步长设置
    move_step = 0.05

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
        position_l, position_r, gripper_l, gripper_r = controller.get_robot_joints()
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
        elif control_signal == '+':
            gripper_l[0] = gripper_l[0] + 0.01
        elif control_signal == '-':
            gripper_l[0] = gripper_l[0] - 0.01
        elif control_signal == 'save':

            ab2ee_left, ab2ee_right, gripper_l, gripper_r = controller.get_ee_pose()
            pose_data = {
                "ab2ee_left": pose_to_mat(ab2ee_left),
                "ab2ee_right": pose_to_mat(ab2ee_right),
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
            joint_l = [-1.14161869,  1.55846848, -2.93737387, -0.08676579,  1.74417887,  0.18531836, -0.88284676]
            joint_r = [-1.04133849,  0.91041665,  2.98185267, -0.56326047,  2.35526518, -0.02474398, -0.70209807]
            controller.set_target_JP(joint_l, joint_r, gripper_l, gripper_r)
            controller.execute()
            time.sleep(0.3)

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

        if (control_signal in ['+', '-']) and err == 0:
            controller.set_target_JP(position_l, position_r, gripper_l, gripper_r)

        err = 0
        controller.execute()
        time.sleep(0.3)

        ab2ee_left, ab2ee_right, gripper_l, gripper_r = controller.get_ee_pose()
        joint_l, joint_r, gripper_l, gripper_r = controller.get_robot_joints()

        # a = 1.053562
        # b = 0.151515
        # print("left gripper:")
        # print(gripper_l)
        # print((gripper_l[0] - b) / a)
        # print("right gripper:")
        # print(gripper_r)
        # print((gripper_r[0] - b) / a)

        # print("left joint:")
        # print(joint_l)
        # print("right joint:")
        # print(joint_r)

    controller.robot._robot.shutdown()
    exit()

    