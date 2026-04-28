import sys
import os

from sympy.geometry.plane import x
dir = os.getcwd()
sys.path.append(dir)

save_dir = "/home/rvsa/codehub/VB-vla/real_world/robot_api"
save_path = os.path.join(save_dir, "saved_poses.npy")

import time
import signal
import numpy as np
from real_world.robot_api.arm.RobotControl_pykin import RobotControl
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

    controller = RobotControl(GUI = True, 
                width_slope=1.053562,
                width_offset=0.151515,
                vel_max=0.8)
    print(f"Server up")

    # 步长设置
    move_step = 0.01

    # 控制变量
    arm = 'l'
    axis = 0 # 0,1,2 for x,y,z
    direction = 'w'
    err = 0
    control_running = True

    controller.execute()

    while control_running:

        '''获取当前关节状态'''
        ee2ab_left, ee2ab_right, gripper_l, gripper_r = controller.get_ee_pose()
        print("REAL: left ee2ab pose:")
        print(ee2ab_left)
        # print("right ee pose:")
        # print(ee2ab_right)

        '''遥控信号输入'''
        control_signal = input(f"请输入要控制的臂(l,r)/要控制的轴(x,y,z)/要控制的方向(w,s)，按q退出. 当前为{arm}/{axis}/{direction}:")
        
        if control_signal == 'l' or control_signal == 'r':
            arm = control_signal
        elif control_signal == 'x' or control_signal == 'y' or control_signal == 'z':
            axis = ['x', 'y', 'z'].index(control_signal)
        elif control_signal == 'w' or control_signal == 's':
            direction = control_signal
        elif control_signal == '':
            pass
        elif control_signal == 'q':
            print("退出...")
            control_running = False
        elif control_signal == 'k':
            print("keep current pose: left:", ee2ab_left)
            # print("keep current pose: right:", ee2ab_right)
            controller.set_target_CP(ee2ab_left, ee2ab_right, gripper_l, gripper_r)

        if (control_signal in ['w', 's', '']) and err == 0:

            if direction == 'w':
                move_step_dir = move_step
            elif direction == 's':
                move_step_dir = -1 * move_step

            if arm == 'l':
                ee2ab_left[axis] += move_step_dir
            elif arm == 'r':
                ee2ab_right[axis] += move_step_dir
            
            print("left target ee2ab pose:")
            print(ee2ab_left)
            # print("right target ee2ab pose:")
            # print(ee2ab_right)

            controller.set_target_CP(ee2ab_left, ee2ab_right, gripper_l, gripper_r)


        err = 0
        controller.execute()
        time.sleep(0.3)


    controller.robot._robot.shutdown()
    exit()

    