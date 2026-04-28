import sys
import os
dir = os.getcwd()
sys.path.append(dir)

from scipy.spatial.transform import Rotation
import numpy as np

from real_world.robot_api.arm.RobotWrapper import RobotWrapper
from real_world.robot_api.sim.PybulletEnv import PybulletEnv
from utils.pose_util import pose_to_mat, mat_to_pose


def pos_orn_to_mat(position : list[float, float, float], quaternion : list[float, float, float, float]) -> np.ndarray:
    # Create rotation matrix from quaternion

    rotation = Rotation.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
    rotation_matrix = rotation.as_matrix()
    
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
    rotation = Rotation.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # x, y, z, w format

    return position, quaternion

class RobotControl:
    def __init__(self,GUI = True, 
    width_slope:float = None,
    width_offset:float = None,
    vel_max: float = None,
    ):

        # 初始化机器人与仿真
        self.robot = RobotWrapper(vel_max=vel_max)
        self.sim = PybulletEnv(GUI)

        # 用来接收运动指令的buffer
        self.action_target = [None, None, None, None]

        # 导入仿真中arm base相对world base的变换，用于ik, fk求解
        self.ab2w_left = self.sim.arm_base_left
        self.ab2w_right = self.sim.arm_base_right

        print("ab2w_left:")
        print(self.ab2w_left)
        print("ab2w_right:")
        print(self.ab2w_right)

        # linear fitting result
        if width_slope is not None:
            self.width_slope = width_slope
        else:
            self.width_slope = 1.053562
        if width_offset is not None:
            self.width_offset = width_offset
        else:
            self.width_offset = 0.151515

        self.component_name = ["left_arm", "right_arm", "left_gripper", "right_gripper"]
    
    def get_robot_joints(self):
        '''return joint angles of left and right arm, gripper width'''
        joint_cur_left = self.robot.get_joint_angle("left_arm")
        joint_cur_right = self.robot.get_joint_angle("right_arm")

        gripper_cur_left = self.robot.get_joint_angle("left_gripper")
        gripper_cur_right = self.robot.get_joint_angle("right_gripper")

        # a = self.width_slope
        # b = self.width_offset

        # gripper_cur_left = [a*gripper_cur_left[0] + b]
        # gripper_cur_right = [a*gripper_cur_right[0] + b]

        
        return joint_cur_left, joint_cur_right, gripper_cur_left, gripper_cur_right
    

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray, list, list]:
        '''return ee pose of left and right arm, gripper width'''
        
        # 存储arm base到world base的变换映射
        ab2w_map = {
            "left_arm": self.ab2w_left,
            "right_arm": self.ab2w_right
        }
        
        # 初始化结果变量
        ee2ab_pose_left = None
        ee2ab_pose_right = None
        gripper_cur_left = None
        gripper_cur_right = None
        
        # 遍历所有组件
        for component in self.component_name:
            if "arm" in component:
                # 对于arm组件，执行FK计算
                joint_cur = self.robot.get_joint_angle(component)
                
                # 计算当前末端位置和姿态
                ee2w_pos, ee2w_orn = self.sim.arm_fk_with_step_sim(component, joint_cur)

                if component == "left_arm":
                    print("REAL:FK,", component, "ee2w pose:")
                    print(mat_to_pose(pos_orn_to_mat(ee2w_pos, ee2w_orn)))
                
                # 计算当前末端位置和姿态相对于world base的变换
                ee2w_mat = pos_orn_to_mat(ee2w_pos, ee2w_orn)
                
                # 计算当前末端位置和姿态相对于arm base的变换
                ab2w_mat = ab2w_map[component]
                ee2ab_mat = np.linalg.inv(ab2w_mat) @ ee2w_mat
                
                # 转换为pose
                if component == "left_arm":
                    ee2ab_pose_left = mat_to_pose(ee2ab_mat)
                    # print(quest2ab_pose_left)
                elif component == "right_arm":
                    ee2ab_pose_right = mat_to_pose(ee2ab_mat)
                    # print(quest2ab_pose_left)
                    
            elif "gripper" in component:
                # 对于gripper组件，只获取关节角度
                gripper_cur = self.robot.get_joint_angle(component)
                
                # 根据组件名称赋值
                a = self.width_slope
                b = self.width_offset
                if component == "left_gripper":
                    # print(gripper_cur[0])
                    gripper_cur_left = [a*gripper_cur[0] + b]
                elif component == "right_gripper":
                    # print(gripper_cur[0])
                    gripper_cur_right = [a*gripper_cur[0] + b]
        
        return ee2ab_pose_left, ee2ab_pose_right, gripper_cur_left, gripper_cur_right

    def set_target_JP(self, joint_left:list, joint_right:list, gripper_left:list, gripper_right:list):
        self.action_target[0] = joint_left
        self.action_target[1] = joint_right

        a = self.width_slope
        b = self.width_offset

        gripper_left_command = [(gripper_left[0] - b) / a]
        gripper_right_command = [(gripper_right[0] - b) / a]

        self.action_target[2] = gripper_left_command
        self.action_target[3] = gripper_right_command

    def set_target_CP(self, ee2ab_target_pose_left:np.ndarray, ee2ab_target_pose_right:np.ndarray, gripper_left:list, gripper_right:list):
        '''input value is the pose of the target position relative to the arm base and true width of the gripper'''
        # get current joints
        joint_cur_left, joint_cur_right, _, _ = self.get_robot_joints()

        # compute FK for current joints
        ee2w_pos_left, ee2w_orn_left = self.sim.arm_fk_with_step_sim("left_arm", joint_cur_left)
        ee2w_pos_right, ee2w_orn_right = self.sim.arm_fk_with_step_sim("right_arm", joint_cur_right)


        # convert pose to matrix
        ee2ab_target_mat_left = pose_to_mat(ee2ab_target_pose_left)
        ee2ab_target_mat_right = pose_to_mat(ee2ab_target_pose_right)
        
        # transform to world base; convert mat to pos and orn
        target_pos_left, target_orn_left = mat_to_pos_orn(self.ab2w_left @ ee2ab_target_mat_left)
        target_pos_right, target_orn_right = mat_to_pos_orn(self.ab2w_right @ ee2ab_target_mat_right)

        # solve ik
        joint_target_left, joint_target_right = self.sim.arm_ik(joint_cur_left, joint_cur_right, 
                                                        ee2w_pos_left, ee2w_orn_left, 
                                                        ee2w_pos_right, ee2w_orn_right)

        print()
        print("Current joint angles: left:", [f"{joint_cur_left[i]:.3f}" for i in range(len(joint_cur_left))], "right:", [f"{joint_cur_right[i]:.3f}" for i in range(len(joint_cur_right))])    
        print("Target joint angles: left:", [f"{joint_target_left[i]:.3f}" for i in range(len(joint_target_left))], "right:", [f"{joint_target_right[i]:.3f}" for i in range(len(joint_target_right))])
        print()

        '''debug'''
        # fk after ik
        left_ee2w_pos, left_ee2w_orn = self.sim.arm_fk_with_step_sim("left_arm", joint_target_left)
        right_ee2w_pos, right_ee2w_orn = self.sim.arm_fk_with_step_sim("right_arm", joint_target_right)
        left_ee2w_mat = pos_orn_to_mat(left_ee2w_pos, left_ee2w_orn)
        right_ee2w_mat = pos_orn_to_mat(right_ee2w_pos, right_ee2w_orn)

        left_ee2ab_pose = mat_to_pose(np.linalg.inv(self.ab2w_left) @ left_ee2w_mat)
        right_ee2ab_pose = mat_to_pose(np.linalg.inv(self.ab2w_right) @ right_ee2w_mat)

        print("SIM:FK after IK, left arm ee2w pose:")
        print(mat_to_pose(left_ee2w_mat))
        print("SIM:FK after IK, left arm ee2ab pose:")
        print(left_ee2ab_pose)
        # print("SIM:FK after IK, right arm ee2ab pose:")
        # print(right_ee2ab_pose)
        print()
        '''end of debug'''
        
        # convert true width to commanded width
        a = self.width_slope
        b = self.width_offset        
        gripper_left_command = [(gripper_left[0] - b) / a]
        gripper_right_command = [(gripper_right[0] - b) / a]

        # update target
        self.action_target[0] = joint_target_left
        self.action_target[1] = joint_target_right
        self.action_target[2] = gripper_left_command
        self.action_target[3] = gripper_right_command


    def execute(self):
        for i in range(len(self.component_name)):
            if self.action_target[i] is not None:
                # clip range for gripper
                if self.component_name[i] == "left_gripper":
                    self.action_target[i][0] = np.clip(self.action_target[i][0], -0.075, 0.0)
                elif self.component_name[i] == "right_gripper":
                    self.action_target[i][0] = np.clip(self.action_target[i][0], -0.075, 0.0)
                # send command to robot
                self.robot.set_joint_angle(self.component_name[i], self.action_target[i])

        # update simulation, only for arm
        for i in [0,1]:
            if self.action_target[i] is not None:
                self.sim.set_sim_joint_angle(self.component_name[i], self.action_target[i])

    def stop(self):
        self.robot._robot.shutdown()