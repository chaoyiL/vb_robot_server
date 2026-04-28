import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation
import numpy as np
import time

def vec_to_matrix(position):
    # Create rotation matrix from quaternion
    
    # Create 4x4 transformation matrix
    matrix = np.eye(4)
    matrix[:3, 3] = position
    
    return matrix

class PybulletEnv:
    def __init__(self, GUI = True):

        # 初始化仿真环境
        if GUI == True:
            self.physicsClient = p.connect(p.GUI)  # 使用GUI进行可视化
        else:
            self.physicsClient = p.connect(p.DIRECT)  # 使用GUI进行可视化
        p.setGravity(0, 0, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf")

        # 配置仿真参数
        self.time_step = 10.0 / 240.0  # 仿真步长
        p.setTimeStep(self.time_step)

        # 设置相机视角
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5]
        )

        # #载入机械臂模型
        # left_base_pos = [-0.2, 0, 0.5]
        # left_base_ori = p.getQuaternionFromEuler([-3.14/2, 0, 3.14/2])
        # right_base_pos = [0.2, 0, 0.5]
        # right_base_ori = p.getQuaternionFromEuler([3.14/2, 0, 3.14/2])
                #载入机械臂模型

        left_base_pos = [0, 0, 0]
        left_base_ori = p.getQuaternionFromEuler([0, 0, 0])
        right_base_pos = [100, 0, 0]
        right_base_ori = p.getQuaternionFromEuler([0, 0, 0])

        self.arm_left = p.loadURDF("real_world/robot_api/assets/arm_left.urdf", 
                        basePosition=left_base_pos, baseOrientation=left_base_ori, 
                        useFixedBase = True)
        self.arm_right = p.loadURDF("real_world/robot_api/assets/arm_right.urdf", 
                            basePosition=right_base_pos, baseOrientation=right_base_ori,
                            useFixedBase = True)
        
        # arm base to world base in sim
        self.arm_base_left = vec_to_matrix(left_base_pos)
        self.arm_base_right = vec_to_matrix(right_base_pos)
        
        # 获取可驱动关节信息（假设前n个为旋转关节）
        self.joint_indices_left = [i for i in range(p.getNumJoints(self.arm_left)) 
                        if p.getJointInfo(self.arm_left, i)[2] == p.JOINT_REVOLUTE]
        print(f"Controllable joints of left arm: {self.joint_indices_left}")

        self.joint_indices_right = [i for i in range(p.getNumJoints(self.arm_right)) 
                        if p.getJointInfo(self.arm_right, i)[2] == p.JOINT_REVOLUTE]
        print(f"Controllable joints of right arm: {self.joint_indices_right}")

        # 设置末端执行器链接索引
        self.end_effector_index = 6
    
    def set_sim_joint_angle(self, name, joint_angles):

        if name == "left_arm":
            for j in self.joint_indices_left:
                angle = joint_angles[j]
                p.resetJointState(self.arm_left, j, angle)

        if name == "right_arm": # sim right arm is opposite to real right arm
            for j in self.joint_indices_right:
                # angle = -joint_angles[j]
                angle = joint_angles[j]
                p.resetJointState(self.arm_right, j, angle)

        #设置转角后步进仿真
        p.stepSimulation()

    def arm_fk_with_step_sim(self, name, position)->tuple[list[float], list[float]]:

        self.set_sim_joint_angle(name, position)

        if name == "left_arm":
            robotID = self.arm_left
            # ab2w_mat = self.arm_base_left
        if name == "right_arm":
            robotID = self.arm_right
            # ab2w_mat = self.arm_base_right

        # 读取末端位姿
        ee2w_state = p.getLinkState(robotID, self.end_effector_index, computeForwardKinematics=True)
        
        ee2w_pos = ee2w_state[0]
        ee2w_orn = ee2w_state[1]

        ee2w_pos = list(ee2w_pos)
        ee2w_orn = list(ee2w_orn)

        # print("starting pose:")
        # print(ee2w_pos, ee2w_orn)

        # ee2w_pos[2] = ee2w_pos[2] + 0.02

        # ee2w_orn_new = ee2w_orn
        # # compute IK
        # joint_angles_left_tuple = p.calculateInverseKinematics(
        #     self.arm_left,
        #     endEffectorLinkIndex=self.end_effector_index,

        #     targetPosition=ee2w_pos,
        #     targetOrientation=ee2w_orn_new,

        #     lowerLimits=[-2*np.pi]*len(self.joint_indices_left),
        #     upperLimits=[2*np.pi]*len(self.joint_indices_left),

        #     maxNumIterations=10000,
        #     residualThreshold=1e-8,
        #     physicsClientId=self.physicsClient
        # )

        # print("current joint angles:\n", position)
        # joint_angles_left = list(joint_angles_left_tuple)
        # print(joint_angles_left)

        # self.set_sim_joint_angle(name, joint_angles_left)
        # ee2w_state = p.getLinkState(self.arm_left, self.end_effector_index, computeForwardKinematics=True)
        
        # ee2w_pos = ee2w_state[0]
        # ee2w_orn = ee2w_state[1]
        # print("ending pose:")
        # print(ee2w_pos, ee2w_orn)

        return ee2w_pos, ee2w_orn

    def arm_ik(self, joint_left: list, joint_right: list, 
            target_pos_left: list, target_orn_left: list, 
            target_pos_right: list, target_orn_right: list):

        # self.set_sim_joint_angle("left_arm", joint_left)
        # self.set_sim_joint_angle("right_arm", joint_right)

        # 计算逆运动学
        joint_angles_left_tuple = p.calculateInverseKinematics(
            self.arm_left,
            endEffectorLinkIndex=self.end_effector_index,

            targetPosition=target_pos_left,
            targetOrientation=target_orn_left,

            lowerLimits=[-np.pi]*len(self.joint_indices_left),
            upperLimits=[np.pi]*len(self.joint_indices_left),

            currentPosition=joint_left,

            maxNumIterations=10000,
            residualThreshold=1e-8,
            physicsClientId=self.physicsClient
        )

        joint_angles_left = list(joint_angles_left_tuple)

        joint_angles_right_tuple = p.calculateInverseKinematics(
            self.arm_right,
            endEffectorLinkIndex=self.end_effector_index,

            targetPosition=target_pos_right,
            targetOrientation=target_orn_right,

            lowerLimits=[-np.pi]*len(self.joint_indices_right),
            upperLimits=[np.pi]*len(self.joint_indices_right),

            currentPosition=joint_right,

            maxNumIterations=100,
            residualThreshold=1e-5,
            physicsClientId=self.physicsClient
        )

        joint_angles_right = list(joint_angles_right_tuple)

        # for i in range(len(joint_angles_right)): # sim right arm is opposite to real right arm
        #     joint_angles_right[i] = -joint_angles_right[i]

        return joint_angles_left, joint_angles_right