import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet as p

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(ROOT_DIR)

from real_world.robot_api.sim.PybulletEnv import PybulletEnv

def quaternion_angle_error(q1, q2):
    """
    计算两个四元数之间的角度误差（弧度）
    q1, q2: [x, y, z, w] 格式的四元数（Pybullet格式）
    """
    # Pybullet使用 [x, y, z, w] 格式，scipy也使用 [x, y, z, w] 格式
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    
    # 计算相对旋转
    r_rel = r2 * r1.inv()
    
    # 获取旋转角度（弧度）
    angle = r_rel.magnitude()
    
    return angle

def test_ik_fk(num_steps=100):
    """
    测试IK和FK的一致性
    
    在循环中：
    1. 给出一组特定的关节角度
    2. 使用FK计算末端位姿
    3. 使用IK从末端位姿反算关节角度
    4. 使用IK返回的关节角度再次计算FK
    5. 比较原始FK结果和IK后的FK结果
    """
    # 初始化仿真环境
    sim = PybulletEnv(GUI=True)
    
    # 定义关节角度的范围（弧度）
    joint_min = np.array([-2.0, -1.5, -2.0, -1.5, -2.0, -1.5, -3.0])
    joint_max = np.array([2.0, 1.5, 2.0, 1.5, 2.0, 1.5, 3.0])
    
    # 统计误差
    pos_errors_left = []
    pos_errors_right = []
    orn_errors_left = []
    orn_errors_right = []
    
    print(f"开始IK/FK测试，共 {num_steps} 步...")
    print("=" * 80)
    
    for step in range(num_steps):
        # 1. 生成一组特定的关节角度（在合理范围内随机生成）
        joint_angles_left = np.random.uniform(joint_min, joint_max)
        joint_angles_right = np.random.uniform(joint_min, joint_max)
        
        # 转换为列表格式
        joint_left_list = joint_angles_left.tolist()
        joint_right_list = joint_angles_right.tolist()
        
        # 2. 使用FK计算末端位姿（原始）
        ee2w_pos_left, ee2w_orn_left = sim.arm_fk_with_step_sim("left_arm", joint_left_list)
        ee2w_pos_right, ee2w_orn_right = sim.arm_fk_with_step_sim("right_arm", joint_right_list)
        
        # 3. 使用IK从末端位姿反算关节角度
        joint_ik_left, joint_ik_right = sim.arm_ik(
            joint_left_list, joint_right_list,
            ee2w_pos_left, ee2w_orn_left,
            ee2w_pos_right, ee2w_orn_right
        )

        # 4. 使用IK返回的关节角度再次计算FK
        ee2w_pos_left_after, ee2w_orn_left_after = sim.arm_fk_with_step_sim("left_arm", joint_ik_left)
        ee2w_pos_right_after, ee2w_orn_right_after = sim.arm_fk_with_step_sim("right_arm", joint_ik_right)
        
        # 5. 计算位置误差（欧氏距离，单位：米）
        pos_error_left = np.linalg.norm(np.array(ee2w_pos_left) - np.array(ee2w_pos_left_after))
        pos_error_right = np.linalg.norm(np.array(ee2w_pos_right) - np.array(ee2w_pos_right_after))
        
        # 6. 计算姿态误差（角度差，单位：弧度）
        orn_error_left = quaternion_angle_error(ee2w_orn_left, ee2w_orn_left_after)
        orn_error_right = quaternion_angle_error(ee2w_orn_right, ee2w_orn_right_after)
        
        pos_errors_left.append(pos_error_left)
        pos_errors_right.append(pos_error_right)
        orn_errors_left.append(orn_error_left)
        orn_errors_right.append(orn_error_right)
        
        # 打印每10步的结果
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step + 1:3d}:")
            print(f"  左臂 - 位置误差: {pos_error_left*1000:.4f} mm, 姿态误差: {np.degrees(orn_error_left):.4f} deg")
            print(f"  右臂 - 位置误差: {pos_error_right*1000:.4f} mm, 姿态误差: {np.degrees(orn_error_right):.4f} deg")
            print()
    
    # 统计结果
    print("=" * 80)
    print("测试完成！统计结果：")
    print(f"左臂位置误差 - 平均: {np.mean(pos_errors_left)*1000:.4f} mm, 最大: {np.max(pos_errors_left)*1000:.4f} mm")
    print(f"右臂位置误差 - 平均: {np.mean(pos_errors_right)*1000:.4f} mm, 最大: {np.max(pos_errors_right)*1000:.4f} mm")
    print(f"左臂姿态误差 - 平均: {np.degrees(np.mean(orn_errors_left)):.4f} deg, 最大: {np.degrees(np.max(orn_errors_left)):.4f} deg")
    print(f"右臂姿态误差 - 平均: {np.degrees(np.mean(orn_errors_right)):.4f} deg, 最大: {np.degrees(np.max(orn_errors_right)):.4f} deg")
    print(f"左臂位置误差 - 标准差: {np.std(pos_errors_left)*1000:.4f} mm")
    print(f"右臂位置误差 - 标准差: {np.std(pos_errors_right)*1000:.4f} mm")
    print(f"左臂姿态误差 - 标准差: {np.degrees(np.std(orn_errors_left)):.4f} deg")
    print(f"右臂姿态误差 - 标准差: {np.degrees(np.std(orn_errors_right)):.4f} deg")
    
    # 判断测试是否通过（位置误差阈值：1mm，姿态误差阈值：1度）
    pos_threshold = 0.001  # 1mm
    orn_threshold = np.radians(1.0)  # 1度
    
    if (np.mean(pos_errors_left) < pos_threshold and 
        np.mean(pos_errors_right) < pos_threshold and
        np.mean(orn_errors_left) < orn_threshold and
        np.mean(orn_errors_right) < orn_threshold):
        print(f"\n✓ 测试通过！误差在允许范围内")
    else:
        print(f"\n✗ 测试未通过！误差超出允许范围")
    
    return pos_errors_left, pos_errors_right, orn_errors_left, orn_errors_right

if __name__ == "__main__":
    np.random.seed(42)  # 设置随机种子以便复现
    pos_errors_left, pos_errors_right, orn_errors_left, orn_errors_right = test_ik_fk(num_steps=100)
