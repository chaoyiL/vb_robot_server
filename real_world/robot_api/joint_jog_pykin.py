"""
joint_jog_pykin.py
==================
交互式手动 jog 工具,用于驱动双臂机械臂并保存关键位姿。

★ 安全护栏 ★
  - jog 命令 (w/s) 步长 0.02 rad,自然安全
  - b/sr 命令前会做关节空间距离检查,大幅度运动需输入 'y' 确认
  - sr 命令会校验 IK 收敛,不收敛拒绝执行
  - signal handler 容忍未初始化状态
  - 任意 abort 路径都会把 action_target 重置回当前关节,防止主循环 execute 把它带过去
"""

import os
import sys
import time
import signal
import numpy as np

dir = os.getcwd()
sys.path.append(dir)

import scipy.spatial.transform as t3d
from real_world.robot_api.arm.RobotControl_pykin import RobotControl
from utils.pose_util import pose_to_mat, mat_to_pose


# ============================================================
# 安全相关参数 (全部在这里调,方便审阅)
# ============================================================
JOG_STEP_RAD            = 0.02   # 单步 jog (≈ 1.15°)
LARGE_MOVE_THRESHOLD    = 0.10   # 单关节超过这个差值视为"大幅度",触发二次确认 (≈ 17°)
IK_CONVERGENCE_POS_TOL  = 0.005  # 5 mm
IK_CONVERGENCE_ROT_TOL  = 1e-2
LOOP_SLEEP_S            = 0.3
VEL_MAX                 = 0.1    # 注意:typhon backend 下此参数无效


# ============================================================
# 工具函数
# ============================================================
def pos_orn_to_mat(position, quaternion_wxyz):
    """quaternion 是 wxyz (scalar-first)"""
    rot = t3d.Rotation.from_quat(quaternion_wxyz, scalar_first=True).as_matrix()
    M = np.eye(4)
    M[:3, :3] = rot
    M[:3, 3] = position
    return M


def mat_to_pos_orn(matrix: np.ndarray) -> np.ndarray:
    """返回 7-vector [x,y,z, qw,qx,qy,qz] (scalar-first)"""
    pos = matrix[:3, 3]
    quat_wxyz = t3d.Rotation.from_matrix(matrix[:3, :3]).as_quat(scalar_first=True)
    return np.concatenate([pos, quat_wxyz])


def confirm_large_move(label: str, joint_l_target, joint_r_target,
                        curr_left, curr_right,
                        threshold: float = LARGE_MOVE_THRESHOLD) -> bool:
    """
    检查目标姿态相对当前的最大关节差。大幅度运动需用户确认。
    Returns: True 通过, False 用户拒绝/异常
    """
    try:
        diff_l = np.max(np.abs(np.array(joint_l_target) - np.array(curr_left)))
        diff_r = np.max(np.abs(np.array(joint_r_target) - np.array(curr_right)))
    except ValueError as e:
        print(f"[SAFETY] {label}: shape mismatch ({e}), 拒绝执行")
        return False

    print(f"\n[SAFETY] {label}:")
    print(f"  left  max joint diff = {diff_l:.4f} rad ({np.degrees(diff_l):.2f}°)")
    print(f"  right max joint diff = {diff_r:.4f} rad ({np.degrees(diff_r):.2f}°)")
    print(f"  current left:  {np.round(curr_left, 3)}")
    print(f"  target  left:  {np.round(joint_l_target, 3)}")
    print(f"  current right: {np.round(curr_right, 3)}")
    print(f"  target  right: {np.round(joint_r_target, 3)}")

    if max(diff_l, diff_r) <= threshold:
        print(f"  ✓ 在阈值 {threshold:.2f} rad 内,直接执行")
        return True

    ans = input(f"  ! 超阈值 ({threshold:.2f} rad), 确认执行? 输入 'y' 继续,其他取消: ").strip().lower()
    if ans == "y":
        print("  → 用户确认,执行")
        return True
    print("  ✗ 用户取消")
    return False


def reset_target_to_current(controller):
    """把 action_target 设回当前关节角,防止主循环末尾 execute() 把上次设的目标带出去。"""
    cur = controller.get_robot_joints()
    controller.set_target_JP(
        cur["left_arm"], cur["right_arm"],
        cur["left_gripper"], cur["right_gripper"],
    )


# ============================================================
# Signal handling
# ============================================================
controller = None
control_running = False


def signal_handler(sig, frame):
    print("\n[SIGINT] 用户按下 Ctrl+C")
    global control_running
    control_running = False
    if controller is not None:
        try:
            controller.robot._robot.shutdown()
            print("[SIGINT] robot shutdown OK")
        except Exception as e:
            print(f"[SIGINT] shutdown 失败: {e}")
    else:
        print("[SIGINT] controller 尚未初始化,跳过 shutdown")
    sys.exit(0)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    save_dir = "manual_snapshots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,
                             f"saved_poses_{time.strftime('%Y.%m.%d_%H.%M.%S')}.npy")

    controller = RobotControl(vel_max=VEL_MAX)
    print(f"[init] Server up. vel_max={VEL_MAX} (注意: typhon backend 下此值无效)")
    print(f"[init] Save target: {save_path}")
    print(f"[init] LARGE_MOVE_THRESHOLD = {LARGE_MOVE_THRESHOLD} rad "
          f"({np.degrees(LARGE_MOVE_THRESHOLD):.1f}°)")

    arm = "l"
    joint = 0
    direction = "w"
    err = 0
    control_running = True

    # 第一次先把目标设为当前姿态,确保后续 execute() 不会发出陈旧目标
    reset_target_to_current(controller)
    controller.execute()

    while control_running:

        '''获取当前关节状态'''
        curr_state = controller.get_robot_joints()
        position_l = curr_state["left_arm"]
        position_r = curr_state["right_arm"]
        gripper_l = curr_state["left_gripper"]
        gripper_r = curr_state["right_gripper"]

        print("\n--- current state ---")
        print(f"left  joint:   {np.round(position_l, 4)}")
        print(f"right joint:   {np.round(position_r, 4)}")
        print(f"left  gripper: {np.round(gripper_l, 4)}")
        print(f"right gripper: {np.round(gripper_r, 4)}")

        '''遥控信号输入'''
        control_signal = input(
            f"\n[{arm}/joint{joint}/{direction}] 命令 (l/r,0-6,w/s,lo/lc/ro/rc,b,sr,save,q): "
        ).strip()

        # ---------------- 模式切换 ----------------
        if control_signal in ("l", "r"):
            arm = control_signal

        elif control_signal in ("w", "s"):
            direction = control_signal

        elif control_signal == "":
            pass  # 空回车 = 重复上一个方向

        elif control_signal == "q":
            print("退出...")
            control_running = False

        elif control_signal in ("0", "1", "2", "3", "4", "5", "6"):
            joint = int(control_signal)

        # ---------------- 夹爪 ----------------
        elif control_signal == "lo":
            gripper_l = [0.05]
        elif control_signal == "lc":
            gripper_l = [0.03]
        elif control_signal == "ro":
            gripper_r = [0.05]
        elif control_signal == "rc":
            gripper_r = [0.03]

        # ---------------- 保存 ----------------
        elif control_signal == "save":
            ee_pose = controller.get_ee_pose()
            # 注意:不再覆盖外层 gripper_l/r,以免影响下次 jog
            pose_data = {
                "ee2rb_left":  ee_pose["left_arm_ee2rb"],
                "ee2rb_right": ee_pose["right_arm_ee2rb"],
                "gripper_l":   ee_pose["left_gripper"],
                "gripper_r":   ee_pose["right_gripper"],
                "joint_l":     position_l,    # 同时存关节角,方便复现
                "joint_r":     position_r,
                "timestamp":   time.time(),
            }
            print("\n[save] pose_data:")
            for k, v in pose_data.items():
                print(f"  {k}: {v}")

            if os.path.exists(save_path):
                existing = np.load(save_path, allow_pickle=True)
                saved_list = [existing.item()] if existing.shape == () else list(existing)
                saved_list.append(pose_data)
            else:
                saved_list = [pose_data]
            np.save(save_path, saved_list)
            print(f"[save] -> {save_path}  (#{len(saved_list)})")

        # ---------------- b: 跳到 home ----------------
        elif control_signal == "b":
            joint_l_home = [-1.46198,  1.12388, -0.253083, 1.37440,  1.798234, -0.560622, 0.254401]
            joint_r_home = [-1.372984, 1.191208, -0.396319, 1.252495, 1.811727, -0.471914, 0.136069]
            gripper_l_home = [0.0391]
            gripper_r_home = [0.0354]

            if confirm_large_move("b: move-to-home",
                                   joint_l_home, joint_r_home,
                                   position_l, position_r):
                controller.set_target_JP(joint_l_home, joint_r_home,
                                         gripper_l_home, gripper_r_home)
            else:
                # 用户取消:把 target 锁回当前,防止 execute 误发
                reset_target_to_current(controller)
                err = 1

        # ---------------- sr: 右臂跟随左臂 (固定相对位姿) ----------------
        elif control_signal == "sr":
            # 注意:l2r_pose 实际是 right→left 的 6-vector(基于历史代码)
            l2r_pose = np.array([
                -4.65774357e-01, -2.38045557e-01, 4.98711966e-04,
                -2.85670998e+00, 1.09655970e+00, 1.99338444e-01,
            ])
            l2r_mat = pose_to_mat(l2r_pose)

            # 1) 取当前左臂 EE
            ee_pose = controller.get_ee_pose()
            ee_l_7v = ee_pose["left_arm_ee2rb"]
            ee_l_mat = pos_orn_to_mat(ee_l_7v[:3], ee_l_7v[3:7])

            # 2) 算右臂目标 EE
            ee_r_7v = mat_to_pos_orn(ee_l_mat @ np.linalg.inv(l2r_mat))

            # 3) 用 IK 算右臂关节,先做收敛检查
            curr_right_arr = np.asarray(position_r)
            joints_r_pred = controller._inverse_kin_silent(
                controller.kin_right, curr_right_arr, ee_r_7v
            )
            # 复算 FK,确认 IK 真的收敛了
            ee_r_check = controller.kin_right.compute_eef_pose(
                controller.kin_right.forward_kin(joints_r_pred)
            )
            pos_err = float(np.linalg.norm(np.asarray(ee_r_check[:3]) - ee_r_7v[:3]))
            rot_err = float(1 - abs(np.dot(ee_r_check[3:], ee_r_7v[3:])))

            if pos_err > IK_CONVERGENCE_POS_TOL or rot_err > IK_CONVERGENCE_ROT_TOL:
                print(f"\n[SAFETY] sr: IK 未收敛 "
                      f"(pos_err={pos_err*1000:.2f}mm, rot_err={rot_err:.2e}),拒绝执行")
                reset_target_to_current(controller)
                err = 1
            else:
                # 用关节空间距离检查代替 CP 直接下发,跳大角度需要确认
                if confirm_large_move("sr: right-arm follow",
                                       position_l, joints_r_pred,
                                       position_l, position_r):
                    # 这里走 JP 而不是 CP,因为我们已经亲自做了 IK + 校验
                    controller.set_target_JP(
                        position_l, joints_r_pred.tolist(),
                        gripper_l, gripper_r,
                    )
                else:
                    reset_target_to_current(controller)
                    err = 1

        else:
            print("输入有误,请重新输入")
            err = 1

        # ---------------- jog 步进 ----------------
        if (control_signal in ["w", "s", ""]) and err == 0:
            move_step_dir = JOG_STEP_RAD if direction == "w" else -JOG_STEP_RAD
            if arm == "l":
                position_l[joint] += move_step_dir
            elif arm == "r":
                position_r[joint] += move_step_dir
            controller.set_target_JP(position_l, position_r, gripper_l, gripper_r)

        # ---------------- 夹爪命令的 target 更新 ----------------
        # (删掉了原来 100 次循环的死代码 - 没 execute 没用)
        if control_signal in ("lo", "lc", "ro", "rc") and err == 0:
            controller.set_target_JP(position_l, position_r, gripper_l, gripper_r)

        err = 0
        controller.execute()
        time.sleep(LOOP_SLEEP_S)

    # 正常退出路径
    try:
        controller.robot._robot.shutdown()
        print("[exit] robot shutdown OK")
    except Exception as e:
        print(f"[exit] shutdown 失败: {e}")
    sys.exit(0)