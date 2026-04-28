from __future__ import annotations

import json
import select
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from scipy.spatial.transform import Rotation
import os

sys.path.append(str(Path(__file__).resolve().parents[2]))

from real_world.robot_api.arm.RobotControl_pykin import RobotControl

def qpos_2_matrix_4x4(qpos, scalar_first=False):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(qpos[3:], scalar_first=scalar_first).as_matrix()
    matrix[:3, 3] = qpos[:3]
    return matrix

def left_coord_2_right_coord(pose):
    # 确保 pose 是 numpy 数组
    pose = np.array(pose)
    
    world_frame = np.array([0, 0, 0, 0, 0, 0, 1])
    world_frame[:3] = np.array([world_frame[0], world_frame[2], world_frame[1]])
    pose[:3] = np.array([pose[0], pose[2], pose[1]])

    Q = np.array([[1, 0, 0],
                [0, 0, 1],
                [0, 1, 0.]])
    rot_base = Rotation.from_quat(world_frame[3:]).as_matrix()
    rot = Rotation.from_quat(pose[3:]).as_matrix()
    rel_rot = Rotation.from_matrix(Q @ (rot_base.T @ rot) @ Q.T) # Is order correct.
    rel_pos = Rotation.from_matrix(Q @ rot_base.T@ Q.T).apply(pose[:3] - world_frame[:3]) # Apply base rotation not relative rotation...
    return rel_pos, rel_rot.as_quat()

class QuestPoseInteractiveClient:
    """
    Collect Quest pose data from a TCP stream.
    - Press 's' + Enter to save everything collected so far to a .npy file.
    - Press 'q' + Enter to quit gracefully.
    """

    def __init__(self, 
    host: str = "localhost", 
    port: int = 7777, 
    # save_dir: str | None = None
    ):
        self.host = host
        self.port = port
        self.socket: socket.socket | None = None
        self.running = False
        self.shutdown_requested = False

        self.latest_pose: dict | None = None
        self.snapshots: list[dict] = []
        self.session_timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
        # self.save_dir = Path(save_dir) if save_dir else None
        # if self.save_dir:
        #     self.save_dir.mkdir(parents=True, exist_ok=True)

        self.controller = RobotControl(vel_max=0.05)
        self.snapshots_robot: list[dict] = []

    def connect(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(1.0)  # short timeout to allow graceful shutdown checks
            self.socket.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"Connection failed ({exc}). Check: ADB forward tcp:7777 tcp:7777")
            return False

    def disconnect(self):
        if self.socket:
            self.socket.close()

    def parse_pose_data(self, json_str: str):
        try:
            data = json.loads(json_str.strip())
            if all(k in data for k in ["head_pose", "left_wrist", "right_wrist", "timestamp"]):
                return data
            return None
        except Exception:  # noqa: BLE001
            return None

    def record_snapshot(self):
        if self.latest_pose is None:
            print("\n[WARN] No pose received yet; cannot save snapshot.")
            return
        self.snapshots.append(self.latest_pose)
        self.snapshots_robot.append(self.controller.get_ee_pose())
        print(f"\n[SNAPSHOT] Captured frame #{len(self.snapshots)}")

    # def finalize_save(self):
    #     if not self.save_dir:
    #         print("[INFO] No save directory provided; skipping final save.")
    #         return
    #     if not self.snapshots:
    #         print("[INFO] No snapshots captured; nothing to save.")
    #         return

    #     filename = f"quest_poses_{self.session_timestamp}_manual.npy"
    #     output_path = self.save_dir / filename
    #     np.save(output_path, np.array(self.snapshots, dtype=object), allow_pickle=True)
    #     print(f"[SAVED] {output_path} ({len(self.snapshots)} frames)")

    #     return output_path

    def poll_keyboard(self):
        """Non-blocking check for keyboard commands."""
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
        except (ValueError, OSError):
            return

        for _ in ready:
            cmd = sys.stdin.readline().strip().lower()
            if cmd == "s":
                self.record_snapshot()
            elif cmd == "q":
                print("\n[INFO] Quit requested by user.")
                self.shutdown_requested = True
                self.running = False
            elif cmd:
                print("\n[INFO] Unknown command. Use 's' to save, 'q' to quit.")

    def run(self):
        if not self.connect():
            return

        self.running = True
        buffer = ""
        buffer_max_size = 10 * 1024 * 1024  # 10MB

        print("\nReceiving Quest pose data... ('s'+Enter to save, 'q'+Enter to quit)")
        # if self.save_dir:
        #     print(f"Save directory: {self.save_dir}")
        # else:
        #     print("[WARN] No save directory provided; will not write files.")

        try:
            while self.running and not self.shutdown_requested:
                try:
                    data = self.socket.recv(1024).decode("utf-8")
                    if not data:
                        print("\n[INFO] No more data received, closing...")
                        break

                    buffer += data
                    if len(buffer) > buffer_max_size:
                        print(f"\n[WARNING] Buffer too large ({len(buffer)} bytes), clearing...")
                        buffer = ""
                        continue

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip():
                            pose_data = self.parse_pose_data(line)
                            if pose_data:
                                self.latest_pose = {
                                    **pose_data,
                                    "timestamp_unix": time.time(),
                                    "timestamp_readable": datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f"),
                                }

                    # Check for user commands every loop
                    self.poll_keyboard()

                except socket.timeout:
                    self.poll_keyboard()
                    continue
                except UnicodeDecodeError:
                    continue
                except (ConnectionResetError, BrokenPipeError):
                    print("\n[INFO] Connection lost")
                    break
            
        except KeyboardInterrupt:
            print("\n\n[INFO] Interrupted by user (Ctrl+C)")
            self.shutdown_requested = True
        finally:
            print("\n[INFO] Closing connection...")
            self.running = False
            self.disconnect()
            # path = self.finalize_save()
            print("[INFO] Shutdown complete.")

            # return path
            return self.snapshots, self.snapshots_robot


def main():
    # default_save_dir = "/home/rvsa/codehub/VB-VLA/tools/get_quest_pose/manual_snapshots"

    client = QuestPoseInteractiveClient(
        # save_dir=str(default_save_dir)
    )
    # path = client.run()
    # return path
    snapshots, snapshots_robot = client.run()
    return snapshots, snapshots_robot

if __name__ == "__main__":
    '''config'''
    save_dir = "/home/rvsa/codehub/VB-VLA/tools/check_hand_eye_mat/pose_and_error"
    quest_2_ee_left = np.load("/home/rvsa/codehub/VB-VLA/quest_2_ee_left_hand_fix_quest.npy")
    quest_2_ee_right = np.load("/home/rvsa/codehub/VB-VLA/quest_2_ee_right_hand_fix_quest.npy")
    # # 将上面两个矩阵求逆
    # quest_2_ee_left = np.linalg.inv(quest_2_ee_left)
    # quest_2_ee_right = np.linalg.inv(quest_2_ee_right)

    snapshots, snapshots_robot = main()

    pose_quest = snapshots
    pose_robot = snapshots_robot
    data_num = len(pose_quest)
    left_quest_position = []
    left_quest_rotation = []
    right_quest_position = []
    right_quest_rotation = []
    left_robot_pose_matrix = []
    right_robot_pose_matrix = []

    for i in range(data_num):
        # 提取 position (字典 -> 数组 [x, y, z])
        left_pos = pose_quest[i]['left_wrist']['position']
        left_quest_position.append(np.array([left_pos['x'], left_pos['y'], left_pos['z']]))
        
        right_pos = pose_quest[i]['right_wrist']['position']
        right_quest_position.append(np.array([right_pos['x'], right_pos['y'], right_pos['z']]))
        
        left_rot = pose_quest[i]['left_wrist']['rotation']
        left_quest_rotation.append(np.array([left_rot['x'], left_rot['y'], left_rot['z'], left_rot['w']]))
        
        right_rot = pose_quest[i]['right_wrist']['rotation']
        right_quest_rotation.append(np.array([right_rot['x'], right_rot['y'], right_rot['z'], right_rot['w']]))

        left_robot_pose_matrix.append(qpos_2_matrix_4x4(pose_robot[i]['left_arm_ee2rb'], scalar_first=True))
        right_robot_pose_matrix.append(qpos_2_matrix_4x4(pose_robot[i]['right_arm_ee2rb'], scalar_first=True))

    # 将 position (3维) 和 rotation (4维) 拼接成 7 维向量
    left_quest_pose_7d = [np.concatenate([left_quest_position[i], left_quest_rotation[i]]) for i in range(data_num)]
    right_quest_pose_7d = [np.concatenate([right_quest_position[i], right_quest_rotation[i]]) for i in range(data_num)]

    # 转换坐标系，返回的元组需要拼接成7维向量
    left_quest_pose_7d_right_coord = []
    right_quest_pose_7d_right_coord = []

    for i in range(data_num):
        rel_pos_l, rel_rot_l = left_coord_2_right_coord(left_quest_pose_7d[i])
        left_quest_pose_7d_right_coord.append(np.concatenate([rel_pos_l, rel_rot_l]))
        
        rel_pos_r, rel_rot_r = left_coord_2_right_coord(right_quest_pose_7d[i])
        right_quest_pose_7d_right_coord.append(np.concatenate([rel_pos_r, rel_rot_r]))

    # 得到quest pose的观测值与计算值
    # （quest左右对调）
    left_quest_mat_got = [qpos_2_matrix_4x4(left_quest_pose_7d_right_coord[i]) for i in range(data_num)]
    right_quest_mat_got = [qpos_2_matrix_4x4(right_quest_pose_7d_right_coord[i]) for i in range(data_num)]
    right2left_mat_got = [
        np.linalg.inv(left_quest_mat_got[i]) @ right_quest_mat_got[i]
        for i in range(data_num)]

    left_quest_mat_cal = [left_robot_pose_matrix[i] @ quest_2_ee_left for i in range(data_num)]
    right_quest_mat_cal = [right_robot_pose_matrix[i] @ quest_2_ee_right for i in range(data_num)]
    left2right_mat_cal = [
        np.linalg.inv(right_quest_mat_cal[i]) @ left_quest_mat_cal[i] 
        for i in range(data_num)]

    # 加入一段代码，计算每个quest pose的观测值与计算值的误差
    error_mat = [right2left_mat_got[i] - left2right_mat_cal[i] for i in range(data_num)]

    # 计算error的平均矩阵
    error_mean = np.mean(error_mat, axis=0)

    print("right2left_mat_got: \n", right2left_mat_got)
    print("left2right_mat_cal: \n", left2right_mat_cal)
    print("error_mean: \n", error_mean)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'right2left_mat_got.npy'), right2left_mat_got)
    np.save(os.path.join(save_dir, 'left2right_mat_cal.npy'), left2right_mat_cal)
    np.save(os.path.join(save_dir, 'error_mat.npy'), error_mat)

    print("process done")