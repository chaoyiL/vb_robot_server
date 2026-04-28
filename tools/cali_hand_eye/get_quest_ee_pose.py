from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import cv2

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.append(str(Path(__file__).resolve().parents[2]))

from real_world.robot_api.arm.RobotControl_pykin import RobotControl


# ──────────────────────────── coordinate helpers ────────────────────────────

def qpos_2_matrix_4x4(qpos: np.ndarray, scalar_first=False) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(qpos[3:], scalar_first=scalar_first).as_matrix()
    matrix[:3, 3] = qpos[:3]
    return matrix


def left_coord_2_right_coord(pose: np.ndarray):
    pose = np.array(pose)
    world_frame = np.array([0, 0, 0, 0, 0, 0, 1], dtype=float)
    world_frame[:3] = np.array([world_frame[0], world_frame[2], world_frame[1]])
    pose[:3] = np.array([pose[0], pose[2], pose[1]])

    Q = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0.]])
    rot_base = Rotation.from_quat(world_frame[3:], scalar_first=False).as_matrix()
    rot = Rotation.from_quat(pose[3:], scalar_first=False).as_matrix()
    rel_rot = Rotation.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
    rel_pos = Rotation.from_matrix(Q @ rot_base.T @ Q.T).apply(
        pose[:3] - world_frame[:3]
    )
    return rel_pos, rel_rot.as_quat()

# 
def extract_rotation_translation(transform_matrix):
    R = transform_matrix[:3, :3]
    t = transform_matrix[:3, 3].reshape(3, 1)
    return R, t

def calibrate_hand_eye(ee_2_base_matrices, world_2_quest):
    R1_list = [] 
    t1_list = [] 
    R2_list = []  
    t2_list = []
    for A, B in zip(ee_2_base_matrices, world_2_quest):
        R1, t1 = extract_rotation_translation(A)
        R2, t2 = extract_rotation_translation(B)
        
        R1_list.append(R1) 
        t1_list.append(t1)
        R2_list.append(R2) 
        t2_list.append(t2)
    
    R1_list = [R.astype(np.float64) for R in R1_list]
    t1_list = [t.astype(np.float64) for t in t1_list]
    R2_list = [R.astype(np.float64) for R in R2_list]
    t2_list = [t.astype(np.float64) for t in t2_list]
    
    R_quest_to_ee, t_quest_to_ee = cv2.calibrateHandEye(
        R1_list, t1_list,
        R2_list, t2_list,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS
        # method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    quest_to_ee = np.eye(4)
    quest_to_ee[:3, :3] = R_quest_to_ee
    quest_to_ee[:3, 3] = t_quest_to_ee.flatten()
    return quest_to_ee

# ──────────────────── Quest TCP receiver (background thread) ────────────────

class QuestPoseReceiver:
    """Background thread that keeps the latest Quest pose up-to-date."""

    def __init__(self, host: str = "localhost", port: int = 7777):
        self.host = host
        self.port = port
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self.latest_pose: dict | None = None

    def connect(self) -> bool:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(1.0)
            self._socket.connect((self.host, self.port))
            self._connected = True
            print(f"[Quest] Connected to {self.host}:{self.port}")
            return True
        except Exception as exc:
            print(f"[Quest] Connection failed ({exc}). Check: adb forward tcp:7777 tcp:7777")
            return False

    def start(self):
        if not self.connect():
            return False
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self._socket:
            self._socket.close()
        print("[Quest] Receiver stopped.")

    def get_latest(self) -> dict | None:
        with self._lock:
            return self.latest_pose

    def _recv_loop(self):
        buf = ""
        buf_max = 10 * 1024 * 1024
        while self._running:
            try:
                data = self._socket.recv(1024).decode("utf-8")
                if not data:
                    print("[Quest] No more data, stream closed.")
                    break
                buf += data
                if len(buf) > buf_max:
                    buf = ""
                    continue
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if not line.strip():
                        continue
                    parsed = self._parse(line)
                    if parsed:
                        with self._lock:
                            self.latest_pose = {
                                **parsed,
                                "timestamp_unix": time.time(),
                                "timestamp_readable": datetime.now().strftime(
                                    "%Y.%m.%d_%H.%M.%S.%f"
                                ),
                            }
            except socket.timeout:
                continue
            except UnicodeDecodeError:
                continue
            except (ConnectionResetError, BrokenPipeError):
                print("[Quest] Connection lost.")
                break

    @staticmethod
    def _parse(json_str: str) -> dict | None:
        try:
            d = json.loads(json_str.strip())
            if all(k in d for k in ("head_pose", "left_wrist", "right_wrist", "timestamp")):
                return d
        except Exception:
            pass
        return None


# ─────────────────── post-processing: quest pose → 4×4 matrix ──────────────

def process_quest_snapshots(snapshots: list[dict]) -> dict[str, list[np.ndarray]]:
    '''
    左手系转右手系；对调左右手数据。
    '''
    left_matrices, right_matrices = [], []
    for snap in snapshots:
        for side, matrices in (("left_wrist", left_matrices), ("right_wrist", right_matrices)):
            p = snap[side]["position"]
            r = snap[side]["rotation"]
            pose_7d = np.array([p["x"], p["y"], p["z"], r["x"], r["y"], r["z"], r["w"]])
            rel_pos, rel_rot = left_coord_2_right_coord(pose_7d)
            mat = qpos_2_matrix_4x4(np.concatenate([rel_pos, rel_rot]), scalar_first=False)
            matrices.append(mat)

    # 左右对调！！！
    return {
        "left_quest_pose_mat": right_matrices,
        "right_quest_pose_mat": left_matrices,
    }

def process_ee_snapshots(snapshots: list[dict]) -> dict[str, list[np.ndarray]]:
    left_matrices, right_matrices = [], []

    for snap in snapshots:
        left_pos_orn = snap["left_arm_ee2rb"]
        right_pos_orn = snap["right_arm_ee2rb"]
        left_mat = qpos_2_matrix_4x4(np.concatenate([left_pos_orn[:3], left_pos_orn[3:]]), scalar_first=True)
        right_mat = qpos_2_matrix_4x4(np.concatenate([right_pos_orn[:3], right_pos_orn[3:]]), scalar_first=True)
        left_matrices.append(left_mat)
        right_matrices.append(right_mat)

    return {
        "left_ee_pose_mat": left_matrices,
        "right_ee_pose_mat": right_matrices,
    }

# ────────────────────────── combined interactive loop ───────────────────────

class CaliDataCollector:
    """
    Unified data collector for hand-eye calibration.

    Runs the robot controller in the foreground (interactive joint control)
    while a background thread keeps the latest Quest pose refreshed.

    Commands (type + Enter):
        l / r          — select left / right arm
        0-6            — select joint index
        w / s          — jog selected joint forward / backward
        Enter (empty)  — repeat last jog
        save           — snapshot BOTH ee-pose and quest-pose
        b              — go to preset "base" joint configuration
        q              — quit and save all collected data
    """

    PRESET_JOINT_L = [-1.91627709, 0.89598147, -0.10594245, 1.37463758, 
                     1.47918844, 0.11891484, 0.15657425] # left arm preset joint configuration
    PRESET_JOINT_R = [-2.52254692, 0.59934129, 0.74494037, 1.34938575, 
                     1.64482704, -0.17661188, 0.20930105] # right arm preset joint configuration

    def __init__(
        self,
        save_dir: str | Path,
        quest_host: str = "localhost",
        quest_port: int = 7777,
        vel_max: float = 0.5,
        move_step: float = 0.02,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.session_ts = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())

        self.quest = QuestPoseReceiver(host=quest_host, port=quest_port)
        self.controller = RobotControl(vel_max=vel_max)
        self.move_step = move_step

        self.ee_snapshots: list[dict] = []
        self.quest_snapshots: list[dict] = []
        self._running = False

    # ── snapshot ──

    def _take_snapshot(self):
        quest_pose = self.quest.get_latest()
        if quest_pose is None:
            print("\n[WARN] Quest pose not available yet, skipping.")
            return

        ee_pose = self.controller.get_ee_pose()

        self.quest_snapshots.append(quest_pose)
        self.ee_snapshots.append(ee_pose)

        idx = len(self.ee_snapshots)
        print(f"\n[SNAPSHOT #{idx}] ee_pose + quest_pose captured")
        # print(f"  ee left:  {ee_pose['left_arm_ee2rb'][:3, 3]}")
        # print(f"  ee right: {ee_pose['right_arm_ee2rb'][:3, 3]}")
        # lp = quest_pose["left_wrist"]["position"]
        # rp = quest_pose["right_wrist"]["position"]
        # print(f"  quest left:  [{lp['x']:.4f}, {lp['y']:.4f}, {lp['z']:.4f}]")
        # print(f"  quest right: [{rp['x']:.4f}, {rp['y']:.4f}, {rp['z']:.4f}]")

    # ── save & calibrate ──

    def _save_and_calibrate(self):
        """Process snapshots, save to disk, and run hand-eye calibration."""
        if not self.ee_snapshots:
            print("[INFO] No snapshots to save; skipping calibration.")
            return

        n = len(self.ee_snapshots)
        result_dir = self.save_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)

        processed_quest = process_quest_snapshots(self.quest_snapshots)
        np.save(result_dir / "left_quest_pose_mat.npy",
                processed_quest["left_quest_pose_mat"])
        np.save(result_dir / "right_quest_pose_mat.npy",
                processed_quest["right_quest_pose_mat"])

        processed_ee = process_ee_snapshots(self.ee_snapshots)
        np.save(result_dir / "left_ee_pose_mat.npy",
                processed_ee["left_ee_pose_mat"])
        np.save(result_dir / "right_ee_pose_mat.npy",
                processed_ee["right_ee_pose_mat"])

        print(f"\n[SAVED] {n} pairs of snapshots → {result_dir}")

        print("[INFO] Calculating hand-eye matrices...")
        for side in ("left", "right"):
            ee2rb_matrices = processed_ee[f"{side}_ee_pose_mat"]
            wb2quest_matrices = [
                np.linalg.inv(m) for m in processed_quest[f"{side}_quest_pose_mat"]
            ]
            print(f"  [{side}] last ee2rb:\n{ee2rb_matrices[-1]}")
            print(f"  [{side}] last wb2quest:\n{wb2quest_matrices[-1]}")

            X = calibrate_hand_eye(ee2rb_matrices, wb2quest_matrices)
            print(f"  [{side}] quest→ee matrix:\n{X}")
            np.save(self.save_dir / f"quest_2_ee_{side}_hand_fix_quest.npy", X)

    # ── trajectory generation ──

    @staticmethod
    def _generate_trajectory(
        base_joints: list[float],
        num_points: int,
        amplitude: float,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Generate a trajectory of joint-space waypoints around a base configuration.

        Each waypoint randomly perturbs all 7 joints within [-amplitude, +amplitude].
        The first waypoint is the base configuration itself (for homing).

        Returns:
            np.ndarray of shape (num_points + 1, 7)  — includes base as index 0.
        """
        rng = np.random.default_rng(seed)
        base = np.array(base_joints, dtype=float)
        offsets = rng.uniform(-amplitude, amplitude, size=(num_points, len(base)))
        waypoints = np.vstack([base[np.newaxis, :], base + offsets])
        return waypoints

    # ── auto mode ──

    def auto_run(
        self,
        num_points: int = 20,
        amplitude: float = 0.15,
        settle_time: float = 1.5,
        seed: int | None = None,
    ):
        """
        Automatic calibration: generate random joint perturbations, move to each,
        and capture paired ee + quest snapshots.

        Args:
            num_points:  Number of calibration waypoints (excluding the home pose).
            amplitude:   Max joint perturbation per axis (radians, default ≈ 8.6°).
            settle_time: Seconds to wait after reaching each waypoint before snapshot.
            seed:        Optional RNG seed for reproducibility.
        """
        if not self.quest.start():
            print("[ERROR] Cannot start Quest receiver; aborting.")
            return

        self.controller.execute()
        self._running = True

        def _shutdown(sig=None, frame=None):
            print("\n[INFO] Auto mode interrupted.")
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)

        traj_l = self._generate_trajectory(self.PRESET_JOINT_L, num_points, amplitude, seed)
        traj_r = self._generate_trajectory(self.PRESET_JOINT_R, num_points, amplitude,
                                           seed + 1 if seed is not None else None)

        js = self.controller.get_robot_joints()
        gripper_l = js["left_gripper"]
        gripper_r = js["right_gripper"]

        print("\n" + "=" * 60)
        print(f"  Auto Calibration — {num_points} waypoints, amplitude={amplitude:.3f} rad")
        print("=" * 60)

        try:
            # Step 0: go home
            print("\n[0/{n}] Moving to home position...".format(n=num_points))
            self.controller.set_target_JP(
                list(traj_l[0]), list(traj_r[0]), gripper_l, gripper_r,
            )
            self.controller.execute()
            time.sleep(settle_time)

            # Wait until Quest stream is live
            print("[INFO] Waiting for Quest pose stream...")
            for _ in range(30):
                if self.quest.get_latest() is not None:
                    break
                time.sleep(0.5)
            if self.quest.get_latest() is None:
                print("[ERROR] Quest pose not available after 15 s; aborting.")
                return

            # Step 1..N: visit each waypoint
            for i in range(1, num_points + 1):
                if not self._running:
                    print("[INFO] Interrupted, stopping early.")
                    break

                print(f"\n[{i}/{num_points}] Moving to waypoint...")
                print(f"  L: {np.round(traj_l[i], 4)}")
                print(f"  R: {np.round(traj_r[i], 4)}")

                self.controller.set_target_JP(
                    list(traj_l[i]), list(traj_r[i]), gripper_l, gripper_r,
                )
                self.controller.execute()
                time.sleep(settle_time)

                self._take_snapshot()

            print(f"\n[DONE] Collected {len(self.ee_snapshots)} snapshots.")

        finally:
            self._running = False
            self.quest.stop()
            self._save_and_calibrate()
            self.controller.robot._robot.shutdown()
            print("[INFO] Shutdown complete.")

    # ── manual mode ──

    def run(self):
        if not self.quest.start():
            print("[ERROR] Cannot start Quest receiver; aborting.")
            return

        self.controller.execute()
        self._running = True

        arm, joint, direction = "l", 0, "w"
        gripper_l, gripper_r = [0.0], [0.0]

        def _shutdown(sig=None, frame=None):
            print("\n[INFO] Shutting down...")
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)

        print("\n" + "=" * 60)
        print("  Hand-Eye Calibration Data Collector")
        print("  Commands: l/r  0-6  w/s  save  b  q")
        print("=" * 60 + "\n")

        try:
            while self._running:
                js = self.controller.get_robot_joints()
                position_l = js["left_arm"]
                position_r = js["right_arm"]
                gripper_l = js["left_gripper"]
                gripper_r = js["right_gripper"]

                print(f"\n--- joints (L): {np.round(position_l, 4)}")
                print(f"--- joints (R): {np.round(position_r, 4)}")

                quest_latest = self.quest.get_latest()
                quest_status = "connected" if quest_latest else "waiting..."
                print(f"--- quest: {quest_status}  |  snapshots: {len(self.ee_snapshots)}")

                cmd = input(
                    f"[{arm}/{joint}/{'↑' if direction == 'w' else '↓'}] > "
                ).strip().lower()

                if cmd == "q":
                    break
                elif cmd in ("l", "r"):
                    arm = cmd
                elif cmd in ("w", "s"):
                    direction = cmd
                elif cmd in [str(i) for i in range(7)]:
                    joint = int(cmd)
                elif cmd == "save":
                    self._take_snapshot()
                elif cmd == "b":
                    self.controller.set_target_JP(
                        list(self.PRESET_JOINT_L), list(self.PRESET_JOINT_R),
                        gripper_l, gripper_r,
                    )
                    self.controller.execute()
                    time.sleep(0.3)
                    continue
                elif cmd == "":
                    pass
                else:
                    print("[INFO] Unknown command. Use: l/r  0-6  w/s  save  b  q")
                    continue

                if cmd in ("w", "s", ""):
                    step = self.move_step if direction == "w" else -self.move_step
                    if arm == "l":
                        position_l[joint] += step
                    else:
                        position_r[joint] += step
                    self.controller.set_target_JP(position_l, position_r, gripper_l, gripper_r)

                self.controller.execute()
                time.sleep(0.3)

        finally:
            self._running = False
            self.quest.stop()
            self._save_and_calibrate()
            self.controller.robot._robot.shutdown()
            print("[INFO] Shutdown complete.")

# ─────────────────────────────── entry point ────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect ee-pose + quest-pose for hand-eye calibration")
    parser.add_argument("--save-dir", type=str,
                        default="/home/rvsa/codehub/VB-VLA/tools/cali_hand_eye")
    parser.add_argument("--quest-host", type=str, default="localhost")
    parser.add_argument("--quest-port", type=int, default=7777)
    parser.add_argument("--vel-max", type=float, default=0.5)
    parser.add_argument("--move-step", type=float, default=0.02)

    parser.add_argument("--auto", type=bool, default=True,
                        help="Run automatic calibration instead of manual mode")
    parser.add_argument("--num-points", type=int, default=50,
                        help="[auto] Number of calibration waypoints")
    parser.add_argument("--amplitude", type=float, default=0.15,
                        help="[auto] Max joint perturbation in radians (default ≈ 8.6°)")
    parser.add_argument("--settle-time", type=float, default=3,
                        help="[auto] Seconds to wait at each waypoint before snapshot")
    parser.add_argument("--seed", type=int, default=None,
                        help="[auto] RNG seed for reproducibility")
    args = parser.parse_args()

    collector = CaliDataCollector(
        save_dir=args.save_dir,
        quest_host=args.quest_host,
        quest_port=args.quest_port,
        vel_max=args.vel_max,
        move_step=args.move_step,
    )

    if args.auto:
        collector.auto_run(
            num_points=args.num_points,
            amplitude=args.amplitude,
            settle_time=args.settle_time,
            seed=args.seed,
        )
    else:
        collector.run()


if __name__ == "__main__":
    main()
