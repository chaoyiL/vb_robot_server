#!/usr/bin/env python3
"""
可视化 LeRobot 数据集中 actions 各维度的变化折线图。

通过逐帧累积相对动作还原绝对动作链（第一帧视作单位矩阵），
并将 [tx, ty, tz, r1(3), r2(3)] 转换为 [x, y, z, rx, ry, rz] (轴角) 表示。

用法:
    python ./tools/draw_action_plot/draw_action_plot.py --repo_id chaoyi/raw_0118_data
    python ./tools/draw_action_plot/draw_action_plot.py --repo_id chaoyi/0118_data_single_smooth --episodes 0 1 2
    python ./tools/draw_action_plot/draw_action_plot.py --repo_id chaoyi/0118_data_single_smooth --episodes 0 --dims_per_robot 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "plots"

LEROBOT_HOME = Path(os.environ.get(
    "LEROBOT_HOME",
    Path.home() / ".cache" / "huggingface" / "lerobot",
))

POSE_LABELS = ["x", "y", "z", "rx", "ry", "rz"]


# ============================================================================
# 旋转矩阵重建 & 位姿累积
# ============================================================================

def _normalize(v, eps=1e-12):
    return v / max(np.linalg.norm(v), eps)


def action9d_to_mat(action_9d: np.ndarray) -> np.ndarray:
    """
    将 9D 相对动作 [tx, ty, tz, r1(3), r2(3)] 还原为 4×4 齐次变换矩阵。
    r1, r2 为旋转矩阵的前两列，经 Gram-Schmidt 正交化后补出第三列。
    """
    pos = action_9d[:3]
    r1_raw, r2_raw = action_9d[3:6], action_9d[6:9]

    b1 = _normalize(r1_raw)
    b2 = r2_raw - np.dot(b1, r2_raw) * b1
    b2 = _normalize(b2)
    b3 = np.cross(b1, b2)

    mat = np.eye(4)
    mat[:3, 0] = b1
    mat[:3, 1] = b2
    mat[:3, 2] = b3
    mat[:3, 3] = pos
    return mat


def mat_to_pose6d(mat: np.ndarray) -> np.ndarray:
    """4×4 齐次矩阵 → [x, y, z, rx, ry, rz] (轴角)"""
    pos = mat[:3, 3]
    rotvec = Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    return np.concatenate([pos, rotvec])


def accumulate_absolute_poses(actions: np.ndarray, dims_per_robot: int = 10):
    """
    逐帧累积相对动作，还原每个机器人的绝对位姿链。

    Args:
        actions: (T, action_dim) 相对动作序列
        dims_per_robot: 每个机器人的动作维度 (9 pose + 1 gripper = 10)

    Returns:
        dict[int, np.ndarray]: robot_idx → (T+1, 6) 绝对位姿 [x,y,z,rx,ry,rz]
        dict[int, np.ndarray]: robot_idx → (T, 1) 夹爪宽度
    """
    T, action_dim = actions.shape
    num_robots = action_dim // dims_per_robot
    gripper_dim = dims_per_robot - 9

    all_poses = {}
    all_grippers = {}

    for robot_idx in range(num_robots):
        offset = robot_idx * dims_per_robot
        cur_mat = np.eye(4)
        poses = [mat_to_pose6d(cur_mat)]
        grippers = []

        for t in range(T):
            act_9 = actions[t, offset:offset + 9]
            gripper = actions[t, offset + 9:offset + 9 + gripper_dim]
            grippers.append(gripper)

            if np.allclose(act_9, 0):
                poses.append(poses[-1].copy())
                continue

            delta = action9d_to_mat(act_9)
            cur_mat = cur_mat @ delta
            poses.append(mat_to_pose6d(cur_mat))

        all_poses[robot_idx] = np.array(poses)       # (T+1, 6)
        all_grippers[robot_idx] = np.array(grippers)  # (T, gripper_dim)

    return all_poses, all_grippers


# ============================================================================
# 数据加载
# ============================================================================

def load_dataset_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)


def load_episode_actions(dataset_root: Path, episode_idx: int, chunks_size: int = 1000) -> np.ndarray:
    chunk = episode_idx // chunks_size
    parquet_path = dataset_root / f"data/chunk-{chunk:03d}/episode_{episode_idx:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet 文件不存在: {parquet_path}")
    df = pd.read_parquet(parquet_path, columns=["actions"])
    return np.stack(df["actions"].values)


# ============================================================================
# 绘图
# ============================================================================

def plot_episode(
    episode_idx: int,
    poses: dict,
    grippers: dict,
    save_dir: Path,
    fps: int = 30,
    repo_id: str = None,
):
    """
    为一个 episode 绘制绝对位姿各维度的折线图，
    每个机器人的 6 个位姿维度 + 夹爪 合成一张大图。
    """
    num_robots = len(poses)

    for robot_idx in range(num_robots):
        pose_data = poses[robot_idx]            # (T+1, 6)
        grip_data = grippers[robot_idx]          # (T, gripper_dim)
        n_frames = pose_data.shape[0]
        gripper_dim = grip_data.shape[1] if grip_data.ndim == 2 else 1

        n_dims = 6 + gripper_dim
        fig, axes = plt.subplots(n_dims, 1, figsize=(14, 2.5 * n_dims), sharex=True)
        if n_dims == 1:
            axes = [axes]

        time_axis = np.arange(n_frames) / fps
        grip_time = np.arange(grip_data.shape[0]) / fps

        for d in range(6):
            ax = axes[d]
            ax.plot(time_axis, pose_data[:, d], linewidth=0.8)
            ax.set_ylabel(POSE_LABELS[d], fontsize=12)
            ax.grid(True, alpha=0.3)

        for g in range(gripper_dim):
            ax = axes[6 + g]
            gdata = grip_data[:, g] if grip_data.ndim == 2 else grip_data
            ax.plot(grip_time, gdata, linewidth=0.8, color="tab:green")
            ax.set_ylabel(f"gripper{'_' + str(g) if gripper_dim > 1 else ''}", fontsize=12)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)", fontsize=12)

        robot_tag = f"_robot{robot_idx}" if num_robots > 1 else ""
        title = f"Episode {episode_idx}{robot_tag}  —  Accumulated Absolute Pose (axis-angle)"
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        fname = save_dir / f"{repo_id}_episode_{episode_idx:04d}{robot_tag}_absolute_pose.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  已保存: {fname.name}")


def plot_raw_actions(
    episode_idx: int,
    actions: np.ndarray,
    dims_per_robot: int,
    save_dir: Path,
    fps: int = 30,
    repo_id: str = None,
):
    """绘制原始相对动作各维度的折线图。"""
    T, action_dim = actions.shape
    num_robots = action_dim // dims_per_robot
    raw_labels = ["tx", "ty", "tz", "r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z", "gripper"]

    for robot_idx in range(num_robots):
        offset = robot_idx * dims_per_robot
        robot_actions = actions[:, offset:offset + dims_per_robot]
        n_dims = min(dims_per_robot, len(raw_labels))

        fig, axes = plt.subplots(n_dims, 1, figsize=(14, 2.5 * n_dims), sharex=True)
        if n_dims == 1:
            axes = [axes]

        time_axis = np.arange(T) / fps
        for d in range(n_dims):
            ax = axes[d]
            ax.plot(time_axis, robot_actions[:, d], linewidth=0.8,
                    color="tab:orange" if d < 3 else ("tab:blue" if d < 9 else "tab:green"))
            label = raw_labels[d] if d < len(raw_labels) else f"dim_{d}"
            ax.set_ylabel(label, fontsize=12)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)", fontsize=12)

        robot_tag = f"_robot{robot_idx}" if num_robots > 1 else ""
        title = f"Episode {episode_idx}{robot_tag}  —  Raw Relative Actions"
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        fname = save_dir / f"{repo_id}_episode_{episode_idx:04d}{robot_tag}_raw_actions.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  已保存: {fname.name}")


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="可视化 LeRobot 数据集 actions 的各维度变化折线图"
    )
    parser.add_argument("--repo_id", type=str, required=True,
                        help="LeRobot 数据集 repo_id，如 chaoyi/0118_data_single_smooth")
    parser.add_argument("--root", type=str, default=None,
                        help="LeRobot 数据集根目录（默认 $LEROBOT_HOME）")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="要可视化的 episode 索引（默认: 前 1 个）")
    parser.add_argument("--dims_per_robot", type=int, default=10,
                        help="每个机器人的动作维度（默认 10 = 9 pose + 1 gripper）")
    parser.add_argument("--plot_raw", action="store_true",
                        help="同时绘制原始相对动作图")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认: tools/draw_action_plot/plots）")
    args = parser.parse_args()

    root = Path(args.root) if args.root else LEROBOT_HOME
    dataset_root = root / args.repo_id
    repo_id = args.repo_id.split("/")[-1]
    if not dataset_root.exists():
        print(f"错误: 数据集目录不存在: {dataset_root}")
        sys.exit(1)

    info = load_dataset_info(dataset_root)
    total_episodes = info["total_episodes"]
    chunks_size = info.get("chunks_size", 1000)
    fps = info.get("fps", 30)
    action_shape = info["features"]["actions"]["shape"]
    action_dim = action_shape[0]

    print(f"数据集: {args.repo_id}")
    print(f"  路径: {dataset_root}")
    print(f"  总 episodes: {total_episodes}, FPS: {fps}, action_dim: {action_dim}")

    episodes = args.episodes if args.episodes is not None else list(range(min(1, total_episodes)))
    save_dir = Path(args.output_dir) if args.output_dir else PLOTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"  将可视化 episodes: {episodes}")
    print(f"  输出目录: {save_dir}\n")

    for ep_idx in episodes:
        if ep_idx >= total_episodes:
            print(f"  跳过 episode {ep_idx}（超出范围 [0, {total_episodes - 1}]）")
            continue

        print(f"处理 Episode {ep_idx} ...")
        actions = load_episode_actions(dataset_root, ep_idx, chunks_size)
        print(f"  帧数: {actions.shape[0]}, 动作维度: {actions.shape[1]}")

        poses, grippers = accumulate_absolute_poses(actions, args.dims_per_robot)
        plot_episode(ep_idx, poses, grippers, save_dir, fps, repo_id)

        if args.plot_raw:
            plot_raw_actions(ep_idx, actions, args.dims_per_robot, save_dir, fps, repo_id)

    print(f"\n完成！所有图片已保存到: {save_dir}")


if __name__ == "__main__":
    main()
