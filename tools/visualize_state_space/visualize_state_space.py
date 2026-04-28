#!/usr/bin/env python3
"""
在一个 3D 空间中可视化 LeRobot 数据集中 state 的整体运动轨迹。

思路：
- 直接从本地 LeRobot parquet 数据中读取 `observation.state` 列（避免依赖远程 HF）。
- 汇总所有 episode 的 state 向量（可限制最大帧数 / 步长抽样）。
- 对高维 state 做 PCA 降到 3 维，在一个 3D 空间中画散点图。

用法示例（在仓库根目录）:

    python tools/visualize_state_space/visualize_state_space.py \\
        --repo_id chaoyi/0118_data_single_smooth

    # 限制最多采样 100k 帧，每隔 2 帧取一个
    python tools/visualize_state_space/visualize_state_space.py \\
        --repo_id chaoyi/0118_data_single_smooth \\
        --max_frames 100000 --stride 2

    # 仅展示每个 episode 的起始（绿点）与结束（红点）位置
    python tools/visualize_state_space/visualize_state_space.py \\
        --repo_id chaoyi/0118_data_single_smooth --start_end_only

    # 自定义数据根目录 & 输出目录
    python tools/visualize_state_space/visualize_state_space.py \\
        --repo_id chaoyi/0118_data_single_smooth \\
        --root /your/lerobot/root \\
        --output_dir /tmp/state_space_plots
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "plots"

LEROBOT_HOME = Path(
    os.environ.get(
        "LEROBOT_HOME",
        Path.home() / ".cache" / "huggingface" / "lerobot",
    )
)


def load_dataset_root(repo_id: str, root: str | None) -> Path:
    """根据 repo_id 和 root 找到本地数据集根目录。"""
    base = Path(root) if root is not None else LEROBOT_HOME
    dataset_root = base / repo_id
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")
    return dataset_root


def load_dataset_info(dataset_root: Path) -> dict | None:
    """读取 meta/info.json（如果存在，仅用于标题等信息）。"""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        with open(info_path) as f:
            return json.load(f)
    except Exception:
        return None


def iter_state_vectors(
    dataset_root: Path,
    *,
    max_frames: int | None = None,
    stride: int = 1,
) -> np.ndarray:
    """
    从所有 episode 的 parquet 文件中收集 `observation.state` 向量。

    返回：
        states: np.ndarray，形状 (N, D)，N 为采样帧数，D 为 state 维度。
    """
    data_root = dataset_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"未找到 data 目录: {data_root}")

    all_states: list[np.ndarray] = []
    total = 0

    # 遍历 chunk-* 目录
    chunk_dirs = sorted(p for p in data_root.glob("chunk-*") if p.is_dir())
    if not chunk_dirs:
        raise FileNotFoundError(f"在 {data_root} 下未找到任何 chunk-* 目录")

    for chunk_dir in chunk_dirs:
        episode_files = sorted(chunk_dir.glob("episode_*.parquet"))
        for ep_path in episode_files:
            # 只读取 `observation.state` 列，避免多余 I/O
            try:
                df = pd.read_parquet(ep_path, columns=["observation.state"])
            except Exception as e:  # pragma: no cover - 防御性
                print(f"[WARN] 读取 {ep_path} 失败: {e}")
                continue

            if "observation.state" not in df.columns:
                print(f"[WARN] 文件 {ep_path} 中不含 'observation.state' 列，跳过。")
                continue

            # parquet 中每一行的 `observation.state` 是一个 numpy 数组
            states = np.stack(df["observation.state"].values)  # (T, D)

            if stride > 1:
                states = states[::stride]

            if max_frames is not None:
                remain = max_frames - total
                if remain <= 0:
                    # 已经够了，直接返回
                    if not all_states:
                        raise RuntimeError("max_frames 太小，导致没有任何样本被收集。")
                    return np.concatenate(all_states, axis=0)
                if states.shape[0] > remain:
                    states = states[:remain]

            if states.size == 0:
                continue

            all_states.append(states)
            total += states.shape[0]

            if max_frames is not None and total >= max_frames:
                return np.concatenate(all_states, axis=0)

    if not all_states:
        raise RuntimeError("没有从任何 episode 中成功收集到 state 数据。")

    return np.concatenate(all_states, axis=0)


def iter_start_end_states(dataset_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    从所有 episode 中只收集每条的「第一帧」和「最后一帧」的 state。

    返回：
        starts: (N_episodes, D)，每个 episode 的起始 state
        ends:   (N_episodes, D)，每个 episode 的结束 state
    """
    data_root = dataset_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"未找到 data 目录: {data_root}")

    chunk_dirs = sorted(p for p in data_root.glob("chunk-*") if p.is_dir())
    if not chunk_dirs:
        raise FileNotFoundError(f"在 {data_root} 下未找到任何 chunk-* 目录")

    start_list: list[np.ndarray] = []
    end_list: list[np.ndarray] = []

    for chunk_dir in chunk_dirs:
        episode_files = sorted(chunk_dir.glob("episode_*.parquet"))
        for ep_path in episode_files:
            try:
                df = pd.read_parquet(ep_path, columns=["observation.state"])
            except Exception as e:
                print(f"[WARN] 读取 {ep_path} 失败: {e}")
                continue
            if "observation.state" not in df.columns:
                continue
            states = np.stack(df["observation.state"].values)  # (T, D)
            if states.shape[0] < 2:
                continue
            start_list.append(states[0:1])
            end_list.append(states[-1:])

    if not start_list:
        raise RuntimeError("没有从任何 episode 中收集到首尾 state。")
    return np.concatenate(start_list, axis=0), np.concatenate(end_list, axis=0)


def pca_to_3d(states: np.ndarray) -> np.ndarray:
    """
    将高维 state 通过 PCA 降到 3 维。

    如果原始维度 <= 3，则直接截取前三维。
    """
    n_samples, dim = states.shape
    if dim <= 3:
        # 直接用前 3 维（不足 3 维时后面补 0）
        coords = np.zeros((n_samples, 3), dtype=np.float32)
        coords[:, :dim] = states
        return coords

    # 中心化
    mean = states.mean(axis=0, keepdims=True)
    X = states - mean

    # SVD 实现 PCA：X = U S V^T，前 3 个主成分在 V^T[:3, :]
    # 为了数值稳定性，使用 full_matrices=False
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    components = vt[:3].T  # (D, 3)

    coords = X @ components  # (N, 3)
    return coords.astype(np.float32)


def plot_state_space(
    coords: np.ndarray,
    output_path: Path,
    *,
    title: str | None = None,
):
    """在 3D 空间中绘制所有 state 的散点图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=1,
        alpha=0.4,
        cmap="viridis",
    )

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_zlabel("PC3", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] 状态空间可视化已保存到: {output_path}")


def plot_start_end_only(
    start_coords: np.ndarray,
    end_coords: np.ndarray,
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    """在 3D 空间中仅绘制每个 episode 的起始（绿点）与结束（红点）位置。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        start_coords[:, 0],
        start_coords[:, 1],
        start_coords[:, 2],
        s=40,
        c="green",
        alpha=0.8,
        label="起始",
        edgecolors="darkgreen",
    )
    ax.scatter(
        end_coords[:, 0],
        end_coords[:, 1],
        end_coords[:, 2],
        s=40,
        c="red",
        alpha=0.8,
        label="结束",
        edgecolors="darkred",
    )

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_zlabel("PC3", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    if title:
        ax.set_title(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] 首尾位置可视化已保存到: {output_path}")


def plot_state_xyz_trajectories(
    dataset_root: Path,
    output_path: Path,
    *,
    stride: int = 1,
    left_indices: tuple[int, int, int] = (0, 1, 2),
    right_indices: tuple[int, int, int] = (7, 8, 9),
) -> None:
    """
    使用 state 的原始维度绘制左右手 3D 轨迹：
    - 左手使用 state[0,1,2] 作为 (x,y,z)，用绿色轨迹
    - 右手使用 state[7,8,9] 作为 (x,y,z)，用红色轨迹
    """
    data_root = dataset_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"未找到 data 目录: {data_root}")

    chunk_dirs = sorted(p for p in data_root.glob("chunk-*") if p.is_dir())
    if not chunk_dirs:
        raise FileNotFoundError(f"在 {data_root} 下未找到任何 chunk-* 目录")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    num_episodes = 0

    for chunk_dir in chunk_dirs:
        episode_files = sorted(chunk_dir.glob("episode_*.parquet"))
        for ep_path in episode_files:
            try:
                df = pd.read_parquet(ep_path, columns=["observation.state"])
            except Exception as e:
                print(f"[WARN] 读取 {ep_path} 失败: {e}")
                continue
            if "observation.state" not in df.columns:
                continue

            states = np.stack(df["observation.state"].values)  # (T, D)
            if states.shape[0] < 2:
                continue

            if stride > 1:
                states = states[::stride]

            # 维度检查，避免索引越界
            dim = states.shape[1]
            if max(left_indices + right_indices) >= dim:
                print(
                    f"[WARN] episode {ep_path} 的 state 维度为 {dim}，"
                    f"不足以索引左右手 xyz，已跳过。"
                )
                continue

            left_xyz = states[:, list(left_indices)]   # (T, 3)
            right_xyz = states[:, list(right_indices)]  # (T, 3)

            if np.linalg.norm(left_xyz[0])>0.01 or np.linalg.norm(right_xyz[0])>0.01:
                print(f"[WARN] episode {ep_path} 的左右手起点不一致，已跳过。")
                print(f"left_xyz[0]: {left_xyz[0]}")
                print(f"right_xyz[0]: {right_xyz[0]}")
                continue

            ax.plot(
                left_xyz[:, 0],
                left_xyz[:, 1],
                left_xyz[:, 2],
                color="green",
                alpha=0.4,
                linewidth=1.0,
            )
            ax.plot(
                right_xyz[:, 0],
                right_xyz[:, 1],
                right_xyz[:, 2],
                color="red",
                alpha=0.4,
                linewidth=1.0,
            )

            num_episodes += 1

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_zlabel("z", fontsize=12)
    ax.set_title(
        f"State Trajectories (xyz) — left: green, right: red\nepisodes={num_episodes}",
        fontsize=13,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] xyz 轨迹可视化已保存到: {output_path}")


def plot_state_xyz_start_end_only(
    dataset_root: Path,
    output_path: Path,
    *,
    stride: int = 1,
    left_indices: tuple[int, int, int] = (0, 1, 2),
    right_indices: tuple[int, int, int] = (7, 8, 9),
) -> None:
    """
    xyz 轨迹模式下，仅展示每条 episode 左右手的起点和终点：
    - 左手：state[0,1,2]，绿色点
    - 右手：state[7,8,9]，红色点
    """
    data_root = dataset_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"未找到 data 目录: {data_root}")

    chunk_dirs = sorted(p for p in data_root.glob("chunk-*") if p.is_dir())
    if not chunk_dirs:
        raise FileNotFoundError(f"在 {data_root} 下未找到任何 chunk-* 目录")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    left_start_points = []
    left_end_points = []
    right_start_points = []
    right_end_points = []
    num_episodes = 0

    for chunk_dir in chunk_dirs:
        episode_files = sorted(chunk_dir.glob("episode_*.parquet"))
        for ep_path in episode_files:
            try:
                df = pd.read_parquet(ep_path, columns=["observation.state"])
            except Exception as e:
                print(f"[WARN] 读取 {ep_path} 失败: {e}")
                continue
            if "observation.state" not in df.columns:
                continue

            states = np.stack(df["observation.state"].values)  # (T, D)
            if states.shape[0] < 2:
                continue

            if stride > 1:
                states = states[::stride]

            dim = states.shape[1]
            if max(left_indices + right_indices) >= dim:
                print(
                    f"[WARN] episode {ep_path} 的 state 维度为 {dim}，"
                    f"不足以索引左右手 xyz，已跳过。"
                )
                continue

            left_xyz = states[:, list(left_indices)]
            right_xyz = states[:, list(right_indices)]

            # 起点和终点
            left_start_points.append(left_xyz[0])
            left_end_points.append(left_xyz[-1])
            right_start_points.append(right_xyz[0])
            right_end_points.append(right_xyz[-1])

            num_episodes += 1

    if left_start_points:
        left_start_points_arr = np.stack(left_start_points)
        ax.scatter(
            left_start_points_arr[:, 0],
            left_start_points_arr[:, 1],
            left_start_points_arr[:, 2],
            c="white",
            s=35,
            alpha=0.8,
            label="left hand (start/end)",
            edgecolors="darkgreen",
        )
    if left_end_points:
        left_end_points_arr = np.stack(left_end_points)
        ax.scatter(
            left_end_points_arr[:, 0],
            left_end_points_arr[:, 1],
            left_end_points_arr[:, 2],
            c="darkgreen",
            s=35,
            alpha=0.8, 
            label="left hand (start/end)",
            edgecolors="darkgreen",
        )
    if right_start_points:
        right_start_points_arr = np.stack(right_start_points)
        ax.scatter(
            right_start_points_arr[:, 0],
            right_start_points_arr[:, 1],
            right_start_points_arr[:, 2],
            c="white",
            s=35,
            alpha=0.8,
            label="right hand (start/end)",
            edgecolors="darkred",
        )
    if right_end_points:
        right_end_points_arr = np.stack(right_end_points)
        ax.scatter(
            right_end_points_arr[:, 0],
            right_end_points_arr[:, 1],
            right_end_points_arr[:, 2],
            c="darkred",
            s=35,
            alpha=0.8,
            label="right hand (start/end)",
            edgecolors="darkred",
        )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_zlabel("z", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(
        f"State xyz start/end — left: green, right: red\nepisodes={num_episodes}",
        fontsize=13,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] xyz 首尾位置可视化已保存到: {output_path}")


def plot_relative_xyz_trajectories(
    dataset_root: Path,
    output_path: Path,
    *,
    stride: int = 1,
    rel_indices: tuple[int, int, int] = (14, 15, 16),
) -> None:
    """
    可视化「每一帧左右手的相对位置」在 3D 空间中的轨迹。

    假设 state 的第 14,15,16 维（即索引 14,15,16）存的是
    「右手相对左手」的位移向量 (dx, dy, dz)，我们将其看成从
    原点 (0,0,0) 指向 (dx,dy,dz) 的点轨迹，并在 3D 空间中画出
    每个 episode 的整条轨迹。
    """
    data_root = dataset_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"未找到 data 目录: {data_root}")

    chunk_dirs = sorted(p for p in data_root.glob("chunk-*") if p.is_dir())
    if not chunk_dirs:
        raise FileNotFoundError(f"在 {data_root} 下未找到任何 chunk-* 目录")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    num_episodes = 0

    for chunk_dir in chunk_dirs:
        episode_files = sorted(chunk_dir.glob("episode_*.parquet"))
        for ep_path in episode_files:
            try:
                df = pd.read_parquet(ep_path, columns=["observation.state"])
            except Exception as e:
                print(f"[WARN] 读取 {ep_path} 失败: {e}")
                continue
            if "observation.state" not in df.columns:
                continue

            states = np.stack(df["observation.state"].values)  # (T, D)
            if states.shape[0] < 2:
                continue

            if stride > 1:
                states = states[::stride]

            dim = states.shape[1]
            if max(rel_indices) >= dim:
                print(
                    f"[WARN] episode {ep_path} 的 state 维度为 {dim}，"
                    f"不足以索引相对位置 xyz (14,15,16)，已跳过。"
                )
                continue

            rel_xyz = states[:, list(rel_indices)]  # (T, 3)

            ax.plot(
                rel_xyz[:, 0],
                rel_xyz[:, 1],
                rel_xyz[:, 2],
                color="purple",
                alpha=0.4,
                linewidth=1.0,
            )

            num_episodes += 1

    ax.set_xlabel("Δx (right - left)", fontsize=12)
    ax.set_ylabel("Δy (right - left)", fontsize=12)
    ax.set_zlabel("Δz (right - left)", fontsize=12)
    ax.set_title(
        f"Relative position (state[14,15,16]) — episodes={num_episodes}",
        fontsize=13,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] 左右手相对位置 xyz 轨迹已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="在一个 3D 空间中可视化 LeRobot 数据集 state 的整体运动轨迹（PCA 降维）。"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        # required=True,
        # default="0118_data",
        default="chaoyi/0118_data_smooth",
        help="LeRobot 数据集 repo_id，如 chaoyi/0118_data_single_smooth",
    )
    parser.add_argument(
        "--root",
        type=str,
        # default="/home/rvsa/codehub/VB-VLA/data/",
        default=None,
        help="LeRobot 数据集根目录",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100000,
        help="最多采样多少帧进行可视化（默认 100000，设置为 0 或负数表示不限制）",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="帧采样步长，例如 2 表示每隔一帧取一帧（默认 1）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认 tools/visualize_state_space/plots）",
    )
    parser.add_argument(
        "--state_xyz_traj",
        type=bool,
        default=True,
        help="使用 state[0,1,2] 和 state[7,8,9] 绘制左右手 3D 轨迹（左=绿，右=红）",
    )
    parser.add_argument(
        "--start_end_only",
        type=bool,
        default=False,
        help="仅展示每个 episode 的首尾位置（PCA 或 xyz 模式下生效）",
    )
    parser.add_argument(
        "--relative_xyz",
        type=bool,
        default=True,
        help="使用 state[14,15,16] 可视化左右手相对位置轨迹（右手相对左手）",
    )

    args = parser.parse_args()

    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None
    stride = max(args.stride, 1)

    dataset_root = load_dataset_root(args.repo_id, args.root)
    info = load_dataset_info(dataset_root)

    print(f"[INFO] 数据集根目录: {dataset_root}")
    if info is not None:
        print(f"[INFO] info.json: {json.dumps(info, indent=2, ensure_ascii=False)[:400]}...")

    output_dir = Path(args.output_dir) if args.output_dir else PLOTS_DIR
    repo_tag = args.repo_id.replace("/", "_")

    if args.relative_xyz:
        print("[INFO] 使用 state[14,15,16] 绘制左右手相对位置 3D 轨迹...")
        output_path = output_dir / f"{repo_tag}_relative_xyz_trajectories.png"
        plot_relative_xyz_trajectories(
            dataset_root,
            output_path,
            stride=stride,
        )
    elif args.state_xyz_traj:
        if args.start_end_only:
            print("[INFO] xyz 模式下，仅绘制左右手首尾位置...")
            output_path = output_dir / f"{repo_tag}_state_xyz_start_end.png"
            plot_state_xyz_start_end_only(
                dataset_root,
                output_path,
                stride=stride,
            )
        else:
            print("[INFO] 使用 state[0,1,2] (左手) 与 state[7,8,9] (右手) 绘制 3D 轨迹...")
            output_path = output_dir / f"{repo_tag}_state_xyz_trajectories.png"
            plot_state_xyz_trajectories(
                dataset_root,
                output_path,
                stride=stride,
            )
    elif args.start_end_only:
        print("[INFO] 仅收集每个 episode 的首尾 state（PCA 模式）...")
        start_states, end_states = iter_start_end_states(dataset_root)
        n_ep = start_states.shape[0]
        print(f"[INFO] 收集到 {n_ep} 个 episode，state 维度: {start_states.shape[1]}")
        # 在同一 PCA 空间中投影首尾
        all_states = np.concatenate([start_states, end_states], axis=0)
        all_coords = pca_to_3d(all_states)
        start_coords = all_coords[:n_ep]
        end_coords = all_coords[n_ep:]
        output_path = output_dir / f"{repo_tag}_state_space_start_end.png"
        title = f"State Space start and end (PCA) — {args.repo_id}  (episodes={n_ep})"
        print("[INFO] 绘制首尾位置（绿=起始，红=结束）...")
        plot_start_end_only(start_coords, end_coords, output_path, title=title)
    else:
        print("[INFO] 开始收集 state 向量...")
        states = iter_state_vectors(
            dataset_root,
            max_frames=max_frames,
            stride=stride,
        )
        print(f"[INFO] 收集到 state 样本数: {states.shape[0]}, 维度: {states.shape[1]}")
        print("[INFO] 开始进行 PCA 降维到 3D...")
        coords = pca_to_3d(states)
        output_path = output_dir / f"{repo_tag}_state_space_3d.png"
        title = f"State Space (PCA) — {args.repo_id}"
        if max_frames is not None:
            title += f"  (N={coords.shape[0]})"
        print("[INFO] 开始绘制 3D 散点图...")
        plot_state_space(coords, output_path, title=title)


if __name__ == "__main__":
    main()

