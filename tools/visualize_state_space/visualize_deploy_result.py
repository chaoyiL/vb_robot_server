#!/usr/bin/env python3
"""
在 3D 空间中对比：
- 指定 LeRobot 数据集中的相对位置轨迹（state 的第 14,15,16 维）
- 指定 deploy 结果目录中的相对位置轨迹

可用于对比「理想/训练数据中的相对位置分布」与「真实部署时机器人执行出的相对位置」。

当前实现对 deploy 结果做了一个约定：
- 假设在 eval 结果根目录下有一个 `relative_xyz.npy`（或 npz 中的 `relative_xyz` 数组），
  形状为 (T, 3)，每一行是 (dx, dy, dz) = state[14:17] 的相对位置。
如果你的真实格式不同，可以在 `load_deploy_relative_xyz` 中改读取逻辑。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

# 确保可以从仓库根目录导入 `utils` 等模块
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.pose_util import pose_to_mat, mat_to_pose  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import open3d as o3d  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - open3d 为可选依赖
    o3d = None


SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "plots"


def load_dataset_relative_xyz(
    dataset_root: Path,
    *,
    stride: int = 1,
    max_frames: int | None = None,
    rel_indices: tuple[int, int, int] = (14, 15, 16),
) -> np.ndarray:
    """
    从本地 LeRobot 数据集中读取 state 的第 14,15,16 维（相对位置）。

    返回：
        rel_xyz: np.ndarray, 形状 (N, 3)
    """
    data_root = dataset_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"未找到 data 目录: {data_root}")

    chunk_dirs = sorted(p for p in data_root.glob("chunk-*") if p.is_dir())
    if not chunk_dirs:
        raise FileNotFoundError(f"在 {data_root} 下未找到任何 chunk-* 目录")

    all_rel: list[np.ndarray] = []
    start_end_rel = []
    start_pose = []
    end_pose = []
    total = 0

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
            rel_pose = states[:, 14:20]  # (T, 6)

            if max_frames is not None:
                remain = max_frames - total
                if remain <= 0:
                    if not all_rel:
                        raise RuntimeError("max_frames 太小，导致没有任何样本被收集。")
                    return np.concatenate(all_rel, axis=0)
                if rel_xyz.shape[0] > remain:
                    rel_xyz = rel_xyz[:remain]

            all_rel.append(rel_xyz)
            start_end_rel.append(rel_xyz[0])
            start_end_rel.append(rel_xyz[-1])
            start_pose.append(rel_pose[0])
            end_pose.append(rel_pose[-1])

            total += rel_xyz.shape[0]

            if max_frames is not None and total >= max_frames:
                return np.concatenate(all_rel, axis=0), np.array(start_end_rel, dtype=np.float32), np.array(start_pose, dtype=np.float32), np.array(end_pose, dtype=np.float32)

    if not all_rel:
        raise RuntimeError("没有从任何 episode 中成功收集到相对位置数据。")

    return np.concatenate(all_rel, axis=0), np.array(start_end_rel, dtype=np.float32), np.array(start_pose, dtype=np.float32), np.array(end_pose, dtype=np.float32)


def load_deploy_relative_xyz(eval_root: Path) -> np.ndarray:
    """
    从 deploy/eval 结果目录中读取相对位置轨迹。

    目录结构约定：
        eval_obs_xxx/
            step_000000/
                robot0_eef_pos.json
                robot1_eef_pos.json
            step_000001/
                ...

    JSON 内容格式示例（与你提供的一致）：
        [
          [x, y, z]
        ]

    我们按 step 顺序遍历所有 step_xxx 目录：
    - 读取左手位置 left_pos = robot0_eef_pos.json[0]
    - 读取左手旋转 left_rot = robot0_eef_rot_axis_angle.json[0]
    - 读取右手位置 right_pos = robot1_eef_pos.json[0]
    - 读取右手旋转 right_rot = robot1_eef_rot_axis_angle.json[0]
    - 相对位置定义为 right - left，对应训练中 state[14,15,16] 的含义。
    """
    if not eval_root.exists():
        raise FileNotFoundError(f"eval 结果目录不存在: {eval_root}")

    step_dirs = sorted(p for p in eval_root.glob("step_*") if p.is_dir())
    if not step_dirs:
        raise FileNotFoundError(f"在 {eval_root} 下未找到任何 step_* 子目录。")

    rel_list: list[np.ndarray] = []

    for step_dir in step_dirs:
        left_pos_path = step_dir / "robot0_eef_pos.json"
        left_rot_path = step_dir / "robot0_eef_rot_axis_angle.json"
        right_pos_path = step_dir / "robot1_eef_pos.json"
        right_rot_path = step_dir / "robot1_eef_rot_axis_angle.json"

        apath = [left_pos_path, left_rot_path, right_pos_path, right_rot_path]
        if not all(path.exists() for path in apath):
            # 某些 step 可能缺文件，直接跳过
            continue

        try:
            with open(left_pos_path, "r") as f:
                left_pos_data = json.load(f)
            with open(left_rot_path, "r") as f:
                left_rot_data = json.load(f)
            with open(right_pos_path, "r") as f:
                right_pos_data = json.load(f)
            with open(right_rot_path, "r") as f:
                right_rot_data = json.load(f)
        except Exception as e:
            print(f"[WARN] 读取 {step_dir} 下的 eef pos JSON 失败: {e}")
            continue

        try:
            left_pos = np.array(left_pos_data[0], dtype=float)   # (3,)
            left_rot = np.array(left_rot_data[0], dtype=float)   # (3,)
            right_pos = np.array(right_pos_data[0], dtype=float)  # (3,)
            right_rot = np.array(right_rot_data[0], dtype=float)  # (3,)
            left = np.concatenate([left_pos, left_rot], axis=0)  # (6,)
            right = np.concatenate([right_pos, right_rot], axis=0)  # (6,)
        except Exception as e:
            print(f"[WARN] 解析 {step_dir} eef pos JSON 失败: {e}")
            continue

        if left.shape != (6,) or right.shape != (6,):
            print(
                f"[WARN] {step_dir} 中 eef pos & rot 形状异常："
                f"left={left.shape}, right={right.shape}，已跳过。"
            )
            continue

        rel = mat_to_pose(np.linalg.inv(pose_to_mat(right)) @ pose_to_mat(left))
        rel_list.append(rel[:3])  # (3,)

    if not rel_list:
        raise RuntimeError(f"未能从 {eval_root} 中收集到任何相对位置数据。")

    return np.array(rel_list, dtype=np.float32)


def plot_comparison(
    dataset_rel: np.ndarray,
    deploy_rel: np.ndarray,
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    """
    在同一张 3D 图中绘制：
    - 训练数据集中的相对位置轨迹（紫色、透明度较高）
    - deploy/eval 结果中的相对位置轨迹（橙色，线更粗）
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # 数据集整体轨迹（散点或者细线）
    ax.plot(
        dataset_rel[:, 0],
        dataset_rel[:, 1],
        dataset_rel[:, 2],
        color="purple",
        alpha=0.35,
        linewidth=1.0,
        label="dataset (state[14,15,16])",
    )

    # deploy 轨迹（粗线）
    ax.plot(
        deploy_rel[:, 0],
        deploy_rel[:, 1],
        deploy_rel[:, 2],
        color="darkorange",
        alpha=0.9,
        linewidth=2.0,
        label="deploy result",
    )

    ax.set_xlabel("Δx (right - left)", fontsize=12)
    ax.set_ylabel("Δy (right - left)", fontsize=12)
    ax.set_zlabel("Δz (right - left)", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)

    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] 对比可视化已保存到: {output_path}")


def render_comparison_open3d(
    dataset_rel: np.ndarray,
    deploy_rel: np.ndarray,
    *,
    start_end_only: bool = False,
) -> None:
    """
    使用 Open3D 在交互式 3D 窗口中渲染两条轨迹。

    - 数据集轨迹：紫色线
    - deploy 轨迹：橙色线
    """
    if o3d is None:
        raise ImportError(
            "未安装 open3d，请先执行 `pip install open3d` 再使用 3D 渲染功能。"
        )

    geometries: list[o3d.geometry.Geometry] = []

    if start_end_only:
        # 只画散点：数据集起点/终点都用紫色点，deploy 起点/终点用橙色点
        pcs = []
        if dataset_rel.shape[0] >= 1:
            ds_pcd = o3d.geometry.PointCloud()
            ds_pcd.points = o3d.utility.Vector3dVector(dataset_rel)
            ds_pcd.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[0.6, 0.2, 0.8]]), (dataset_rel.shape[0], 1))
            )
            pcs.append(ds_pcd)
        if deploy_rel.shape[0] >= 1:
            dep_pcd = o3d.geometry.PointCloud()
            dep_pcd.points = o3d.utility.Vector3dVector(deploy_rel)
            dep_pcd.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[1.0, 0.55, 0.0]]), (deploy_rel.shape[0], 1))
            )
            pcs.append(dep_pcd)
        if not pcs:
            raise RuntimeError("没有足够的点来渲染散点。")
        geometries.extend(pcs)
    else:
        # 默认仍然渲染完整轨迹折线
        def make_lineset(points: np.ndarray, color: tuple[float, float, float]):
            pts = o3d.utility.Vector3dVector(points)
            lines = [[i, i + 1] for i in range(points.shape[0] - 1)]
            line_set = o3d.geometry.LineSet(
                points=pts,
                lines=o3d.utility.Vector2iVector(lines),
            )
            colors = np.tile(np.array(color, dtype=float), (len(lines), 1))
            line_set.colors = o3d.utility.Vector3dVector(colors)
            return line_set

        if dataset_rel.shape[0] >= 2:
            ds_lines = make_lineset(dataset_rel, (0.6, 0.2, 0.8))  # purple
            geometries.append(ds_lines)

        if deploy_rel.shape[0] >= 2:
            dep_lines = make_lineset(deploy_rel, (1.0, 0.55, 0.0))  # orange
            geometries.append(dep_lines)

        if not geometries:
            raise RuntimeError("没有足够的点来渲染轨迹。")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Relative xyz trajectories (dataset vs deploy)",
        width=960,
        height=720,
    )


def _extract_start_end(arr: np.ndarray) -> np.ndarray:
    """
    仅保留给定轨迹的起点和终点。
    如果长度为 0 -> 抛错；长度为 1 -> 原样返回。
    """
    if arr.shape[0] == 0:
        raise ValueError("轨迹长度为 0，无法提取起始/结束点。")
    if arr.shape[0] == 1:
        return arr.copy()
    return np.stack([arr[0], arr[-1]], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "在 3D 空间中对比 LeRobot 数据集的相对位置轨迹（state[14,15,16]）"
            "与指定 deploy 结果目录中的相对位置轨迹。"
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/rvsa/codehub/VB-VLA/data/0118_data_smooth",
        help="LeRobot 本地数据集根目录",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="/home/rvsa/codehub/VB-VLA/eval_obs_data/eval_obs_20260316_230002",
        help="deploy/eval 结果目录",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="从数据集中采样 state 的步长（默认 1）",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=50000,
        help="从数据集中最多采样多少帧相对位置（默认 50000，<=0 表示不限制）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图片路径（默认保存在 tools/visualize_state_space/plots 下自动命名）",
    )
    parser.add_argument(
        "--show_3d",
        type=bool,
        default=True,
        help="使用 open3d 在交互式 3D 窗口中渲染轨迹（需要安装 open3d）",
    )
    parser.add_argument(
        "--start_end_only",
        type=bool,
        default=True,
        help="仅使用每条轨迹的起点和终点进行绘制与 3D 渲染",
    )

    quest_2_ee_left = np.load("/home/rvsa/codehub/VB-VLA/quest_2_ee_left_hand_fix_quest.npy")
    quest_2_ee_right = np.load("/home/rvsa/codehub/VB-VLA/quest_2_ee_right_hand_fix_quest.npy")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    eval_root = Path(args.eval_dir).expanduser().resolve()

    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None
    stride = max(args.stride, 1)

    print(f"[INFO] 数据集根目录: {dataset_root}")
    print(f"[INFO] eval 结果目录: {eval_root}")

    dataset_rel, data_start_end_rel, data_start_pose, data_end_pose = load_dataset_relative_xyz(
        dataset_root,
        stride=stride,
        max_frames=max_frames,
    )
    print(f"[INFO] 从数据集中采样到 {dataset_rel.shape[0]} 帧相对位置。")

    deploy_rel = load_deploy_relative_xyz(eval_root)
    print(f"[INFO] 从 deploy 结果中加载到 {deploy_rel.shape[0]} 帧相对位置。")

    # 根据模式决定是使用全轨迹还是只用起始/结束点
    if args.start_end_only:
        print("[INFO] 仅使用轨迹的起点和终点进行可视化。")
        dataset_rel_vis = data_start_end_rel
        print(data_start_end_rel.shape)
        deploy_rel_vis = _extract_start_end(deploy_rel)
        # 计算点云中心
        data_start_center = np.mean(data_start_pose, axis=0)
        data_end_center = np.mean(data_end_pose, axis=0)
        print(f"l2r start pose 点云中心 (quest): {data_start_center}")
        data_start_center_ee = mat_to_pose(
            quest_2_ee_right @ pose_to_mat(data_start_center) @ np.linalg.inv(quest_2_ee_left)
            )
        print(f"l2r start pose 点云中心 (ee): {data_start_center_ee}")
        print(f"l2r end pose 点云中心 (quest): {data_end_center}")

    else:
        dataset_rel_vis = dataset_rel
        deploy_rel_vis = deploy_rel

    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
    else:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        tag = f"{dataset_root.name}__{eval_root.name}"
        output_path = PLOTS_DIR / f"{tag}_dataset_vs_deploy_relative_xyz.png"

    title = (
        f"Relative xyz comparison\n"
        f"dataset: {dataset_root.name}   eval: {eval_root.name}"
    )
    plot_comparison(dataset_rel_vis, deploy_rel_vis, output_path, title=title)

    if args.show_3d:
        print("[INFO] 使用 open3d 打开交互式 3D 窗口渲染轨迹/散点...")
        render_comparison_open3d(
            dataset_rel_vis,
            deploy_rel_vis,
            start_end_only=args.start_end_only,
        )


if __name__ == "__main__":
    main()

