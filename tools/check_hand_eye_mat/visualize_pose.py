"""
Visualize the last-frame 4x4 homogeneous transformation matrices
stored in pose_and_error/ as 3-D coordinate frames.

Saves the result as an image under tools/check_hand_eye_mat/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3-D projection
from scipy.spatial.transform import Rotation


def draw_frame(
    ax: plt.Axes,
    T: np.ndarray,
    label: str,
    length: float = 0.08,
    lw: float = 2.5,
):
    """Draw a coordinate frame defined by a 4×4 homogeneous matrix *T*."""
    origin = T[:3, 3]
    R = T[:3, :3]

    colours = ("r", "g", "b")
    axis_labels = ("X", "Y", "Z")
    for i, (c, al) in enumerate(zip(colours, axis_labels)):
        direction = R[:, i] * length
        ax.quiver(
            *origin, *direction,
            color=c, linewidth=lw, arrow_length_ratio=0.15,
        )

    ax.text(*origin, f"  {label}", fontsize=9, fontweight="bold")


def set_equal_aspect(ax: plt.Axes, points: np.ndarray, margin: float = 0.15):
    """Force equal aspect ratio on a 3-D axes so frames look undistorted."""
    mid = points.mean(axis=0)
    span = (points.max(axis=0) - points.min(axis=0)).max() / 2 + margin
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)


def pose_info_text(T: np.ndarray) -> str:
    """Return a human-readable summary of position + Euler angles."""
    pos = T[:3, 3]
    euler = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    return (
        f"pos = [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]\n"
        f"rpy = [{euler[0]:+.1f}°, {euler[1]:+.1f}°, {euler[2]:+.1f}°]"
    )


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "pose_and_error"

    l2r_cal = np.load(data_dir / "left2right_mat_cal.npy", allow_pickle=True)
    r2l_got = np.load(data_dir / "right2left_mat_got.npy", allow_pickle=True)

    T_l2r = l2r_cal[-1]  # last frame
    T_r2l = r2l_got[-1]

    T_origin = np.eye(4)

    all_positions = np.array([
        T_origin[:3, 3],
        T_l2r[:3, 3],
        T_r2l[:3, 3],
    ])

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    draw_frame(ax, T_origin, "Origin", length=0.10, lw=3.0)
    draw_frame(ax, T_l2r, "left2right_cal", length=0.08)
    draw_frame(ax, T_r2l, "right2left_got", length=0.08)

    for T, color, marker, name in [
        (T_l2r, "darkorange", "o", "left2right_cal"),
        (T_r2l, "dodgerblue", "s", "right2left_got"),
    ]:
        p = T[:3, 3]
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]],
                "--", color=color, alpha=0.5, linewidth=1.2)
        ax.scatter(*p, color=color, s=60, marker=marker, zorder=5)

    set_equal_aspect(ax, all_positions)

    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.set_zlabel("Z", fontsize=11)
    ax.set_title("Pose Visualization (last frame)", fontsize=14, pad=20)

    info = (
        f"left2right_mat_cal:\n{pose_info_text(T_l2r)}\n\n"
        f"right2left_mat_got:\n{pose_info_text(T_r2l)}"
    )
    fig.text(0.02, 0.02, info, fontsize=9, family="monospace",
             verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7))

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="darkorange", marker="o", ls="--",
                       label="left2right_cal"),
            plt.Line2D([0], [0], color="dodgerblue", marker="s", ls="--",
                       label="right2left_got"),
        ],
        loc="upper right", fontsize=10,
    )

    out_path = base_dir / "pose_visualization.png"
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
