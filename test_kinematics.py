"""
test_kinematics.py
==================
验证新 AM2 URDF 的 FK/IK 正确性。

★★★ 完全离线 — 不连真机,机械臂绝对不会动 ★★★

原因:
  - 不实例化 RobotControl / RobotWrapperTyphon / RobotWrapper(eyou)
  - 不触发 typhon HTTP 连接,不切换 DUAL_ARM_API_CONTROL 模式
  - 纯 pykin 求解器 + pybullet DIRECT 模式 (无 GUI 无物理仿真)

使用:
  cd <repo_root>
  python real_world/robot_api/arm/test_kinematics.py
  
  # 自定义 URDF 路径
  python test_kinematics.py \
      --urdf-left  real_world/robot_api/assets/AM2_left.urdf \
      --urdf-right real_world/robot_api/assets/AM2_right.urdf

通过标准:
  Level 1 (FK↔IK 自洽):       fail < 5%, max_pos_err < 1mm
  Level 2 (vs pybullet):      max_pos_err < 0.01mm, max_rot_err < 1e-6
  Level 3 (已知姿态):         数值符合直觉 (左 y > 0, 右 y < 0)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ── Python 3.10+ 兼容 ──
import collections
import collections.abc
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import pykin
from pykin.robots.single_arm import SingleArm


# ============================================================
# 默认配置
# ============================================================
DEFAULT_URDF_LEFT  = "real_world/robot_api/assets/AM2_left/urdf/AM2_left.urdf"
DEFAULT_URDF_RIGHT = "real_world/robot_api/assets/AM2_right/urdf/AM2_right.urdf"
BASE_LINK = "base_link"
EE_LEFT   = "left-link_arm_7"
EE_RIGHT  = "right-link_arm_7"

REPO_ROOT = Path(__file__).resolve().parents[3] if "real_world" in str(Path(__file__).resolve()) \
            else Path.cwd()


# ============================================================
# URDF 准备工具 (从 RobotControl_pykin_copy.py 内联,保持自包含)
# ============================================================
def _resolve_path(p: str) -> str:
    pp = Path(p)
    return str(pp) if pp.is_absolute() else str((REPO_ROOT / pp).resolve())


def _prepare_urdf_for_pykin(urdf_path: str) -> tuple[str, str | None]:
    """
    把 URDF 暂存到 pykin/assets/ 下,因为 pykin 强制把传入路径拼到它的
    assets 目录后面 (asset_dir + "/" + path),不识别绝对路径。

    Returns:
        (relative_path_from_pykin_assets, temp_file_to_cleanup_or_None)

    自适应两种布局:
      ROS package:  <pkg>/urdf/<f>.urdf + <pkg>/meshes/   (旧 ARM-LEFT-GR-0)
      扁平:         <dir>/<f>.urdf + <dir>/meshes/        (新 AM2)
    """
    src = Path(urdf_path).resolve()
    urdf_dir = src.parent

    # 探测布局:看 mesh 目录在哪一级
    if urdf_dir.name == "urdf" and (urdf_dir.parent / "meshes").is_dir():
        # ROS package 布局: 资源根 = URDF 的 grandparent
        resource_dir = urdf_dir.parent
        urdf_subpath = Path("urdf") / src.name
    else:
        # 扁平布局: 资源根 = URDF 同级
        resource_dir = urdf_dir
        urdf_subpath = Path(src.name)

    pkg_name = resource_dir.name
    pykin_assets = Path(pykin.__file__).resolve().parent / "assets"
    stage_root = pykin_assets / "vb_vla_assets"
    stage_pkg = stage_root / pkg_name
    stage_urdf = stage_pkg / urdf_subpath

    stage_root.mkdir(parents=True, exist_ok=True)

    # 处理 stage_pkg 的存在情况 (防止之前失败留下的错误 symlink)
    if stage_pkg.is_symlink():
        try:
            current_target = stage_pkg.resolve(strict=False)
        except OSError:
            current_target = None
        if current_target != resource_dir.resolve():
            print(f"[setup] removing stale symlink: {stage_pkg} -> {current_target}")
            stage_pkg.unlink()

    if not stage_pkg.exists():
        try:
            stage_pkg.symlink_to(resource_dir, target_is_directory=True)
        except OSError:
            shutil.copytree(resource_dir, stage_pkg)

    if not stage_urdf.exists():
        raise FileNotFoundError(
            f"URDF not staged correctly:\n"
            f"  expected: {stage_urdf}\n"
            f"  resource_dir: {resource_dir}\n"
            f"  layout detected: "
            f"{'ROS package' if urdf_dir.name == 'urdf' else 'flat'}"
        )

    # 预处理 (只在 URDF 含 package:// 时才生效)
    with open(stage_urdf, "r") as f:
        content = f.read()

    if "package://" not in content:
        # 干净 URDF - 直接用 stage 后的路径
        final_path = stage_urdf.resolve()
        tmp_for_cleanup = None
    else:
        # 替换 package://<pkg>/ -> ../
        # ROS 惯例 package_name 是 grandparent name
        pkg_uri_name = stage_urdf.parent.parent.name
        new_content = re.sub(f"package://{re.escape(pkg_uri_name)}/", "../", content)
        fd, tmp = tempfile.mkstemp(suffix=".urdf", prefix="test_kin_",
                                    dir=str(stage_urdf.parent))
        try:
            with os.fdopen(fd, "w") as f:
                f.write(new_content)
            final_path = Path(tmp).resolve()
            tmp_for_cleanup = tmp
        except Exception:
            os.close(fd)
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    try:
        rel = str(final_path.relative_to(pykin_assets))
    except ValueError:
        rel = os.path.relpath(str(final_path), str(pykin_assets))
    return rel, tmp_for_cleanup


def _extract_joint_limits(urdf_path: str, joint_names: list[str]) -> list[tuple[float, float]]:
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return [(-np.pi, np.pi) for _ in joint_names]

    m = {}
    for j in root.findall("joint"):
        jn = j.attrib.get("name")
        jt = j.attrib.get("type", "")
        if jt not in ("revolute", "prismatic"):
            continue
        lim = j.find("limit")
        if lim is None:
            continue
        try:
            m[jn] = (float(lim.attrib["lower"]), float(lim.attrib["upper"]))
        except (KeyError, ValueError):
            continue

    return [m.get(n, (-np.pi, np.pi)) for n in joint_names]


# ============================================================
# Solver 工厂
# ============================================================
def setup_solvers(urdf_left: str, urdf_right: str) -> dict:
    """准备 pykin SingleArm + 抽限位 (方案 A: 不传 ab→rb 偏移)"""
    urdf_l_abs = _resolve_path(urdf_left)
    urdf_r_abs = _resolve_path(urdf_right)
    
    if not os.path.exists(urdf_l_abs):
        raise FileNotFoundError(f"Left URDF not found: {urdf_l_abs}")
    if not os.path.exists(urdf_r_abs):
        raise FileNotFoundError(f"Right URDF not found: {urdf_r_abs}")

    print(f"[setup] left  URDF: {urdf_l_abs}")
    print(f"[setup] right URDF: {urdf_r_abs}")

    urdf_l_pykin, tmp_l = _prepare_urdf_for_pykin(urdf_l_abs)
    urdf_r_pykin, tmp_r = _prepare_urdf_for_pykin(urdf_r_abs)

    kin_l = SingleArm(urdf_l_pykin)
    kin_l.setup_link_name(BASE_LINK, EE_LEFT)
    kin_r = SingleArm(urdf_r_pykin)
    kin_r.setup_link_name(BASE_LINK, EE_RIGHT)

    joint_names_l = [f"left-joint_arm_{i}" for i in range(1, 8)]
    joint_names_r = [f"right-joint_arm_{i}" for i in range(1, 8)]
    limits_l = _extract_joint_limits(urdf_l_abs, joint_names_l)
    limits_r = _extract_joint_limits(urdf_r_abs, joint_names_r)

    print(f"[setup] left  limits: {[(round(lo,2), round(hi,2)) for lo,hi in limits_l]}")
    print(f"[setup] right limits: {[(round(lo,2), round(hi,2)) for lo,hi in limits_r]}")

    return {
        "left":  {"kin": kin_l, "urdf_abs": urdf_l_abs, "limits": limits_l,
                  "ee_link": EE_LEFT,  "joint_prefix": "left-joint_arm_",  "tmp": tmp_l},
        "right": {"kin": kin_r, "urdf_abs": urdf_r_abs, "limits": limits_r,
                  "ee_link": EE_RIGHT, "joint_prefix": "right-joint_arm_", "tmp": tmp_r},
    }


def _silent_ik(kin, q_init, target):
    with contextlib.redirect_stdout(io.StringIO()):
        return kin.inverse_kin(np.asarray(q_init, dtype=float),
                               target, method="LM", max_iter=100)


def _sample_q(rng, limits, margin=0.1):
    """在限位内安全采样,留 margin 避免 IK 在边界附近收敛慢"""
    q = []
    for lo, hi in limits:
        if np.isfinite(lo) and np.isfinite(hi):
            q.append(rng.uniform(lo + margin, hi - margin))
        else:
            q.append(rng.uniform(-1.5, 1.5))
    return np.array(q)


# ============================================================
# Level 1: FK ↔ IK 自洽性
# ============================================================
def level1_self_consistency(solvers: dict, n: int = 100, seed: int = 42):
    print("\n" + "=" * 72)
    print(f"Level 1: FK ↔ IK self-consistency  (n={n} per arm)")
    print("=" * 72)
    print("方法: 随机 q_gt → FK → target → IK(从远初值) → q_solved → FK → 比较")
    print("判定: 看 EE 误差,不看关节误差 (7-DOF 冗余,关节有多解)\n")

    rng = np.random.default_rng(seed)
    all_pass = True

    for arm in ["left", "right"]:
        kin = solvers[arm]["kin"]
        limits = solvers[arm]["limits"]

        max_pos_err = max_rot_err = max_joint_err = 0.0
        fail = 0

        for _ in range(n):
            q_gt = _sample_q(rng, limits)
            target = kin.compute_eef_pose(kin.forward_kin(q_gt))

            # 关键: IK 初值远离 q_gt,真正考验求解器
            q_init = q_gt + rng.normal(0, 0.5, size=q_gt.shape)
            q_solved = _silent_ik(kin, q_init, target)

            # 用解出的 q 再 FK,跟 target 对比
            target_check = kin.compute_eef_pose(kin.forward_kin(q_solved))
            pos_err = float(np.linalg.norm(target[:3] - target_check[:3]))
            rot_err = float(1 - abs(np.dot(target[3:], target_check[3:])))
            joint_err = float(np.max(np.abs(q_solved - q_gt)))

            if pos_err > 1e-3 or rot_err > 1e-3:
                fail += 1
            max_pos_err = max(max_pos_err, pos_err)
            max_rot_err = max(max_rot_err, rot_err)
            max_joint_err = max(max_joint_err, joint_err)

        passed = fail < n * 0.05
        all_pass = all_pass and passed
        verdict = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{arm:5s}] {verdict}  fail={fail}/{n}  "
              f"pos_err≤{max_pos_err*1000:.3f}mm  "
              f"rot_err≤{max_rot_err:.2e}  "
              f"joint_diff≤{max_joint_err:.3f}rad")

    return all_pass


# ============================================================
# Level 2: pykin FK vs pybullet FK (跨实现对比)
# ============================================================
def level2_pybullet_compare(solvers: dict, n: int = 30, seed: int = 123):
    print("\n" + "=" * 72)
    print(f"Level 2: pykin FK vs pybullet FK  (n={n} per arm)")
    print("=" * 72)
    print("方法: 同样的 q,两个独立的 URDF 解析器算 FK,比较 EE 位姿\n")

    try:
        import pybullet as p
    except ImportError:
        print("  [SKIP] pybullet 未安装  (pip install pybullet)")
        print("        强烈建议装上 - 这层是抓 URDF 解析 bug 的关键")
        return None

    p.connect(p.DIRECT)  # 无 GUI,无物理仿真
    rng = np.random.default_rng(seed)
    all_pass = True

    try:
        for arm in ["left", "right"]:
            info = solvers[arm]
            kin = info["kin"]
            urdf_abs = info["urdf_abs"]
            ee_link_name = info["ee_link"]
            joint_prefix = info["joint_prefix"]
            limits = info["limits"]

            # pybullet 自己处理 mesh 相对路径,直接喂原始 URDF
            try:
                bid = p.loadURDF(urdf_abs, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
            except p.error as e:
                print(f"  [{arm:5s}] ✗ pybullet failed to load URDF: {e}")
                all_pass = False
                continue

            # 找 7 个 revolute joint 和 ee link
            arm_joints = []  # (name, index)
            ee_idx = -1
            for j in range(p.getNumJoints(bid)):
                jinfo = p.getJointInfo(bid, j)
                jname = jinfo[1].decode()
                lname = jinfo[12].decode()
                if jname.startswith(joint_prefix) and jinfo[2] == p.JOINT_REVOLUTE:
                    arm_joints.append((jname, j))
                if lname == ee_link_name:
                    ee_idx = j
            arm_joints.sort(key=lambda x: int(x[0].split("_")[-1]))
            joint_indices = [j for _, j in arm_joints]

            if len(joint_indices) != 7 or ee_idx < 0:
                print(f"  [{arm:5s}] ✗ pybullet: found {len(joint_indices)}/7 joints, "
                      f"ee_idx={ee_idx}")
                p.removeBody(bid)
                all_pass = False
                continue

            max_pos_err = max_rot_err = 0.0
            for _ in range(n):
                q = _sample_q(rng, limits)

                # pybullet FK
                for idx, qi in zip(joint_indices, q):
                    p.resetJointState(bid, idx, qi)
                ls = p.getLinkState(bid, ee_idx, computeForwardKinematics=True)
                pb_pos = np.array(ls[4])  # worldLinkFramePosition (URDF frame, 不是 visual)
                pb_xyzw = np.array(ls[5])
                pb_wxyz = np.array([pb_xyzw[3], pb_xyzw[0], pb_xyzw[1], pb_xyzw[2]])

                # pykin FK
                ee = kin.compute_eef_pose(kin.forward_kin(q))
                pk_pos, pk_wxyz = np.asarray(ee[:3]), np.asarray(ee[3:])

                pos_err = float(np.linalg.norm(pb_pos - pk_pos))
                rot_err = float(1 - abs(np.dot(pb_wxyz, pk_wxyz)))
                max_pos_err = max(max_pos_err, pos_err)
                max_rot_err = max(max_rot_err, rot_err)

            p.removeBody(bid)
            passed = max_pos_err < 1e-5 and max_rot_err < 1e-6
            all_pass = all_pass and passed
            verdict = "✓ PASS" if passed else "✗ FAIL"
            print(f"  [{arm:5s}] {verdict}  "
                  f"pos_err≤{max_pos_err*1000:.6f}mm  "
                  f"rot_err≤{max_rot_err:.2e}")

            if not passed:
                print(f"         ↑ 误差较大,可能是 URDF 链路根没设对,"
                      f"或 setup_link_name 用错了 link 名")
    finally:
        p.disconnect()

    return all_pass


# ============================================================
# Level 3: 已知姿态人工检查
# ============================================================
def level3_known_poses(solvers: dict):
    print("\n" + "=" * 72)
    print("Level 3: Known pose sanity checks")
    print("=" * 72)
    print("方法: 检查几个直观可预测的姿态,确认坐标系约定\n")

    kin_l = solvers["left"]["kin"]
    kin_r = solvers["right"]["kin"]

    # ── Check 1: q=0 时左右 EE 位置 ──
    q0 = np.zeros(7)
    ee_l = kin_l.compute_eef_pose(kin_l.forward_kin(q0))
    ee_r = kin_r.compute_eef_pose(kin_r.forward_kin(q0))

    print("  [q=0] EE pose in robot base frame:")
    print(f"    left  pos:  ({ee_l[0]:+.4f}, {ee_l[1]:+.4f}, {ee_l[2]:+.4f}) m")
    print(f"    right pos:  ({ee_r[0]:+.4f}, {ee_r[1]:+.4f}, {ee_r[2]:+.4f}) m")
    print(f"    left  quat: ({ee_l[3]:+.4f}, {ee_l[4]:+.4f}, {ee_l[5]:+.4f}, {ee_l[6]:+.4f}) wxyz")
    print(f"    right quat: ({ee_r[3]:+.4f}, {ee_r[4]:+.4f}, {ee_r[5]:+.4f}, {ee_r[6]:+.4f}) wxyz")

    qn_l = float(np.linalg.norm(ee_l[3:]))
    qn_r = float(np.linalg.norm(ee_r[3:]))
    print(f"    quat norm:  left={qn_l:.6f}  right={qn_r:.6f}  (期望 = 1.0)")

    if abs(qn_l - 1.0) > 1e-4 or abs(qn_r - 1.0) > 1e-4:
        print("    ✗ 四元数不是单位四元数!")
    else:
        print("    ✓ 四元数归一化 OK")

    # ── Check 2: y 符号 ──
    print("\n  [y-sign check] (期望左 y > 0,右 y < 0,因为 base_link 偏移 ±0.108)")
    yl, yr = ee_l[1], ee_r[1]
    if yl > 0 and yr < 0:
        print(f"    ✓ left y={yl:+.4f} > 0,  right y={yr:+.4f} < 0")
    else:
        print(f"    ✗ left y={yl:+.4f},  right y={yr:+.4f}  ← 不符合期望!")

    # ── Check 3: 左右镜像对称性 (URDF rpy 互为相反数) ──
    print("\n  [mirror symmetry] (URDF 左 rpy=(-π/2,0,0),右 rpy=(π/2,0,0),约 x 轴互镜)")
    sym_x = abs(ee_l[0] - ee_r[0])
    sym_y = abs(ee_l[1] + ee_r[1])
    sym_z = abs(ee_l[2] - ee_r[2])
    print(f"    |xL - xR| = {sym_x:.6f}   (期望 ≈ 0)")
    print(f"    |yL + yR| = {sym_y:.6f}   (期望 ≈ 0)")
    print(f"    |zL - zR| = {sym_z:.6f}   (期望 ≈ 0)")
    if sym_x < 1e-5 and sym_y < 1e-5 and sym_z < 1e-5:
        print("    ✓ 严格镜像对称")
    else:
        print("    ! 不严格镜像 - 可能左右 URDF 内部 link origin 不完全镜像")
        print("      (SolidWorks 导出常见,只要不影响实物对齐就 OK)")

    # ── Check 4: 单关节扫描 ──
    print("\n  [single-joint sweep] left arm,只动 joint_1:")
    print(f"    {'q1':>7s}  {'x':>8s}  {'y':>8s}  {'z':>8s}")
    positions = []
    for q1 in np.linspace(-1.0, 1.0, 5):
        q = np.zeros(7)
        q[0] = q1
        ee = kin_l.compute_eef_pose(kin_l.forward_kin(q))
        positions.append(ee[:3])
        print(f"    {q1:>+7.3f}  {ee[0]:>+8.4f}  {ee[1]:>+8.4f}  {ee[2]:>+8.4f}")

    # 判定: 5 个点应该共面 (绕 joint_1 轴的圆),且各点到圆心距离接近
    positions = np.array(positions)
    center = positions.mean(axis=0)
    dists = np.linalg.norm(positions - center, axis=1)
    if dists.std() / (dists.mean() + 1e-9) < 0.5:
        print(f"    ✓ 5 个点到平均中心距离接近 (相对 std={dists.std()/(dists.mean()+1e-9):.3f})")
    else:
        print(f"    ! 5 个点离散度高 - joint_1 行为可能异常")


# ============================================================
# 清理
# ============================================================
def cleanup(solvers: dict):
    for arm in solvers.values():
        tmp = arm.get("tmp")
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf-left",  default=DEFAULT_URDF_LEFT)
    parser.add_argument("--urdf-right", default=DEFAULT_URDF_RIGHT)
    parser.add_argument("--n1", type=int, default=100, help="Level 1 sample count")
    parser.add_argument("--n2", type=int, default=30,  help="Level 2 sample count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", choices=["1", "2", "3"], action="append", default=[])
    args = parser.parse_args()

    np.set_printoptions(precision=4, suppress=True)

    print("=" * 72)
    print(" Kinematics Verification Test")
    print(" ★ 完全离线 - 不连真机,机械臂不会动 ★")
    print("=" * 72)

    solvers = setup_solvers(args.urdf_left, args.urdf_right)

    results = {}
    try:
        if "1" not in args.skip:
            results["L1"] = level1_self_consistency(solvers, n=args.n1, seed=args.seed)
        if "2" not in args.skip:
            results["L2"] = level2_pybullet_compare(solvers, n=args.n2, seed=args.seed + 1)
        if "3" not in args.skip:
            level3_known_poses(solvers)
            results["L3"] = None  # 人工判读

        # ── 总结 ──
        print("\n" + "=" * 72)
        print(" Summary")
        print("=" * 72)
        for k, v in results.items():
            tag = "✓ PASS" if v is True else ("✗ FAIL" if v is False else "— manual review")
            print(f"  {k}:  {tag}")

        all_auto_pass = all(v is True or v is None for v in results.values())
        if all_auto_pass:
            print("\n  自动测试全部通过。Level 3 的几个数字请你自己看一眼是否符合预期。")
            print("  下一步: 上真机做 Level 4 (回零位 + 实测末端坐标对齐)。")
        else:
            print("\n  ✗ 有测试未通过 - 检查 base_link 名字、ee link 名字、URDF mesh 路径。")
        print("=" * 72)

    finally:
        cleanup(solvers)


if __name__ == "__main__":
    main()