"""
RobotControl_pykin.py
---------------------
AM2 双臂机械臂的运动学控制层。

设计 (方案 A):
    - URDF 已拆成左右两个单臂文件, 每个的根都是 base_link
    - pykin SingleArm 加载时不传 ab2rb 偏移 (Transform 用单位变换)
    - setup_link_name("base_link", "left-link_tcp"), 让 IK/FK 链
      从 URDF 根开始, 自动包含 JOINT_L00 的肩部偏移
    - 结果: FK 输出 = ^rbT_eef, IK 输入 = ^rbT_eef, 完全自洽

接口:
    - get_robot_joints() -> dict
    - get_ee_pose()      -> dict, ^rbT_eef 形式 (7-vector wxyz)
    - set_target_JP(...)
    - set_target_CP(target_pose, single_arm_mode=False)
    - execute()
    - stop()

EE 位姿格式 (与旧版兼容):
    7-vector [x, y, z, qw, qx, qy, qz]  (scalar-first 四元数)

可配置项 (优先级: 显式参数 > YAML > 默认值):
    - urdf_path_left/right
    - base_link_left/right (默认 "base_link")
    - ee_link_left/right (默认 "left-link_tcp" / "right-link_tcp")
    - robot_type ("typhon" | "eyou")
"""

import os
import sys
import io
import re
import time
import shutil
import tempfile
import contextlib
from pathlib import Path

# 让 `python real_world/.../RobotControl_pykin.py` 直接跑也能解析包路径
# 把仓库根加到 sys.path
_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

import numpy as np
import yaml
import xml.etree.ElementTree as ET

import collections
import collections.abc
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import transforms3d as t3d
import pykin
from pykin.robots.single_arm import SingleArm
from pykin.kinematics import transform as t_utils

from real_world.robot_api.arm.RobotWrapper_typhon import RobotWrapperTyphon

# 旧 backend 是 optional 的 (eyou 用 rb_python)
try:
    from real_world.robot_api.arm.RobotWrapper import RobotWrapper as RobotWrapperEyou
    EYOU_AVAILABLE = True
except ImportError:
    EYOU_AVAILABLE = False


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]


# ============================================================
# Helpers
# ============================================================
def _resolve_repo_path(path_like: str) -> str:
    p = Path(path_like)
    return str(p) if p.is_absolute() else str((REPO_ROOT / p).resolve())


def _load_yaml_config(config_path: str) -> dict:
    p = Path(config_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def preprocess_urdf(urdf_path: str) -> str:
    """把 ROS package:// URI 替换成相对路径, 写到临时文件返回路径"""
    with open(urdf_path, "r") as f:
        content = f.read()
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    parent_dir = os.path.dirname(urdf_dir)
    package_name = os.path.basename(parent_dir)
    content = re.sub(f"package://{re.escape(package_name)}/", "../", content)
    fd, tmp_path = tempfile.mkstemp(suffix=".urdf", prefix="preprocessed_",
                                    dir=urdf_dir)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return tmp_path
    except Exception:
        os.close(fd)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def prepare_urdf_for_pykin(urdf_path: str) -> tuple[str, str]:
    """
    pykin 要求 URDF 在它的 assets 目录下 (它内部用相对路径加载 mesh)
    这里 symlink 整个 package 到 pykin/assets/vb_vla_assets/, 然后预处理.
    返回 (相对 pykin/assets 的路径, 临时文件绝对路径)
    """
    src = Path(urdf_path).resolve()
    pkg_dir = src.parent.parent
    pkg_name = pkg_dir.name

    pykin_assets = Path(pykin.__file__).resolve().parent / "assets"
    stage_root = pykin_assets / "vb_vla_assets"
    stage_pkg = stage_root / pkg_name
    stage_urdf = stage_pkg / "urdf" / src.name

    stage_root.mkdir(parents=True, exist_ok=True)
    if not stage_pkg.exists():
        try:
            stage_pkg.symlink_to(pkg_dir, target_is_directory=True)
        except OSError:
            shutil.copytree(pkg_dir, stage_pkg)

    if not stage_urdf.exists():
        raise FileNotFoundError(f"URDF not staged: {stage_urdf}")

    preprocessed = Path(preprocess_urdf(str(stage_urdf))).resolve()
    try:
        rel = str(preprocessed.relative_to(pykin_assets))
    except ValueError:
        rel = os.path.relpath(str(preprocessed), str(pykin_assets))
    return rel, str(preprocessed)


# ============================================================
# RobotControl
# ============================================================
class RobotControl:
    """
    AM2 双臂机械臂运动学控制层.

    Args:
        vel_max: 关节最大速度 (typhon 下未使用, 保留是为兼容旧接口)
        urdf_path_left/right: 单臂 URDF 路径 (拆分后的)
        base_link_left/right: pykin IK/FK 链的根 link 名 (方案 A 推荐 'base_link')
        ee_link_left/right: pykin IK/FK 链的末端 link 名 (推荐 'left-link_tcp')
        robot_type: 'typhon' (AM2) 或 'eyou' (旧机械臂)
        config_path: YAML 配置, 用来集中管理上面这些参数
        robot_wrapper: 直接注入的 wrapper 实例 (覆盖 robot_type)
    """

    def __init__(
        self,
        vel_max: float = None,
        urdf_path_left: str = None,
        urdf_path_right: str = None,
        base_link_left: str = "base_link",
        base_link_right: str = "base_link",
        ee_link_left: str = "left-link_arm_7",
        ee_link_right: str = "right-link_arm_7",
        robot_type: str = "typhon",
        config_path: str = "configs/am2.yaml",
        robot_wrapper=None,
        left_joint_names: list[str] | None = None,
        right_joint_names: list[str] | None = None,
    ):
        # ── 加载配置 ──
        cfg = _load_yaml_config(config_path) if config_path else {}
        def _g(k, default): return cfg.get(k, default)

        urdf_left  = urdf_path_left  or _g("urdf_path_left",
            "real_world/robot_api/assets/AM2_left/urdf/AM2_left.urdf")
        urdf_right = urdf_path_right or _g("urdf_path_right",
            "real_world/robot_api/assets/AM2_right/urdf/AM2_right.urdf")
        # 方案 A: 默认固定从 base_link 起链，避免被 YAML 中旧配置意外覆盖。
        # 如需特殊根链路，显式通过构造参数传入 base_link_left/right。
        base_l = base_link_left
        base_r = base_link_right
        ee_l   = ee_link_left    if ee_link_left   != "left-link_tcp"  else _g("ee_link_left",  "left-link_tcp")
        ee_r   = ee_link_right   if ee_link_right  != "right-link_tcp" else _g("ee_link_right", "right-link_tcp")
        rtype  = _g("robot_type", robot_type)
        vel    = vel_max if vel_max is not None else _g("vel_max", 0.08)

        # ── 选择底层 wrapper ──
        if robot_wrapper is not None:
            self.robot = robot_wrapper
            self._robot_type = "typhon" if isinstance(robot_wrapper, RobotWrapperTyphon) else "eyou"
        elif rtype == "typhon":
            typhon_cfg = _g("typhon", {})
            self.robot = RobotWrapperTyphon(
                base_url=typhon_cfg.get("base_url", "http://192.168.100.100:8081"),
                timeout=typhon_cfg.get("timeout", 5.0),
                auto_enter_control_mode=typhon_cfg.get("auto_enter_control_mode", True),
            )
            self._robot_type = "typhon"
        elif rtype == "eyou":
            if not EYOU_AVAILABLE:
                raise RuntimeError("rb_python not installed; eyou backend unavailable")
            self.robot = RobotWrapperEyou(vel_max=vel)
            self._robot_type = "eyou"
        else:
            raise ValueError(f"Unknown robot_type: {rtype}")

        # ── 命令缓冲 ──
        self.action_target = dict(
            left_arm=None, right_arm=None,
            left_gripper=None, right_gripper=None,
        )

        # ── pykin URDF 准备 ──
        urdf_l_pykin, urdf_l_tmp = prepare_urdf_for_pykin(_resolve_repo_path(urdf_left))
        urdf_r_pykin, urdf_r_tmp = prepare_urdf_for_pykin(_resolve_repo_path(urdf_right))
        self._temp_urdf_files = [urdf_l_tmp, urdf_r_tmp]

        # ── pykin SingleArm (方案 A: 不传 ab2rb 偏移) ──
        # 单臂 URDF 已经从 base_link 开始, JOINT_L00/R00 在 URDF 内部,
        # pykin 加载后从 base_link 走到 tcp 自动包含所有偏移.
        # SingleArm 第二个参数留单位变换, 表示 "URDF 根 = world 原点".
        self.kin_left  = SingleArm(urdf_l_pykin)
        self.kin_right = SingleArm(urdf_r_pykin)

        self.kin_left.setup_link_name(base_l, ee_l)
        self.kin_right.setup_link_name(base_r, ee_r)

        # ── 关节限位 (从 URDF 抽取, 用于 set_target_JP 时 clip 防止超限) ──
        self._joint_layout = {
            "left_arm":  left_joint_names  or _g("left_joint_names",
                [f"JOINT_L0{i}" for i in range(1, 8)]),
            "right_arm": right_joint_names or _g("right_joint_names",
                [f"JOINT_R0{i}" for i in range(1, 8)]),
        }
        self._joint_limits = {
            "left_arm":  self._extract_joint_limits(_resolve_repo_path(urdf_left),  self._joint_layout["left_arm"]),
            "right_arm": self._extract_joint_limits(_resolve_repo_path(urdf_right), self._joint_layout["right_arm"]),
        }

        self._joint_shape_logged = False
        print(f"[RobotControl] initialized "
              f"(backend={self._robot_type}, base_link={base_l}, ee={ee_l})")

    # ============================================================
    # URDF 限位抽取
    # ============================================================
    @staticmethod
    def _extract_joint_limits(urdf_path: str, joint_names: list[str]) -> list[tuple[float, float]]:
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
        except Exception as e:
            print(f"[WARN] parse URDF for limits failed: {e}")
            return [(-np.inf, np.inf) for _ in joint_names]

        m: dict[str, tuple[float, float]] = {}
        for j in root.findall("joint"):
            jn = j.attrib.get("name")
            jt = j.attrib.get("type", "")
            if not jn or jt not in ("revolute", "prismatic"):
                continue
            lim = j.find("limit")
            if lim is None:
                continue
            lo = lim.attrib.get("lower")
            hi = lim.attrib.get("upper")
            if lo is None or hi is None:
                continue
            try:
                m[jn] = (float(lo), float(hi))
            except ValueError:
                pass

        out = []
        for n in joint_names:
            if n in m:
                out.append(m[n])
            else:
                print(f"[WARN] no limit for {n}, using unbounded")
                out.append((-np.inf, np.inf))
        return out

    def sanitize_joint_targets(self, arm_name: str, joints, strategy: str = "clip"):
        """clip 关节目标到限位内. strategy='skip' 时超限返回 None."""
        arr = np.asarray(joints, dtype=float).copy()
        limits = self._joint_limits.get(arm_name, [])
        if len(arr) != len(limits):
            return arr.tolist()

        exceeded = False
        for i, (lo, hi) in enumerate(limits):
            if np.isfinite(lo) and arr[i] < lo:
                exceeded = True
                if strategy == "clip":
                    print(f"[WARN] {arm_name}[{i}] {arr[i]:.4f} < {lo:.4f}, clipped")
                    arr[i] = lo
            if np.isfinite(hi) and arr[i] > hi:
                exceeded = True
                if strategy == "clip":
                    print(f"[WARN] {arm_name}[{i}] {arr[i]:.4f} > {hi:.4f}, clipped")
                    arr[i] = hi

        if exceeded and strategy == "skip":
            return None
        return arr.tolist()

    # ============================================================
    # 主接口
    # ============================================================
    def get_robot_joints(self) -> dict[str, list[float]]:
        out = {
            "left_arm":      np.asarray(self.robot.get_joint_angle("left_arm"),     dtype=float).tolist(),
            "right_arm":     np.asarray(self.robot.get_joint_angle("right_arm"),    dtype=float).tolist(),
            "left_gripper":  np.asarray(self.robot.get_joint_angle("left_gripper"), dtype=float).tolist(),
            "right_gripper": np.asarray(self.robot.get_joint_angle("right_gripper"),dtype=float).tolist(),
        }
        if not self._joint_shape_logged:
            print("[INFO] get_robot_joints structure (logged once):")
            for k, v in out.items():
                print(f"  {k}: len={len(v)}")
            self._joint_shape_logged = True
        return out

    def get_ee_pose(self) -> dict[str, np.ndarray]:
        """
        返回左右臂 ee 在 base 系下的位姿.

        方案 A 下 pykin FK 链从 base_link 开始, JOINT_L00/R00 包含在内,
        compute_eef_pose 输出就是 ^rbT_eef, 直接返回, 不做任何额外变换.

        格式: 7-vector [x, y, z, qw, qx, qy, qz]  (scalar-first 四元数)
        """
        joints = self.get_robot_joints()

        fk_l = self.kin_left.forward_kin(np.asarray(joints["left_arm"],  dtype=float))
        fk_r = self.kin_right.forward_kin(np.asarray(joints["right_arm"], dtype=float))

        ee_l = self.kin_left.compute_eef_pose(fk_l)
        ee_r = self.kin_right.compute_eef_pose(fk_r)

        return {
            "left_arm_ee2rb":  np.asarray(ee_l, dtype=float),
            "right_arm_ee2rb": np.asarray(ee_r, dtype=float),
            "left_gripper":  joints["left_gripper"],
            "right_gripper": joints["right_gripper"],
        }

    def set_target_JP(
        self,
        joint_left: np.ndarray,
        joint_right: np.ndarray = None,
        gripper_left: np.ndarray = None,
        gripper_right: np.ndarray = None,
    ):
        """关节空间目标设置. 自动 clip 到关节限位."""
        safe_l = self.sanitize_joint_targets("left_arm", joint_left, "clip")
        self.action_target["left_arm"] = safe_l if safe_l is not None else list(joint_left)
        self.action_target["left_gripper"] = gripper_left

        if joint_right is not None:
            safe_r = self.sanitize_joint_targets("right_arm", joint_right, "clip")
            self.action_target["right_arm"] = safe_r if safe_r is not None else list(joint_right)
        if gripper_right is not None:
            self.action_target["right_gripper"] = gripper_right

    def set_target_CP(self, target_pose: dict, single_arm_mode: bool = False):
        """
        笛卡尔空间目标设置.

        target_pose 里的 *_ee2rb 都是 ^rbT_eef (与 get_ee_pose 输出格式一致).
        方案 A 下 pykin IK 期望的也是 ^rbT_eef, 直接传入.
        """
        joints = self.get_robot_joints()

        # 左臂
        target_l = target_pose["left_arm_ee2rb"]
        joints_l = self._inverse_kin_silent(self.kin_left, joints["left_arm"], target_l)

        if single_arm_mode:
            self.action_target["left_arm"] = joints_l
            self.action_target["left_gripper"] = target_pose.get("left_gripper")
        else:
            target_r = target_pose["right_arm_ee2rb"]
            joints_r = self._inverse_kin_silent(self.kin_right, joints["right_arm"], target_r)
            self.set_target_JP(
                joints_l, joints_r,
                target_pose.get("left_gripper"),
                target_pose.get("right_gripper"),
            )

    @staticmethod
    def _inverse_kin_silent(kin: SingleArm, current_joints, target):
        """静默版 IK, 屏蔽 pykin 的迭代日志."""
        with contextlib.redirect_stdout(io.StringIO()):
            return kin.inverse_kin(np.asarray(current_joints, dtype=float),
                                   target, method="LM", max_iter=100)

    def execute(self):
        for name, joints in self.action_target.items():
            if joints is not None:
                self.robot.set_joint_angle(name, joints)

    def stop(self):
        if hasattr(self.robot, "_robot") and hasattr(self.robot._robot, "shutdown"):
            self.robot._robot.shutdown()
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        for f in getattr(self, "_temp_urdf_files", []):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass

    def __del__(self):
        self._cleanup_temp_files()


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    rc = RobotControl(vel_max=0.08)

    print("\n=== Joints ===")
    js = rc.get_robot_joints()
    for k, v in js.items():
        print(f"  {k}: {np.round(v, 4)}")

    print("\n=== EE Pose ===")
    ee = rc.get_ee_pose()
    for k in ["left_arm_ee2rb", "right_arm_ee2rb"]:
        v = ee[k]
        print(f"  {k}:")
        print(f"    pos:  {np.round(v[:3], 4)}")
        print(f"    quat: {np.round(v[3:], 4)} (wxyz)")
        print(f"    ||quat||: {np.linalg.norm(v[3:]):.6f}")
        print(f"    distance from base: {np.linalg.norm(v[:3]):.4f} m")

    print("\n=== IK ↔ FK self-consistency ===")
    target = ee["left_arm_ee2rb"].copy()
    js_before = np.asarray(js["left_arm"], dtype=float)

    rc.set_target_CP({
        "left_arm_ee2rb":  target,
        "right_arm_ee2rb": ee["right_arm_ee2rb"],
        "left_gripper":  js["left_gripper"],
        "right_gripper": js["right_gripper"],
    })
    js_solved = np.asarray(rc.action_target["left_arm"], dtype=float)
    diff = js_solved - js_before

    print(f"  current joints: {np.round(js_before, 4)}")
    print(f"  IK solved:      {np.round(js_solved, 4)}")
    print(f"  max abs diff:   {np.max(np.abs(diff)):.6f} rad")

    if np.max(np.abs(diff)) < 0.001:
        print("  ✓ IK ↔ FK 自洽")
    else:
        print("  ✗ FAILED — IK 和 FK 坐标系约定不一致")