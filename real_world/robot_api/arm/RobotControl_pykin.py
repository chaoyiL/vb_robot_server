import sys
import os
from time import time
from pathlib import Path
dir = os.getcwd()
sys.path.append(dir)

import numpy as np
import time
import io
import contextlib

from real_world.robot_api.arm.RobotWrapper import RobotWrapper
from utils.pose_util import pose_to_mat, mat_to_pose
# 修复 Python 3.10+ 中 collections.Iterable 的兼容性问题
import collections
import collections.abc
# 为了兼容旧版本的库，需要将 collections.abc.Iterable 添加到 collections 模块
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable

from pykin.robots.single_arm import SingleArm
from pykin.robots.robot import Robot
from pykin.kinematics import transform as t_utils

import transforms3d as t3d
import tempfile
import re
import shutil
import pykin

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]


def _resolve_repo_path(path_like: str) -> str:
    """Resolve a repo-relative path to an absolute path."""
    path = Path(path_like)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def preprocess_urdf(urdf_path: str) -> str:
    """
    Preprocess URDF file to replace ROS package:// URIs with relative paths.
    Returns path to the preprocessed URDF file (temporary file).
    """
    # Read the original URDF
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    
    # Get the directory containing the URDF
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    # Get the parent directory (where meshes/ folder is located)
    parent_dir = os.path.dirname(urdf_dir)
    
    # Extract package name from path (e.g., ARM-LEFT-GR-0 or ARM-RIGHT-GR-0)
    # The package name is typically the directory name containing the urdf folder
    package_name = os.path.basename(parent_dir)
    
    # Replace all package://PACKAGE_NAME/ references with relative paths
    # Since meshes/ and other resources are at the same level as urdf/, use ../
    escaped_package_name = re.escape(package_name)
    # Replace package://PACKAGE_NAME/ with ../ (relative to urdf/ directory)
    urdf_content = re.sub(f'package://{escaped_package_name}/', '../', urdf_content)
    
    # Create a temporary file for the preprocessed URDF
    temp_fd, temp_path = tempfile.mkstemp(suffix='.urdf', prefix='preprocessed_', dir=urdf_dir)
    try:
        with os.fdopen(temp_fd, 'w') as f:
            f.write(urdf_content)
        return temp_path
    except Exception as e:
        os.close(temp_fd)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def prepare_urdf_for_pykin(urdf_path: str) -> tuple[str, str]:
    """
    Prepare URDF for pykin and return:
    1) path relative to pykin/assets (for SingleArm input)
    2) temp file absolute path (for cleanup)
    """
    src_urdf = Path(urdf_path).resolve()
    package_dir = src_urdf.parent.parent
    package_name = package_dir.name

    pykin_assets_dir = Path(pykin.__file__).resolve().parent / "assets"
    stage_root = pykin_assets_dir / "vb_vla_assets"
    stage_pkg_dir = stage_root / package_name
    stage_urdf = stage_pkg_dir / "urdf" / src_urdf.name

    stage_root.mkdir(parents=True, exist_ok=True)
    if not stage_pkg_dir.exists():
        try:
            # Prefer symlink to avoid copying heavy mesh files.
            stage_pkg_dir.symlink_to(package_dir, target_is_directory=True)
        except OSError:
            shutil.copytree(package_dir, stage_pkg_dir)

    if not stage_urdf.exists():
        raise FileNotFoundError(f"Staged URDF not found: {stage_urdf}")

    preprocessed_abs = Path(preprocess_urdf(str(stage_urdf))).resolve()
    try:
        rel_to_pykin_assets = str(preprocessed_abs.relative_to(pykin_assets_dir))
    except ValueError:
        # If stage_pkg_dir is a symlink, the temp URDF may resolve outside pykin/assets.
        # pykin still accepts a relative path containing ".." segments.
        rel_to_pykin_assets = os.path.relpath(str(preprocessed_abs), str(pykin_assets_dir))
    return rel_to_pykin_assets, str(preprocessed_abs)


def pos_orn_to_mat(position : list[float, float, float], quaternion : list[float, float, float, float]) -> np.ndarray:
    # Create rotation matrix from quaternion

    rotation_matrix = t3d.quaternions.quat2mat(quaternion)
    
    # Create 4x4 transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = position
    
    return matrix

def mat_to_pos_orn(matrix : np.ndarray) -> tuple[list[float, float, float], list[float, float, float, float]]:
    # Extract position
    position = matrix[:3, 3]
    
    # Extract rotation matrix and convert to quaternion
    rotation_matrix = matrix[:3, :3]
    quaternion = t3d.quaternions.mat2quat(rotation_matrix)  # x, y, z, w format

    return position, quaternion

class RobotControl:
    def __init__(self,
    vel_max: float = None,
    urdf_path_left = "assets/ARM-LEFT-GR-0/urdf/ARM-LEFT-GR-0.urdf",
    urdf_path_right = "assets/ARM-RIGHT-GR-0/urdf/ARM-RIGHT-GR-0.urdf"
    ):
        # class for robot control
        self.robot = RobotWrapper(vel_max=vel_max)

        # buffer for action target
        self.action_target = dict[str, None](left_arm=None, right_arm=None, left_gripper=None, right_gripper=None)

        left_base_pos = [0, 0.116, 0]
        right_base_pos = [0, -0.116, 0]
        left_base_euler = [-np.pi/2, 0, 0]
        right_base_euler = [np.pi/2, 0, 0]

        # arm base to robot base
        self.ab2rb_left = np.eye(4)
        self.ab2rb_left[:3, 3] = left_base_pos
        self.ab2rb_left[:3, :3] = t3d.euler.euler2mat(*left_base_euler)
        self.ab2rb_right = np.eye(4)
        self.ab2rb_right[:3, 3] = right_base_pos
        self.ab2rb_right[:3, :3] = t3d.euler.euler2mat(*right_base_euler)

        # Preprocess URDF files to replace package:// URIs with relative paths
        urdf_left_for_pykin, urdf_left_tmp = prepare_urdf_for_pykin(_resolve_repo_path(urdf_path_left))
        urdf_right_for_pykin, urdf_right_tmp = prepare_urdf_for_pykin(_resolve_repo_path(urdf_path_right))
        
        # Store temp file paths for cleanup
        self._temp_urdf_files = [urdf_left_tmp, urdf_right_tmp]

        #class for ik/fk
        self.kin_left = SingleArm(urdf_left_for_pykin, t_utils.Transform(rot=t3d.euler.euler2quat(*left_base_euler), pos=self.ab2rb_left[:3, 3]))
        self.kin_right = SingleArm(urdf_right_for_pykin, t_utils.Transform(rot=t3d.euler.euler2quat(*right_base_euler), pos=self.ab2rb_right[:3, 3]))
        self.kin_left.setup_link_name("left-link_arm_base", "left-link_arm_7")
        self.kin_right.setup_link_name("right-link_arm_base", "right-link_arm_7")

    def get_robot_joints(self) -> dict[str, np.ndarray]:
        '''return joint angles of left and right arm, gripper width'''
        joint_cur_left = np.array(self.robot.get_joint_angle("left_arm"))
        joint_cur_right = np.array(self.robot.get_joint_angle("right_arm"))

        gripper_cur_left = np.array(self.robot.get_joint_angle("left_gripper"))
        gripper_cur_right = np.array(self.robot.get_joint_angle("right_gripper"))

        robot_joints: dict[str, np.ndarray] = {
            "left_arm": joint_cur_left,
            "right_arm": joint_cur_right,
            "left_gripper": gripper_cur_left,
            "right_gripper": gripper_cur_right
        }
        return robot_joints
    

    def get_ee_pose(self) -> dict[str, np.ndarray]:
        '''return ee pose of left and right arm, gripper width'''
        
        robot_joints = self.get_robot_joints()

        fk_left = self.kin_left.forward_kin(robot_joints["left_arm"])
        fk_right = self.kin_right.forward_kin(robot_joints["right_arm"])

        ee2rb_pose_left = self.kin_left.compute_eef_pose(fk_left)
        ee2rb_pose_right = self.kin_right.compute_eef_pose(fk_right)

        ee_pose: dict[str, np.ndarray] = {
            "left_arm_ee2rb": ee2rb_pose_left,
            "right_arm_ee2rb": ee2rb_pose_right,
            "left_gripper": robot_joints["left_gripper"],
            "right_gripper": robot_joints["right_gripper"],
        }

        return ee_pose


    def set_target_JP(self, joint_left:np.ndarray, joint_right:np.ndarray=None, gripper_left:np.ndarray=None, gripper_right:np.ndarray=None):
        """
        Set target joint positions for bimanual or single arm robot.
        
        Args:
            joint_left: Left arm joint angles
            joint_right: Right arm joint angles (None for single arm mode)
            gripper_left: Left gripper position
            gripper_right: Right gripper position (None for single arm mode)
        """
        self.action_target["left_arm"] = joint_left
        self.action_target["left_gripper"] = gripper_left
        
        # Only set right arm if provided (bimanual mode)
        if joint_right is not None:
            self.action_target["right_arm"] = joint_right
        if gripper_right is not None:
            self.action_target["right_gripper"] = gripper_right

    def _inverse_kin_silent(self, kin_solver: SingleArm, current_joints: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        # pykin 的 IK 内部会 print 迭代日志，这里静默以避免刷屏。
        with contextlib.redirect_stdout(io.StringIO()):
            return kin_solver.inverse_kin(current_joints, target_pose, method="LM", max_iter=100)


    def set_target_CP(self, target_pose:dict[str, np.ndarray], single_arm_mode:bool=False):
        
        # get current joints
        robot_joints: dict[str, np.ndarray] = self.get_robot_joints()

        # Always compute left arm IK
        ee2ab_target_pose_left = target_pose["left_arm_ee2rb"]
        # quat in pykin is w x y z !!!
        joints_left: np.ndarray = self._inverse_kin_silent(self.kin_left, robot_joints["left_arm"], ee2ab_target_pose_left)
        
        if single_arm_mode:
            # Single arm mode - only set left arm, don't update right arm action_target
            self.action_target["left_arm"] = joints_left
            self.action_target["left_gripper"] = target_pose["left_gripper"]
            # Don't modify right arm targets - they remain None or previous value
        else:
            # Bimanual mode - set both arms
            ee2ab_target_pose_right = target_pose["right_arm_ee2rb"]
            joints_right: np.ndarray = self._inverse_kin_silent(self.kin_right, robot_joints["right_arm"], ee2ab_target_pose_right)
            self.set_target_JP(joints_left, joints_right, target_pose["left_gripper"], target_pose["right_gripper"])


    def execute(self):
        for component_name, target_joints in self.action_target.items():
            if target_joints is not None:
                self.robot.set_joint_angle(component_name, target_joints)


    def stop(self):
        self.robot._robot.shutdown()
        self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Clean up temporary URDF files"""
        if hasattr(self, '_temp_urdf_files'):
            for temp_file in self._temp_urdf_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception:
                    pass  # Ignore errors during cleanup
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self._cleanup_temp_files()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    robot = RobotControl(vel_max=0.08)

    # get current joints
    # curr_joints = robot.get_robot_joints()
    # print(curr_joints)

    save_dir = "real_world/robot_api"
    save_path = os.path.join(save_dir, f"saved_poses.npy")
    os.makedirs(save_dir, exist_ok=True)

    curr_ee_pose = robot.get_ee_pose()


    # If file exists, append to list; else, create new list
    if os.path.exists(save_path):
        existing_data = np.load(save_path, allow_pickle=True)
        if isinstance(existing_data, np.ndarray) and len(existing_data.shape) == 0:
            saved_list = [existing_data.item()]
        else:
            saved_list = list(existing_data)
        saved_list.append(curr_ee_pose)
    else:
        saved_list = [curr_ee_pose]

    print("current ee pose:", curr_ee_pose)
    np.save(save_path, saved_list)
    print(f"Saved current pose data to {save_path}")
