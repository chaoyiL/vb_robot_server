# 修复 Python 3.10+ 中 collections.Iterable 的兼容性问题
import collections
import collections.abc
# 为了兼容旧版本的库，需要将 collections.abc.Iterable 添加到 collections 模块
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable

import numpy as np

from pykin.robots.single_arm import SingleArm
from pykin.kinematics import transform as t_utils
from pykin.utils import plot_utils as p_utils

urdf_path = "/home/rvsa/codehub/VB-vla/real_world/robot_api/assets/arm_left.urdf"

robot = SingleArm(urdf_path, t_utils.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

init_thetas = np.random.randn(robot.arm_dof)
target_thetas = init_thetas + np.random.randn(robot.arm_dof) * 0.001
np.set_printoptions(precision=3, suppress=True)

robot.setup_link_name("left-link_arm_base", "left-link_arm_7")
print("Initial Thetas: ", init_thetas)
print("Target Thetas: ", target_thetas)

robot.set_transform(target_thetas)
# _, ax = p_utils.init_3d_figure("FK")
# p_utils.plot_robot(ax=ax, robot=robot, geom="visual", only_visible_geom=True)

fk = robot.forward_kin(target_thetas)
target_pose = robot.compute_eef_pose(fk)

print("Target Pose: \n", target_pose)
joints = robot.inverse_kin(init_thetas, target_pose, method="NR")
print("IK Thetas: ", joints)

robot.set_transform(joints)
target_pose_ik = robot.compute_eef_pose(robot.forward_kin(joints))
print("Target Pose from IK: \n", target_pose_ik)
# _, ax = p_utils.init_3d_figure("IK")
# p_utils.plot_robot(ax=ax, robot=robot, geom="visual", only_visible_geom=True)
# p_utils.show_figure()