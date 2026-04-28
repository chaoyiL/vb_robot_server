from typing import Optional, List
import pathlib
import numpy as np
import time
import shutil
import math

from multiprocessing.managers import SharedMemoryManager
# from real_world.rokae.rokae_interpolation_controller import RokaeInterpolationController
# from real_world.pgi.pgi_controller import PGIController
from real_world.robot_api.arm.Controller import Controller
from real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder

from utils.common.cv2_util import get_image_transform

from utils.interpolation_util import get_interp1d, PoseInterpolator
from utils.pose_util import pose_to_mat, mat_to_pose, pose_to_pos_quat, pos_quat_to_pose
from utils.cv_util import draw_fisheye_mask

import cv2
import time

from real_world.robot_api.arm.RobotControl_pykin import RobotControl

class BimanualUmiEnv:
    def __init__(self, 
            # required params
            cam_path=None,
            data_type='vision',
            fps_num_points=256,
            control_frequency=10,
            controller_frequency=100,
            obs_image_resolution=(224,224),

            max_obs_buffer_size=60,
            obs_float32=False,
            camera_obs_latency=0.125,

            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,

            use_fisheye_mask=False,
            fisheye_mask_radius=400,
            fisheye_mask_center=None,
            fisheye_mask_fill_color=(0, 0, 0),

            shm_manager=None,
            quest_2_ee_left:np.ndarray = None,
            quest_2_ee_right:np.ndarray = None,
            width_slope:float = None,
            width_offset:float = None,
            vel_max:float = None,
            single_arm_mode:bool = False,
            ):

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        resolution = list()
        capture_fps = list()
        cap_buffer_size = list()
        video_recorder = list()
        transform = list()

        for idx in range(len(cam_path)):
            res = (3840, 800)
            fps = 20
            buf = 3
            bit_rate = 6000*1000
            
            # 创建transform函数，处理3840x800图像的裁剪和resize
            def create_transform_func(idx, res, data_type, obs_image_resolution, obs_float32):
                is_right = (idx == 1)  # idx=0是left hand
                
                def tf4k(data, input_res=res,
                        use_mask=use_fisheye_mask,
                        mask_radius=fisheye_mask_radius,
                        mask_center=fisheye_mask_center,
                        mask_fill_color=fisheye_mask_fill_color):
                    img = data['color']  # 3840x800 BGR图像

                    # Apply mask to all visual cameras (non-tactile mode has only visual cameras)
                    if use_mask:
                        # Apply mask before resize, consistent with training data processing order
                        img = draw_fisheye_mask(
                            img, 
                            radius=mask_radius,
                            center=mask_center,
                            fill_color=mask_fill_color
                        )
                    
                    # 裁剪参数
                    CROP_WIDTH = 1280
                    TOTAL_WIDTH = 3840
                    
                    # 验证图像尺寸
                    h, w = img.shape[:2]
                    if w != TOTAL_WIDTH or h != 800:
                        raise ValueError(f"Expected image size {TOTAL_WIDTH}x800, got {w}x{h}")
                    
                    # 裁剪成三部分
                    left_tactile = img[:, 0:CROP_WIDTH]           # 左侧触觉 (1280x800)
                    visual = img[:, CROP_WIDTH:2*CROP_WIDTH]      # 中间视觉 (1280x800)
                    right_tactile = img[:, 2*CROP_WIDTH:3*CROP_WIDTH]  # 右侧触觉 (1280x800)

                    # Process
                    left_tactile = cv2.rotate(left_tactile, cv2.ROTATE_180)
                    visual = visual
                    right_tactile = right_tactile
                    
                    # # left hand的visual旋转180度
                    # if is_right:
                    #     visual = cv2.rotate(visual, cv2.ROTATE_180)
                    
                    # Resize函数
                    f = get_image_transform(
                        input_res=(CROP_WIDTH, 800),  # visual部分是1280x800
                        output_res=obs_image_resolution, 
                        bgr_to_rgb=True)
                    
                    # 处理visual（总是需要）
                    visual_resized = f(visual)
                    if obs_float32:
                        visual_resized = visual_resized.astype(np.float32) / 255
                    data['color'] = visual_resized  # 统一存为color
                    
                    # 根据data_type决定是否处理tactile
                    if data_type == 'vitac':
                        left_tactile_resized = f(left_tactile)
                        right_tactile_resized = f(right_tactile)
                        if obs_float32:
                            left_tactile_resized = left_tactile_resized.astype(np.float32) / 255
                            right_tactile_resized = right_tactile_resized.astype(np.float32) / 255
                        data['left_tactile'] = left_tactile_resized
                        data['right_tactile'] = right_tactile_resized
                    
                    return data
                
                return tf4k
            
            transform.append(create_transform_func(idx, res, data_type, obs_image_resolution, obs_float32))
        
            resolution.append(res)
            capture_fps.append(fps)
            cap_buffer_size.append(buf)
            video_recorder.append(VideoRecorder.create_hevc_nvenc(  # TODO: why use hevc
                fps=fps,
                input_pix_fmt='bgr24',
                bit_rate=bit_rate
            ))

        camera = MultiUvcCamera(
            dev_video_paths=cam_path,
            shm_manager=shm_manager,
            resolution=resolution,
            capture_fps=capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            get_max_k=max_obs_buffer_size,
            receive_latency=camera_obs_latency,
            cap_buffer_size=cap_buffer_size,
            transform=transform,
            video_recorder=video_recorder,
            verbose=False
        )

        self.camera = camera
        self.single_arm_mode = single_arm_mode
        self.controller = Controller(shm_manager=shm_manager, 
            vel_max=vel_max,
            frequency=controller_frequency,
            receive_latency=camera_obs_latency,
            single_arm_mode=self.single_arm_mode,
        )
        self.quest_2_ee_left = quest_2_ee_left
        self.quest_2_ee_right = quest_2_ee_right
        self.width_slope = width_slope
        self.width_offset = width_offset
        self.data_type = data_type
        self.cam_path = cam_path
        self.control_frequency = control_frequency
        self.last_camera_data = None

        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        self._last_log_time = {}

    def _rate_limited_log(self, key: str, message: str, interval_sec: float = 2.0) -> None:
        now = time.monotonic()
        last_log_time = self._last_log_time.get(key, 0.0)
        if now - last_log_time >= interval_sec:
            print(message)
            self._last_log_time[key] = now

    # ======== start-stop API =============
    #### 待修改
    @property
    def is_ready(self):
        ready_flag_camera = self.camera.is_ready
        ready_flag = ready_flag_camera and self.controller.is_ready
        return ready_flag
        
    def start(self, wait=True):
        self.camera.start(wait=False)
        self.controller.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.camera.stop(wait=False)
        self.controller.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.camera.start_wait()
        self.controller.start_wait()
    
    def stop_wait(self):
        self.camera.stop_wait()
        self.controller.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ==========
    def get_obs(self) -> dict:
        """
        Timestamp alignment policy
        We assume the cameras used for obs are always [0, k - 1], where k is the number of robots
        All other cameras, find corresponding frame with the nearest timestamp
        All low-dim observations, interpolate with respect to 'current' time
        """

        "observation dict"
        assert self.is_ready

        # get data
        # 60 Hz, camera_calibrated_timestamp (note: cameras capture at 30Hz, but using 60Hz for interpolation)
        k = math.ceil(
            self.camera_obs_horizon * self.camera_down_sample_steps \
            * (20 / self.control_frequency)) + 2 # here 2 is adjustable, typically 1 should be enough
        # print('==>k  ', k, self.camera_obs_horizon, self.camera_down_sample_steps, self.control_frequency)

        'camera obs'
        self.last_camera_data = self.camera.get(
            k=k, 
            out=self.last_camera_data)
        # print("camera get time:", time.time() - start_time)

        # select align_camera_idx based on calibrated timestamps
        # The timestamps are already calibrated in UvcCamera, so we use them directly
        num_obs_cameras = len(self.cam_path)
        align_camera_idx = None
        running_best_error = np.inf

        for camera_idx in range(num_obs_cameras):
            this_error = 0
            this_timestamp = self.last_camera_data[camera_idx]['timestamp'][-1]
            for other_camera_idx in range(num_obs_cameras):
                if other_camera_idx == camera_idx:
                    continue
                other_timestep_idx = -1
                while True:
                    if self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx] < this_timestamp:
                        this_error += this_timestamp - self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx]
                        break
                    other_timestep_idx -= 1
            if align_camera_idx is None or this_error < running_best_error:
                running_best_error = this_error
                align_camera_idx = camera_idx

        last_timestamp = self.last_camera_data[align_camera_idx]['timestamp'][-1]
        
        dt = 1 / self.control_frequency

        # align camera obs timestamps
        # Since timestamps are already calibrated in UvcCamera, we can use them directly
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)

        camera_obs = dict()
        for camera_idx, value in self.last_camera_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in camera_obs_timestamps:
                nn_idx = np.argmin(np.abs(this_timestamps - t))
                # Optional: Add warning for large timestamp mismatches
                # if np.abs(this_timestamps - t)[nn_idx] > 1.0 / 60:
                #     print(f'WARNING: Large timestamp mismatch for camera {camera_idx}: {np.abs(this_timestamps - t)[nn_idx]:.4f}s')
                this_idxs.append(nn_idx)
            # remap key - 简化逻辑
            # camera_idx=0是left hand, camera_idx=1是right hand
            hand_idx = camera_idx  # 0=left hand (camera0), 1=right hand (camera1)

            # 提取visual (总是存在，存为color)
            camera_obs[f'camera{hand_idx}_rgb'] = value['color'][...,:3][this_idxs]

            # 如果是vitac模式，还需要提取tactile
            if self.data_type == 'vitac':
                if 'left_tactile' in value:
                    camera_obs[f'camera{hand_idx}_left_tactile'] = value['left_tactile'][...,:3][this_idxs]
                if 'right_tactile' in value:
                    camera_obs[f'camera{hand_idx}_right_tactile'] = value['right_tactile'][...,:3][this_idxs]

        '''robot obs'''
        last_robot_data = self.controller.get_all_state()

        # obs_data to return (it only includes camera data at this stage)
        obs_data = dict(camera_obs)
        # include camera timesteps
        obs_data['timestamp'] = camera_obs_timestamps

        # align robot obs
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)

        # convert ee pose to quest pose
        quest_pose_left = mat_to_pose(pose_to_mat(last_robot_data['ee_pose_left']) @ self.quest_2_ee_left)
        robot_pose_left_interpolator = PoseInterpolator(
            t = last_robot_data['robot_timestamp'],
            x = quest_pose_left
        )
        robot_pose_left = robot_pose_left_interpolator(robot_obs_timestamps)
        if not self.single_arm_mode:
            quest_pose_right = mat_to_pose(pose_to_mat(last_robot_data['ee_pose_right']) @ self.quest_2_ee_right)
            robot_pose_right_interpolator = PoseInterpolator(
                t = last_robot_data['robot_timestamp'],
                x = quest_pose_right
            )
            robot_pose_right = robot_pose_right_interpolator(robot_obs_timestamps)
        else:
            robot_pose_right = None
        
        robot_obs = {
            'robot0_eef_pos': robot_pose_left[...,:3],
            'robot0_eef_rot_axis_angle': robot_pose_left[...,3:],
            'robot1_eef_pos': robot_pose_right[...,:3] if robot_pose_right is not None else None,
            'robot1_eef_rot_axis_angle': robot_pose_right[...,3:] if robot_pose_right is not None else None
        }
        obs_data.update(robot_obs)

        '''gripper obs'''
        # align gripper obs
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)

        # convert commanded gripper width to actual gripper width
        commanded_gripper_width_left = last_robot_data['gripper_pose_left']
        actual_gripper_width_left = self.width_slope * commanded_gripper_width_left + self.width_offset
        gripper_left_interpolator = get_interp1d(
            t= last_robot_data['robot_timestamp'],
            x= actual_gripper_width_left
        )
        gripper_left = gripper_left_interpolator(gripper_obs_timestamps)
        if not self.single_arm_mode:
            commanded_gripper_width_right = last_robot_data['gripper_pose_right']
            actual_gripper_width_right = self.width_slope * commanded_gripper_width_right + self.width_offset
            gripper_right_interpolator = get_interp1d(
                t= last_robot_data['robot_timestamp'],
                x= actual_gripper_width_right
            )
            gripper_right = gripper_right_interpolator(gripper_obs_timestamps)
        else:
            gripper_right = None

        gripper_obs = {
            'robot0_gripper_width': gripper_left,
            'robot1_gripper_width': gripper_right if gripper_right is not None else None
        }
        obs_data.update(gripper_obs)

        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray):
        
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # # 更新动作序列，确保全都是新动作
        # receive_time = time.time()
        # is_new = timestamps > receive_time
        # new_actions = actions[is_new]
        # new_timestamps = timestamps[is_new]

        # print(f"[env] exec {len(new_actions)}/{len(actions)} actions")
        # print("[env] receive_time:", int(receive_time * 1000) % 1000)
        # print("[env] new_timestamps:")
        # print([int(new_timestamps[i] * 1000) % 1000 for i in range(len(new_timestamps))])

        if len(actions) != 0:
            for i in range(len(actions)):
                index_left = 0
                index_right = 1

                quest_left_action = actions[i, 7 * index_left + 0: 7 * index_left + 6]
                gripper_left_action = actions[i, 7 * index_left + 6]
                # convert quest pose to ee pose
                target_ee_pose_left = mat_to_pose(pose_to_mat(quest_left_action) @ (np.linalg.inv(self.quest_2_ee_left)))
                # convert actual gripper width to commanded gripper width
                commanded_gripper_width_left = [(gripper_left_action - self.width_offset) / self.width_slope]
                if not self.single_arm_mode:
                    quest_right_action = actions[i, 7 * index_right + 0: 7 * index_right + 6]
                    gripper_right_action = actions[i, 7 * index_right + 6]
                    target_ee_pose_right = mat_to_pose(pose_to_mat(quest_right_action) @ (np.linalg.inv(self.quest_2_ee_right)))
                    commanded_gripper_width_right = [(gripper_right_action - self.width_offset) / self.width_slope]
                else:
                    target_ee_pose_right = None
                    commanded_gripper_width_right = None

                # DEBUG: 在这里检查 quest 相对 当前动作 的 dist
                # 使用 get_obs 获取机械臂当前位姿
                # curr_obs = self.get_obs()
                # curr_pos_left = curr_obs['robot0_eef_pos'][-1]
                # curr_rot_left = curr_obs['robot0_eef_rot_axis_angle'][-1]
                # curr_pos_right = curr_obs['robot1_eef_pos'][-1]
                # curr_rot_right = curr_obs['robot1_eef_rot_axis_angle'][-1]
                # curr_pose_left_mat = pose_to_mat(np.concatenate([curr_pos_left, curr_rot_left], axis=-1))
                # curr_pose_right_mat = pose_to_mat(np.concatenate([curr_pos_right, curr_rot_right], axis=-1))
                # 获取相对位姿
                # quest_left_action_mat = pose_to_mat(quest_left_action)
                # quest_right_action_mat = pose_to_mat(quest_right_action)
                # quest_left_action_rel_mat = np.linalg.inv(curr_pose_left_mat) @ quest_left_action_mat
                # quest_right_action_rel_mat = np.linalg.inv(curr_pose_right_mat) @ quest_right_action_mat
                # # 计算并输出Quest坐标系下的相对位姿
                # dist_left = np.linalg.norm(quest_left_action_rel_mat[:3, 3])
                # dist_right = np.linalg.norm(quest_right_action_rel_mat[:3, 3])
                # print(f"[env] #{i} action dist_left: {dist_left}, dist_right: {dist_right}")

                # HACK: 将 target_ee_pose_left 改为左手在curr_pose的基础上增加x，右手不动
                # curr_ee_pose_left_mat = curr_pose_left_mat @ (np.linalg.inv(self.quest_2_ee_left))
                # curr_ee_pose_right_mat = curr_pose_right_mat @ (np.linalg.inv(self.quest_2_ee_right))
                # target_ee_pose_left = mat_to_pose(curr_ee_pose_left_mat)
                # target_ee_pose_left[0] += 0.005
                # target_ee_pose_right = mat_to_pose(curr_ee_pose_right_mat)

                # 把目标动作序列 依次 发送给控制器
                if timestamps[i] < time.time():
                    self._rate_limited_log("late_action", "[env] action is too late")
                else:
                    self.controller.schedule_waypoint(
                        pose_left=target_ee_pose_left,
                        gripper_left=commanded_gripper_width_left,
                        pose_right=target_ee_pose_right,
                        gripper_right=commanded_gripper_width_right,
                            target_time=timestamps[i]
                    )
        else:
            self._rate_limited_log("no_action", "[env] no action received")
    
    def get_debug_info(self):
        return self.controller.get_debug_info()

    
