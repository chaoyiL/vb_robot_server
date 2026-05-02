import gc
import sys
import os
dir = os.getcwd()
sys.path.append(dir)

import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

import numpy as np
# from scipy.spatial.transform import Rotation

from policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from real_world.robot_api.arm.RobotControl_pykin import RobotControl

from utils.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from utils.pose_util import mat_to_pose, pose_to_mat, pose_to_pos_quat, pos_quat_to_pose
from utils.precise_sleep import precise_wait

class CustomError(Exception):
    def __init__(self, message):
        self.message = message

# 用于控制机器人状态的指令类，包括停机、行动和 SCHEDULE_WAYPOINT（？）三种状态
class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

class Controller(mp.Process):
    def __init__(self,
                shm_manager: SharedMemoryManager, # 多进程控制
                launch_timeout = 3,
                verbose = False,

                vel_max:float = 0.1,

                frequency : int = 100,
                get_max_k : int = None,
                max_pos_speed : float = 0.25,
                max_rot_speed : float = 0.16,
                receive_latency : float = 0.0,
                single_arm_mode: bool = False,
                ):
        
        super().__init__(name="arm_controller") # 直接调用父类 mp.Process 的初始化函数，初始化该进程

        # 进程参数初始化
        self.verbose = verbose # 用来控制是否输出调试信息的变量
        self.launch_timeout = launch_timeout # 用来控制进程开始前的等待时间
        self.single_arm_mode = single_arm_mode

        # 控制参数初始化
        self.frequency = frequency

        if get_max_k is None:
            get_max_k = int(frequency * 5)
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.receive_latency = receive_latency

        # 创建用于储存控制信息的队列，控制信息格式如example所示
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose_left': np.zeros((6,), dtype=np.float64),
            'target_pose_right': np.zeros((6,), dtype=np.float64),
            'target_gripper_left': np.zeros((1,), dtype=np.float64),
            'target_gripper_right': np.zeros((1,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }

        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=int(512 * 1e5)
        )

        # 创建用于储存反馈信息的队列，反馈信息内容及格式如example所示
        example = {
            'ee_pose_left': np.zeros((6,), dtype=np.float64),
            'ee_pose_right': np.zeros((6,), dtype=np.float64),
            'gripper_pose_left': np.zeros((1,), dtype=np.float64),
            'gripper_pose_right': np.zeros((1,), dtype=np.float64),
            'robot_receive_timestamp': time.time(),
            'robot_timestamp': time.time()
        }

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # DEBUG BUFFER
        if not self.single_arm_mode:
            self.side = ["left", "right"]
        else:
            self.side = ["left"]
        self.para = ["x", "y", "z", "rx", "ry", "rz", "g"]
        example = dict()
        for side in self.side:
            for para in self.para:
                example[f"ee_pose_{side}_{para}"] = 0.0
                example[f"target_pose_{side}_{para}"] = 0.0
        example["time"] = 0.0

        self.input_queue_debug = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=int(10 * 1024 * 1024)
        )

        # Pre-compute debug keys to avoid string formatting in the hot loop
        self._debug_keys_per_side = {
            side: [(f"ee_pose_{side}_{para}", f"target_pose_{side}_{para}") for para in self.para]
            for side in self.side
        }
        self._debug_message = {"time": 0.0}
        for side in self.side:
            for ee_key, tgt_key in self._debug_keys_per_side[side]:
                self._debug_message[ee_key] = 0.0
                self._debug_message[tgt_key] = 0.0

        # 一些变量赋值
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.vel_max = vel_max
        self.single_arm_mode = single_arm_mode
        
        #Downsample: protect loop interval
        self._debug_downsample = 200  # write debug at ~33 Hz instead of every iteration

    '''===进程控制==='''

    # 进程控制函数（开始/结束）
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[ArmController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value # ？
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # 进程确认函数（等待本进程启动/等待子进程结束）
    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)    # Block until ready_event is set or timeout
        assert self.is_alive()                        # Ensure controller process or thread is started and active
    
    def stop_wait(self):
        self.join()
    
    # 返回该进程是否已准备好
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    '''===主要功能API：控制机器人运动==='''
    
    # 安排经过点？
    def schedule_waypoint(self, pose_left:list, pose_right:list, gripper_left:list, gripper_right:list, target_time:float):

        pose_left = np.array(pose_left)
        assert pose_left.shape == (6,)
        gripper_left = np.array(gripper_left)
        assert gripper_left.shape == (1,)

        if not self.single_arm_mode:
            pose_right = np.array(pose_right)
            assert pose_right.shape == (6,)
            gripper_right = np.array(gripper_right)
            assert gripper_right.shape == (1,)
        else:
            # only for occupancy check, never used
            pose_right = np.zeros((6,))
            gripper_right = np.zeros((1,))

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose_left': pose_left,
            'target_pose_right': pose_right,
            'target_gripper_left': gripper_left,
            'target_gripper_right': gripper_right,
            'target_time': target_time
        }

        self.input_queue.put(message)
    
    def renew_debug_buffer(self, message):
        self.input_queue_debug.put(message)

    '''===主要功能API：获取机器人状态；状态将在控制主循环中上传==='''

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    def get_debug_info(self):
        return self.input_queue_debug.get_all()
    
    def run(self):

        robot_control = RobotControl(
            vel_max=self.vel_max
        )

        try:
            if self.verbose:
                print(f"[PositionalController] Connect to robot")

            dt_controller = 1. / self.frequency

            '''INITIALIZE'''
            # initialize time and last waypoint time
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            curr_ee_pose = robot_control.get_ee_pose()

            # get initial ee pose and gripper pose
            curr_ee_pos_quat_left = curr_ee_pose["left_arm_ee2rb"]
            curr_ee_pose_left = pos_quat_to_pose(curr_ee_pos_quat_left[:3], curr_ee_pos_quat_left[3:])
            target_gripper_left = curr_ee_pose["left_gripper"] # pre-set gripper pose, because we won't use interpolator for grippers

            # initialize pose interpolator
            pose_interp_left = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_ee_pose_left]
            )

            if not self.single_arm_mode:
                curr_ee_pos_quat_right = curr_ee_pose["right_arm_ee2rb"]
                curr_ee_pose_right = pos_quat_to_pose(curr_ee_pos_quat_right[:3], curr_ee_pos_quat_right[3:])
                target_gripper_right = curr_ee_pose["right_gripper"] 
                pose_interp_right = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_ee_pose_right]
            )
            else:
                curr_ee_pos_quat_right = None
                curr_ee_pose_right = None
                target_gripper_right = None
                pose_interp_right = None

            '''MAIN LOOP'''
            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            cmd_pose = None
            gc_interval = self.frequency  # collect once per second
            gc.disable()

            while keep_running:
                t_now = time.monotonic()

                # DEBUG
                # get current ee pose and gripper pose
                curr_ee_pose = robot_control.get_ee_pose()

                curr_ee_pos_quat_left = curr_ee_pose["left_arm_ee2rb"]
                curr_gripper_left = curr_ee_pose["left_gripper"]
                curr_ee_pose_left = pos_quat_to_pose(curr_ee_pos_quat_left[:3], curr_ee_pos_quat_left[3:])
                if not self.single_arm_mode:
                    curr_ee_pos_quat_right = curr_ee_pose["right_arm_ee2rb"]
                    curr_gripper_right = curr_ee_pose["right_gripper"]
                    curr_ee_pose_right = pos_quat_to_pose(curr_ee_pos_quat_right[:3], curr_ee_pos_quat_right[3:])
                else:
                    curr_ee_pos_quat_right = None
                    curr_gripper_right = None
                    curr_ee_pose_right = None

                # update robot state to ring buffer
                t_recv = time.time()
                state = {
                    'ee_pose_left': curr_ee_pose_left,
                    'ee_pose_right': curr_ee_pose_right,
                    'gripper_pose_left': curr_gripper_left,
                    'gripper_pose_right': curr_gripper_right,
                    'robot_receive_timestamp': t_recv,
                    'robot_timestamp': t_recv - self.receive_latency,
                }
                self.ring_buffer.put(state)
                
                # get target ee mat (cache interpolation result for reuse in debug)
                target_pose_interp_left = pose_interp_left(t_now)
                target_ee_pos_quat_left = pose_to_pos_quat(target_pose_interp_left)
                if not self.single_arm_mode:
                    target_pose_interp_right = pose_interp_right(t_now)
                    target_ee_pos_quat_right = pose_to_pos_quat(target_pose_interp_right)
                else:
                    target_pose_interp_right = None
                    target_ee_pos_quat_right = None
                
                if cmd_pose is not None:
                    target_gripper_left = cmd_pose["left_gripper"]
                    if not self.single_arm_mode:    
                        target_gripper_right = cmd_pose["right_gripper"]
                    else:
                        target_gripper_right = None

                # set target pose and execute robot
                target_pose = {
                    "left_arm_ee2rb": target_ee_pos_quat_left,
                    "right_arm_ee2rb": target_ee_pos_quat_right,
                    "left_gripper": target_gripper_left,
                    "right_gripper": target_gripper_right,
                }

                robot_control.set_target_CP(target_pose, single_arm_mode = self.single_arm_mode)
                robot_control.execute()
                
                # DEBUG: throttled to reduce hot-loop overhead
                if iter_idx % self._debug_downsample == 0:
                    for side in self.side:
                        if side == "left":
                            _ee_pose = curr_ee_pose_left
                            _gripper = curr_gripper_left
                            _interp_result = target_pose_interp_left
                            _gripper_target = target_gripper_left
                        else:
                            _ee_pose = curr_ee_pose_right
                            _gripper = curr_gripper_right
                            _interp_result = target_pose_interp_right
                            _gripper_target = target_gripper_right

                        keys = self._debug_keys_per_side[side]
                        for para_idx, para in enumerate(self.para):
                            ee_key, tgt_key = keys[para_idx]
                            if para == "g":
                                self._debug_message[ee_key] = _gripper
                                self._debug_message[tgt_key] = _gripper_target
                            else:
                                self._debug_message[ee_key] = _ee_pose[para_idx]
                                self._debug_message[tgt_key] = _interp_result[para_idx]

                    self._debug_message["time"] = t_now - t_start
                    self.renew_debug_buffer(self._debug_message)
                
                # Get a command from input queue. The period of getting command is dt_controller
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                except Exception:
                    n_cmd = 0

                # If commands are received, put it into the interpolator
                for i in range(n_cmd):
                    command = {key: value[i] for key, value in commands.items()}
                    cmd = command['cmd']

                    if cmd == Command.SCHEDULE_WAYPOINT.value:
                        cmd_pose = {
                            "left_arm_ee2rb": command['target_pose_left'],
                            "right_arm_ee2rb": command['target_pose_right'],
                            "left_gripper": command['target_gripper_left'],
                            "right_gripper": command['target_gripper_right'],
                        }

                        # The timestamp of the received single frame target action is global time
                        target_time = float(command['target_time'])

                        # Convert global time to monotonic time, for subsequent interpolation
                        target_time = time.monotonic() - time.time() + target_time
                        # The time at the end of the loop (start time + single loop duration),
                        curr_time = t_now + dt_controller

                        if target_time <= curr_time:
                            print("[controller] action is too late")
                        else:
                            print("[controller] target_time, curr_time:", target_time, curr_time)
                            print("[controller] time :", target_time - curr_time)

                        # Interpolator: If the target time is behind the current time, no action is executed
                        pose_interp_left = pose_interp_left.schedule_waypoint(
                            pose=cmd_pose["left_arm_ee2rb"],
                            time=target_time,
                            # max_pos_speed=self.max_pos_speed,
                            # max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time,
                        )
                        if not self.single_arm_mode:
                            pose_interp_right = pose_interp_right.schedule_waypoint(
                                pose=cmd_pose["right_arm_ee2rb"],
                                time=target_time, #ACTION TIME
                                # max_pos_speed=self.max_pos_speed,
                                # max_rot_speed=self.max_rot_speed,
                                curr_time=curr_time, # CURRENT TIME
                                last_waypoint_time=last_waypoint_time,
                            )
                        # Update the latest target time
                        last_waypoint_time = target_time

                    else:
                        keep_running = False
                        break

                # regulate frequency with absolute time grid + precise sleep
                t_cycle_end = t_start + (iter_idx + 1) * dt_controller
                if time.monotonic() < t_cycle_end:
                    precise_wait(t_cycle_end)
                else:
                    print("[controller] loop speed error, please slow down the controller frequency")
                    t_start = time.monotonic() - (iter_idx + 1) * dt_controller

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # deterministic GC: collect during the sleep window to avoid random pauses
                if iter_idx % gc_interval == 0:
                    gc.collect()

        except Exception as e:
            print(f"Exception occurred: {e}")
            raise
        finally:
            gc.enable()
            print('\n\n\n\nterminate_current_policy\n\n\n\n\n')

            robot_control.stop()
            del robot_control
            self.ready_event.set() # in case of exception before first loop