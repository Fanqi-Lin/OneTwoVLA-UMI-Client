"""
Usage:
(umi): python umi_client.py \
--robot_config=example/eval_robots_config.yaml \
-o /path/to/output/dir/ \
--frequency 5 -j \
--temporal_agg -si 1 \
-ins "$instruction" \
-state_horizon 3,15 -action_down_sample_steps 3 -getitem_type necessary \
--remote_port 8000

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly!

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import os

import json
import time
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
import scipy.spatial.transform as st
import yaml
from omegaconf import OmegaConf
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from PIL import ImageDraw
import copy

from umi.common.cv_util import FisheyeRectConverter, parse_fisheye_intrinsics
from umi.common.pose_util import mat_to_pose, pose_to_mat, mat_to_pose10d
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from umi.common.precise_sleep import precise_wait
from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.keystroke_counter import Key, KeyCode, KeystrokeCounter
from umi.real_world.real_inference_util import get_real_umi_action
from umi.real_world.spacemouse_shared_memory import Spacemouse

OmegaConf.register_new_resolver("eval", eval, replace=True)


def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta


def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal

                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))

def get_pi_obs_dict(env_obs, start_image, bbox_list, getitem_type, obs_pose_repr, tx_robot1_robot0, episode_start_pose, instruction):
    low_dim_dict, return_dict = {}, {'prompt': instruction}
    history_length = env_obs['camera0_rgb'].shape[0]

    for i in range(history_length):
        return_dict[f'image_{i + 1}'] = env_obs['camera0_rgb'][i]
    if start_image is not None:
        return_dict['condition'] = {
            'detect': bbox_list,
            'episode_start_image': start_image
        }

    pose_mat = pose_to_mat(np.concatenate([env_obs['robot0_eef_pos'], env_obs['robot0_eef_rot_axis_angle']], axis=-1))
    start_pose = episode_start_pose[0]
    start_pose_mat = pose_to_mat(start_pose)
    rel_obs_pose_mat = convert_pose_mat_rep(
        pose_mat,
        base_pose_mat=start_pose_mat,
        pose_rep=obs_pose_repr,
        backward=False)
    rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
    low_dim_dict['eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

    obs_pose_mat = convert_pose_mat_rep(
        pose_mat, 
        base_pose_mat=pose_mat[-1],
        pose_rep=obs_pose_repr,
        backward=False)

    obs_pose = mat_to_pose10d(obs_pose_mat)

    low_dim_dict['eef_pos'] = obs_pose[:,:3]
    low_dim_dict['eef_rot_axis_angle'] = obs_pose[:,3:]
    low_dim_dict['gripper_width'] = env_obs['robot0_gripper_width']

    # concat low dim features to get state
    # state = np.concatenate([low_dim_dict[key] for key in low_dim_dict.keys()], axis=-1).flatten()
    if getitem_type == 'default':
        key_sequence = ['eef_pos', 'eef_rot_axis_angle', 'eef_rot_axis_angle_wrt_start', 'gripper_width']
        return_dict['state'] = np.concatenate([low_dim_dict[key].flatten() for key in key_sequence], axis=-1).astype(np.float32)
    elif getitem_type == 'necessary':
        return_dict['state'] = np.concatenate([low_dim_dict['eef_pos'][:-1].flatten(), low_dim_dict['eef_rot_axis_angle'][:-1].flatten(), \
                                                                low_dim_dict['eef_rot_axis_angle_wrt_start'][-1], low_dim_dict['gripper_width'].flatten()], axis=-1).astype(np.float32)
    elif getitem_type == 'shortest':
        history_rel_pose = mat_to_pose(obs_pose_mat)[:-1]
        history_rel_pose = np.concatenate([history_rel_pose, low_dim_dict['gripper_width'][:-1]], axis=-1)
        current_start_rel_pose = mat_to_pose(rel_obs_pose_mat)[-1]
        return_dict['state'] = np.concatenate([history_rel_pose.flatten(), current_start_rel_pose, low_dim_dict['gripper_width'][-1]], axis=-1).astype(np.float32)
    else:
        raise ValueError('getitem_type should be one of default, necessary, shortest')
    return return_dict


def draw_image(image, bbox_list):
    tmp_image = copy.deepcopy(image)
    draw = ImageDraw.Draw(tmp_image)
    for i, bbox in enumerate(bbox_list):
        if i == 0:
            color = 'red'
        elif i == 1:
            color = 'green'
        real_bbox = np.array(bbox) * 224
        draw.rectangle(real_bbox.tolist(), outline=color, width=6)
    tmp_image.show()
    response_code = input('Please input response code, 1 for success, 2 for reverse, 3 for fail: ')
    tmp_image.close()
    return response_code
    

@click.command()
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='0')
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--max_timesteps', '-mt', default=5_000, help='Max steps for each epoch.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_swap', is_flag=True, default=False)
@click.option('--temporal_agg', is_flag=True, default=False)
@click.option('--instruction', '-ins', default='', help='Instruction for the task')
@click.option('--remote_host', '-rh', default='localhost', help='Remote host for the policy server')
@click.option('--remote_port', '-rp', default=8000, help='Remote port for the policy server')
@click.option('--state_horizon', '-state_horizon', type=str, default='3,15', help='state down sample steps')
@click.option('--image_horizon', '-image_horizon', type=str, default='', help='image down sample steps')
@click.option('--action_down_sample_steps', '-action_down_sample_steps', type=int, default=3, help='action down sample steps')
@click.option('--getitem_type', '-getitem_type', type=str, default='necessary', help='`getitem_type` in the dataset config of the training code')
def main(output, robot_config, match_camera,
    camera_reorder,
    init_joints,
    steps_per_inference, max_duration, max_timesteps,
    frequency, command_latency,
    no_mirror, sim_fov, camera_intrinsics,
    mirror_swap, temporal_agg, instruction, remote_host, remote_port, state_horizon, image_horizon, action_down_sample_steps, getitem_type):
    max_gripper_width = 0.09
    gripper_speed = 0.2

    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))

    # load left-right robot relative transform
    tx_left_right = np.array(robot_config_data['tx_left_right'])
    tx_robot1_robot0 = tx_left_right

    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']

    state_horizon = [int(x) for x in state_horizon.split(',')]
    state_down_sample_steps = [0] + [x // action_down_sample_steps for x in state_horizon]
    state_down_sample_steps = state_down_sample_steps[::-1]

    image_horizon =  [] if image_horizon == '' else [int(x) for x in image_horizon.split(',')]
    image_down_sample_steps = [0] + [x // action_down_sample_steps for x in image_horizon]
    image_down_sample_steps = image_down_sample_steps[::-1]
    
    print("Instruction:", instruction)

    # setup experiment
    dt = 1/frequency

    obs_res = (224, 224)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm, \
            KeystrokeCounter() as key_counter, \
            BimanualUmiEnv(
                output_dir=output,
                robots_config=robots_config,
                grippers_config=grippers_config,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                multi_cam_vis_resolution=(1080, 1080),
                # latency
                camera_obs_latency=0.17,

                # downsample
                camera_down_sample_steps=image_down_sample_steps,
                robot_down_sample_steps=state_down_sample_steps,
                gripper_down_sample_steps=state_down_sample_steps,

                # obs
                camera_obs_horizon=image_horizon,
                robot_obs_horizon=state_horizon,
                gripper_obs_horizon=state_horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_swap=mirror_swap,
                # action
                max_pos_speed=2.0,
                max_rot_speed=6.0,
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            print("Connecting to policy server")
            policy_client = websocket_client_policy.WebsocketClientPolicy(remote_host, remote_port)

            obs_pose_rep = 'relative'
            action_pose_repr = 'relative'
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)

            print("Warming up policy inference")
            obs = env.get_obs()
            episode_start_pose = list()

            start_image, bbox_list = None, None

            for robot_id in range(len(robots_config)):
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)
            # compile mode like 'reduce-overhead' might need more than one iteration to warm up
            for _ in tqdm(range(1), desc='Warming up', leave=False):
                obs_dict_np = get_pi_obs_dict(
                    env_obs=obs, start_image=start_image, bbox_list=bbox_list, getitem_type=getitem_type,
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=episode_start_pose, instruction=instruction)
                obs_dict_np['is_warm_up'] = np.array([True])
                # for warm up, policy will always return an action
                action = policy_client.infer(obs_dict_np)['actions']
                assert action.shape[-1] == 10 * len(robots_config)
                action = get_real_umi_action(action, obs, action_pose_repr)
                action_horizon = action.shape[0]
                action_dim = action.shape[-1]
                assert action.shape[-1] == 7 * len(robots_config)

            print('Ready!')
            if_first_time = True
            all_time_actions = np.zeros((max_timesteps, max_timesteps + action_horizon, action_dim))
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                robot_states = env.get_robot_state()
                target_pose = np.stack([rs['ActualTCPPose'] for rs in robot_states])
                # target_pose = np.stack([rs['TargetTCPPose'] for rs in robot_states])

                gripper_states = env.get_gripper_state()
                gripper_target_pos = np.asarray([gs['gripper_position'] for gs in gripper_states])

                control_robot_idx_list = [0]

                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera{match_camera}_rgb'][-1]
                    
                    obs_left_img = obs['camera0_rgb'][-1]
                    obs_right_img = obs['camera0_rgb'][-1]
                    vis_img = np.concatenate([obs_left_img, obs_right_img, vis_img], axis=1)

                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0,0,0)
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    _ = cv2.pollKey()
                    press_events = key_counter.get_press_events()
                    start_policy = False
                    if init_joints and if_first_time:
                        if_first_time = False
                        break
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            # Exit human control loop
                            # hand control over to the policy
                            start_policy = True

                        elif key_stroke == Key.backspace:
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                        elif key_stroke == KeyCode(char='a'):
                            control_robot_idx_list = list(range(target_pose.shape[0]))
                        elif key_stroke == KeyCode(char='1'):
                            control_robot_idx_list = [0]
                        elif key_stroke == KeyCode(char='2'):
                            control_robot_idx_list = [1]

                    if start_policy:
                        break

                    precise_wait(t_sample)
                    # get teleop command
                    sm_state = sm.get_motion_state_transformed()
                    # print(sm_state)
                    dpos = sm_state[:3] * (0.5 / frequency) / 2.
                    drot_xyz = sm_state[3:] * (1.5 / frequency) / 2.

                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    for robot_idx in control_robot_idx_list:
                        target_pose[robot_idx, :3] += dpos
                        target_pose[robot_idx, 3:] = (drot * st.Rotation.from_rotvec(
                            target_pose[robot_idx, 3:])).as_rotvec()

                    dpos = 0
                    if sm.is_button_pressed(0):
                        # close gripper
                        dpos = -gripper_speed / frequency
                    if sm.is_button_pressed(1):
                        dpos = gripper_speed / frequency
                    for robot_idx in control_robot_idx_list:
                        gripper_target_pos[robot_idx] = np.clip(gripper_target_pos[robot_idx] + dpos, 0, max_gripper_width)

                    # solve collision with table
                    for robot_idx in control_robot_idx_list:
                        solve_table_collision(
                            ee_pose=target_pose[robot_idx],
                            gripper_width=gripper_target_pos[robot_idx],
                            height_threshold=robots_config[robot_idx]['height_threshold'])

                    # solve collison between two robots
                    solve_sphere_collision(
                        ee_poses=target_pose,
                        robots_config=robots_config
                    )

                    action = np.zeros((7 * target_pose.shape[0],))

                    for robot_idx in range(target_pose.shape[0]):
                        action[7 * robot_idx + 0: 7 * robot_idx + 6] = target_pose[robot_idx]
                        action[7 * robot_idx + 6] = gripper_target_pos[robot_idx]

                    # execute teleop command
                    action = thinking_pose if thinking_pose is not None else action
                    env.exec_actions(
                        actions=[action],
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        compensate_latency=False)
                    # break
                    precise_wait(t_cycle_end)
                    iter_idx += 1

                # ========== policy control loop ==============
                try:
                    # start episode
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # get current pose
                    obs = env.get_obs()
                    episode_start_pose = list()
                    for robot_id in range(len(robots_config)):
                        pose = np.concatenate([
                            obs[f'robot{robot_id}_eef_pos'],
                            obs[f'robot{robot_id}_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        episode_start_pose.append(pose)

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    inference_idx = steps_per_inference
                    thinking_pose = None

                    cycle_idx = 0
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (cycle_idx + 1) * dt

                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        s = time.time()
                        obs_dict_np = get_pi_obs_dict(
                            env_obs=obs, start_image=start_image, bbox_list=bbox_list, getitem_type=getitem_type,
                            obs_pose_repr=obs_pose_rep,
                            tx_robot1_robot0=tx_robot1_robot0,
                            episode_start_pose=episode_start_pose, instruction=instruction)
                        policy_return_dict = policy_client.infer(obs_dict_np)
                        if policy_return_dict.get('isthinking', False):
                            if thinking_pose is None:
                                print('Thinking...')
                                # the robot stays still while thinking
                                if iter_idx > 0:
                                    thinking_pose = all_time_actions[[iter_idx - 1], iter_idx - 1 + 2]
                                else:
                                    thinking_pose = np.concatenate([obs['robot0_eef_pos'][-1], obs['robot0_eef_rot_axis_angle'][-1], obs['robot0_gripper_width'][-1]])
                            action = np.tile(thinking_pose, (5, 1))
                        else:
                            if thinking_pose is not None:
                                thinking_pose = None
                            raw_action = policy_return_dict['actions']
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)  # (16, 7)
                        print('Inference latency:', time.time() - s)
                        if thinking_pose is None:
                            all_time_actions[[iter_idx], iter_idx:iter_idx + action_horizon] = action

                        if inference_idx == steps_per_inference:
                            inference_idx = 0

                            if thinking_pose is not None:
                                this_target_poses = action
                            elif temporal_agg:
                                # temporal ensemble
                                action_seq_for_curr_step = all_time_actions[:, iter_idx:iter_idx + action_horizon]
                                target_pose_list = []
                                for i in range(action_horizon):
                                    ensemble_num = 8
                                    actions_for_curr_step = action_seq_for_curr_step[max(0, iter_idx - ensemble_num + 1):iter_idx + 1, i]
                                    actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                                    actions_for_curr_step = actions_for_curr_step[actions_populated]

                                    k = -0.01
                                    exp_weights = np.exp(k * np.arange(len(actions_for_curr_step)))
                                    exp_weights = exp_weights / exp_weights.sum()
                                    weighted_rotvec = R.from_rotvec(np.array(actions_for_curr_step)[:, 3:6]).mean(weights=exp_weights).as_rotvec()
                                    weighted_action = (actions_for_curr_step * exp_weights[:, np.newaxis]).sum(axis=0, keepdims=True)
                                    weighted_action[0][3:6] = weighted_rotvec
                                    target_pose_list.append(weighted_action)
                                this_target_poses = np.concatenate(target_pose_list, axis=0)
                            else:
                                this_target_poses = action

                            assert this_target_poses.shape[1] == len(robots_config) * 7
                            for target_pose in this_target_poses:
                                for robot_idx in range(len(robots_config)):
                                    solve_table_collision(
                                        ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                                        gripper_width=target_pose[robot_idx * 7 + 6],
                                        height_threshold=robots_config[robot_idx]['height_threshold']
                                    )

                                # solve collison between two robots
                                solve_sphere_collision(
                                    ee_poses=target_pose.reshape([len(robots_config), -1]),
                                    robots_config=robots_config
                                )

                            # deal with timing
                            # the same step actions are always the target for
                            action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
                            action_exec_latency = 0.01
                            curr_time = time.time()
                            is_new = action_timestamps > (curr_time + action_exec_latency)
                            if np.sum(is_new) == 0:
                                # exceeded time budget, still do something
                                this_target_poses = this_target_poses[[-1]]  # (1, 7)
                                # schedule on next available step
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamp = eval_t_start + (next_step_idx) * dt
                                print('Over budget', action_timestamp - curr_time)
                                action_timestamps = np.array([action_timestamp])
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]

                            # execute actions

                            env.exec_actions(
                                actions=this_target_poses,
                                timestamps=action_timestamps,
                                compensate_latency=True,
                            )

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        obs_left_img = obs['camera0_rgb'][-1]
                        obs_right_img = obs['camera0_rgb'][-1]
                        vis_img = np.concatenate([obs_left_img, obs_right_img], axis=1)
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])

                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode
                                # Hand control back to human
                                thinking_pose = this_target_poses[1]
                                print('Stopped.')
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if iter_idx + 1 >= max_timesteps:
                            print("Max Timesteps reached.")
                            stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        cycle_idx += 1
                        if thinking_pose is None:
                            iter_idx += 1
                        inference_idx += 1

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()

                print("Stopped.")


# %%
if __name__ == '__main__':
    main()
