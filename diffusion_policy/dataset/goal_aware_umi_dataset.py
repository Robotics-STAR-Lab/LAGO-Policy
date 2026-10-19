"""
Goal-Aware UMI Dataset

Extended from UMI Dataset to provide goal labels for Goal-Aware Diffusion Policy.
Goal is defined as the next skill point (gripper change position).

Goal/Skill definition:
- Goal frame: first frame within skill_window (30 frames) before gripper change
  AND within goal_distance_threshold (5cm) of the gripper change position
- Skill interval: [goal_frame, change + skill_end_window(10 frames)]
  - skill_start = goal_frame (satisfies 5cm + 30 frames condition)
  - skill_end = gripper_change + 10 frames

Goal assignment:
- Before skill: goal = goal_frame of the next gripper change
- Inside skill: goal = next frame EE pose (reactive tracking)
- After last skill: goal = next frame EE pose
"""

import copy
from typing import Dict, Optional, List, Tuple
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pose_util import pose_to_mat, mat_to_pose10d

register_codecs()


class GoalAwareUmiDataset(BaseDataset):
    """
    UMI Dataset with goal prediction labels.
    
    Goal is defined as the pose at the next gripper change event.
    This represents the task-relevant interaction goal.
    """
    
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            cache_dir: Optional[str] = None,
            pose_repr: dict = {},
            action_padding: bool = False,
            repeat_frame_prob: float = 0.0,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_duration: float = None,
            # Skill detection
            gripper_change_threshold: float = 0.01,
            min_change_interval: int = 15,
            skill_window: int = 30,  # Max frames before gripper change to search for goal
            skill_end_window: int = 10,  # Frames after gripper change for skill end
            # Goal configuration
            goal_dim: int = 10,  # pos(3) + rot6d(6) + gripper(1)
            goal_distance_threshold: float = 0.05,  # 3cm - max distance from goal to gripper change
        ):
        self.pose_repr = pose_repr
        self.obs_pose_repr = pose_repr.get('obs_pose_repr', 'abs')
        self.action_pose_repr = pose_repr.get('action_pose_repr', 'abs')
        self.gripper_change_threshold = gripper_change_threshold
        self.min_change_interval = min_change_interval
        self.skill_window = skill_window
        self.skill_end_window = skill_end_window
        self.goal_dim = goal_dim
        self.goal_distance_threshold = goal_distance_threshold
        self.shape_meta = shape_meta
        
        # Load replay buffer
        print('Loading replay buffer from disk...')
        with zarr.ZipStore(dataset_path, mode='r') as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store,
                store=zarr.MemoryStore()
            )
        print('Loaded!')
        
        # Parse shape meta (same as umi_dataset.py)
        self.num_robot = 0
        rgb_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta['obs']
        
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                rgb_keys.append(key)
            elif obs_type == 'low_dim':
                lowdim_keys.append(key)
            
            if key.endswith('eef_pos'):
                self.num_robot += 1
            
            key_horizon[key] = attr['horizon']
            key_latency_steps[key] = attr['latency_steps']
            key_down_sample_steps[key] = attr['down_sample_steps']
        
        # Action settings
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']
        
        # Get validation mask
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        
        # Build sampler_lowdim_keys (same as umi_dataset.py)
        sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if 'wrt' not in key:
                sampler_lowdim_keys.append(key)
        
        for key in replay_buffer.keys():
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                sampler_lowdim_keys.append(key)
                query_key = key.split('_')[0] + '_eef_pos'
                key_horizon[key] = shape_meta['obs'][query_key]['horizon']
                key_latency_steps[key] = shape_meta['obs'][query_key]['latency_steps']
                key_down_sample_steps[key] = shape_meta['obs'][query_key]['down_sample_steps']
        
        # Create sampler
        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob
        )
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.sampler_lowdim_keys = sampler_lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.train_mask = train_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        
        # Store for goal computation
        self.gripper_width = replay_buffer['robot0_gripper_width'][:, 0]
        self.eef_pos = replay_buffer['robot0_eef_pos'][:]
        self.eef_rot = replay_buffer['robot0_eef_rot_axis_angle'][:]
        self.episode_ends = replay_buffer.episode_ends[:]
        
        # Detect gripper changes and build goal map
        print('Detecting gripper changes and building goal map...')
        print(f'  skill_window: {self.skill_window} frames (max search for goal/skill start)')
        print(f'  skill_end_window: {self.skill_end_window} frames (after gripper change)')
        print(f'  goal_distance_threshold: {self.goal_distance_threshold*100:.1f}cm')
        self.gripper_changes = self._detect_gripper_changes()
        self.goal_map = self._build_goal_map()
        
        # Print goal selection statistics
        self._print_goal_statistics()
        
        print(f'Dataset ready: {len(self.sampler)} samples, '
              f'{len(self.gripper_changes)} gripper changes')
        
        self.threadpool_limits_is_applied = False
    
    def _print_goal_statistics(self):
        """Print statistics about goal selection."""
        if not self.gripper_changes:
            print('  No gripper changes detected')
            return
        
        distances = []
        frame_offsets = []
        
        for change_idx, ep_idx in self.gripper_changes:
            start_idx = 0 if ep_idx == 0 else self.episode_ends[ep_idx - 1]
            goal_idx = self._find_goal_frame_for_change(change_idx, start_idx)
            
            # Distance from goal to change
            change_pos = self.eef_pos[change_idx]
            goal_pos = self.eef_pos[goal_idx]
            dist = np.linalg.norm(goal_pos - change_pos)
            distances.append(dist)
            
            # Frame offset (how many frames before change)
            offset = change_idx - goal_idx
            frame_offsets.append(offset)
        
        distances = np.array(distances)
        frame_offsets = np.array(frame_offsets)
        
        within_threshold = np.sum(distances <= self.goal_distance_threshold)
        
        print(f'  Goal selection statistics ({len(self.gripper_changes)} changes):')
        print(f'    Distance to change: {distances.mean()*100:.2f}cm avg, '
              f'{distances.min()*100:.2f}-{distances.max()*100:.2f}cm range')
        print(f'    Frame offset: {frame_offsets.mean():.1f} avg, '
              f'{frame_offsets.min()}-{frame_offsets.max()} range')
        print(f'    Within {self.goal_distance_threshold*100:.0f}cm threshold: '
              f'{within_threshold}/{len(self.gripper_changes)} '
              f'({within_threshold/len(self.gripper_changes)*100:.1f}%)')
    
    def _detect_gripper_changes(self, for_validation: bool = False) -> List[Tuple[int, int]]:
        """Detect frames where gripper changes (skill points)."""
        gripper_changes = []
        
        for ep_idx in range(len(self.episode_ends)):
            if for_validation:
                if not self.val_mask[ep_idx]:
                    continue
            else:
                if self.val_mask[ep_idx]:
                    continue
            
            start_idx = 0 if ep_idx == 0 else self.episode_ends[ep_idx - 1]
            end_idx = self.episode_ends[ep_idx]
            
            last_change_frame = -self.min_change_interval - 1
            
            for i in range(start_idx + 1, end_idx):
                if i - last_change_frame < self.min_change_interval:
                    continue
                
                delta = abs(self.gripper_width[i] - self.gripper_width[i - 1])
                if delta > self.gripper_change_threshold:
                    gripper_changes.append((i, ep_idx))
                    last_change_frame = i
        
        return gripper_changes
    
    def _find_goal_frame_for_change(self, change_idx: int, start_idx: int) -> int:
        """
        Find the goal frame for a gripper change event.
        
        Goal must satisfy two conditions:
        1. Within skill_window (30 frames) before the gripper change
        2. EE position within goal_distance_threshold (5cm) of the gripper change position
        
        Search from change_idx - skill_window towards change_idx.
        Returns the first frame that satisfies both conditions.
        If no frame satisfies the distance condition, return change_idx - skill_window.
        """
        change_pos = self.eef_pos[change_idx]
        earliest_frame = max(start_idx, change_idx - self.skill_window)
        
        # Search from earliest_frame towards change_idx
        for frame_idx in range(earliest_frame, change_idx):
            frame_pos = self.eef_pos[frame_idx]
            dist = np.linalg.norm(frame_pos - change_pos)
            if dist <= self.goal_distance_threshold:
                return frame_idx
        
        # If no frame within distance threshold, use earliest frame (skill_window before)
        return earliest_frame
    
    def _build_goal_map(self) -> Dict[int, int]:
        """
        Build mapping from each frame to its goal frame.
        
        Skill interval: [goal_frame, change + skill_end_window]
        - goal_frame: first frame within skill_window(30) AND goal_distance_threshold(3cm)
        - skill_end: gripper_change + skill_end_window (10 frames)
        
        Goal assignment:
        - Before skill: goal = goal_frame of next gripper change
        - Inside skill: goal = next frame EE pose (reactive tracking)
        - After last skill: goal = next frame EE pose
        """
        goal_map = {}
        
        # Group changes by episode
        ep_changes = {}
        for global_idx, ep_idx in self.gripper_changes:
            if ep_idx not in ep_changes:
                ep_changes[ep_idx] = []
            ep_changes[ep_idx].append(global_idx)
        
        # Pre-compute goal frames for each change
        change_to_goal = {}
        for global_idx, ep_idx in self.gripper_changes:
            start_idx = 0 if ep_idx == 0 else self.episode_ends[ep_idx - 1]
            change_to_goal[global_idx] = self._find_goal_frame_for_change(global_idx, start_idx)
        
        for ep_idx in range(len(self.episode_ends)):
            if self.val_mask[ep_idx]:
                continue
            
            start_idx = 0 if ep_idx == 0 else self.episode_ends[ep_idx - 1]
            end_idx = self.episode_ends[ep_idx]
            
            changes = sorted(ep_changes.get(ep_idx, []))
            
            # Build skill intervals:
            # - skill_start: goal frame (satisfies 3cm AND within skill_window)
            # - skill_end: change_idx + skill_end_window (10 frames after gripper change)
            skill_intervals = []
            for change_idx in changes:
                # skill_start = goal frame (same as _find_goal_frame_for_change)
                skill_start = change_to_goal[change_idx]
                # skill_end = change + skill_end_window (default 10 frames)
                skill_end = min(end_idx, change_idx + self.skill_end_window + 1)
                skill_intervals.append((skill_start, skill_end, change_idx))
            
            # Find last skill end
            last_skill_end = skill_intervals[-1][1] if skill_intervals else start_idx
            
            for i in range(start_idx, end_idx):
                goal_idx = None
                
                # Check if inside any skill interval
                in_skill = False
                for skill_start, skill_end, change_idx in skill_intervals:
                    if skill_start <= i < skill_end:
                        in_skill = True
                        break
                
                if in_skill:
                    # Inside skill: goal = next frame
                    goal_idx = min(i + 1, end_idx - 1)
                elif i >= last_skill_end:
                    # After last skill: goal = next frame
                    goal_idx = min(i + 1, end_idx - 1)
                else:
                    # Before skill: find goal using pre-computed map
                    for change_idx in changes:
                        if change_idx > i:
                            goal_idx = change_to_goal[change_idx]
                            break
                    
                    # Fallback
                    if goal_idx is None:
                        goal_idx = min(i + 1, end_idx - 1)
                
                goal_map[i] = goal_idx
        
        return goal_map
    
    def _get_goal_pose(self, goal_idx: int) -> np.ndarray:
        """Get goal pose in pose10d format."""
        pos = self.eef_pos[goal_idx]
        rot = self.eef_rot[goal_idx]
        gripper = self.gripper_width[goal_idx:goal_idx + 1]
        
        pose6d = np.concatenate([pos, rot])
        pose_mat = pose_to_mat(pose6d)
        pose10d = mat_to_pose10d(pose_mat[np.newaxis])[0]
        goal = np.concatenate([pose10d, gripper]).astype(np.float32)
        
        return goal
    
    def get_validation_dataset(self):
        """Get validation dataset."""
        val_set = copy.copy(self)
        
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=0.0
        )
        
        val_set.gripper_changes = self._detect_gripper_changes(for_validation=True)
        val_set.goal_map = val_set._build_goal_map_validation()
        
        print(f'Validation dataset: {len(val_set.sampler)} samples')
        return val_set
    
    def _build_goal_map_validation(self) -> Dict[int, int]:
        """Build goal map for validation episodes."""
        goal_map = {}
        
        ep_changes = {}
        for global_idx, ep_idx in self.gripper_changes:
            if ep_idx not in ep_changes:
                ep_changes[ep_idx] = []
            ep_changes[ep_idx].append(global_idx)
        
        # Pre-compute goal frames for each change
        change_to_goal = {}
        for global_idx, ep_idx in self.gripper_changes:
            start_idx = 0 if ep_idx == 0 else self.episode_ends[ep_idx - 1]
            change_to_goal[global_idx] = self._find_goal_frame_for_change(global_idx, start_idx)
        
        for ep_idx in range(len(self.episode_ends)):
            if not self.val_mask[ep_idx]:
                continue
            
            start_idx = 0 if ep_idx == 0 else self.episode_ends[ep_idx - 1]
            end_idx = self.episode_ends[ep_idx]
            
            changes = sorted(ep_changes.get(ep_idx, []))
            
            # Build skill intervals (same logic as _build_goal_map)
            skill_intervals = []
            for change_idx in changes:
                skill_start = change_to_goal[change_idx]
                skill_end = min(end_idx, change_idx + self.skill_end_window + 1)
                skill_intervals.append((skill_start, skill_end, change_idx))
            
            last_skill_end = skill_intervals[-1][1] if skill_intervals else start_idx
            
            for i in range(start_idx, end_idx):
                goal_idx = None
                
                in_skill = False
                for skill_start, skill_end, change_idx in skill_intervals:
                    if skill_start <= i < skill_end:
                        in_skill = True
                        break
                
                if in_skill:
                    goal_idx = min(i + 1, end_idx - 1)
                elif i >= last_skill_end:
                    goal_idx = min(i + 1, end_idx - 1)
                else:
                    # Before skill: find goal using pre-computed map
                    for change_idx in changes:
                        if change_idx > i:
                            goal_idx = change_to_goal[change_idx]
                            break
                    
                    if goal_idx is None:
                        goal_idx = min(i + 1, end_idx - 1)
                
                goal_map[i] = goal_idx
        
        return goal_map
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """Get normalizer including goal normalizer."""
        normalizer = LinearNormalizer()
        
        # Build action data for normalizer (same as umi_dataset.py)
        action_data = []
        for ep_idx in range(len(self.episode_ends)):
            if not self.train_mask[ep_idx]:
                continue
            start = 0 if ep_idx == 0 else self.episode_ends[ep_idx - 1]
            end = self.episode_ends[ep_idx]
            
            for i in range(start, end):
                pos = self.eef_pos[i]
                rot = self.eef_rot[i]
                gripper = self.gripper_width[i:i + 1]
                
                pose6d = np.concatenate([pos, rot])
                pose_mat = pose_to_mat(pose6d)
                pose10d = mat_to_pose10d(pose_mat[np.newaxis])[0]
                action = np.concatenate([pose10d, gripper])
                action_data.append(action)
        
        action_data = np.stack(action_data)
        
        # Action normalizer: pos(3) + rot6d(6) + gripper(1)
        pos_norm = get_range_normalizer_from_stat(array_to_stats(action_data[:, :3]))
        rot_norm = get_identity_normalizer_from_stat(array_to_stats(action_data[:, 3:9]))
        grip_norm = get_range_normalizer_from_stat(array_to_stats(action_data[:, 9:10]))
        normalizer['action'] = concatenate_normalizer([pos_norm, rot_norm, grip_norm])
        
        # Goal normalizer (same format as action)
        goal_data = []
        for global_idx, goal_idx in self.goal_map.items():
            pos = self.eef_pos[goal_idx]
            rot = self.eef_rot[goal_idx]
            gripper = self.gripper_width[goal_idx:goal_idx + 1]
            
            pose6d = np.concatenate([pos, rot])
            pose_mat = pose_to_mat(pose6d)
            pose10d = mat_to_pose10d(pose_mat[np.newaxis])[0]
            goal = np.concatenate([pose10d, gripper])
            goal_data.append(goal)
        
        if goal_data:
            goal_data = np.stack(goal_data)
            goal_pos_norm = get_range_normalizer_from_stat(array_to_stats(goal_data[:, :3]))
            goal_rot_norm = get_identity_normalizer_from_stat(array_to_stats(goal_data[:, 3:9]))
            goal_grip_norm = get_range_normalizer_from_stat(array_to_stats(goal_data[:, 9:10]))
            normalizer['goal'] = concatenate_normalizer([goal_pos_norm, goal_rot_norm, goal_grip_norm])
        
        # Low-dim observation normalizers
        for key in self.lowdim_keys:
            data = self.replay_buffer[key][:]
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Use only training data
            train_data = []
            for ep_idx in range(len(self.episode_ends)):
                if not self.train_mask[ep_idx]:
                    continue
                start = 0 if ep_idx == 0 else self.episode_ends[ep_idx - 1]
                end = self.episode_ends[ep_idx]
                train_data.append(data[start:end])
            
            if train_data:
                train_data = np.concatenate(train_data)
                stat = array_to_stats(train_data)
                if 'pos' in key:
                    normalizer[key] = get_range_normalizer_from_stat(stat)
                else:
                    normalizer[key] = get_identity_normalizer_from_stat(stat)
        
        # Image normalizers
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        
        print(f"Action normalizer: pos_range={action_data[:, :3].min(axis=0)} to {action_data[:, :3].max(axis=0)}")
        
        return normalizer
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        
        # Get data from sampler (same as umi_dataset.py)
        data = self.sampler.sample_sequence(idx)
        
        # Build observation dict
        obs_dict = {}
        
        # RGB observations
        for key in self.rgb_keys:
            if key not in data:
                continue
            imgs = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.0
            obs_dict[key] = imgs
            del data[key]
        
        # Low-dim observations
        for key in self.lowdim_keys:
            if key in data:
                obs_dict[key] = data[key].astype(np.float32)
        
        # Process poses (same as umi_dataset.py)
        for robot_id in range(self.num_robot):
            pos_key = f'robot{robot_id}_eef_pos'
            rot_key = f'robot{robot_id}_eef_rot_axis_angle'
            
            if pos_key in obs_dict and rot_key in obs_dict:
                pose_mat = pose_to_mat(np.concatenate([
                    obs_dict[pos_key],
                    obs_dict[rot_key]
                ], axis=-1))
                
                obs_pose_mat = convert_pose_mat_rep(
                    pose_mat,
                    base_pose_mat=pose_mat[-1],
                    pose_rep=self.obs_pose_repr,
                    backward=False)
                
                obs_pose = mat_to_pose10d(obs_pose_mat)
                obs_dict[pos_key] = obs_pose[:, :3].astype(np.float32)
                obs_dict[rot_key] = obs_pose[:, 3:].astype(np.float32)
        
        # Build action (same as umi_dataset.py)
        raw_action = data['action']
        action_mat = pose_to_mat(raw_action[:, :6])
        
        # For absolute pose repr, just convert to pose10d
        action_pose = mat_to_pose10d(action_mat)
        action_gripper = raw_action[:, 6:7]
        action = np.concatenate([action_pose, action_gripper], axis=-1).astype(np.float32)
        
        # Get goal
        # Find the decision/current frame for this sample
        sample_indices = self.sampler.indices[idx]
        frame_idx = sample_indices[0]  # decision/current frame (last obs frame = first action frame)
        
        current_idx = frame_idx
        current_idx = max(0, min(current_idx, len(self.gripper_width) - 1))
        
        goal_idx = self.goal_map.get(current_idx, current_idx)
        goal = self._get_goal_pose(goal_idx)
        
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action),
            'goal': torch.from_numpy(goal),
        }
        
        return torch_data
