"""
Goal-Aware Diffusion Policy

This implements the goal-directed paradigm from the Efficient Policy paper:
- Normal Diffusion Policy for action prediction
- Goal prediction head at UNet bottleneck for task-relevant goal prediction
- Joint training with diffusion loss + goal loss

The goal is predicted from bottleneck features using:
    ĝt = gψ(GAP(ht))
where GAP is global average pooling over temporal dimension.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class GoalAwareDiffusionPolicy(BaseImagePolicy):
    """
    Goal-Aware Diffusion Policy that jointly predicts actions and task-relevant goals.
    
    Based on the Efficient Policy paper:
    - Diffusion Policy predicts action chunks
    - Goal prediction head at UNet bottleneck predicts interaction goals
    - Goals can be used for collision-free trajectory planning
    """
    
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            num_inference_steps: int = None,
            obs_as_global_cond: bool = True,
            crop_shape: tuple = (76, 76),
            diffusion_step_embed_dim: int = 256,
            down_dims: tuple = (256, 512, 1024),
            kernel_size: int = 5,
            n_groups: int = 8,
            cond_predict_scale: bool = True,
            obs_encoder_group_norm: bool = False,
            eval_fixed_crop: bool = False,
            transforms=None,
            # CFG parameters
            future_cond_horizon: int = 0,
            cf_drop_prob: float = 0.0,
            # Goal prediction parameters
            goal_dim: int = 10,  # pos(3) + rot6d(6) + gripper(1)
            goal_loss_weight: float = 0.1,
            **kwargs):
        super().__init__()
        
        # Parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        # Get robomimic config for obs encoder
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw
        
        ObsUtils.initialize_obs_utils_with_config(config)
        
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu',
        )
        
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features)
            )
        
        if eval_fixed_crop and hasattr(rmbn, 'CropRandomizer'):
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )
        
        # Create diffusion model with goal prediction head
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim
        global_cond_dim = obs_feature_dim * n_obs_steps if obs_as_global_cond else None
        
        # CFG
        self.future_cond_horizon = max(0, future_cond_horizon)
        self.cf_drop_prob = cf_drop_prob
        self.future_cond_dim = self.future_cond_horizon * action_dim
        
        # Create UNet with goal prediction head
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            future_cond_dim=self.future_cond_dim,
            predict_goal=(goal_dim > 0),
            goal_dim=goal_dim,
        )
        
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.transforms = None if transforms is None else torch.nn.Sequential(*transforms)
        self.kwargs = kwargs
        
        # Goal prediction
        self.goal_dim = goal_dim
        self.goal_loss_weight = goal_loss_weight
        
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # Last action prediction for future conditioning
        self.last_action_pred = None
        
        print(f"GoalAwareDiffusionPolicy initialized:")
        print(f"  Diffusion params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Vision params: {sum(p.numel() for p in self.obs_encoder.parameters()):,}")
        print(f"  Goal dim: {goal_dim}")
        print(f"  Goal loss weight: {goal_loss_weight}")
        if self.future_cond_dim > 0:
            print(f"  CFG: future_cond_horizon={self.future_cond_horizon}, cf_drop_prob={cf_drop_prob}")
    
    def conditional_sample(self,
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            future_cond=None,
            return_goal=False,
            generator=None,
            **kwargs):
        """
        Conditional diffusion sampling with optional goal prediction.
        """
        model = self.model
        scheduler = self.noise_scheduler
        
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
        
        scheduler.set_timesteps(self.num_inference_steps)
        
        goal_pred = None
        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            
            if return_goal and t == scheduler.timesteps[0]:
                # Get goal prediction from first step (when noise is highest)
                # Returns tuple (output, goal_pred) when return_goal=True
                model_output, goal_pred = model(trajectory, t,
                    local_cond=local_cond, global_cond=global_cond,
                    future_cond=future_cond, return_goal=True)
            else:
                model_output = model(trajectory, t,
                    local_cond=local_cond, global_cond=global_cond,
                    future_cond=future_cond)
            
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample
        
        trajectory[condition_mask] = condition_data[condition_mask]
        
        if return_goal:
            return trajectory, goal_pred
        return trajectory
    
    def _future_cond_from_overlap(self, naction_pred: torch.Tensor, B: int) -> torch.Tensor:
        """
        Build the future-action condition A^c_t from a (normalized) predicted action
        sequence by taking the POST-EXECUTION OVERLAP -- the head of the next chunk.

        Under asynchronous inference the world advances by exactly H^e = T_a
        (n_action_steps) between cycles, so the slice that becomes the HEAD of the
        NEXT chunk starts at offset s + H^e, where s = n_obs_steps - 1 is the
        execution start index (executed chunk = action_pred[:, s:s+T_a]). This mirrors
        training, where compute_loss conditions on the chunk head a_{delta:delta+H^f}
        of the predicted sequence (delta ~ Uniform{0,1,2}); the only difference is the
        small train-time delta jitter that makes the policy robust to async misalignment.

        With T_p=16, T_o=2 (s=1), T_a=8, T_f=4 -> slice [9:13], in-bounds; a safe
        clamp/pad (mirroring the hybrid policy) handles configs where the slice would
        exceed the prediction horizon.
        """
        T = naction_pred.shape[1]
        Da = naction_pred.shape[-1]
        s = self.n_obs_steps - 1                  # execution start index
        Hf = self.future_cond_horizon            # H^f (future condition horizon)
        future_start = s + self.n_action_steps   # head of the next chunk (offset T_a)
        future_end = future_start + Hf
        if future_end <= T:
            future_actions = naction_pred[:, future_start:future_end, :]  # [B, H^f, Da]
        else:
            # Edge case: not enough future steps -> pad with the last predicted action.
            available = max(0, T - future_start)
            if available > 0:
                partial_future = naction_pred[:, future_start:T, :]       # [B, available, Da]
                last_action = naction_pred[:, -1:, :].expand(B, Hf - available, Da)
                future_actions = torch.cat([partial_future, last_action], dim=1)
            else:
                future_actions = naction_pred[:, -1:, :].expand(B, Hf, Da)
        return future_actions.reshape(B, -1)     # [B, future_cond_dim]
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                       future_cond: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Predict action and goal from observations.
        
        Returns:
            result: dict containing:
                - 'action': predicted actions [B, n_action_steps, action_dim]
                - 'action_pred': full action prediction [B, horizon, action_dim]
                - 'goal': predicted goal [B, goal_dim] (unnormalized)
                - 'future_cond_new': new future condition for next inference
        """
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps
        
        device = self.device
        dtype = self.dtype
        
        # Handle future_cond
        if future_cond is not None and self.future_cond_dim > 0:
            future_cond = future_cond.to(device)
            if future_cond.dim() == 1:
                future_cond = future_cond.unsqueeze(0)
        elif self.future_cond_dim > 0 and self.last_action_pred is not None:
            # Derive A^c_t from the previous prediction's POST-EXECUTION OVERLAP (the
            # head of the new chunk), matching training's chunk-head condition rather
            # than the chunk tail.
            future_cond = self._future_cond_from_overlap(self.last_action_pred, B)
        
        # Encode observations
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]).to(dtype=dtype))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            raise NotImplementedError("Only obs_as_global_cond=True is supported")
        
        # Sample with goal prediction
        nsample, goal_pred = self.conditional_sample(
            cond_data, cond_mask,
            local_cond=local_cond, global_cond=global_cond,
            future_cond=future_cond,
            return_goal=True,
            **self.kwargs)
        
        # Unnormalize
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        
        # Store for next CFG
        self.last_action_pred = naction_pred.detach()
        
        # Extract action steps
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        # Unnormalize goal if available
        if goal_pred is not None and 'goal' in self.normalizer.params_dict:
            goal = self.normalizer['goal'].unnormalize(goal_pred)
        else:
            goal = goal_pred
        
        # Build future_cond_new for next inference: the post-execution overlap (head
        # of the next chunk, offset = execution horizon T_a), aligned with the training
        # chunk-head A^c_t instead of the chunk tail.
        future_cond_new = None
        if self.future_cond_dim > 0:
            future_cond_new = self._future_cond_from_overlap(naction_pred, B)
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'goal': goal,
            'future_cond_new': future_cond_new,
        }
        return result
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute combined diffusion loss + goal prediction loss.
        
        L = LDP + λgoal * Lgoal
        """
        obs_dict = batch['obs']
        action = batch['action']
        goal = batch.get('goal')  # [B, goal_dim]
        
        nobs = self.normalizer.normalize(obs_dict)
        naction = self.normalizer['action'].normalize(action)
        
        if goal is not None and 'goal' in self.normalizer.params_dict:
            ngoal = self.normalizer['goal'].normalize(goal)
        else:
            ngoal = goal
        
        B = naction.shape[0]
        T = self.horizon
        device = self.device
        dtype = self.dtype
        
        # Ensure correct dtype
        naction = naction.to(dtype=dtype)
        if ngoal is not None:
            ngoal = ngoal.to(dtype=dtype)
        
        # Encode observations
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]).to(dtype=dtype))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(B, -1)
        
        # CFG: Build future condition a^c with per-sample delay randomization + dropout
        #
        # Paper Algorithm 1 / LA-CFG: a^c is the CHUNK HEAD taken from a per-sample
        # random offset delta ~ Uniform{0,1,2} of the predicted (not-yet-executed)
        # action chunk:  a^c = a_{t+delta : t+delta+H^f}.
        # (delta in {0,1,2}, H^f=4, horizon=16 -> max gathered index 5 < 16, in-bounds)
        future_cond = None
        if self.future_cond_dim > 0:
            # per-sample random start offset delta ~ Uniform{0, 1, 2}
            start = torch.randint(0, 3, (B,), device=device)
            Hf = self.future_cond_horizon  # H^f

            # a^c = a_{t+delta : t+delta+H^f} per-sample with randomized start
            steps = torch.arange(Hf, device=device)
            idx = (start[:, None] + steps[None, :]).unsqueeze(-1)        # [B, H^f, 1]
            idx = idx.expand(-1, -1, naction.shape[-1])                  # [B, H^f, action_dim]
            future_seq = torch.gather(naction, dim=1, index=idx)         # [B, H^f, action_dim]
            future_cond = future_seq.reshape(B, -1).to(dtype=dtype)      # [B, future_cond_dim]

            # Classifier-Free Guidance dropout: randomly zero out some a^c
            if self.training and self.cf_drop_prob > 0:
                drop_mask = torch.rand(B, device=device) < self.cf_drop_prob
                future_cond[drop_mask] = 0.0
        
        # Sample noise and timesteps
        noise = torch.randn(naction.shape, device=device, dtype=dtype)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        
        # Add noise
        noisy_action = self.noise_scheduler.add_noise(naction, noise, timesteps)
        
        # Predict noise with goal (returns tuple when return_goal=True)
        noise_pred, goal_pred = self.model(
            sample=noisy_action,
            timestep=timesteps,
            local_cond=None,
            global_cond=global_cond,
            future_cond=future_cond,
            return_goal=True,
        )
        
        # Diffusion loss
        diffusion_loss = F.mse_loss(noise_pred, noise)
        
        # Goal loss
        goal_loss = torch.tensor(0.0, device=device)
        if ngoal is not None and goal_pred is not None:
            goal_loss = F.mse_loss(goal_pred, ngoal.to(dtype=goal_pred.dtype))
        
        # Total loss
        total_loss = diffusion_loss + self.goal_loss_weight * goal_loss
        
        return {
            'loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'goal_loss': goal_loss,
        }
    
    def reset(self):
        """Reset policy state for new episode."""
        self.last_action_pred = None
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
