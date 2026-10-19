from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
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

# Check if CropRandomizer exists in robomimic (may be removed in newer versions)
_HAS_ROBOMIMIC_CROP_RANDOMIZER = hasattr(rmbn, 'CropRandomizer')


class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            transforms=None,
            # CFG (Classifier-Free Guidance) parameters
            future_cond_horizon=0,  # Number of future action steps for condition
            cf_drop_prob=0.0,  # Classifier-free dropout probability
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
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

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop and _HAS_ROBOMIMIC_CROP_RANDOMIZER:
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

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        # CFG: Compute future condition dimension
        self.future_cond_horizon = future_cond_horizon
        self.cf_drop_prob = cf_drop_prob
        if self.future_cond_horizon < 0:
            self.future_cond_horizon = 0
        self.future_cond_dim = self.future_cond_horizon * action_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            future_cond_dim=self.future_cond_dim  # CFG: pass future_cond_dim
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

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
        if self.future_cond_dim > 0:
            print(f"CFG enabled: future_cond_horizon={self.future_cond_horizon}, "
                  f"future_cond_dim={self.future_cond_dim}, cf_drop_prob={self.cf_drop_prob}")
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            future_cond=None,  # CFG: future action condition
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output (with optional future_cond for CFG)
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond,
                future_cond=future_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                        future_cond: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Predict action from observations with optional future condition for CFG.
        
        Args:
            obs_dict: must include "obs" key
            future_cond: Optional future action condition [B, future_cond_dim] (normalized)
                - If provided, use conditional branch
                - If None, use unconditional branch
        
        Returns:
            result: dict containing:
                - 'action': predicted actions [B, n_action_steps, action_dim]
                - 'action_pred': full action prediction [B, horizon, action_dim]
                - 'future_cond_new': new future condition for next inference [B, future_cond_dim]
                  (None if future_cond_dim == 0)
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # Handle future_cond if provided and CFG is enabled
        if future_cond is not None and self.future_cond_dim > 0:
            future_cond = future_cond.to(device)
            if future_cond.dim() == 1:
                future_cond = future_cond.unsqueeze(0)
            assert future_cond.shape[0] == B, f"future_cond batch {future_cond.shape[0]} != obs batch {B}"
            assert future_cond.shape[1] == self.future_cond_dim, \
                f"future_cond dim {future_cond.shape[1]} != expected {self.future_cond_dim}"

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling (with optional future_cond for CFG)
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            future_cond=future_cond if self.future_cond_dim > 0 else None,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # Extract new future_cond for next inference
        # After executing n_action_steps, we need future_cond for next timestep
        # future_cond_new = normalized action prediction at [start+He : start+He+Hf]
        future_cond_new = None
        if self.future_cond_dim > 0:
            # He = execution horizon T_a (== n_action_steps). Between asynchronous
            # cycles the world advances by exactly T_a steps, so the overlap that
            # becomes the HEAD of the next chunk starts at offset start + T_a. Using
            # n_action_steps (not a hardcoded 6) aligns the deploy-time A^c_t with the
            # training chunk-head condition: compute_loss conditions on the chunk head
            # a_{delta:delta+T_f} of the predicted sequence (delta ~ Uniform{0,1,2}).
            He = self.n_action_steps
            Hf = self.future_cond_horizon  # future condition horizon
            # Post-execution overlap (head of the next chunk), normalized; this becomes
            # future_cond (A^c_t) for the next inference cycle.
            future_start = start + He
            future_end = future_start + Hf
            if future_end <= T:
                future_actions = naction_pred[:, future_start:future_end, :]  # [B, Hf, Da]
                future_cond_new = future_actions.reshape(B, -1)  # [B, future_cond_dim]
            else:
                # Handle edge case where there aren't enough future steps
                # Pad with the last action
                available = T - future_start
                if available > 0:
                    partial_future = naction_pred[:, future_start:T, :]  # [B, available, Da]
                    last_action = naction_pred[:, -1:, :].expand(B, Hf - available, Da)
                    future_actions = torch.cat([partial_future, last_action], dim=1)
                else:
                    future_actions = naction_pred[:, -1:, :].expand(B, Hf, Da)
                future_cond_new = future_actions.reshape(B, -1)
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'future_cond_new': future_cond_new
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch

        # image augmentation
        if self.transforms is not None:
            for key, val in batch['obs'].items():
                if key.startswith('camera'):
                    raw_shape = val.shape
                    batch['obs'][key] = self.transforms(val.reshape(-1, *raw_shape[2:])).reshape(raw_shape)
        
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        ##########################################################################
        # === CFG: Construct future condition a^c for each sample ===
        #
        # Symbol definitions:
        #   H^e = n_action_steps (execution horizon)
        #   H^f = future_cond_horizon (future condition horizon)
        #   H = horizon (prediction horizon)
        #
        # Timeline analysis (assuming current timestep is t):
        #   - Previous timestep (t - H^e) predicted a_{t-H^e : t-H^e+H}
        #   - Executed first H^e steps: a_{t-H^e : t}
        #   - Remaining unexecuted portion is a_{t : t-H^e+H}
        #   - The first H^f steps of this become a^c (the chunk head)
        #
        # Delay randomization (paper Algorithm 1):
        #   - a^c is taken from a per-sample random offset delta ~ Uniform{0,1,2}
        #     at the BEGINNING of the predicted chunk:
        #       a^c = a_{t+delta : t+delta+H^f}
        #   - This exposes the denoiser to delta-shifted future-action conditions,
        #     making it robust to temporal misalignment under async inference.
        #
        # In dataset:
        #   - nactions[i] corresponds to action sequence a_{t : t+H} at timestep t
        #   - a^c = nactions[:, delta:delta+H^f] with per-sample delta (via gather)
        #
        # nactions shape: [B, horizon, action_dim]
        # (with delta in {0,1,2}, H^f=4, horizon=16 -> max index 5 < 16, in-bounds)

        future_cond = None
        if self.future_cond_dim > 0:
            # per-sample random start offset delta ~ Uniform{0, 1, 2}
            start = torch.randint(0, 3, (batch_size,), device=nactions.device)
            Hf = self.future_cond_horizon  # H^f

            # a^c = a_{t+delta : t+delta+H^f} per-sample with randomized start
            steps = torch.arange(Hf, device=nactions.device)
            idx = (start[:, None] + steps[None, :]).unsqueeze(-1)        # [B, H^f, 1]
            idx = idx.expand(-1, -1, nactions.shape[-1])                 # [B, H^f, action_dim]
            future_seq = torch.gather(nactions, dim=1, index=idx)        # [B, H^f, action_dim]
            future_cond = future_seq.reshape(batch_size, -1)             # [B, future_cond_dim]

            # Classifier-Free Guidance dropout: randomly zero out some a^c
            if self.cf_drop_prob > 0.0 and self.training:
                drop_mask = (torch.rand(batch_size, device=nactions.device) < self.cf_drop_prob)
                drop_mask = drop_mask.float().unsqueeze(1)               # [B, 1]
                future_cond = future_cond * (1.0 - drop_mask)
        ##########################################################################

        # Predict the noise residual (with optional future_cond for CFG)
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond, future_cond=future_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
