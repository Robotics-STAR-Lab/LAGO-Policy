"""
Training workspace for Goal-Aware Diffusion Policy.

Trains the policy to jointly predict actions and task-relevant goals.
"""

import os
import copy
import pathlib
import hydra
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.goal_aware_diffusion_policy import GoalAwareDiffusionPolicy
from diffusion_policy.dataset.goal_aware_umi_dataset import GoalAwareUmiDataset
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.common.json_logger import JsonLogger

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainGoalAwareWorkspace(BaseWorkspace):
    """
    Workspace for training Goal-Aware Diffusion Policy.
    """
    include_keys = ['global_step', 'epoch']
    
    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        
        self.cfg = cfg
        self._output_dir = cfg.output_dir
        
        # Setup device
        self.device = torch.device(cfg.training.device)
        
        # Build policy (don't move to device yet - wait for normalizer)
        policy: GoalAwareDiffusionPolicy = hydra.utils.instantiate(cfg.policy)
        
        # Setup EMA
        ema_model = None
        if cfg.training.use_ema:
            ema_model = copy.deepcopy(policy)
        
        self.policy = policy
        self.ema_model = ema_model
        self.global_step = 0
        self.epoch = 0
    
    def run(self):
        cfg = self.cfg
        
        # Build dataset
        print("Loading dataset...")
        dataset: GoalAwareUmiDataset = hydra.utils.instantiate(cfg.task.dataset)
        
        # Build normalizer and set it BEFORE moving to device
        normalizer = dataset.get_normalizer()
        self.policy.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
        
        # Now move to device (normalizer will be moved together)
        self.policy.to(self.device)
        if self.ema_model is not None:
            self.ema_model.to(self.device)
        
        # Setup optimizer (after moving to device)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=cfg.training.lr,
            betas=cfg.training.betas,
            weight_decay=cfg.training.weight_decay
        )
        
        # Get validation dataset
        val_dataset = dataset.get_validation_dataset()
        
        # Build dataloaders
        train_loader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            persistent_workers=cfg.training.num_workers > 0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True
        )
        
        # Setup LR scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=len(train_loader) * cfg.training.num_epochs
        )
        
        # Setup EMA
        ema = None
        if self.ema_model is not None:
            ema = EMAModel(
                model=self.ema_model,
                power=cfg.training.ema_power
            )
        
        # Setup logging
        if cfg.logging.mode == 'online':
            wandb.init(
                project=cfg.logging.project,
                name=cfg.logging.name,
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=self._output_dir
            )
        
        json_logger = JsonLogger(os.path.join(self._output_dir, 'logs.json.txt'))
        json_logger.start()
        
        # Training loop
        for epoch in range(self.epoch, cfg.training.num_epochs):
            self.epoch = epoch
            self.policy.train()
            
            epoch_losses = {'loss': [], 'diffusion_loss': [], 'goal_loss': []}
            
            with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
                for batch in pbar:
                    # Move batch to device and convert to float32
                    batch = {
                        'obs': dict_apply(batch['obs'], lambda x: x.to(self.device, dtype=torch.float32)),
                        'action': batch['action'].to(self.device, dtype=torch.float32),
                        'goal': batch['goal'].to(self.device, dtype=torch.float32),
                    }
                    
                    # Compute loss
                    loss_dict = self.policy.compute_loss(batch)
                    loss = loss_dict['loss']
                    
                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        cfg.training.max_grad_norm
                    )
                    self.optimizer.step()
                    lr_scheduler.step()
                    
                    # Update EMA
                    if ema is not None:
                        ema.step(self.policy)
                    
                    # Log
                    for key, value in loss_dict.items():
                        epoch_losses[key].append(value.item())
                    
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'diff': f"{loss_dict['diffusion_loss'].item():.4f}",
                        'goal': f"{loss_dict['goal_loss'].item():.4f}"
                    })
                    
                    self.global_step += 1
            
            # Compute epoch averages
            avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
            
            # Validation
            self.policy.eval()
            val_losses = {'loss': [], 'diffusion_loss': [], 'goal_loss': []}
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {
                        'obs': dict_apply(batch['obs'], lambda x: x.to(self.device, dtype=torch.float32)),
                        'action': batch['action'].to(self.device, dtype=torch.float32),
                        'goal': batch['goal'].to(self.device, dtype=torch.float32),
                    }
                    loss_dict = self.policy.compute_loss(batch)
                    for key, value in loss_dict.items():
                        val_losses[key].append(value.item())
            
            avg_val_losses = {k: sum(v) / len(v) if v else 0 for k, v in val_losses.items()}
            
            # Log to wandb
            log_dict = {
                'epoch': epoch,
                'train/loss': avg_losses['loss'],
                'train/diffusion_loss': avg_losses['diffusion_loss'],
                'train/goal_loss': avg_losses['goal_loss'],
                'val/loss': avg_val_losses['loss'],
                'val/diffusion_loss': avg_val_losses['diffusion_loss'],
                'val/goal_loss': avg_val_losses['goal_loss'],
                'lr': lr_scheduler.get_last_lr()[0]
            }
            
            if cfg.logging.mode == 'online':
                wandb.log(log_dict, step=self.global_step)
            
            json_logger.log(log_dict)
            
            print(f"Epoch {epoch}: train_loss={avg_losses['loss']:.4f}, "
                  f"val_loss={avg_val_losses['loss']:.4f}, "
                  f"goal_loss={avg_losses['goal_loss']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % cfg.training.checkpoint_every == 0:
                self.save_checkpoint(
                    tag=f"epoch={epoch:04d}-val_loss={avg_val_losses['loss']:.5f}"
                )
        
        # Final save
        self.save_checkpoint(tag="latest")
        
        if cfg.logging.mode == 'online':
            wandb.finish()
    
    def save_checkpoint(self, tag='latest'):
        """Save checkpoint."""
        path = os.path.join(self._output_dir, 'checkpoints', f'{tag}.ckpt')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state_dict = {
            'cfg': OmegaConf.to_container(self.cfg, resolve=True),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'policy': self.policy.state_dict(),
        }
        
        if self.ema_model is not None:
            state_dict['ema_model'] = self.ema_model.state_dict()
        
        torch.save(state_dict, path)
        print(f"Saved checkpoint: {path}")
