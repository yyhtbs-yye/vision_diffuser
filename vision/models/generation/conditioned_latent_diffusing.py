import torch
import torch.nn as nn
from vision.models.generation.latent_diffusing import LatentDiffusionModel
from vision.condition_generators.simple_generating import SimpleGenerator

import psutil
import os

class ConditionedLatentDiffusionModel(LatentDiffusionModel):
    
    def __init__(self, model_config, train_config, validation_config):
        super().__init__(model_config, train_config, validation_config)
        
        # Condition mapper processes conditioning inputs (text, class labels, etc.)
        self.condition_encoder = self.build_module(model_config['condition_encoder']) if 'condition_encoder' in model_config else nn.Identity()

        self.condition_generator = SimpleGenerator(model_config['condition_generator']['config']['dims'])


    def on_train_epoch_start(self):
        main_process = psutil.Process(os.getpid())
        print(f"Epoch {self.current_epoch}, Main RSS: {main_process.memory_info().rss / 1024**2:.2f} MB")
        children = main_process.children(recursive=True)
        for i, child in enumerate(children):
            try:
                print(f"Worker {i}, PID {child.pid}, RSS: {child.memory_info().rss / 1024**2:.2f} MB")
            except psutil.NoSuchProcess:
                pass

    def forward(self, z0, conditions=None):
        """Generate images from noise with conditioning."""
        with torch.no_grad():
            network_in_use = self.network_ema if self.use_ema and hasattr(self, 'network_ema') else self.network

            encoder_hidden_states = self.condition_encoder(conditions)
            
            hz1 = self.solver.solve(network_in_use, z0, encoder_hidden_states)
            hx1 = self.decode_latents(hz1)
            result = torch.clamp(hx1, -1, 1)
        
        return result
        
    def training_step(self, batch, batch_idx):
        x1 = batch['gt']
        conditions = batch['cond']  # Get conditioning from batch
        
        batch_size, c, gt_h, gt_w = x1.shape
        
        # Encode ground truth images to latent space
        with torch.no_grad():
            z1 = self.encode_images(x1)
            encoder_hidden_states = self.condition_encoder(conditions)
        
        # Initialize random noise in latent space
        z0 = torch.randn_like(z1)

        # Sample random timesteps
        timesteps = self.scheduler.sample_timesteps(batch_size, self.device)
        
        # Add noise to latents according to noise schedule
        zt = self.scheduler.perturb(z1, z0, timesteps)

        targets = self.scheduler.get_targets(z1, z0, timesteps)
        
        # Pass mapped condition to the network via encoder_hidden_states [B, T, D]
        predictions = self.network(sample=zt, timestep=timesteps, encoder_hidden_states=encoder_hidden_states)['sample']
        
        weights = self.scheduler.get_loss_weights(timesteps)
        loss = torch.mean(torch.mean((predictions - targets) ** 2, dim=[1, 2, 3]) * weights)

        # Manually optimize
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        
        # Log metrics
        self.log("train/noise_mse", loss, prog_bar=True)
        
        # Update EMA
        if self.use_ema and self.global_step >= self.ema_start:
            self._update_ema()
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        x1 = batch['gt']  # Get conditioning from batch
        
        batch_size, c, gt_h, gt_w = x1.shape

        conditions = self.condition_generator.generate(batch_size).to(self.device)
        
        with torch.no_grad():
            z1 = self.encode_images(x1)
            z0 = torch.randn_like(z1)

            # Generate images from noise without conditioning !!!! conditions=torch.zeros(batch_size, 1, 9) is HARD CODED, needs to be fixed
            hx1 = self.forward(z0, conditions=conditions)

            if self.use_visualizer:
                named_imgs = {
                    'groundtruth': x1[:self.num_vis_samples] if self.num_vis_samples is not None else x1,
                    'generated': hx1[:self.num_vis_samples] if self.num_vis_samples is not None else hx1,
                }
                self._visualize_validation(named_imgs, batch_idx, texts=[str(it).replace('tensor', '') for it in conditions])

        return {"val_loss": 0}
    
    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        if self.condition_encoder is not None:
            checkpoint['condition_encoder'] = self.condition_encoder.state_dict()

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if self.condition_encoder is not None and 'condition_encoder' in checkpoint:
            self.condition_encoder.load_state_dict(checkpoint['condition_encoder'])

