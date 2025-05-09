import torch
from vision.models.generation.base import BaseDiffusionModel

class LatentDiffusionModel(BaseDiffusionModel):
    
    def __init__(self, model_config, train_config, validation_config):
        super().__init__(model_config, train_config, validation_config)
        

        # Optional encoder/decoder for latent diffusion models
        self.encoder = self.build_module(model_config['encoder']) if 'encoder' in model_config else None

        # disable gradient tracking for encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, z0):
        with torch.no_grad():
            network_in_use = self.model_ema if self.use_ema and hasattr(self, 'network_ema') else self.model
            hz1 = self.solver.solve(network_in_use, z0)
            hx1 = self.decode_latents(hz1)
            result = torch.clamp(hx1, -1, 1)
        
        return result
        
    def training_step(self, batch, batch_idx):
        x1 = batch['gt']
        
        batch_size, c, gt_h, gt_w = x1.shape
        
        # Encode ground truth images to latent space
        with torch.no_grad():
            z1 = self.encode_images(x1)
        
        # Initialize random noise in latent space
        z0 = torch.randn_like(z1)

        # Sample random timesteps
        timesteps = self.scheduler.sample_timesteps(batch_size, self.device)
        
        # Add noise to latents according to noise schedule
        zt = self.scheduler.perturb(z1, z0, timesteps)

        targets = self.scheduler.get_targets(z1, z0, timesteps)
        predictions = self.model(zt, timesteps)['sample']
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
        x1 = batch['gt']
        batch_size, c, gt_h, gt_w = x1.shape
        
        with torch.no_grad():
            z1 = self.encode_images(x1)
            z0 = torch.randn_like(z1)
            
            timesteps = self.scheduler.sample_timesteps(batch_size, self.device)
            zt = self.scheduler.perturb(z1, z0, timesteps)

            network_in_use = self.model_ema if self.use_ema and hasattr(self, 'network_ema') else self.model

            targets = self.scheduler.get_targets(z1, z0, timesteps)
            predictions = network_in_use(zt, timesteps)['sample']

            loss_weights = self.scheduler.get_loss_weights(timesteps)
            noise_mse = torch.mean(torch.mean((predictions - targets) ** 2, dim=[1, 2, 3]) * loss_weights)

            self.log("val/noise_mse", noise_mse, on_step=False, on_epoch=True)

            # Generate images from noise
            hx1 = self.forward(z0)

            for metric_name, metric in self.metrics.items():
                metric_val = metric(hx1, x1)  # Compare against original pixel space images
                self.log(f"val/{metric_name}", metric_val, on_step=False, on_epoch=True)

            if self.use_visualizer:
                named_imgs = {
                    'groundtruth': x1[:self.num_vis_samples] if self.num_vis_samples is not None else x1,
                    'generated': hx1[:self.num_vis_samples] if self.num_vis_samples is not None else hx1,
                }
                self._visualize_validation(named_imgs, batch_idx)

        return {"val_loss": noise_mse}
    
    def decode_latents(self, latents):
        """Decode latents to images using the VAE."""
        # Scale latents according to VAE configuration
        latents = 1 / self.encoder.config.scaling_factor * latents
        image = self.encoder.decode(latents)
            
        return image
    def encode_images(self, images):
        """Encode images to latents using the VAE."""
        # Encode images to latent space
        latents = self.encoder.encode(images)
        latents = latents * self.encoder.config.scaling_factor
        
        return latents
    
    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        if self.encoder is not None:
            checkpoint['encoder'] = self.encoder.state_dict()

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if self.encoder is not None and 'encoder' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder'])
