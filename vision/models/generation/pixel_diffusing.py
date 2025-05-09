import torch
from vision.models.base.base_diffusing import BaseDiffusionBoat

class PixelDiffusionBoat(BaseDiffusionBoat):
    
    def forward(self, x0):

        network_in_use = self.models['model_ema'] if self.use_ema and 'model_ema' in self.models else self.models['model']

        hx1 = self.models['solver'].solve(network_in_use, x0)
        
        return torch.clamp(hx1, -1, 1)
        
    def training_step(self, batch, batch_idx):

        x1 = batch['gt']
        
        batch_size, c, gt_h, gt_w = x1.shape
        
        # Add noise to images
        x0 = torch.randn_like(x1)

        # Sample random timesteps
        timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)
        
        # Add noise to images according to noise schedule
        xt = self.models['scheduler'].perturb(x1, x0, timesteps)

        targets = self.models['scheduler'].get_targets(x1, x0, timesteps)
        predictions = self.models['model'](xt, timesteps)['sample']
        weights = self.models['scheduler'].get_loss_weights(timesteps)

        loss = self.losses['model'](predictions, targets, weights=weights)

        self._step(self, loss)
        
        # Log metrics
        self.trainer.log("train/noise_mse", loss, prog_bar=True)
        
        # Update EMA
        if self.use_ema and self.trainer.global_step >= self.ema_start:
            self._update_ema()
        
        return loss
    
    def validation_step(self, batch, batch_idx):

        x1 = batch['gt']
        batch_size, c, gt_h, gt_w = x1.shape
        
        with torch.no_grad():
            timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)

            x0 = torch.randn_like(x1)

            xt = self.models['scheduler'].perturb(x1, x0, timesteps)

            network_in_use = self.models['model'] if self.use_ema and hasattr(self, 'network_ema') else self.models['model']

            targets = self.models['scheduler'].get_targets(x1, x0, timesteps)
            predictions = network_in_use(xt, timesteps)['sample']

            weights = self.models['scheduler'].get_loss_weights(timesteps)

            loss = self.losses['model'](predictions, targets, weights=weights)

            self.trainer.log("val/noise_mse", loss)

            hx1 = self.forward(x0)

            self._log_metrics(hx1, x1)

            if self.use_visualizer:
                named_imgs = {
                    'groundtruth': x1,
                    'generated': hx1,
                }
                self._visualize_validation(named_imgs, batch_idx)

        return loss

    # def _prepare_visualization(self):
    #     super()._prepare()
        
    #     # Set up the model
    #     self.models['model'].eval()
    #     if self.use_ema:
    #         self.models['model_ema'].eval()
        
    #     # Set up the scheduler
    #     self.models['scheduler'].set_timesteps(self.num_timesteps)
        
    #     # Set up the loss function
    #     self.losses['model'].to(self.device)