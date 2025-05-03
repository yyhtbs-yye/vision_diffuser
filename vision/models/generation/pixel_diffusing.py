import torch
from vision.models.generation.base import BaseDiffusionModel
class PixelDiffusionModel(BaseDiffusionModel):
    
    def forward(self, x0):
        network_in_use = self.network_ema if self.use_ema and hasattr(self, 'network_ema') else self.network
        hx1 = self.solver.solve(network_in_use, x0)
        return torch.clamp(hx1, -1, 1)
        
    def training_step(self, batch, batch_idx):

        x1 = batch['gt']
        
        batch_size, c, gt_h, gt_w = x1.shape
        
        # Add noise to images
        x0 = torch.randn_like(x1)

        # Sample random timesteps
        timesteps = self.scheduler.sample_timesteps(batch_size, self.device)
        
        # Add noise to images according to noise schedule
        xt = self.scheduler.perturb(x1, x0, timesteps)

        targets = self.scheduler.get_targets(x1, x0, timesteps)
        predictions = self.network(xt, timesteps)['sample']
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
            timesteps = self.scheduler.sample_timesteps(batch_size, self.device)

            x0 = torch.randn_like(x1)

            xt = self.scheduler.perturb(x1, x0, timesteps)

            network_in_use = self.network_ema if self.use_ema and hasattr(self, 'network_ema') else self.network

            targets = self.scheduler.get_targets(x1, x0, timesteps)
            predictions = network_in_use(xt, timesteps)['sample']

            loss_weights = self.scheduler.get_loss_weights(timesteps)

            noise_mse = torch.mean(torch.mean((predictions - targets) ** 2, dim=[1, 2, 3]) * loss_weights)

            self.log("val/noise_mse", noise_mse, on_step=False, on_epoch=True)

            hx1 = self.forward(x0)

            for metric_name, metric in self.metrics.items():
                metric_val = metric(hx1, x1)
                self.log(f"val/{metric_name}", metric_val, on_step=False, on_epoch=True)

            if self.use_visualizer:
                named_imgs = {
                    'groundtruth': x1[:self.num_vis_samples] if self.num_vis_samples is not None else x1,
                    'generated': hx1[:self.num_vis_samples] if self.num_vis_samples is not None else hx1,
                }
                self._visualize_validation(named_imgs, batch_idx)

        return {"val_loss": noise_mse}

