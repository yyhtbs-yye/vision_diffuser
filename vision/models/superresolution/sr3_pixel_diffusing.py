import torch
import torch.nn.functional as F

from vision.models.generation.pixel_diffusing import BaseDiffusionModel
class SR3PixelDiffusionModel(BaseDiffusionModel):

    def forward(self, x0, lc0):
        with torch.no_grad():
            B, C, H, W = x0.shape
            
            # !!!!this can be improved using SRCNN and PixelShuffle
            c0 = F.interpolate(lc0, size=(H, W), mode='bilinear', align_corners=False)
            network_in_use = self.network_ema if self.use_ema and hasattr(self, 'network_ema') else self.network
            hx1 = self.solver.solve(network_in_use, x0, c0)

        return torch.clamp(hx1, -1, 1)

    def training_step(self, batch, batch_idx):

        x1 = batch['gt']
        lc0 = batch['lq']
        B, C, H, W = x1.shape

        c0 = F.interpolate(lc0, size=(H, W), mode='bilinear', align_corners=False)

        # Add x0 to images
        x0 = torch.randn_like(x1)
        
        # Sample random timesteps
        timesteps = self.scheduler.sample_timesteps(B, self.device)
        
        # Add noise to images according to noise schedule
        xt = self.scheduler.perturb(x1, x0, timesteps)
        
        cxt = torch.cat((xt, c0), dim=1)

        targets = self.scheduler.get_targets(x1, x0, timesteps)
        predictions = self.network(cxt, timesteps)['sample']
        weights = self.scheduler.get_loss_weights(timesteps)
        
        loss = torch.mean(torch.mean((predictions - targets) ** 2, dim=[1, 2, 3]) * weights)
        
        # Manually optimize
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        
        # Update EMA model after each step
        if self.use_ema and self.global_step >= self.ema_start:
            self._update_ema()
        
        return loss
    
    def validation_step(self, batch, batch_idx):

        with torch.no_grad():
            x1, lc0 = batch['gt'], batch['lq']

            _, _, H, W = x1.shape
            
            x0 = torch.randn_like(x1)

            hx1 = self.forward(x0, lc0)

            img_mse = F.mse_loss(hx1, x1)

            self.log("val/img_mse", img_mse, on_step=False, on_epoch=True)

            for metric_name, metric in self.metrics.items():
                metric_val = metric(hx1, x1)
                self.log(f"val/{metric_name}", metric_val, on_step=False, on_epoch=True)

            if self.use_visualizer:
                named_imgs = {
                    'groundtruth': x1[:self.num_vis_samples],
                    'low-quality': F.interpolate(lc0[:self.num_vis_samples], 
                                                      size=(H, W), 
                                                      mode='bilinear', 
                                                      align_corners=False),
                    'sr-quality': hx1[:self.num_vis_samples],
                }
                self._visualize_validation(named_imgs, batch_idx)

        return {"val_loss": img_mse}
