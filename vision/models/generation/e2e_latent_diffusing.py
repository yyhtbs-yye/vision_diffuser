import torch
import torch.nn.functional as F
from vision.models.generation.base import BaseDiffusionModel
from vision.models.generation.latent_diffusing import LatentDiffusionModel


class End2endTrainableLatentDiffusion(LatentDiffusionModel):

    def __init__(self, model_config, train_config, validation_config):
        BaseDiffusionModel.__init__(self, model_config, train_config, validation_config)

        # Optional encoder/decoder for latent diffusion models
        self.encoder = self.build_module(model_config['encoder']) if 'encoder' in model_config else None

        self.noise_estimation_weight = train_config['loss'].get('noise_estimation_weight', 0.4)
        self.image_denoise_weight = train_config['loss'].get('image_denoise_weight', 0.3)
        self.image_decode_weight = train_config['loss'].get('image_decode_weight', 0.3)
    
    def training_step(self, batch, batch_idx):
        x1 = batch['gt']
        
        batch_size, c, gt_h, gt_w = x1.shape
        
        # Encode ground truth images to latent space
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
        
        noise_mse = torch.mean(torch.mean((predictions - targets) ** 2, dim=[1, 2, 3]) * weights)

        hz1 = self.scheduler.reverse(zt, predictions, timesteps)

        hx1 = self.decode_latents(hz1)

        d_hx1 = self.decode_latents(self.encode_images(x1))
        
        image_mse = F.mse_loss(hx1, x1)
        vae_image_mse = F.mse_loss(d_hx1, x1)
        total_loss = self.noise_estimation_weight * noise_mse \
                        + self.image_denoise_weight * image_mse \
                            + self.image_decode_weight * vae_image_mse

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()

        self.log_dict({
            "train/total_loss": total_loss,
            "train/noise_mse": noise_mse,
            "train/image_mse": image_mse,
            "train/vae_image_mse": vae_image_mse
        }, prog_bar=True)

        if self.use_ema and self.global_step >= self.ema_start:
            self._update_ema()

        return total_loss
    