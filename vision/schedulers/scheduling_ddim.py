from diffusers import DDIMScheduler
import torch

class DDIMScheduler(DDIMScheduler):

    def sample_timesteps(self, batch_size, device, low=0, high=None):
        
        high = high or self.config.num_train_timesteps
        return torch.randint(low, high, (batch_size,), device=device).long()
    
    def reverse(self, noisy_imgs, noise, timesteps):
        # Ensure timesteps are a tensor
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, device=noisy_imgs.device)
            
        # Get the noise coefficients used in the forward process
        alphas = self.alphas_cumprod[timesteps].to(noisy_imgs.device)
        sigmas = (1 - alphas).sqrt()
        
        # Apply the inverse operation
        denoised_imgs = (noisy_imgs - noise * sigmas.view(-1, 1, 1, 1)) / alphas.sqrt().view(-1, 1, 1, 1)
        return denoised_imgs
    
    def perturb(self, imgs, noise, timesteps):
        return self.add_noise(imgs, noise, timesteps)

    def get_targets(self, imgs, noises, timesteps):
        return noises
    
    def get_loss_weights(self, timesteps):
        return torch.ones_like(timesteps)