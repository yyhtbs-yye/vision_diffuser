from trainer.boats.basic_boat import BaseBoat
from trainer.utils.build_components import build_module, build_modules, build_optimizer

from vision.visualizers.basic_visualizer import visualize_image_dict

from copy import deepcopy

class BaseDiffusionBoat(BaseBoat):

    def __init__(self, boat_config=None, optimizer_config=None, valid_config=None, use_ema=False):
        super().__init__()
        
        self.models['model'] = build_module(boat_config['model'])
        self.models['scheduler'] = build_module(boat_config['scheduler'])
        self.models['solver'] = build_module(boat_config['solver'])

        self.boat_config = boat_config
        self.optimizer_config = optimizer_config
        self.valid_config = valid_config
        self.use_ema = self.optimizer_config.get('use_ema', False)
        
        if use_ema:
            self._setup_ema()

    def attach_trainer(self, trainer):
        self.trainer = trainer

    def configure_optimizers(self):
        self.optimizers['model'] = build_optimizer(self.models['model'], self.optimizer_config['model'])
        self.lr_schedulers['model'] = build_module(self.optimizer_config.get('lr_scheduler', {}).get('model', None))

    def configure_losses(self):
        self.losses['model'] = build_module(self.boat_config.get('loss', {}).get('model', None))

    def configure_metrics(self):
        self.metrics = build_modules(self.valid_config.get('metrics', {}))

    def _setup_ema(self):
        """Set up Exponential Moving Average (EMA) model."""

        if self.use_ema:
            self.models['model_ema'] = deepcopy(self.models['model'])
            for param in self.models['model_ema'].parameters():
                param.requires_grad = False
    
        self.ema_decay = self.use_ema.get('ema_decay', 0.999)
        self.ema_start = self.use_ema.get('ema_start', 1000)

    def _update_ema(self):
        """Update EMA model parameters."""
        if not self.use_ema: return

        for ema_param, param in zip(self.models['model_ema'].parameters(), self.models['model'].parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        for ema_buffer, buffer in zip(self.models['model_ema'].buffers(), self.models['model'].buffers()):
            ema_buffer.data.copy_(buffer.data)

    def _step(self, loss):
        # Manually optimize
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

    def _log_metrics(self, predictions, targets):
        for metric_name, metric in self.metrics.items():
            metric_val = metric(predictions, targets)
            self.trainer.log(f"val/{metric_name}", metric_val)

    def _visualize_validation(self, named_imgs, batch_idx, num_vis_samples=4, first_batch_only=True, texts=None):
        """Visualize validation results."""
        if first_batch_only and batch_idx == 0:
            # Limit the number of samples to visualize
            for key in named_imgs.keys():
                if named_imgs[key].shape[0] > num_vis_samples:
                    named_imgs[key] = named_imgs[key][:num_vis_samples]
            
            # Log visualizations to the experiment tracker
            visualize_image_dict(
                logger=self.logger.experiment,
                images_dict=named_imgs,
                keys=list(named_imgs.keys()),
                global_step=self.global_step,
                wnb=(0.5, 0.5),
                prefix='val',
                texts=texts,
            )