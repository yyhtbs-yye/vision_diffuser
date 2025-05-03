import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy
import importlib
from abc import ABC, abstractmethod
from vision.visualizers.basic_visualizer import visualize_comparisons
from vision.utils.metric import setup_metrics

class BaseDiffusionModel(pl.LightningModule, ABC):
    """Base class for diffusion models, providing common functionality."""

    @staticmethod
    def get_class(path, name):
        """Dynamically import a class from a module."""
        module = importlib.import_module(path)
        return getattr(module, name)

    @staticmethod
    def build_module(config):
        """Build a module from configuration."""
        class_ = BaseDiffusionModel.get_class(config['path'], config['name'])
        if not hasattr(class_, 'from_config'):
            raise AttributeError(f"Class {config['name']} must implement 'from_config' method")
        
        if 'pretrained' in config:
            return class_.from_pretrained(config['pretrained'])
        else:
            return class_.from_config(config['config'])

    def __init__(self, model_config, train_config, validation_config):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=['model_config'])

        # Initialize core components
        self.network = self.build_module(model_config['network'])
        self.scheduler = self.build_module(model_config['scheduler'])
        self.solver = self.build_module(model_config['solver'])

        self.optimizer_config = train_config['optimizer']
        self.metric_config = validation_config.get('metrics', {})
        self.metrics = setup_metrics(self.metric_config, None)
        self._setup_ema()

        self.use_visualizer = validation_config.get('use_visualizer', True)
        self.num_vis_samples = validation_config.get('num_vis_samples', 4)

    @abstractmethod
    def forward(self, noise):
        """Generate images from noise."""
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """Define the training step."""
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """Define the validation step."""
        pass

    def configure_optimizers(self):
        """Configure the optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config.get('learning_rate', 1e-4),
            betas=tuple(self.optimizer_config.get('betas', [0.9, 0.999])),
            weight_decay=self.optimizer_config.get('weight_decay', 0.0)
        )
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """Save model components to checkpoint."""
        checkpoint['network'] = self.network.state_dict()

        if self.use_ema and hasattr(self, 'network_ema'):
            checkpoint['network_ema'] = self.network_ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        """Load model components from checkpoint."""
        if 'network' in checkpoint:
            self.network.load_state_dict(checkpoint['network'])
        if self.use_ema and 'network_ema' in checkpoint and hasattr(self, 'network_ema'):
            self.network_ema.load_state_dict(checkpoint['network_ema'])

    def _setup_ema(self):
        """Set up Exponential Moving Average (EMA) model."""
        self.use_ema = self.optimizer_config.get('use_ema', False)

        if self.use_ema:
            self.network_ema = deepcopy(self.network)
            for param in self.network_ema.parameters():
                param.requires_grad = False
    
        self.ema_decay = self.optimizer_config.get('ema_decay', 0.999)
        self.ema_start = self.optimizer_config.get('ema_start', 1000)

    def _update_ema(self):
        """Update EMA model parameters."""
        if not self.use_ema:
            return
        for ema_param, param in zip(self.network_ema.parameters(), self.network.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        for ema_buffer, buffer in zip(self.network_ema.buffers(), self.network.buffers()):
            ema_buffer.data.copy_(buffer.data)

    def _visualize_validation(self, named_imgs, batch_idx, texts=None):
        """Visualize validation results."""
        if batch_idx == 0:
            
            visualize_comparisons(
                logger=self.logger.experiment,
                images_dict=named_imgs,
                keys=list(named_imgs.keys()),
                global_step=self.global_step,
                wnb=(0.5, 0.5),
                prefix='val',
                texts=texts,
            )