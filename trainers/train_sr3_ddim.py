import os
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger 

from vision.data_modules.image_data_module import ImageDataModule
from vision.models.superresolution.sr3_pixel_diffusing import SR3PixelDiffusionModel

from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta

torch.set_float32_matmul_precision('high')

config_path = "configs/train_sr3_ddim_16_128.yaml"  # Updated config path

# Load YAML configuration
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract configurations
model_config = config['model']
validation_config = config['validation']
train_config = config['train']
data_config = config['data']
logging_config = config['logging']
checkpoint_config = config['checkpoint']

# Create a new model or load from checkpoint, this is from the previous checkpoint
checkpoint_path = "work_dirs/sr3_ddim/sr3_16_128/checkpoints/last-v1.ckpt" 

if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = SR3PixelDiffusionModel.load_from_checkpoint(
        checkpoint_path, 
        model_config=model_config,
        train_config=train_config,
        validation_config=validation_config
    )
else:
    print("Creating new model")
    model = SR3PixelDiffusionModel(
        model_config=model_config,
        train_config=train_config,
        validation_config=validation_config
    )

# Create data module
data_module = ImageDataModule(data_config)

# Set up logger
logger = TensorBoardLogger(
    save_dir=logging_config['log_dir'],
    name=logging_config['experiment_name']
)

# Set up callbacks
callbacks = []

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(logging_config['log_dir'], 
                         logging_config['experiment_name'], 
                         'checkpoints'),
    filename='{epoch:02d}-{' + checkpoint_config['save_best_metric'] + ':.4f}',
    monitor=checkpoint_config['save_best_metric'],
    mode=checkpoint_config['save_best_mode'],
    save_top_k=checkpoint_config['save_top_k'],
    save_last=checkpoint_config['save_last']
)
callbacks.append(checkpoint_callback)

# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks.append(lr_monitor)

# Initialize trainer
trainer = Trainer(
    max_epochs=train_config['max_epochs'],
    accelerator="auto",
    devices="0,3,6,7" if torch.cuda.is_available() else None,
    strategy=DDPStrategy(
        find_unused_parameters=False,  # Optional: Disable if not needed
        timeout=timedelta(seconds=3600)  # Set timeout to 1 hour
    ),

    logger=logger,
    callbacks=callbacks,
    val_check_interval=train_config['val_check_interval'],
    log_every_n_steps=logging_config['log_every_n_steps']
)

# Train model
trainer.fit(model=model, datamodule=data_module)