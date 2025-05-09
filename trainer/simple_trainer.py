import os
import torch
from pathlib import Path
from tqdm import tqdm
from trainer.utils.path_helpers import determine_run_folder
from trainer.utils.state_load_save import load_state, save_state
class Trainer:
    def __init__(
        self, boat, max_epochs=10, device='cpu', 
        callbacks=None, log_every_n_steps=50,
        val_check_steps=None, val_check_epochs=None,
        state_save_steps=None, state_save_epochs=None,
        logger=None,
        root_dir='./', resume_from=None,
    ):
        self.max_epochs = max_epochs
        self.device = torch.device(device)
        self.callbacks = callbacks or []
        self.log_every_n_steps = log_every_n_steps
        self.val_check_steps = val_check_steps
        self.val_check_epochs = val_check_epochs
        self.state_save_steps = state_save_steps
        self.state_save_epochs = state_save_epochs
        self.resume_from = resume_from
        self.root_dir = Path(root_dir)
        self.logger = logger
        
        # Set up the run folder at initialization
        self.run_folder = determine_run_folder(self.root_dir)

        boat.configure_optimizers()
        
        # Resume from checkpoint if specified
        if resume_from:
            resume_from = Path(resume_from)
            boat, metadata = boat.load_state(resume_from)
            self.global_step = metadata['global_step']
            self.start_epoch = metadata['epoch']
        else:
            self.global_step = 0
            self.start_epoch = 0

        self.boat = boat

    def fit(self, train_dataloader, val_dataloader=None):
        
        self.boat.to(self.device)

        for cb in self.callbacks:
            cb.on_train_start(self, self.boat)

        for epoch in range(self.start_epoch, self.max_epochs):
            for cb in self.callbacks:
                cb.on_epoch_start(self, self.boat, epoch)

            self.boat.train()
            for batch_idx, batch in tqdm(enumerate(train_dataloader)):
                self.global_step += 1
                    
                batch = self._move_batch_to_device(batch)

                for cb in self.callbacks:
                    cb.on_batch_start(self, self.boat, batch, batch_idx)
                
                loss = self.boat.training_step(batch, batch_idx)

                self.boat.lr_scheduling_step()

                for cb in self.callbacks:
                    cb.on_batch_end(self, self.boat, batch, batch_idx, loss)

                if self.val_check_steps is not None and self.global_step % self.val_check_steps == 0:
                    if  val_dataloader is not None: 
                        self._run_validation(self.boat, val_dataloader, self.global_step)

                if self.state_save_steps is not None and self.global_step % self.state_save_steps == 0:
                        self.boat.save_state(self.run_folder, 'boat_state', global_step=self.global_step+1)

            if self.val_check_epochs is not None and epoch % self.val_check_epochs == 0:
                if val_dataloader is not None:
                    self._run_validation(val_dataloader, self.global_step, end_of_epoch=True)

            if self.state_save_epochs is not None and epoch % self.state_save_epochs == 0:
                self.boat.save_state(self.run_folder, 'boat_state', epoch=epoch+1)

            # epoch end
            for cb in self.callbacks:
                cb.on_epoch_end(self, self.boat, epoch)

        # on train end
        for cb in self.callbacks:
            cb.on_train_end(self, self.boat)

    def _run_validation(self, val_dataloader, global_step, end_of_epoch=False):

        for cb in self.callbacks:
            cb.on_validation_start(self, self.boat)

        self.boat.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch = self._move_batch_to_device(batch)
                out = self.boat.validation_step(batch, batch_idx)
                for cb in self.callbacks:
                    cb.on_validation_batch_end(self, self.boat, batch, batch_idx, outputs=out)

        for cb in self.callbacks:
            cb.on_validation_end(self, self.boat)

    def _move_batch_to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return [self._move_batch_to_device(x) for x in batch]
        elif isinstance(batch, dict):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        elif hasattr(batch, 'to'):
            return batch.to(self.device)
        else:
            return batch
