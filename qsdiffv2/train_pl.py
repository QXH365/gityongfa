import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from glob import glob
import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
# --- PyTorch Lightning Imports ---
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

# --- Original Project Imports ---
from models.epsnet import get_model
from utils.datasets import ConformationDataset
from utils.misc import seed_all, get_new_log_dir
from utils.common import get_optimizer, get_scheduler

# Suppress a known harmless warning from PyTorch Geometric and Lightning
import warnings
warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*")

# ------------------------------------------------------------------------------------
# 1. The LightningModule: Encapsulates Model, Training, and Optimization Logic
# ------------------------------------------------------------------------------------
class ConformationLitModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Save config to self.hparams and make it accessible everywhere
        self.save_hyperparameters(config)
        self.config = config # Use self.config for easier access
        self.model = get_model(self.config.model)

    def forward(self, batch):
        # The forward pass can be defined if needed, but for this model,
        # the main logic is in get_loss.
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        loss, loss_global, loss_local = self.model.get_loss(
            batch=batch,
            anneal_power=self.config.train.anneal_power,
        )

        # Log metrics using self.log
        self.log('train/loss', loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch.num_graphs)
        self.log('train/loss_global', loss_global, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch.num_graphs)
        self.log('train/loss_local', loss_local, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch.num_graphs)
        
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.model.get_loss(
            batch=batch,
            anneal_power=self.config.train.anneal_power,
        )
        
        # Log validation loss for each batch
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config.train.optimizer, self)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.train.max_iters, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 核心：将更新间隔设置为'step'
                "frequency": 1,      # 每个step都更新
            },
        }
        '''
        scheduler = get_scheduler(self.config.train.scheduler, optimizer)
        # For schedulers like ReduceLROnPlateau, Lightning needs this format
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_epoch",  # Monitor the validation loss
                    "frequency": 1
                },
            }
        else:
            return [optimizer], [scheduler]
        '''


# ------------------------------------------------------------------------------------
# 2. The LightningDataModule: Encapsulates Data Loading and Preparation
# ------------------------------------------------------------------------------------
class ConformationDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # This is called on every GPU in DDP
        full_dataset = ConformationDataset(self.config.dataset.train, transform=None)
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # Use a fixed generator for reproducible splits
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.train.seed)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.config.train.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        # The original code used batch_size=1 for validation, we'll keep that
        return DataLoader(
            self.val_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=self.config.train.num_workers,
            pin_memory=True
        )

# ------------------------------------------------------------------------------------
# 3. Main Execution Script
# ------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the .yml config file.")
    parser.add_argument('--logdir', type=str, default='./logs_lightning')
    # --- Trainer arguments can be added here or passed via command line ---
    # Example: parser.add_argument('--devices', type=int, default=1)
    args = parser.parse_args()

    # --- Load Configuration ---
    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(args.config).split('.')[0]
    seed_all(config.train.seed)
    
    # Add num_workers to config if not present for the DataModule
    if 'num_workers' not in config.train:
        config.train.num_workers = 4

    # --- Setup Data and Model Modules ---
    data_module = ConformationDataModule(config)
    model_module = ConformationLitModule(config)

    # --- Setup Logging and Checkpointing ---
    logger = TensorBoardLogger(save_dir=args.logdir, name=config_name, default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, 'checkpoints'),
        monitor='val/loss',
        save_top_k=5,
        mode='min',
        filename='iter={step}-val_loss={val/loss:.4f}'
    )
    progress_bar_callback = TQDMProgressBar(refresh_rate=10)

    # --- Initialize and Run the Trainer ---
    # The Trainer handles multi-GPU training, checkpointing, logging, etc.
    trainer = pl.Trainer(
        # Multi-GPU training flags
        accelerator='gpu',    # Use 'gpu' or 'cpu'
        devices=-1,           # Use all available GPUs (-1) or specify a number [0, 1, 2]
        strategy='ddp',       # Distributed Data Parallel for multi-GPU
        
        # Training configuration
        max_steps=config.train.max_iters,
        val_check_interval=config.train.val_freq,
        check_val_every_n_epoch=None, # Use step-based validation
        gradient_clip_val=config.train.max_grad_norm,
        
        # Logging and callbacks
        logger=logger,
        callbacks=[checkpoint_callback, progress_bar_callback]
    )

    print("--- Starting Training with PyTorch Lightning ---")
    trainer.fit(model_module, datamodule=data_module)