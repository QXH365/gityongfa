import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm

from fusion_module import AdvancedFusionModule
from cib_core import AdaptiveInformationBottleneck
from scaffold_encoder import MultiResolutionScaffoldEncoder
from sidechain_predictor import AdaptiveSideChainClassifier


class CombinedLossFunction(nn.Module):
    def __init__(
        self,
        prediction_weight: float = 1.0,
        cib_weight: float = 0.1,
        scaffold_weight: float = 0.05,
        sidechain_weight: float = 0.15,
        consistency_weight: float = 0.02,
        diversity_weight: float = 0.01
    ):
        super().__init__()
        self.prediction_weight = prediction_weight
        self.cib_weight = cib_weight
        self.scaffold_weight = scaffold_weight
        self.sidechain_weight = sidechain_weight
        self.consistency_weight = consistency_weight
        self.diversity_weight = diversity_weight
        
    def compute_prediction_loss(
        self,
        predicted_logits: torch.Tensor,
        target_edges: torch.Tensor,
        scaffold_mask: torch.Tensor,
        loss_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        loss_mask = (1 - scaffold_mask).float()
        
        prediction_loss = F.cross_entropy(
            predicted_logits.view(-1, predicted_logits.size(-1)),
            target_edges.view(-1),
            reduction='none',
            weight=loss_weights,
            label_smoothing=0.1
        )
        
        prediction_loss = prediction_loss.view(target_edges.shape)
        masked_loss = prediction_loss * loss_mask
        
        return masked_loss.sum() / (loss_mask.sum() + 1e-8)
    
    def compute_consistency_loss(
        self,
        predicted_logits: torch.Tensor,
        module_outputs: Dict[str, Any]
    ) -> torch.Tensor:
        
        if 'sidechain' not in module_outputs:
            return torch.tensor(0.0, device=predicted_logits.device)
        
        sidechain_predictions = module_outputs['sidechain']['predictions']
        
        edge_probs = F.softmax(predicted_logits, dim=-1)
        sidechain_probs = F.softmax(sidechain_predictions, dim=-1)
        
        consistency_loss = F.kl_div(
            F.log_softmax(predicted_logits, dim=-1),
            sidechain_probs,
            reduction='batchmean'
        )
        
        return consistency_loss
    
    def compute_diversity_loss(
        self,
        predicted_logits: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        
        edge_probs = F.softmax(predicted_logits, dim=-1)
        
        batch_mean_probs = edge_probs.view(batch_size, -1, edge_probs.size(-1)).mean(dim=1)
        
        entropy = -torch.sum(batch_mean_probs * torch.log(batch_mean_probs + 1e-8), dim=-1)
        diversity_loss = -entropy.mean()
        
        return diversity_loss
    
    def forward(
        self,
        predicted_logits: torch.Tensor,
        target_edges: torch.Tensor,
        scaffold_mask: torch.Tensor,
        module_outputs: Dict[str, Any],
        loss_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        batch_size = target_edges.size(0)
        
        prediction_loss = self.compute_prediction_loss(
            predicted_logits, target_edges, scaffold_mask, loss_weights
        )
        
        cib_loss = torch.tensor(0.0, device=predicted_logits.device)
        if 'cib' in module_outputs and 'loss' in module_outputs['cib']:
            cib_loss = module_outputs['cib']['loss']
        
        scaffold_loss = torch.tensor(0.0, device=predicted_logits.device)
        if 'scaffold' in module_outputs and 'importance' in module_outputs['scaffold']:
            importance = module_outputs['scaffold']['importance']
            scaffold_loss = F.binary_cross_entropy(
                importance.squeeze(-1),
                torch.ones_like(importance.squeeze(-1)) * 0.5
            )
        
        sidechain_loss = torch.tensor(0.0, device=predicted_logits.device)
        if 'sidechain' in module_outputs:
            sidechain_predictions = module_outputs['sidechain']['predictions']
            node_targets = target_edges.diagonal(dim1=-2, dim2=-1)
            sidechain_loss = F.cross_entropy(
                sidechain_predictions.view(-1, sidechain_predictions.size(-1)),
                node_targets.view(-1),
                reduction='mean'
            )
        
        consistency_loss = self.compute_consistency_loss(predicted_logits, module_outputs)
        diversity_loss = self.compute_diversity_loss(predicted_logits, batch_size)
        
        total_loss = (
            self.prediction_weight * prediction_loss +
            self.cib_weight * cib_loss +
            self.scaffold_weight * scaffold_loss +
            self.sidechain_weight * sidechain_loss +
            self.consistency_weight * consistency_loss +
            self.diversity_weight * diversity_loss
        )
        
        loss_components = {
            'prediction_loss': prediction_loss,
            'cib_loss': cib_loss,
            'scaffold_loss': scaffold_loss,
            'sidechain_loss': sidechain_loss,
            'consistency_loss': consistency_loss,
            'diversity_loss': diversity_loss
        }
        
        return total_loss, loss_components


class MultiStageTrainingScheduler:
    def __init__(
        self,
        total_epochs: int,
        warmup_epochs: int = 10,
        cib_ramp_epochs: int = 20,
        full_training_epochs: int = None
    ):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.cib_ramp_epochs = cib_ramp_epochs
        self.full_training_epochs = full_training_epochs or (total_epochs - warmup_epochs - cib_ramp_epochs)
        
    def get_stage_weights(self, epoch: int) -> Dict[str, float]:
        if epoch < self.warmup_epochs:
            return {
                'prediction_weight': 1.0,
                'cib_weight': 0.0,
                'scaffold_weight': 0.0,
                'sidechain_weight': 0.0,
                'consistency_weight': 0.0,
                'diversity_weight': 0.0
            }
        elif epoch < self.warmup_epochs + self.cib_ramp_epochs:
            progress = (epoch - self.warmup_epochs) / self.cib_ramp_epochs
            return {
                'prediction_weight': 1.0,
                'cib_weight': 0.1 * progress,
                'scaffold_weight': 0.05 * progress,
                'sidechain_weight': 0.05 * progress,
                'consistency_weight': 0.01 * progress,
                'diversity_weight': 0.005 * progress
            }
        else:
            return {
                'prediction_weight': 1.0,
                'cib_weight': 0.1,
                'scaffold_weight': 0.05,
                'sidechain_weight': 0.15,
                'consistency_weight': 0.02,
                'diversity_weight': 0.01
            }


class PerformanceMonitor:
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history = []
        self.best_metrics = {}
        
        self.logger = logging.getLogger('performance_monitor')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.save_dir / 'performance.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def update_metrics(
        self,
        epoch: int,
        phase: str,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        
        metrics_entry = {
            'epoch': epoch,
            'phase': phase,
            'metrics': metrics,
            'is_best': is_best
        }
        
        self.metrics_history.append(metrics_entry)
        
        log_message = f"Epoch {epoch} ({phase}): "
        for key, value in metrics.items():
            log_message += f"{key}={value:.4f} "
        
        if is_best:
            log_message += " [BEST]"
            self.best_metrics[phase] = metrics.copy()
        
        self.logger.info(log_message)
        
    def save_metrics(self):
        metrics_file = self.save_dir / 'metrics_history.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        best_metrics_file = self.save_dir / 'best_metrics.json'
        with open(best_metrics_file, 'w') as f:
            json.dump(self.best_metrics, f, indent=2)


class AdvancedMolecularDiffusionTrainer(pl.LightningModule):
    def __init__(
        self,
        base_diffusion_model: nn.Module,
        node_vocab_size: int,
        edge_vocab_size: int,
        spectrum_dim: int = 600,
        hidden_dim: int = 512,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        total_epochs: int = 100,
        use_multistage: bool = True,
        integration_strategy: str = 'adaptive'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['base_diffusion_model'])
        
        self.base_model = base_diffusion_model
        
        self.fusion_module = AdvancedFusionModule(
            node_vocab_size=node_vocab_size,
            edge_vocab_size=edge_vocab_size,
            spectrum_dim=spectrum_dim,
            hidden_dim=hidden_dim,
            integration_strategy=integration_strategy
        )
        
        self.combined_loss = CombinedLossFunction()
        
        if use_multistage:
            self.scheduler = MultiStageTrainingScheduler(total_epochs)
        else:
            self.scheduler = None
        
        self.performance_monitor = PerformanceMonitor(Path('./training_logs'))
        
        self.automatic_optimization = True
        
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.total_epochs,
            eta_min=1e-7
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def _compute_accuracy_metrics(
        self,
        predicted_logits: torch.Tensor,
        target_edges: torch.Tensor,
        scaffold_mask: torch.Tensor
    ) -> Dict[str, float]:
        
        with torch.no_grad():
            pred_edges = predicted_logits.argmax(dim=-1)
            loss_mask = (1 - scaffold_mask).float()
            
            correct_predictions = (pred_edges == target_edges).float() * loss_mask
            accuracy = correct_predictions.sum() / (loss_mask.sum() + 1e-8)
            
            batch_size = target_edges.size(0)
            correct_matrix = ((pred_edges == target_edges) | scaffold_mask.bool()).float()
            molecule_correct = correct_matrix.view(batch_size, -1).all(dim=1).float()
            molecule_accuracy = molecule_correct.mean()
            
            edge_type_accuracy = {}
            for edge_type in range(predicted_logits.size(-1)):
                edge_mask = (target_edges == edge_type).float() * loss_mask
                if edge_mask.sum() > 0:
                    type_correct = ((pred_edges == edge_type).float() * edge_mask).sum()
                    edge_type_accuracy[f'edge_type_{edge_type}_acc'] = (type_correct / edge_mask.sum()).item()
            
            return {
                'accuracy': accuracy.item(),
                'molecule_accuracy': molecule_accuracy.item(),
                **edge_type_accuracy
            }
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        node_features = batch["node_features"]
        target_edges = batch["adjacency_matrix"]
        scaffold_mask = batch["scaffold_mask"]
        spectrum = batch["spectrum"]
        
        batch_size = node_features.shape[0]
        device = node_features.device
        
        timesteps = torch.randint(1, 1001, (batch_size,), device=device)
        
        noisy_edges = self.base_model.add_noise(target_edges, timesteps)
        noisy_edges = torch.where(scaffold_mask.bool(), target_edges, noisy_edges)
        
        edge_index = self._create_edge_index(noisy_edges)
        
        predicted_logits, module_outputs = self.fusion_module(
            base_model=self.base_model,
            node_features=node_features,
            edge_features=noisy_edges,
            edge_index=edge_index,
            spectrum=spectrum,
            scaffold_mask=scaffold_mask,
            timestep=timesteps
        )
        
        if self.scheduler:
            stage_weights = self.scheduler.get_stage_weights(self.current_epoch)
            self.combined_loss.prediction_weight = stage_weights['prediction_weight']
            self.combined_loss.cib_weight = stage_weights['cib_weight']
            self.combined_loss.scaffold_weight = stage_weights['scaffold_weight']
            self.combined_loss.sidechain_weight = stage_weights['sidechain_weight']
            self.combined_loss.consistency_weight = stage_weights['consistency_weight']
            self.combined_loss.diversity_weight = stage_weights['diversity_weight']
        
        total_loss, loss_components = self.combined_loss(
            predicted_logits, target_edges, scaffold_mask, module_outputs
        )
        
        accuracy_metrics = self._compute_accuracy_metrics(
            predicted_logits, target_edges, scaffold_mask
        )
        
        self.log('train_loss', total_loss, prog_bar=True, sync_dist=True)
        for component, value in loss_components.items():
            self.log(f'train_{component}', value, prog_bar=False, sync_dist=True)
        
        for metric, value in accuracy_metrics.items():
            self.log(f'train_{metric}', value, prog_bar=True, sync_dist=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        node_features = batch["node_features"]
        target_edges = batch["adjacency_matrix"]
        scaffold_mask = batch["scaffold_mask"]
        spectrum = batch["spectrum"]
        
        batch_size = node_features.shape[0]
        device = node_features.device
        
        timesteps = torch.full((batch_size,), 500, device=device)
        
        noisy_edges = self.base_model.add_noise(target_edges, timesteps)
        noisy_edges = torch.where(scaffold_mask.bool(), target_edges, noisy_edges)
        
        edge_index = self._create_edge_index(noisy_edges)
        
        predicted_logits, module_outputs = self.fusion_module(
            base_model=self.base_model,
            node_features=node_features,
            edge_features=noisy_edges,
            edge_index=edge_index,
            spectrum=spectrum,
            scaffold_mask=scaffold_mask,
            timestep=timesteps
        )
        
        total_loss, loss_components = self.combined_loss(
            predicted_logits, target_edges, scaffold_mask, module_outputs
        )
        
        accuracy_metrics = self._compute_accuracy_metrics(
            predicted_logits, target_edges, scaffold_mask
        )
        
        self.log('val_loss', total_loss, prog_bar=True, sync_dist=True)
        for component, value in loss_components.items():
            self.log(f'val_{component}', value, prog_bar=False, sync_dist=True)
        
        for metric, value in accuracy_metrics.items():
            self.log(f'val_{metric}', value, prog_bar=True, sync_dist=True)
        
        return total_loss
    
    def _create_edge_index(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = adjacency_matrix.shape
        
        edge_indices = []
        for b in range(batch_size):
            adj = adjacency_matrix[b]
            edge_idx = torch.nonzero(adj > 0, as_tuple=False).t()
            if edge_idx.size(1) > 0:
                edge_idx[0] += b * num_nodes
                edge_idx[1] += b * num_nodes
            edge_indices.append(edge_idx)
        
        if edge_indices:
            return torch.cat(edge_indices, dim=1)
        else:
            return torch.empty((2, 0), dtype=torch.long, device=adjacency_matrix.device)
    
    def on_train_epoch_end(self):
        metrics = {}
        for key, value in self.trainer.callback_metrics.items():
            if key.startswith('train_'):
                metrics[key] = value.item() if torch.is_tensor(value) else value
        
        self.performance_monitor.update_metrics(
            epoch=self.current_epoch,
            phase='train',
            metrics=metrics
        )
    
    def on_validation_epoch_end(self):
        metrics = {}
        for key, value in self.trainer.callback_metrics.items():
            if key.startswith('val_'):
                metrics[key] = value.item() if torch.is_tensor(value) else value
        
        is_best = False
        if 'val_loss' in metrics:
            current_val_loss = metrics['val_loss']
            if not hasattr(self, 'best_val_loss') or current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                is_best = True
        
        self.performance_monitor.update_metrics(
            epoch=self.current_epoch,
            phase='validation',
            metrics=metrics,
            is_best=is_best
        )
        
        self.performance_monitor.save_metrics()


class AdvancedTrainingPipeline:
    def __init__(
        self,
        base_model: nn.Module,
        config: Dict[str, Any],
        save_dir: str = './advanced_training_results'
    ):
        self.base_model = base_model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AdvancedTrainingPipeline')
        
    def create_trainer(
        self,
        train_dataloader,
        val_dataloader,
        max_epochs: int = 100
    ) -> AdvancedMolecularDiffusionTrainer:
        
        trainer_model = AdvancedMolecularDiffusionTrainer(
            base_diffusion_model=self.base_model,
            **self.config,
            total_epochs=max_epochs
        )
        
        return trainer_model
    
    def run_training(
        self,
        train_dataloader,
        val_dataloader,
        max_epochs: int = 100,
        gpus: int = 1,
        precision: int = 32
    ):
        
        self.logger.info("Starting advanced molecular diffusion training")
        self.logger.info(f"Configuration: {self.config}")
        
        trainer_model = self.create_trainer(train_dataloader, val_dataloader, max_epochs)
        
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
        from pytorch_lightning.loggers import TensorBoardLogger
        
        callbacks = [
            ModelCheckpoint(
                dirpath=self.save_dir / 'checkpoints',
                filename='advanced-diffusion-{epoch:02d}-{val_loss:.3f}',
                monitor='val_loss',
                save_top_k=5,
                mode='min'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                mode='min',
                verbose=True
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]
        
        logger = TensorBoardLogger(
            save_dir=str(self.save_dir),
            name='advanced_diffusion_logs'
        )
        
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='gpu' if gpus > 0 else 'cpu',
            devices=gpus if gpus > 0 else 'auto',
            precision=precision,
            callbacks=callbacks,
            logger=logger,
            gradient_clip_val=1.0,
            accumulate_grad_batches=2,
            log_every_n_steps=50,
            check_val_every_n_epoch=1
        )
        
        trainer.fit(trainer_model, train_dataloader, val_dataloader)
        
        self.logger.info("Training completed successfully")
        
        best_model_path = trainer.checkpoint_callback.best_model_path
        self.logger.info(f"Best model saved at: {best_model_path}")
        
        return trainer_model, best_model_path