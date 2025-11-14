"""
Training utilities
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Optional, Dict, Callable
import logging
import pickle
from tqdm import tqdm
import time

from .metrics import SegmentationMetrics, ClassificationMetrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Unified trainer for classification and segmentation tasks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        mixed_precision: bool = False,
        gradient_accumulation_steps: int = 1,
        grad_clip_norm: Optional[float] = None,
        checkpoint_dir: Optional[Path] = None,
        task: str = 'segmentation'  # 'segmentation' or 'classification'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            scheduler: Learning rate scheduler (optional)
            mixed_precision: Use mixed precision training (FP16)
            gradient_accumulation_steps: Number of steps to accumulate gradients
            grad_clip_norm: Gradient clipping norm (optional)
            checkpoint_dir: Directory to save checkpoints
            task: Task type ('segmentation' or 'classification')
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_dir = checkpoint_dir
        self.task = task
        
        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        # Best metrics for checkpointing
        if task == 'classification':
            self.best_metric = float('-inf')
        else:
            self.best_metric = 0.0
        self.best_epoch = 0
        
        logger.info(f"Initialized Trainer:")
        logger.info(f"  Task: {task}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Mixed Precision: {mixed_precision}")
        logger.info(f"  Gradient Accumulation: {gradient_accumulation_steps}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        log_interval: int = 50
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            log_interval: Logging interval (batches)
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        running_loss = 0.0
        num_batches = len(train_loader)
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            images = batch['image'].to(self.device)
            
            if self.task == 'segmentation':
                targets = batch['mask'].to(self.device).long()  # Ensure Long dtype for cross-entropy
            else:  # classification
                targets = batch['label'].to(self.device)
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    if self.grad_clip_norm:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.grad_clip_norm:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update running loss
            running_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}'})
            
            # Log
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = running_loss / (batch_idx + 1)
                logger.debug(f'Epoch {epoch} [{batch_idx+1}/{num_batches}] - Loss: {avg_loss:.4f}')
        
        # Calculate average loss
        avg_loss = running_loss / num_batches
        
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        metrics_tracker: Optional[object] = None,
        epoch: int = 0
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            metrics_tracker: Metrics tracker (SegmentationMetrics or ClassificationMetrics)
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        running_loss = 0.0
        num_batches = len(val_loader)
        
        # Reset metrics
        if metrics_tracker:
            metrics_tracker.reset()
        
        # Progress bar
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]', leave=False)
        
        for batch in pbar:
            # Get data
            images = batch['image'].to(self.device)
            
            if self.task == 'segmentation':
                targets = batch['mask'].to(self.device).long()  # Ensure Long dtype for cross-entropy
            else:  # classification
                targets = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Update running loss
            running_loss += loss.item()
            
            # Update metrics
            if metrics_tracker:
                metrics_tracker.update(outputs, targets)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        avg_loss = running_loss / num_batches
        results = {'loss': avg_loss}
        
        if metrics_tracker:
            metrics = metrics_tracker.compute()
            results.update(metrics)
        
        return results
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        metrics_class: Optional[object] = None,
        num_classes: int = 10,
        class_names: Optional[list] = None,
        log_interval: int = 50,
        save_best_only: bool = True,
        early_stopping_patience: int = None,
        checkpoint_metric: str = 'loss'
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            metrics_class: Metrics class (SegmentationMetrics or ClassificationMetrics)
            num_classes: Number of classes
            class_names: List of class names
            log_interval: Logging interval (batches)
            save_best_only: Save only the best model
            early_stopping_patience: Early stopping patience (epochs)
            checkpoint_metric: Metric to use for checkpointing ('loss', 'accuracy', 'mean_iou', etc.)
        """
        logger.info(f"\nStarting training for {num_epochs} epochs...")
        logger.info(f"  Training samples: {len(train_loader.dataset)}")
        logger.info(f"  Validation samples: {len(val_loader.dataset)}")
        logger.info(f"  Batch size: {train_loader.batch_size}")
        logger.info(f"  Checkpoint metric: {checkpoint_metric}\n")
        
        # Initialize metrics tracker
        if metrics_class:
            metrics_tracker = metrics_class(num_classes, class_names)
        else:
            metrics_tracker = None
        
        # Early stopping counter
        epochs_without_improvement = 0
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch, log_interval)
            
            # Validate
            val_metrics = self.validate(val_loader, metrics_tracker, epoch)
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            
            for key, value in val_metrics.items():
                if key != 'loss':
                    history_key = f'val_{key}'
                    if history_key not in self.history:
                        self.history[history_key] = []
                    self.history[history_key].append(value)
            
            # Log
            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"\nEpoch {epoch}/{num_epochs} - {epoch_time:.2f}s - LR: {current_lr:.6f}")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"  Val Loss:   {val_metrics['loss']:.4f}")
            
            if metrics_tracker:
                if self.task == 'segmentation':
                    logger.info(f"  Val mIoU:   {val_metrics.get('mean_iou', 0):.4f}")
                    logger.info(f"  Val Acc:    {val_metrics.get('pixel_accuracy', 0):.4f}")
                else:
                    logger.info(f"  Val Acc:    {val_metrics.get('accuracy', 0):.4f}")
                    logger.info(f"  Val F1:     {val_metrics.get('weighted_f1', 0):.4f}")
            
            # Determine if this is the best epoch
            if checkpoint_metric == 'loss':
                current_metric = val_metrics['loss']
                is_best = current_metric < self.best_metric
            else:
                current_metric = val_metrics.get(checkpoint_metric, 0)
                is_best = current_metric > self.best_metric
            
            # Save checkpoint
            if self.checkpoint_dir:
                if is_best:
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                    logger.info(f"  ✓ New best model saved! {checkpoint_metric}: {current_metric:.4f}")
                    epochs_without_improvement = 0
                elif not save_best_only:
                    self.save_checkpoint(epoch, val_metrics, is_best=False)
                else:
                    epochs_without_improvement += 1
            
            # Early stopping
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                logger.info(f"Best {checkpoint_metric}: {self.best_metric:.4f} at epoch {self.best_epoch}")
                break
        
        logger.info(f"\nTraining completed!")
        logger.info(f"Best {checkpoint_metric}: {self.best_metric:.4f} at epoch {self.best_epoch}")
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        if not self.checkpoint_dir:
            return
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save best model
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.debug(f"Saved best checkpoint to {checkpoint_path}")
        
        # Save last model
        checkpoint_path = self.checkpoint_dir / 'last_model.pth'
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: Path, load_optimizer: bool = True):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except (RuntimeError, pickle.UnpicklingError):
            # PyTorch 2.6 sets weights_only=True by default; fall back to full loading
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

