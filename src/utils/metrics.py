"""
Metrics for evaluation
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import logging

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """
    Metrics for semantic segmentation evaluation.
    
    Computes:
    - Pixel Accuracy
    - Mean IoU (mIoU)
    - Per-class IoU
    - Per-class Precision, Recall, F1-score
    - Confusion Matrix
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize metrics.
        
        Args:
            num_classes: Number of classes
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.confusion_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a batch of predictions.
        
        Args:
            pred: Predicted labels (batch_size, H, W) or (batch_size, num_classes, H, W)
            target: Ground truth labels (batch_size, H, W)
        """
        # Convert logits to labels if needed
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        
        # Move to CPU and convert to numpy
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        # Filter out ignore_index if present
        valid_mask = (target >= 0) & (target < self.num_classes)
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # Update confusion matrix
        self.confusion_mat += confusion_matrix(
            target, pred,
            labels=np.arange(self.num_classes)
        )
        
        self.total_samples += len(target)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Pixel Accuracy
        pixel_acc = np.diag(self.confusion_mat).sum() / (self.confusion_mat.sum() + 1e-10)
        metrics['pixel_accuracy'] = pixel_acc
        
        # Per-class metrics
        class_ious = []
        class_precisions = []
        class_recalls = []
        class_f1s = []
        
        for i in range(self.num_classes):
            # True Positives, False Positives, False Negatives
            tp = self.confusion_mat[i, i]
            fp = self.confusion_mat[:, i].sum() - tp
            fn = self.confusion_mat[i, :].sum() - tp
            
            # IoU
            iou = tp / (tp + fp + fn + 1e-10)
            class_ious.append(iou)
            metrics[f'iou_{self.class_names[i]}'] = iou
            
            # Precision, Recall, F1
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            class_precisions.append(precision)
            class_recalls.append(recall)
            class_f1s.append(f1)
            
            metrics[f'precision_{self.class_names[i]}'] = precision
            metrics[f'recall_{self.class_names[i]}'] = recall
            metrics[f'f1_{self.class_names[i]}'] = f1
        
        # Mean metrics
        metrics['mean_iou'] = np.mean(class_ious)
        metrics['mean_precision'] = np.mean(class_precisions)
        metrics['mean_recall'] = np.mean(class_recalls)
        metrics['mean_f1'] = np.mean(class_f1s)
        
        # Weighted metrics (weighted by support)
        class_supports = self.confusion_mat.sum(axis=1)
        total_support = class_supports.sum()
        
        if total_support > 0:
            weights = class_supports / total_support
            metrics['weighted_iou'] = np.sum(np.array(class_ious) * weights)
            metrics['weighted_f1'] = np.sum(np.array(class_f1s) * weights)
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return self.confusion_mat
    
    def print_metrics(self, metrics: Optional[Dict[str, float]] = None):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics (if None, compute first)
        """
        if metrics is None:
            metrics = self.compute()
        
        logger.info("\n" + "="*70)
        logger.info("Segmentation Metrics")
        logger.info("="*70)
        
        # Overall metrics
        logger.info("\nOverall Metrics:")
        logger.info(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        logger.info(f"  Mean IoU:       {metrics['mean_iou']:.4f}")
        logger.info(f"  Mean F1:        {metrics['mean_f1']:.4f}")
        logger.info(f"  Weighted IoU:   {metrics.get('weighted_iou', 0):.4f}")
        
        # Per-class metrics
        logger.info("\nPer-Class Metrics:")
        logger.info(f"{'Class':<25} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        logger.info("-"*70)
        
        for i, class_name in enumerate(self.class_names):
            iou = metrics[f'iou_{class_name}']
            precision = metrics[f'precision_{class_name}']
            recall = metrics[f'recall_{class_name}']
            f1 = metrics[f'f1_{class_name}']
            
            logger.info(f"{class_name:<25} {iou:>8.4f} {precision:>10.4f} {recall:>8.4f} {f1:>8.4f}")
        
        logger.info("="*70 + "\n")


class ClassificationMetrics:
    """
    Metrics for classification evaluation.
    
    Computes:
    - Accuracy
    - Per-class Precision, Recall, F1-score
    - Confusion Matrix
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize metrics.
        
        Args:
            num_classes: Number of classes
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a batch of predictions.
        
        Args:
            pred: Predicted logits (batch_size, num_classes) or labels (batch_size,)
            target: Ground truth labels (batch_size,)
        """
        # Convert logits to labels if needed
        if pred.dim() == 2:
            pred = pred.argmax(dim=1)
        
        # Move to CPU and convert to numpy
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        self.predictions.extend(pred.tolist())
        self.targets.extend(target.tolist())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metrics
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        # Overall accuracy
        accuracy = (predictions == targets).mean()
        metrics['accuracy'] = accuracy
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions,
            labels=np.arange(self.num_classes),
            zero_division=0
        )
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision[i]
            metrics[f'recall_{class_name}'] = recall[i]
            metrics[f'f1_{class_name}'] = f1[i]
            metrics[f'support_{class_name}'] = support[i]
        
        # Mean metrics
        metrics['mean_precision'] = precision.mean()
        metrics['mean_recall'] = recall.mean()
        metrics['mean_f1'] = f1.mean()
        
        # Weighted metrics
        metrics['weighted_precision'] = (precision * support).sum() / (support.sum() + 1e-10)
        metrics['weighted_recall'] = (recall * support).sum() / (support.sum() + 1e-10)
        metrics['weighted_f1'] = (f1 * support).sum() / (support.sum() + 1e-10)
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        cm = confusion_matrix(targets, predictions, labels=np.arange(self.num_classes))
        return cm
    
    def print_metrics(self, metrics: Optional[Dict[str, float]] = None):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics (if None, compute first)
        """
        if metrics is None:
            metrics = self.compute()
        
        logger.info("\n" + "="*70)
        logger.info("Classification Metrics")
        logger.info("="*70)
        
        # Overall metrics
        logger.info("\nOverall Metrics:")
        logger.info(f"  Accuracy:          {metrics['accuracy']:.4f}")
        logger.info(f"  Weighted F1:       {metrics['weighted_f1']:.4f}")
        logger.info(f"  Mean F1:           {metrics['mean_f1']:.4f}")
        
        # Per-class metrics
        logger.info("\nPer-Class Metrics:")
        logger.info(f"{'Class':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
        logger.info("-"*70)
        
        for i, class_name in enumerate(self.class_names):
            precision = metrics[f'precision_{class_name}']
            recall = metrics[f'recall_{class_name}']
            f1 = metrics[f'f1_{class_name}']
            support = metrics[f'support_{class_name}']
            
            logger.info(f"{class_name:<25} {precision:>10.4f} {recall:>8.4f} {f1:>8.4f} {support:>8.0f}")
        
        logger.info("="*70 + "\n")

