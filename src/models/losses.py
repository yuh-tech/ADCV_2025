"""
Loss functions for semantic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    
    Dice coefficient is a popular metric for segmentation tasks.
    Dice Loss = 1 - Dice Coefficient
    """
    
    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_index: Class index to ignore in loss calculation
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice Loss.
        
        Args:
            pred: Predicted logits of shape (batch_size, num_classes, H, W)
            target: Ground truth labels of shape (batch_size, H, W)
            
        Returns:
            Dice loss value
        """
        # Apply softmax to get probabilities
        pred = F.softmax(pred, dim=1)
        
        # Convert target to one-hot encoding
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # Flatten spatial dimensions
        pred = pred.reshape(pred.shape[0], pred.shape[1], -1)
        target_one_hot = target_one_hot.reshape(target_one_hot.shape[0], target_one_hot.shape[1], -1)
        
        # Calculate Dice coefficient for each class
        intersection = (pred * target_one_hot).sum(dim=2)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Average across classes and batch
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = -100
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights (optional)
            gamma: Focusing parameter
            ignore_index: Class index to ignore
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal Loss.
        
        Args:
            pred: Predicted logits of shape (batch_size, num_classes, H, W)
            target: Ground truth labels of shape (batch_size, H, W)
            
        Returns:
            Focal loss value
        """
        # Calculate cross-entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none', ignore_index=self.ignore_index)
        
        # Get probabilities
        p = F.softmax(pred, dim=1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Calculate focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            alpha_t = self.alpha.gather(0, target.flatten()).reshape(target.shape)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combination of Cross-Entropy and Dice Loss.
    
    Total Loss = ce_weight * CE_Loss + dice_weight * Dice_Loss
    """
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ):
        """
        Initialize Combined Loss.
        
        Args:
            ce_weight: Weight for cross-entropy loss
            dice_weight: Weight for dice loss
            class_weights: Class weights for cross-entropy loss
            ignore_index: Class index to ignore
        """
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        
        # Cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )
        
        # Dice loss
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        
        logger.info(f"Initialized CombinedLoss: ce_weight={ce_weight}, dice_weight={dice_weight}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            pred: Predicted logits of shape (batch_size, num_classes, H, W)
            target: Ground truth labels of shape (batch_size, H, W)
            
        Returns:
            Combined loss value
        """
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        total_loss = self.ce_weight * ce + self.dice_weight * dice
        
        return total_loss


def create_loss_function(
    loss_type: str,
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    device: str = 'cuda'
):
    """
    Create loss function based on configuration.
    
    Args:
        loss_type: Type of loss ('ce', 'dice', 'combined', 'focal')
        num_classes: Number of classes
        class_weights: Optional class weights
        ce_weight: Weight for CE in combined loss
        dice_weight: Weight for Dice in combined loss
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        device: Device to place the loss on
        
    Returns:
        Loss function
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    if loss_type == 'ce':
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type == 'dice':
        loss_fn = DiceLoss()
    
    elif loss_type == 'combined':
        loss_fn = CombinedLoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            class_weights=class_weights
        )
    
    elif loss_type == 'focal':
        alpha = torch.ones(num_classes) * focal_alpha if class_weights is None else class_weights
        loss_fn = FocalLoss(alpha=alpha.to(device), gamma=focal_gamma)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    logger.info(f"Created loss function: {loss_type}")
    
    return loss_fn

