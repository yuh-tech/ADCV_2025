"""
Visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


# Color palette for visualization (10 classes)
COLOR_PALETTE = np.array([
    [255, 255, 0],      # 0: AnnualCrop - Yellow
    [34, 139, 34],      # 1: Forest - ForestGreen
    [144, 238, 144],    # 2: HerbaceousVegetation - LightGreen
    [128, 128, 128],    # 3: Highway - Gray
    [105, 105, 105],    # 4: Industrial - DimGray
    [173, 255, 47],     # 5: Pasture - GreenYellow
    [255, 165, 0],      # 6: PermanentCrop - Orange
    [255, 0, 0],        # 7: Residential - Red
    [0, 0, 255],        # 8: River - Blue
    [0, 191, 255],      # 9: SeaLake - DeepSkyBlue
], dtype=np.uint8)


def mask_to_rgb(mask: np.ndarray, color_palette: np.ndarray = COLOR_PALETTE) -> np.ndarray:
    """
    Convert segmentation mask to RGB image.
    
    Args:
        mask: Segmentation mask (H, W) with class indices
        color_palette: Color palette (num_classes, 3)
        
    Returns:
        RGB image (H, W, 3)
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx in range(len(color_palette)):
        rgb[mask == class_idx] = color_palette[class_idx]
    
    return rgb


def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Denormalize image from ImageNet normalization.
    
    Args:
        image: Normalized image (3, H, W) or (H, W, 3) - torch.Tensor or np.ndarray
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized image in [0, 1] range as numpy array with shape (H, W, 3)
    """
    # Convert torch.Tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if image.shape[0] == 3:
        # Convert from (3, H, W) to (H, W, 3)
        image = np.transpose(image, (1, 2, 0))
    
    mean = np.array(mean)
    std = np.array(std)
    
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    return image


def visualize_segmentation(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
    save_path: Optional[Path] = None,
    num_samples: int = 4,
    figsize: tuple = (15, 10)
):
    """
    Visualize segmentation results.
    
    Args:
        images: Batch of images (batch_size, 3, H, W)
        predictions: Predicted masks (batch_size, num_classes, H, W) or (batch_size, H, W)
        targets: Ground truth masks (batch_size, H, W)
        class_names: List of class names
        save_path: Path to save the figure
        num_samples: Number of samples to visualize
        figsize: Figure size
    """
    # Move to CPU and convert to numpy
    images = images.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Convert predictions to class labels if needed
    if predictions.dim() == 4:
        predictions = predictions.argmax(dim=1)
    predictions = predictions.cpu().numpy()
    
    # Limit number of samples
    num_samples = min(num_samples, len(images))
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize image
        img = denormalize_image(images[i])
        
        # Convert masks to RGB
        pred_rgb = mask_to_rgb(predictions[i])
        target_rgb = mask_to_rgb(targets[i])
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(target_rgb)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 10),
    normalize: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix (num_classes, num_classes)
        class_names: List of class names
        save_path: Path to save the figure
        figsize: Figure size
        normalize: If True, normalize by row (ground truth)
    """
    if normalize:
        cm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-10)
        fmt = '.2f'
    else:
        cm = confusion_matrix
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    figsize: tuple = (15, 5)
):
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc', etc.
        save_path: Path to save the figure
        figsize: Figure size
    """
    metrics = {}
    for key in history.keys():
        if key.startswith('train_'):
            metric_name = key.replace('train_', '')
            if f'val_{metric_name}' in history:
                metrics[metric_name] = {
                    'train': history[key],
                    'val': history[f'val_{metric_name}']
                }
    
    if not metrics:
        logger.warning("No matching train/val metrics found in history")
        return
    
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    if num_metrics == 1:
        axes = [axes]
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        epochs = range(1, len(values['train']) + 1)
        
        axes[i].plot(epochs, values['train'], 'b-', label=f'Train {metric_name}')
        axes[i].plot(epochs, values['val'], 'r-', label=f'Val {metric_name}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric_name.replace('_', ' ').title())
        axes[i].set_title(f'{metric_name.replace("_", " ").title()} Curve')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
    
    plt.close()


def plot_class_distribution(
    class_counts: Dict[int, int],
    class_names: List[str],
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6)
):
    """
    Plot class distribution.
    
    Args:
        class_counts: Dictionary mapping class indices to counts
        class_names: List of class names
        save_path: Path to save the figure
        figsize: Figure size
    """
    counts = [class_counts.get(i, 0) for i in range(len(class_names))]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(class_names)), counts, color='skyblue', edgecolor='black')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved class distribution to {save_path}")
    
    plt.close()


def create_legend(class_names: List[str], save_path: Optional[Path] = None, figsize: tuple = (8, 6)):
    """
    Create a legend showing class colors.
    
    Args:
        class_names: List of class names
        save_path: Path to save the figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # Create legend patches
    from matplotlib.patches import Patch
    legend_patches = []
    
    for i, class_name in enumerate(class_names):
        color = COLOR_PALETTE[i] / 255.0  # Normalize to [0, 1]
        patch = Patch(color=color, label=class_name)
        legend_patches.append(patch)
    
    ax.legend(handles=legend_patches, loc='center', fontsize=12, frameon=True)
    plt.title('Class Color Legend', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved legend to {save_path}")
    
    plt.close()

