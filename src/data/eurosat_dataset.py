"""
EuroSAT RGB Dataset for Encoder Pre-training
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from pathlib import Path
from typing import Optional, Callable
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class EuroSATDataset(Dataset):
    """
    PyTorch Dataset wrapper for EuroSAT RGB dataset.
    
    EuroSAT is organized in folders by class, making it compatible with
    torchvision.datasets.ImageFolder.
    """
    
    def __init__(
        self,
        root_dir: Path,
        split: str = 'train',
        transform: Optional[Callable] = None
    ):
        """
        Initialize EuroSAT dataset.
        
        Args:
            root_dir: Root directory of EuroSAT dataset (should contain train/val/test folders)
            split: Dataset split ('train', 'val', or 'test')
            transform: Optional transform to be applied on a sample
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Path to split folder
        split_dir = self.root_dir / split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Use ImageFolder to load the dataset
        self.dataset = datasets.ImageFolder(str(split_dir))
        
        # Class names
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        
        logger.info(f"Loaded EuroSAT {split} dataset: {len(self)} samples")
        logger.info(f"Classes: {self.classes}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with keys:
                - 'image': Image tensor
                - 'label': Class label
        """
        image, label = self.dataset[idx]
        
        # Convert PIL image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image).astype(np.float32) / 255.0
        
        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return {
            'image': image,
            'label': label
        }
    
    def get_class_distribution(self):
        """Get class distribution in the dataset."""
        class_counts = {}
        for _, label in self.dataset.samples:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        logger.info(f"Class distribution for {self.split} split:")
        for class_name, count in sorted(class_counts.items()):
            logger.info(f"  {class_name}: {count} samples")
        
        return class_counts


def create_eurosat_dataloaders(
    root_dir: Path,
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = True
):
    """
    Create train, validation, and test dataloaders for EuroSAT.
    
    Args:
        root_dir: Root directory of EuroSAT dataset
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = EuroSATDataset(root_dir, 'train', train_transform)
    val_dataset = EuroSATDataset(root_dir, 'val', val_transform)
    test_dataset = EuroSATDataset(root_dir, 'test', val_transform)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Created EuroSAT dataloaders:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

