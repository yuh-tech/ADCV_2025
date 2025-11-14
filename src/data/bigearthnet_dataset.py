"""
BigEarthNet Dataset for Semantic Segmentation
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Dict, Callable
import logging

from .utils import (
    load_sentinel2_rgb,
    load_reference_map,
    find_patch_folder,
    find_reference_map,
    validate_data_integrity
)

logger = logging.getLogger(__name__)


class BigEarthNetSegmentationDataset(Dataset):
    """
    PyTorch Dataset for BigEarthNet semantic segmentation.
    
    Loads RGB images from Sentinel-2 bands and converts CORINE reference maps
    to EuroSAT class indices.
    """
    
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_folders: List[Path],
        reference_maps_folder: Path,
        corine_to_eurosat_mapping: Dict[int, int],
        transform: Optional[Callable] = None,
        num_classes: int = 10,
        validate_data: bool = True,
        cache_data: bool = False
    ):
        """
        Initialize BigEarthNet dataset.
        
        Args:
            metadata_df: Pandas DataFrame with columns ['patch_id', 'labels', 'split', 'country']
            data_folders: List of folders containing BigEarthNet patches
            reference_maps_folder: Folder containing reference maps
            corine_to_eurosat_mapping: Dictionary mapping CORINE codes to EuroSAT indices
            transform: Optional transform to be applied on a sample
            num_classes: Number of target classes
            validate_data: If True, validate data integrity when loading
            cache_data: If True, cache loaded data in memory (requires significant RAM)
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.data_folders = [Path(f) for f in data_folders]
        self.reference_maps_folder = Path(reference_maps_folder)
        self.corine_to_eurosat_mapping = corine_to_eurosat_mapping
        self.transform = transform
        self.num_classes = num_classes
        self.validate_data = validate_data
        self.cache_data = cache_data
        
        # Cache for loaded data
        self.data_cache = {} if cache_data else None
        
        # Statistics
        self.failed_samples = []
        
        logger.info(f"Initialized BigEarthNetSegmentationDataset with {len(self)} samples")
        logger.info(f"Data folders: {len(self.data_folders)}")
        logger.info(f"Reference maps folder: {self.reference_maps_folder}")
        logger.info(f"Caching enabled: {cache_data}")
    
    def __len__(self) -> int:
        return len(self.metadata_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with keys:
                - 'image': RGB image tensor
                - 'mask': Segmentation mask tensor
                - 'patch_id': Patch ID string
        """
        # Check cache
        if self.cache_data and idx in self.data_cache:
            return self.data_cache[idx]
        
        # Get metadata
        row = self.metadata_df.iloc[idx]
        patch_id = row['patch_id']
        
        try:
            # Find patch folder
            patch_folder = find_patch_folder(patch_id, self.data_folders)
            
            # Load RGB image
            rgb = load_sentinel2_rgb(patch_folder, normalize=True)
            
            # Load reference map
            ref_map_path = find_reference_map(patch_id, self.reference_maps_folder)
            mask = load_reference_map(
                ref_map_path,
                self.corine_to_eurosat_mapping,
                default_class=0
            )
            
            # Validate data integrity
            if self.validate_data:
                if not validate_data_integrity(rgb, mask, self.num_classes):
                    logger.warning(f"Data validation failed for patch {patch_id}")
                    # Return a dummy sample or raise exception
                    raise ValueError(f"Invalid data for patch {patch_id}")
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=rgb, mask=mask)
                rgb = transformed['image']
                mask = transformed['mask']
            else:
                # Convert to tensors if no transform is provided
                rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
                mask = torch.from_numpy(mask).long()
            
            sample = {
                'image': rgb,
                'mask': mask,
                'patch_id': patch_id
            }
            
            # Cache if enabled
            if self.cache_data:
                self.data_cache[idx] = sample
            
            return sample
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} (patch_id: {patch_id}): {e}")
            self.failed_samples.append((idx, patch_id, str(e)))
            
            # Return a dummy sample to avoid breaking the training loop
            # In production, you might want to skip this sample or handle it differently
            dummy_image = torch.zeros((3, 120, 120), dtype=torch.float32)
            dummy_mask = torch.zeros((120, 120), dtype=torch.long)
            
            return {
                'image': dummy_image,
                'mask': dummy_mask,
                'patch_id': f"FAILED_{patch_id}"
            }
    
    def get_failed_samples(self) -> List:
        """Get list of failed samples during loading."""
        return self.failed_samples
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        Compute class distribution across the dataset.
        
        Note: This requires loading all masks, which can be time-consuming.
        
        Returns:
            Dictionary mapping class indices to pixel counts
        """
        logger.info("Computing class distribution...")
        class_counts = {i: 0 for i in range(self.num_classes)}
        
        for idx in range(len(self)):
            try:
                sample = self[idx]
                mask = sample['mask']
                
                if isinstance(mask, torch.Tensor):
                    mask = mask.numpy()
                
                unique, counts = np.unique(mask, return_counts=True)
                for cls, count in zip(unique, counts):
                    if 0 <= cls < self.num_classes:
                        class_counts[int(cls)] += int(count)
                        
            except Exception as e:
                logger.error(f"Error computing distribution for sample {idx}: {e}")
                continue
        
        logger.info("Class distribution:")
        for cls, count in class_counts.items():
            logger.info(f"  Class {cls}: {count:,} pixels")
        
        return class_counts


def create_bigearthnet_dataloaders(
    metadata_path: Path,
    data_folders: List[Path],
    reference_maps_folder: Path,
    corine_to_eurosat_mapping: Dict[int, int],
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    num_classes: int = 10,
    max_train_samples: Optional[int] = None
):
    """
    Create train, validation, and test dataloaders for BigEarthNet.
    
    Args:
        metadata_path: Path to metadata.parquet file
        data_folders: List of BigEarthNet data folders
        reference_maps_folder: Folder containing reference maps
        corine_to_eurosat_mapping: CORINE to EuroSAT mapping dictionary
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        num_classes: Number of classes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load metadata
    logger.info(f"Loading metadata from {metadata_path}")
    metadata_df = pd.read_parquet(metadata_path)
    
    # Split by split column
    train_df = metadata_df[metadata_df['split'] == 'train'].reset_index(drop=True)
    val_df = metadata_df[metadata_df['split'] == 'validation'].reset_index(drop=True)
    test_df = metadata_df[metadata_df['split'] == 'test'].reset_index(drop=True)
    
    # Limit training samples if specified (for faster training/testing)
    if max_train_samples is not None and max_train_samples > 0:
        if len(train_df) > max_train_samples:
            logger.info(f"Limiting training samples from {len(train_df)} to {max_train_samples} for faster training")
            train_df = train_df.head(max_train_samples).reset_index(drop=True)
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Create datasets
    train_dataset = BigEarthNetSegmentationDataset(
        train_df, data_folders, reference_maps_folder,
        corine_to_eurosat_mapping, train_transform, num_classes
    )
    
    val_dataset = BigEarthNetSegmentationDataset(
        val_df, data_folders, reference_maps_folder,
        corine_to_eurosat_mapping, val_transform, num_classes
    )
    
    test_dataset = BigEarthNetSegmentationDataset(
        test_df, data_folders, reference_maps_folder,
        corine_to_eurosat_mapping, val_transform, num_classes
    )
    
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
    
    return train_loader, val_loader, test_loader

