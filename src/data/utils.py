"""
Utility functions for data loading and preprocessing
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Union, Dict
import logging

logger = logging.getLogger(__name__)


def load_sentinel2_rgb(patch_folder: Union[str, Path], normalize: bool = True) -> np.ndarray:
    """
    Load RGB image from Sentinel-2 bands (B04, B03, B02).
    
    Args:
        patch_folder: Path to the patch folder containing band files
        normalize: If True, normalize values to [0, 1] range
        
    Returns:
        RGB image as numpy array with shape (H, W, 3)
        
    Raises:
        FileNotFoundError: If any of the required band files is missing
        ValueError: If bands have inconsistent shapes
    """
    patch_folder = Path(patch_folder)
    patch_id = patch_folder.name
    
    # Define band file paths
    band_files = {
        'red': patch_folder / f"{patch_id}_B04.tif",
        'green': patch_folder / f"{patch_id}_B03.tif",
        'blue': patch_folder / f"{patch_id}_B02.tif"
    }
    
    # Check if all band files exist
    for band_name, band_file in band_files.items():
        if not band_file.exists():
            raise FileNotFoundError(f"Band file not found: {band_file}")
    
    # Load bands
    bands = []
    for band_name in ['red', 'green', 'blue']:
        try:
            with rasterio.open(band_files[band_name]) as src:
                band_data = src.read(1).astype(np.float32)
                bands.append(band_data)
        except Exception as e:
            logger.error(f"Error loading {band_name} band from {band_files[band_name]}: {e}")
            raise
    
    # Stack bands into RGB image
    rgb = np.stack(bands, axis=-1)  # Shape: (H, W, 3)
    
    # Normalize if requested
    if normalize:
        rgb = rgb / 10000.0  # Sentinel-2 reflectance values are in range [0, 10000]
        rgb = np.clip(rgb, 0, 1)  # Clip any values outside [0, 1]
    
    return rgb


def convert_corine_to_eurosat(corine_mask: np.ndarray, 
                               mapping: Dict[int, int],
                               default_class: int = 0) -> np.ndarray:
    """
    Convert CORINE Land Cover codes to EuroSAT class indices.
    
    Args:
        corine_mask: Mask with CORINE codes (e.g., 211, 231, 311)
        mapping: Dictionary mapping CORINE codes to EuroSAT indices
        default_class: Default class for unmapped CORINE codes
        
    Returns:
        Mask with EuroSAT class indices (0-9)
    """
    eurosat_mask = np.full_like(corine_mask, default_class, dtype=np.int64)
    
    # Convert each CORINE code to EuroSAT index
    for corine_code, eurosat_idx in mapping.items():
        eurosat_mask[corine_mask == corine_code] = eurosat_idx
    
    # Log warning if there are unmapped codes
    unmapped_codes = np.unique(corine_mask)
    unmapped_codes = [code for code in unmapped_codes if code not in mapping and code != 0]
    if len(unmapped_codes) > 0:
        logger.warning(f"Found unmapped CORINE codes: {unmapped_codes}. "
                      f"These will be assigned to class {default_class}.")
    
    return eurosat_mask


def load_reference_map(reference_map_path: Union[str, Path],
                       mapping: Dict[int, int],
                       default_class: int = 0) -> np.ndarray:
    """
    Load reference map (segmentation mask) and convert to EuroSAT classes.
    
    Args:
        reference_map_path: Path to the reference map GeoTIFF file
        mapping: Dictionary mapping CORINE codes to EuroSAT indices
        default_class: Default class for unmapped CORINE codes
        
    Returns:
        Segmentation mask with EuroSAT class indices (0-9)
        
    Raises:
        FileNotFoundError: If reference map file doesn't exist
    """
    reference_map_path = Path(reference_map_path)
    
    if not reference_map_path.exists():
        raise FileNotFoundError(f"Reference map not found: {reference_map_path}")
    
    try:
        with rasterio.open(reference_map_path) as src:
            corine_mask = src.read(1)
    except Exception as e:
        logger.error(f"Error loading reference map from {reference_map_path}: {e}")
        raise
    
    # Convert CORINE codes to EuroSAT indices
    eurosat_mask = convert_corine_to_eurosat(corine_mask, mapping, default_class)
    
    return eurosat_mask


def find_patch_folder(patch_id: str, search_folders: list) -> Path:
    """
    Find the patch folder in one of the search folders.
    
    Args:
        patch_id: Patch ID (e.g., 'S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57')
        search_folders: List of folders to search in
        
    Returns:
        Path to the patch folder
        
    Raises:
        FileNotFoundError: If patch folder is not found
    """
    # Extract tile name from patch_id (everything before the last two underscore-separated parts)
    parts = patch_id.split('_')
    tile_name = '_'.join(parts[:-2])
    
    for search_folder in search_folders:
        search_folder = Path(search_folder)
        
        # Construct potential path
        tile_folder = search_folder / tile_name
        patch_folder = tile_folder / patch_id
        
        if patch_folder.exists():
            return patch_folder
    
    raise FileNotFoundError(f"Patch folder not found for {patch_id} in {len(search_folders)} search folders")


def find_reference_map(patch_id: str, reference_maps_folder: Union[str, Path]) -> Path:
    """
    Find the reference map file for a given patch.
    
    Args:
        patch_id: Patch ID
        reference_maps_folder: Root folder containing reference maps
        
    Returns:
        Path to the reference map file
        
    Raises:
        FileNotFoundError: If reference map is not found
    """
    reference_maps_folder = Path(reference_maps_folder)
    
    # Extract tile name
    parts = patch_id.split('_')
    tile_name = '_'.join(parts[:-2])
    
    # Construct reference map path
    ref_map_path = reference_maps_folder / tile_name / f"{patch_id}_reference_map.tif"
    
    if not ref_map_path.exists():
        raise FileNotFoundError(f"Reference map not found: {ref_map_path}")
    
    return ref_map_path


def compute_class_weights(masks: list, num_classes: int, method: str = 'inverse_frequency') -> np.ndarray:
    """
    Compute class weights for handling class imbalance.
    
    Args:
        masks: List of segmentation masks
        num_classes: Number of classes
        method: Weighting method ('inverse_frequency' or 'effective_number')
        
    Returns:
        Class weights as numpy array
    """
    # Count pixels per class
    class_counts = np.zeros(num_classes, dtype=np.int64)
    
    for mask in masks:
        unique, counts = np.unique(mask, return_counts=True)
        for cls, count in zip(unique, counts):
            if 0 <= cls < num_classes:
                class_counts[cls] += count
    
    if method == 'inverse_frequency':
        # Inverse frequency weighting
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (num_classes * class_counts + 1e-6)
        
    elif method == 'effective_number':
        # Effective number of samples (from paper: Class-Balanced Loss Based on Effective Number of Samples)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        class_weights = (1.0 - beta) / (effective_num + 1e-6)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights to have mean = 1
    class_weights = class_weights / class_weights.mean()
    
    logger.info(f"Computed class weights using {method}:")
    for i, weight in enumerate(class_weights):
        logger.info(f"  Class {i}: weight={weight:.4f}, count={class_counts[i]}")
    
    return class_weights.astype(np.float32)


def validate_data_integrity(image: np.ndarray, mask: np.ndarray, num_classes: int) -> bool:
    """
    Validate data integrity for image-mask pair.
    
    Args:
        image: RGB image
        mask: Segmentation mask
        num_classes: Number of expected classes
        
    Returns:
        True if data is valid, False otherwise
    """
    # Check shapes
    if image.shape[:2] != mask.shape[:2]:
        logger.warning(f"Image and mask shape mismatch: {image.shape} vs {mask.shape}")
        return False
    
    # Check value ranges
    if image.min() < 0 or image.max() > 1:
        logger.warning(f"Image values out of range [0, 1]: [{image.min()}, {image.max()}]")
        return False
    
    # Check mask values
    unique_classes = np.unique(mask)
    if unique_classes.min() < 0 or unique_classes.max() >= num_classes:
        logger.warning(f"Mask values out of range [0, {num_classes-1}]: {unique_classes}")
        return False
    
    # Check for NaN or Inf
    if np.isnan(image).any() or np.isinf(image).any():
        logger.warning("Image contains NaN or Inf values")
        return False
    
    return True

