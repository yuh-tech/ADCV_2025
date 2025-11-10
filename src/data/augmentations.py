"""
Data augmentation pipelines for training and validation
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_augmentation(input_size: int, strength: str = 'medium', is_segmentation: bool = True):
    """
    Get training augmentation pipeline.
    
    Args:
        input_size: Target image size (will be resized to input_size x input_size)
        strength: Augmentation strength ('light', 'medium', 'strong')
        is_segmentation: If True, augmentations will be applied to both image and mask
        
    Returns:
        Albumentations Compose object
    """
    
    if strength == 'light':
        augmentations = [
            A.Resize(input_size, input_size, interpolation=1, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
        ]
    
    elif strength == 'medium':
        augmentations = [
            A.Resize(input_size, input_size, interpolation=1, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.3
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
        ]
        
        if is_segmentation:
            augmentations.extend([
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=0.2
                ),
            ])
    
    elif strength == 'strong':
        augmentations = [
            A.Resize(input_size, input_size, interpolation=1, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=30,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),
        ]
        
        if is_segmentation:
            augmentations.extend([
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=0.3
                ),
                A.GridDistortion(p=0.2),
                A.OpticalDistortion(p=0.2),
            ])
    
    else:
        raise ValueError(f"Unknown augmentation strength: {strength}")
    
    # Add normalization and tensor conversion
    augmentations.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225],   # ImageNet stds
            always_apply=True
        ),
        ToTensorV2(),
    ])
    
    return A.Compose(augmentations)


def get_val_augmentation(input_size: int):
    """
    Get validation augmentation pipeline (only resize and normalize).
    
    Args:
        input_size: Target image size
        
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(input_size, input_size, interpolation=1, always_apply=True),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            always_apply=True
        ),
        ToTensorV2(),
    ])


def get_classification_train_augmentation(input_size: int, strength: str = 'medium'):
    """
    Get training augmentation for classification task (EuroSAT pre-training).
    
    Args:
        input_size: Target image size
        strength: Augmentation strength
        
    Returns:
        Albumentations Compose object
    """
    return get_train_augmentation(input_size, strength, is_segmentation=False)


def get_segmentation_train_augmentation(input_size: int, strength: str = 'medium'):
    """
    Get training augmentation for segmentation task (U-Net training).
    
    Args:
        input_size: Target image size
        strength: Augmentation strength
        
    Returns:
        Albumentations Compose object
    """
    return get_train_augmentation(input_size, strength, is_segmentation=True)

