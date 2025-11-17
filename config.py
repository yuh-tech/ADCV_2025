"""
Configuration file for Land Cover Segmentation Project
Contains all hyperparameters, paths, and settings
"""

import os
from pathlib import Path

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

def is_kaggle_environment():
    """Check if running on Kaggle."""
    return os.path.exists('/kaggle/input')

IS_KAGGLE = is_kaggle_environment()

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Root directories
PROJECT_ROOT = Path(__file__).parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"

if IS_KAGGLE:
    # Kaggle environment
    print("🌐 Running on Kaggle environment")
    
    # Data directories (read-only in Kaggle input)
    KAGGLE_INPUT = Path("/kaggle/input")
    
    # Working directory for extracted/processed data
    KAGGLE_WORKING = Path("/kaggle/working")
    DATA_DIR = KAGGLE_WORKING / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Output directories
    OUTPUTS_DIR = KAGGLE_WORKING / "outputs"
    
    # Metadata (read from input)
    METADATA_PATH = KAGGLE_INPUT / "bigearthnet-s2-metadata" / "metadata.parquet"
    
    # Reference Maps
    REFERENCE_MAPS_ARCHIVE = KAGGLE_INPUT / "bigearthnet-s2-referencesmap" / "Reference_Maps.tar.zst"
    # Extract to /kaggle/working/data/ directory
    # After extraction: /kaggle/working/data/Reference_Maps/ (single level)
    REFERENCE_MAPS_FOLDER = DATA_DIR / "Reference_Maps"
    
    print(f"  📍 Reference Maps: {REFERENCE_MAPS_FOLDER}")
    print(f"     Exists: {REFERENCE_MAPS_FOLDER.exists()}")
    
    # BigEarthNet folders (multiple datasets: bigearthnetv2-s2-0 through bigearthnetv2-s2-5)
    BIGEARTHNET_FOLDERS = []
    for i in range(6):  # bigearthnetv2-s2-0 to bigearthnetv2-s2-5
        folder = KAGGLE_INPUT / f"bigearthnetv2-s2-{i}" / "BigEarthNet-S2"
        if folder.exists():
            BIGEARTHNET_FOLDERS.append(folder)
    
    # If no numbered folders found, try single folder
    if not BIGEARTHNET_FOLDERS:
        single_folder = KAGGLE_INPUT / "bigearthnet-s2" / "BigEarthNet-S2"
        if single_folder.exists():
            BIGEARTHNET_FOLDERS.append(single_folder)
    
    # EuroSAT path
    EUROSAT_PATH = KAGGLE_INPUT / "rgbeurosat" / "RBG"
    if not EUROSAT_PATH.exists():
        # Try alternative path
        EUROSAT_PATH = KAGGLE_INPUT / "eurosat" / "RBG"
    
    # Pretrained weights from Kaggle input (uploaded from Stage 1 training)
    STAGE1_WEIGHTS_DIR = KAGGLE_INPUT / "model-weight-stage-1" / "stage1"
    PRETRAINED_ENCODER_PATH = STAGE1_WEIGHTS_DIR / "encoder_pretrained.pth"
    STAGE1_BEST_MODEL_PATH = STAGE1_WEIGHTS_DIR / "best_model.pth"
    
    # BIFOLD BigEarthNet pretrained (cached by huggingface_hub)
    BIFOLD_CACHE_DIR = KAGGLE_WORKING / "pretrained_models"
    BIFOLD_REPO_ID = "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"
    
    print(f"  📁 Found {len(BIGEARTHNET_FOLDERS)} BigEarthNet folders")
    print(f"  📍 Metadata: {METADATA_PATH.exists()}")
    print(f"  📍 EuroSAT: {EUROSAT_PATH.exists()}")
    print(f"  📍 Pretrained Encoder (Stage1): {PRETRAINED_ENCODER_PATH.exists() if PRETRAINED_ENCODER_PATH else False}")
    print(f"  📍 BIFOLD available: Can be downloaded from HuggingFace")
    
else:
    # Local environment
    print("💻 Running on local environment")
    
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    
    # Data paths
    METADATA_PATH = DATA_DIR / "metadata.parquet"
    REFERENCE_MAPS_ARCHIVE = DATA_DIR / "Reference_Maps.tar.zst"
    REFERENCE_MAPS_FOLDER = DATA_DIR / "Reference_Maps"
    
    # BigEarthNet folders
    BIGEARTHNET_FOLDERS = [
        DATA_DIR / "BigEarthNet-S2",
    ]
    
    # EuroSAT path
    EUROSAT_PATH = DATA_DIR / "rgbeurosat" / "RBG"
    
    # Pretrained weights from local training
    STAGE1_WEIGHTS_DIR = OUTPUTS_DIR / "stage1"
    PRETRAINED_ENCODER_PATH = STAGE1_WEIGHTS_DIR / "encoder_pretrained.pth"
    STAGE1_BEST_MODEL_PATH = STAGE1_WEIGHTS_DIR / "best_model.pth"
    
    # BIFOLD BigEarthNet pretrained
    BIFOLD_CACHE_DIR = PROJECT_ROOT / "pretrained_models"
    BIFOLD_REPO_ID = "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"

# Common output directories
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, VISUALIZATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Target classes (EuroSAT - 10 classes)
CLASS_NAMES = [
    "AnnualCrop",           # 0
    "Forest",               # 1
    "HerbaceousVegetation", # 2
    "Highway",              # 3
    "Industrial",           # 4
    "Pasture",              # 5
    "PermanentCrop",        # 6
    "Residential",          # 7
    "River",                # 8
    "SeaLake",              # 9
]

NUM_CLASSES = len(CLASS_NAMES)

# CORINE to EuroSAT mapping
CORINE_TO_EUROSAT = {
    # Agricultural areas → AnnualCrop (0)
    211: 0,  # Non-irrigated arable land
    212: 0,  # Permanently irrigated land
    213: 0,  # Rice fields
    
    # Permanent crops → PermanentCrop (6)
    221: 6,  # Vineyards
    222: 6,  # Fruit trees and berry plantations
    223: 6,  # Olive groves
    241: 6,  # Annual crops associated with permanent crops
    242: 6,  # Complex cultivation patterns
    243: 6,  # Land principally occupied by agriculture
    244: 6,  # Agro-forestry areas
    
    # Pastures → Pasture (5)
    231: 5,  # Pastures
    
    # Forest areas → Forest (1)
    311: 1,  # Broad-leaved forest
    312: 1,  # Coniferous forest
    313: 1,  # Mixed forest
    
    # Shrub/herbaceous → HerbaceousVegetation (2)
    321: 2,  # Natural grasslands
    322: 2,  # Moors and heathland
    323: 2,  # Sclerophyllous vegetation
    324: 2,  # Transitional woodland-shrub
    331: 2,  # Beaches, dunes, sands
    332: 2,  # Bare rocks
    333: 2,  # Sparsely vegetated areas
    334: 2,  # Burnt areas
    335: 2,  # Glaciers and perpetual snow
    
    # Wetlands → HerbaceousVegetation (2)
    411: 2,  # Inland marshes
    412: 2,  # Peat bogs
    421: 2,  # Salt marshes
    422: 2,  # Salines
    423: 2,  # Intertidal flats
    
    # Urban fabric → Residential (7)
    111: 7,  # Continuous urban fabric
    112: 7,  # Discontinuous urban fabric
    
    # Industrial/Commercial → Industrial (4)
    121: 4,  # Industrial or commercial units
    122: 4,  # Road and rail networks
    123: 4,  # Port areas
    124: 4,  # Airports
    
    # Infrastructure → Highway (3)
    131: 3,  # Mineral extraction sites
    132: 3,  # Dump sites
    133: 3,  # Construction sites
    141: 3,  # Green urban areas
    142: 3,  # Sport and leisure facilities
    
    # Water bodies → River (8) / SeaLake (9)
    511: 8,  # Water courses (rivers, canals)
    512: 9,  # Water bodies (lakes)
    521: 9,  # Coastal lagoons
    522: 9,  # Estuaries
    523: 9,  # Sea and ocean
}

# Image specifications
BIGEARTHNET_IMAGE_SIZE = 120
EUROSAT_IMAGE_SIZE = 64
SENTINEL2_NORMALIZE_FACTOR = 10000.0

# RGB bands mapping (Sentinel-2)
RGB_BANDS = {
    'red': 'B04',
    'green': 'B03',
    'blue': 'B02'
}

# ============================================================================
# STAGE 1: ENCODER PRE-TRAINING (EuroSAT)
# ============================================================================

STAGE1_CONFIG = {
    'model_name': 'resnet50',  # Options: 'resnet18', 'resnet50', 'efficientnet_b0', etc.
    'pretrained': True,  # Use ImageNet pretrained weights
    'input_size': EUROSAT_IMAGE_SIZE,
    'num_classes': NUM_CLASSES,
    
    # Training parameters
    'batch_size': 64,
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'optimizer': 'adamw',
    
    # Learning rate scheduling
    'scheduler': 'cosine',  # Options: 'cosine', 'step', 'plateau'
    'warmup_epochs': 5,
    'min_lr': 1e-6,
    
    # Data augmentation
    'use_augmentation': True,
    'augmentation_strength': 'medium',  # Options: 'light', 'medium', 'strong'
    
    # Early stopping
    'early_stopping_patience': 15,
    'early_stopping_min_delta': 0.001,
    
    # Checkpointing
    'save_best_only': True,
    'save_frequency': 5,  # Save every N epochs
}

# ============================================================================
# STAGE 2: U-NET TRAINING (BigEarthNet)
# ============================================================================

STAGE2_CONFIG = {
    'encoder_name': 'resnet50',  # Must match Stage 1
    'encoder_weights': 'bifold',  # Options: 'stage1', 'bifold', 'imagenet', None
    'input_size': BIGEARTHNET_IMAGE_SIZE,
    'num_classes': NUM_CLASSES,
    
    # Training parameters
    'batch_size': 32,  # Tăng từ 16 lên 32 để tăng tốc
    'num_epochs': 50,
    'gradient_accumulation_steps': 1,  # Giảm xuống 1 vì đã tăng batch_size
    'mixed_precision': True,  # Use FP16 training
    
    # Dataset sampling (ĐỂ TRAIN NHANH HƠN)
    'use_subset': True,  # Sử dụng subset để test nhanh
    'subset_fraction': 0.05,  # Giảm xuống 5% để load nhanh hơn
    'filter_missing_patches': False,  # TẮT để load nhanh - dataset sẽ handle errors
    
    # Two-phase training
    'freeze_encoder_epochs': 10,  # Phase 2.1: Train decoder only
    'encoder_lr': 1e-5,  # Lower LR for pre-trained encoder
    'decoder_lr': 1e-3,  # Higher LR for decoder
    
    # Optimizer
    'optimizer': 'adamw',
    'weight_decay': 1e-4,
    'betas': (0.9, 0.999),
    
    # Learning rate scheduling
    'scheduler': 'cosine',
    'warmup_epochs': 3,
    'min_lr': 1e-6,
    
    # Loss function
    'loss_type': 'combined',  # Options: 'ce', 'dice', 'combined', 'focal'
    'ce_weight': 1.0,
    'dice_weight': 1.0,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'class_weights': None,  # Will be computed from data if None
    
    # Data augmentation
    'use_augmentation': True,
    'augmentation_strength': 'medium',
    
    # Class balancing
    'use_class_balancing': True,
    'oversample_minority_classes': False,
    
    # Early stopping
    'early_stopping_patience': 20,
    'early_stopping_min_delta': 0.001,
    
    # Checkpointing
    'save_best_only': True,
    'save_frequency': 5,
    'checkpoint_metric': 'val_miou',  # Options: 'val_loss', 'val_miou', 'val_accuracy'
}

# ============================================================================
# DATA LOADING
# ============================================================================

DATALOADER_CONFIG = {
    'num_workers': 4,  # Tăng từ 2 lên 4 (Kaggle có 2 CPU cores, nhưng có thể dùng 4 workers)
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 4,  # Tăng từ 2 lên 4 để prefetch nhiều batch hơn
}

# ============================================================================
# AUGMENTATION SETTINGS
# ============================================================================

AUGMENTATION_CONFIG = {
    'light': {
        'horizontal_flip_prob': 0.5,
        'vertical_flip_prob': 0.5,
        'rotation_limit': 15,
        'brightness_limit': 0.1,
        'contrast_limit': 0.1,
    },
    'medium': {
        'horizontal_flip_prob': 0.5,
        'vertical_flip_prob': 0.5,
        'rotation_limit': 30,
        'brightness_limit': 0.2,
        'contrast_limit': 0.2,
        'hue_shift_limit': 10,
        'sat_shift_limit': 20,
        'shift_scale_rotate_prob': 0.3,
        'elastic_transform_prob': 0.2,
    },
    'strong': {
        'horizontal_flip_prob': 0.5,
        'vertical_flip_prob': 0.5,
        'rotation_limit': 45,
        'brightness_limit': 0.3,
        'contrast_limit': 0.3,
        'hue_shift_limit': 20,
        'sat_shift_limit': 30,
        'shift_scale_rotate_prob': 0.5,
        'elastic_transform_prob': 0.3,
        'grid_distortion_prob': 0.2,
        'optical_distortion_prob': 0.2,
    }
}

# ============================================================================
# EVALUATION
# ============================================================================

EVAL_CONFIG = {
    'metrics': ['accuracy', 'miou', 'precision', 'recall', 'f1'],
    'per_class_metrics': True,
    'confusion_matrix': True,
    'save_predictions': True,
    'num_visualization_samples': 20,
}

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

EXPERIMENT_CONFIG = {
    'use_tensorboard': True,
    'use_wandb': False,  # Set to True if you want to use Weights & Biases
    'project_name': 'land-cover-segmentation',
    'experiment_name': None,  # Will be auto-generated if None
}

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

SEED = 42

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# ============================================================================
# LOGGING
# ============================================================================

LOGGING_CONFIG = {
    'log_interval': 50,  # Log every N batches
    'log_level': 'INFO',  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_experiment_name(stage='stage1'):
    """Generate experiment name with timestamp"""
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{stage}_{timestamp}"

def print_config(config_dict, title="Configuration"):
    """Pretty print configuration"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    for key, value in config_dict.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k:30s}: {v}")
        else:
            print(f"{key:30s}: {value}")
    print(f"{'='*60}\n")

