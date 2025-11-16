"""
Stage 2: Train SegNet with BigEarthNet for Semantic Segmentation

This script trains a SegNet model with pre-trained encoder from Stage 1
on the BigEarthNet dataset for semantic segmentation.

IMPORTANT: This script uses SegNet architecture from src/models/segnet.py
Specifically: SegNetWithPretrainedEncoder class
"""

import torch
import numpy as np
import random
from pathlib import Path
import argparse

from config import (
    METADATA_PATH, BIGEARTHNET_FOLDERS, REFERENCE_MAPS_FOLDER,
    CHECKPOINTS_DIR, LOGS_DIR, VISUALIZATIONS_DIR,
    STAGE2_CONFIG, DATALOADER_CONFIG, CLASS_NAMES, NUM_CLASSES,
    CORINE_TO_EUROSAT, DEVICE, SEED
)
from src.data import (
    create_bigearthnet_dataloaders,
    get_segmentation_train_augmentation,
    get_val_augmentation
)
# Import SegNet model from segnet.py
from src.models.segnet import SegNetWithPretrainedEncoder
from src.models.losses import create_loss_function
from src.utils import (
    setup_logger,
    SegmentationMetrics,
    Trainer,
    plot_training_curves,
    plot_confusion_matrix,
    visualize_segmentation
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizer(model, config):
    """Create optimizer with different learning rates for encoder and decoder."""
    encoder_params = list(model.get_encoder_parameters())
    decoder_params = list(model.get_decoder_parameters())
    
    param_groups = [
        {'params': encoder_params, 'lr': config['encoder_lr']},
        {'params': decoder_params, 'lr': config['decoder_lr']}
    ]
    
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=config['weight_decay'],
            betas=config['betas']
        )
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            weight_decay=config['weight_decay'],
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on configuration."""
    if config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=config['min_lr']
        )
    elif config['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=15,
            gamma=0.5
        )
    elif config['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=config['min_lr']
        )
    else:
        scheduler = None
    
    return scheduler


def main(args=None):
    """Main training function."""
    if args is None:
        # Create default args if not provided
        class DefaultArgs:
            max_samples = None
            num_workers = None
        args = DefaultArgs()
    
    # Setup logging
    log_file = LOGS_DIR / 'stage2_segnet_training.log'
    logger = setup_logger('stage2_segnet', log_file, 'INFO')
    
    logger.info("="*70)
    logger.info("Stage 2: SegNet Training with BigEarthNet")
    logger.info("="*70)
    
    # Set random seed
    set_seed(SEED)
    logger.info(f"Random seed set to {SEED}")
    
    # Log GPU information
    logger.info("\n" + "-"*70)
    logger.info("GPU Information")
    logger.info("-"*70)
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.current_device()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU memory: {gpu_memory:.2f} GB")
        logger.info("✅ Training will use GPU")
    else:
        logger.warning("⚠️  CUDA not available. Training will use CPU (very slow!)")
        logger.warning("   Consider enabling GPU in Kaggle notebook settings")
    
    # Check if BigEarthNet data exists
    if not METADATA_PATH.exists():
        logger.error(f"Metadata file not found at {METADATA_PATH}")
        logger.error("Please ensure BigEarthNet metadata is available")
        return
    
    if not REFERENCE_MAPS_FOLDER.exists():
        logger.error(f"Reference maps folder not found at {REFERENCE_MAPS_FOLDER}")
        logger.error("Please extract the Reference_Maps.tar.zst file")
        return
    
    # Create augmentation pipelines
    logger.info("\nCreating data augmentation pipelines...")
    train_transform = get_segmentation_train_augmentation(
        input_size=STAGE2_CONFIG['input_size'],
        strength=STAGE2_CONFIG['augmentation_strength']
    )
    val_transform = get_val_augmentation(input_size=STAGE2_CONFIG['input_size'])
    
    # Create dataloaders
    logger.info("Loading BigEarthNet dataset...")
    try:
        # Use custom num_workers if provided, otherwise use config default
        num_workers = args.num_workers if args.num_workers is not None else DATALOADER_CONFIG['num_workers']
        
        train_loader, val_loader, test_loader = create_bigearthnet_dataloaders(
            metadata_path=METADATA_PATH,
            data_folders=BIGEARTHNET_FOLDERS,
            reference_maps_folder=REFERENCE_MAPS_FOLDER,
            corine_to_eurosat_mapping=CORINE_TO_EUROSAT,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=STAGE2_CONFIG['batch_size'],
            num_workers=num_workers,
            pin_memory=DATALOADER_CONFIG['pin_memory'],
            num_classes=NUM_CLASSES,
            max_train_samples=args.max_samples  # Limit training samples for faster training
        )
    except Exception as e:
        logger.error(f"Error loading BigEarthNet dataset: {e}")
        logger.error("Please check your data paths and ensure all files are available")
        return
    
    # Create model
    logger.info(f"\nCreating SegNet model with encoder: {STAGE2_CONFIG['encoder_name']}")
    
    # Determine encoder weights source
    encoder_weights_path = None
    if STAGE2_CONFIG['encoder_weights'] == 'stage1':
        encoder_weights_path = CHECKPOINTS_DIR / 'stage1' / 'encoder_pretrained.pth'
        if not encoder_weights_path.exists():
            logger.warning(f"Pre-trained encoder not found at {encoder_weights_path}")
            logger.warning("Will use ImageNet pre-trained weights instead")
            encoder_weights_path = None
    
    model = SegNetWithPretrainedEncoder(
        encoder_name=STAGE2_CONFIG['encoder_name'],
        num_classes=NUM_CLASSES,
        encoder_pretrained=(STAGE2_CONFIG['encoder_weights'] == 'imagenet'),
        encoder_weights_path=encoder_weights_path,
        dropout=0.1,
        freeze_encoder=(STAGE2_CONFIG['freeze_encoder_epochs'] > 0)
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    logger.info("\nCreating optimizer and scheduler...")
    optimizer = create_optimizer(model, STAGE2_CONFIG)
    scheduler = create_scheduler(optimizer, STAGE2_CONFIG)
    
    # Create loss function
    logger.info("Creating loss function...")
    criterion = create_loss_function(
        loss_type=STAGE2_CONFIG['loss_type'],
        num_classes=NUM_CLASSES,
        class_weights=None,  # Can be computed from data if needed
        ce_weight=STAGE2_CONFIG['ce_weight'],
        dice_weight=STAGE2_CONFIG['dice_weight'],
        focal_alpha=STAGE2_CONFIG['focal_alpha'],
        focal_gamma=STAGE2_CONFIG['focal_gamma'],
        device=DEVICE
    )
    
    # Create checkpoint directory
    stage2_checkpoint_dir = CHECKPOINTS_DIR / 'stage2_segnet'
    stage2_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        scheduler=scheduler,
        mixed_precision=STAGE2_CONFIG['mixed_precision'],
        gradient_accumulation_steps=STAGE2_CONFIG['gradient_accumulation_steps'],
        checkpoint_dir=stage2_checkpoint_dir,
        task='segmentation'
    )
    
    # Phase 2.1: Train with frozen encoder (if configured)
    if STAGE2_CONFIG['freeze_encoder_epochs'] > 0:
        logger.info("\n" + "="*70)
        logger.info(f"Phase 2.1: Training decoder with frozen encoder ({STAGE2_CONFIG['freeze_encoder_epochs']} epochs)")
        logger.info("="*70)
        
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=STAGE2_CONFIG['freeze_encoder_epochs'],
            metrics_class=SegmentationMetrics,
            num_classes=NUM_CLASSES,
            class_names=CLASS_NAMES,
            log_interval=50,
            save_best_only=True,
            checkpoint_metric=STAGE2_CONFIG['checkpoint_metric']
        )
        
        # Unfreeze encoder
        logger.info("\nUnfreezing encoder for fine-tuning...")
        model.unfreeze_encoder()
        
        # Update optimizer with encoder parameters
        optimizer = create_optimizer(model, STAGE2_CONFIG)
        scheduler = create_scheduler(optimizer, STAGE2_CONFIG)
        trainer.optimizer = optimizer
        trainer.scheduler = scheduler
    
    # Phase 2.2: Fine-tune entire model
    logger.info("\n" + "="*70)
    logger.info("Phase 2.2: Fine-tuning entire SegNet")
    logger.info("="*70)
    
    remaining_epochs = STAGE2_CONFIG['num_epochs'] - STAGE2_CONFIG['freeze_encoder_epochs']
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=remaining_epochs,
        metrics_class=SegmentationMetrics,
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
        log_interval=50,
        save_best_only=STAGE2_CONFIG['save_best_only'],
        early_stopping_patience=STAGE2_CONFIG['early_stopping_patience'],
        checkpoint_metric=STAGE2_CONFIG['checkpoint_metric']
    )
    
    # Evaluate on test set
    logger.info("\n" + "="*70)
    logger.info("Evaluating on test set...")
    logger.info("="*70)
    
    # Load best model
    best_checkpoint = stage2_checkpoint_dir / 'best_model.pth'
    if best_checkpoint.exists():
        trainer.load_checkpoint(best_checkpoint, load_optimizer=False)
    
    # Test
    test_metrics_tracker = SegmentationMetrics(NUM_CLASSES, CLASS_NAMES)
    test_metrics = trainer.validate(test_loader, test_metrics_tracker)
    
    logger.info("\nTest Results:")
    logger.info(f"  Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"  Test mIoU: {test_metrics['mean_iou']:.4f}")
    logger.info(f"  Test Pixel Accuracy: {test_metrics['pixel_accuracy']:.4f}")
    
    # Print detailed metrics
    test_metrics_tracker.print_metrics()
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    # Confusion matrix
    cm = test_metrics_tracker.get_confusion_matrix()
    cm_path = VISUALIZATIONS_DIR / 'stage2_segnet_confusion_matrix.png'
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)
    
    # Training curves
    curves_path = VISUALIZATIONS_DIR / 'stage2_segnet_training_curves.png'
    plot_training_curves(trainer.history, curves_path)
    
    # Segmentation predictions
    logger.info("Generating prediction visualizations...")
    model.eval()
    with torch.no_grad():
        # Get a batch from test set
        batch = next(iter(test_loader))
        images = batch['image'].to(DEVICE)
        masks = batch['mask']
        
        # Predict
        predictions = model(images)
        
        # Visualize
        vis_path = VISUALIZATIONS_DIR / 'stage2_segnet_predictions.png'
        visualize_segmentation(
            images=images,
            predictions=predictions,
            targets=masks,
            class_names=CLASS_NAMES,
            save_path=vis_path,
            num_samples=min(8, len(images))
        )
    
    logger.info("\nStage 2 SegNet training completed successfully!")
    logger.info(f"Best model saved at: {best_checkpoint}")
    logger.info("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2: Train SegNet with BigEarthNet')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--encoder-lr', type=float, help='Encoder learning rate')
    parser.add_argument('--decoder-lr', type=float, help='Decoder learning rate')
    parser.add_argument('--freeze-epochs', type=int, help='Epochs to freeze encoder')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of training samples (for faster testing, None = use all)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of data loading workers (None = use config default)')
    
    args = parser.parse_args()
    
    # Override config with command-line arguments
    if args.batch_size:
        STAGE2_CONFIG['batch_size'] = args.batch_size
    if args.epochs:
        STAGE2_CONFIG['num_epochs'] = args.epochs
    if args.encoder_lr:
        STAGE2_CONFIG['encoder_lr'] = args.encoder_lr
    if args.decoder_lr:
        STAGE2_CONFIG['decoder_lr'] = args.decoder_lr
    if args.freeze_epochs is not None:
        STAGE2_CONFIG['freeze_encoder_epochs'] = args.freeze_epochs
    
    main(args)


