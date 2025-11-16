"""
Stage 2: Train DeepLabV3+ with BigEarthNet for Semantic Segmentation

This script trains a DeepLabV3+ model with pre-trained encoder from Stage 1
on the BigEarthNet dataset for semantic segmentation.

IMPORTANT: This script uses DeepLabV3+ architecture from src/models/deeplabv3plus.py
Specifically: DeepLabV3PlusWithPretrainedEncoder class
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

# Import DeepLabV3+
from src.models.deeplabv3plus import DeepLabV3PlusWithPretrainedEncoder
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
        {'params': decoder_params, 'lr': config['decoder_lr']},
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
        class DefaultArgs:
            max_samples = None
            num_workers = None
        args = DefaultArgs()

    # Setup logger
    log_file = LOGS_DIR / 'stage2_deeplabv3plus_training.log'
    logger = setup_logger('stage2_deeplabv3plus', log_file, 'INFO')

    logger.info("=" * 70)
    logger.info("Stage 2: DeepLabV3+ Training with BigEarthNet")
    logger.info("=" * 70)

    # Seed
    set_seed(SEED)
    logger.info(f"Random seed set to {SEED}")

    # Check dataset
    if not METADATA_PATH.exists():
        logger.error(f"Metadata file not found at {METADATA_PATH}")
        return

    if not REFERENCE_MAPS_FOLDER.exists():
        logger.error(f"Reference maps folder missing: {REFERENCE_MAPS_FOLDER}")
        return

    # Augmentation pipelines
    logger.info("Creating data augmentation pipelines...")
    train_transform = get_segmentation_train_augmentation(
        input_size=STAGE2_CONFIG['input_size'],
        strength=STAGE2_CONFIG['augmentation_strength']
    )
    val_transform = get_val_augmentation(input_size=STAGE2_CONFIG['input_size'])

    # Create dataloaders
    logger.info("Loading BigEarthNet dataset...")
    num_workers = args.num_workers if args.num_workers is not None else DATALOADER_CONFIG['num_workers']

    try:
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
            max_train_samples=args.max_samples
        )
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    # Create model
    logger.info(f"Creating DeepLabV3+ model with encoder: {STAGE2_CONFIG['encoder_name']}")

    encoder_weights_path = None
    if STAGE2_CONFIG['encoder_weights'] == 'stage1':
        encoder_weights_path = CHECKPOINTS_DIR / "stage1" / "encoder_pretrained.pth"

        if not encoder_weights_path.exists():
            logger.warning(f"Stage 1 weights not found at: {encoder_weights_path}")
            encoder_weights_path = None

    model = DeepLabV3PlusWithPretrainedEncoder(
        encoder_name=STAGE2_CONFIG['encoder_name'],
        num_classes=NUM_CLASSES,
        encoder_pretrained=(STAGE2_CONFIG['encoder_weights'] == 'imagenet'),
        encoder_weights_path=encoder_weights_path,
        freeze_encoder=(STAGE2_CONFIG['freeze_encoder_epochs'] > 0)
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Optimizer & Scheduler
    logger.info("Creating optimizer & scheduler...")
    optimizer = create_optimizer(model, STAGE2_CONFIG)
    scheduler = create_scheduler(optimizer, STAGE2_CONFIG)

    # Loss
    criterion = create_loss_function(
        loss_type=STAGE2_CONFIG['loss_type'],
        num_classes=NUM_CLASSES,
        class_weights=None,
        ce_weight=STAGE2_CONFIG['ce_weight'],
        dice_weight=STAGE2_CONFIG['dice_weight'],
        focal_alpha=STAGE2_CONFIG['focal_alpha'],
        focal_gamma=STAGE2_CONFIG['focal_gamma'],
        device=DEVICE
    )

    # Checkpoint dir
    ckpt_dir = CHECKPOINTS_DIR / "stage2_deeplabv3plus"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        mixed_precision=STAGE2_CONFIG['mixed_precision'],
        gradient_accumulation_steps=STAGE2_CONFIG['gradient_accumulation_steps'],
        checkpoint_dir=ckpt_dir,
        task="segmentation"
    )

    # ============================
    # Phase 2.1 — Freeze encoder
    # ============================
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

        # Unfreeze
        logger.info("Unfreezing encoder...")
        model.unfreeze_encoder()

        # Re-create optimizer/scheduler
        optimizer = create_optimizer(model, STAGE2_CONFIG)
        scheduler = create_scheduler(optimizer, STAGE2_CONFIG)
        trainer.optimizer = optimizer
        trainer.scheduler = scheduler

    # ============================
    # Phase 2.2 — Fine-tune
    # ============================
    logger.info("\n" + "="*70)
    logger.info("Phase 2.2: Fine-tuning entire DeepLabV3+")
    logger.info("="*70)

    ft_epochs = STAGE2_CONFIG['num_epochs'] - STAGE2_CONFIG['freeze_encoder_epochs']

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=ft_epochs,
        metrics_class=SegmentationMetrics,
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
        log_interval=50,
        save_best_only=STAGE2_CONFIG['save_best_only'],
        early_stopping_patience=STAGE2_CONFIG['early_stopping_patience'],
        checkpoint_metric=STAGE2_CONFIG['checkpoint_metric']
    )

    # ============================
    # Final Test Evaluation
    # ============================
    logger.info("\n" + "="*70)
    logger.info("Evaluating on test set...")
    logger.info("="*70)

    best_ckpt = ckpt_dir / "best_model.pth"
    if best_ckpt.exists():
        trainer.load_checkpoint(best_ckpt, load_optimizer=False)

    test_metric_obj = SegmentationMetrics(NUM_CLASSES, CLASS_NAMES)
    test_metrics = trainer.validate(test_loader, test_metric_obj)

    logger.info(f"  Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"  Test mIoU: {test_metrics['mean_iou']:.4f}")
    logger.info(f"  Test Pixel Accuracy: {test_metrics['pixel_accuracy']:.4f}")

    test_metric_obj.print_metrics()

    # Plot confusion matrix
    cm = test_metric_obj.get_confusion_matrix()
    cm_path = VISUALIZATIONS_DIR / "stage2_deeplabv3plus_confusion_matrix.png"
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)

    # Plot training curves
    curves_path = VISUALIZATIONS_DIR / "stage2_deeplabv3plus_training_curves.png"
    plot_training_curves(trainer.history, curves_path)

    # Visualize predictions
    logger.info("Generating prediction visualizations...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        imgs = batch['image'].to(DEVICE)
        masks = batch['mask']

        preds = model(imgs)

        vis_path = VISUALIZATIONS_DIR / "stage2_deeplabv3plus_predictions.png"
        visualize_segmentation(
            images=imgs,
            predictions=preds,
            targets=masks,
            class_names=CLASS_NAMES,
            save_path=vis_path,
            num_samples=min(8, len(imgs))
        )

    logger.info("\nStage 2 DeepLabV3+ training completed successfully!")
    logger.info(f"Best checkpoint saved at: {best_ckpt}")
    logger.info("="*70)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Train DeepLabV3+ with BigEarthNet")

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--encoder-lr", type=float)
    parser.add_argument("--decoder-lr", type=float)
    parser.add_argument("--freeze-epochs", type=int)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)

    args = parser.parse_args()

    # Override config
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
