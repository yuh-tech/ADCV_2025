"""
Stage 2: Train DeepLabV3+ with BigEarthNet for Semantic Segmentation

This script trains a DeepLabV3+ model (MobileNetV3 backbone) on the BigEarthNet dataset.

NOTE:
- The DeepLabV3Plus class implemented in src/models/deeplabv3plus.py uses MobileNetV3 backbone.
- Stage 1 encoder checkpoints (ResNet) cannot be loaded into MobileNetV3; if STAGE2_CONFIG['encoder_weights']=='stage1'
  the script will log a warning and continue using ImageNet pretrained MobileNetV3 (or random init).
"""

import torch
import numpy as np
import random
from pathlib import Path
import argparse
import logging

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
from src.models.deeplabv3plus import DeepLabV3Plus
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


def _get_encoder_and_decoder_param_lists(model):
    """
    Return (encoder_params, decoder_params) lists.
    We treat `model.low_level` and `model.high_level` as encoder if present.
    Otherwise, fallback: encoder_params = [], decoder_params = all params.
    """
    encoder_params = []
    decoder_params = []

    # attempt detection of MobileNetV3-style attributes
    if hasattr(model, 'low_level') and hasattr(model, 'high_level'):
        # collect encoder params
        for p in model.low_level.parameters():
            encoder_params.append(p)
        for p in model.high_level.parameters():
            encoder_params.append(p)

        # decoder = all remaining parameters
        encoder_set = set([id(p) for p in encoder_params])
        for p in model.parameters():
            if id(p) not in encoder_set:
                decoder_params.append(p)
    else:
        # fallback: no encoder/decoder split
        decoder_params = list(model.parameters())

    return encoder_params, decoder_params


def create_optimizer(model, config):
    """
    Create optimizer. If encoder/decoder lr specified and model exposes encoder parts,
    create two param groups; otherwise use single group.
    """
    encoder_params, decoder_params = _get_encoder_and_decoder_param_lists(model)

    # If encoder/decoder split exists and config contains lr for both, create groups
    if encoder_params and 'encoder_lr' in config and 'decoder_lr' in config:
        param_groups = [
            {'params': encoder_params, 'lr': config['encoder_lr']},
            {'params': decoder_params, 'lr': config['decoder_lr']},
        ]
    else:
        param_groups = [{'params': decoder_params, 'lr': config.get('learning_rate', 1e-3)}]

    opt_name = config.get('optimizer', 'adam').lower()
    if opt_name == 'adam':
        optimizer = torch.optim.Adam(param_groups, weight_decay=config.get('weight_decay', 1e-4))
    elif opt_name == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.get('weight_decay', 1e-4),
                                      betas=config.get('betas', (0.9, 0.999)))
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=config.get('momentum', 0.9),
                                    weight_decay=config.get('weight_decay', 1e-4))
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on configuration."""
    sched = config.get('scheduler', None)
    if sched == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.get('num_epochs', 100), eta_min=config.get('min_lr', 1e-6)
        )
    elif sched == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.get('step_size', 15),
                                                    gamma=config.get('step_gamma', 0.5))
    elif sched == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=config.get('plateau_factor', 0.5),
            patience=config.get('plateau_patience', 5), min_lr=config.get('min_lr', 1e-6)
        )
    else:
        scheduler = None

    return scheduler


def freeze_encoder_weights(model):
    """Set requires_grad=False for encoder parameters (if detected)."""
    if hasattr(model, 'low_level') and hasattr(model, 'high_level'):
        for p in model.low_level.parameters():
            p.requires_grad = False
        for p in model.high_level.parameters():
            p.requires_grad = False
        logging.getLogger(__name__).info("Encoder parameters frozen.")
    else:
        logging.getLogger(__name__).warning(
            "Model does not expose low_level/high_level attributes; cannot freeze encoder.")


def unfreeze_encoder_weights(model):
    """Set requires_grad=True for encoder parameters (if detected)."""
    if hasattr(model, 'low_level') and hasattr(model, 'high_level'):
        for p in model.low_level.parameters():
            p.requires_grad = True
        for p in model.high_level.parameters():
            p.requires_grad = True
        logging.getLogger(__name__).info("Encoder parameters unfrozen.")
    else:
        logging.getLogger(__name__).warning(
            "Model does not expose low_level/high_level attributes; cannot unfreeze encoder.")


def main(args=None):
    """Main training function."""
    if args is None:
        class DefaultArgs:
            max_samples = None
            num_workers = None

        args = DefaultArgs()

    # Setup logger
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
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
        strength=STAGE2_CONFIG.get('augmentation_strength', 1.0)
    )
    val_transform = get_val_augmentation(input_size=STAGE2_CONFIG['input_size'])

    # Create dataloaders
    logger.info("Loading BigEarthNet dataset...")
    num_workers = args.num_workers if args.num_workers is not None else DATALOADER_CONFIG.get('num_workers', 4)

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
            pin_memory=DATALOADER_CONFIG.get('pin_memory', True),
            num_classes=NUM_CLASSES,
        )
    except Exception as e:
        logger.exception(f"Error loading dataset: {e}")
        return

    # Create model
    logger.info(f"Creating DeepLabV3+ model (MobileNetV3 backbone).")
    encoder_weights_path = None
    if STAGE2_CONFIG.get('encoder_weights', 'imagenet') == 'stage1':
        encoder_weights_path = CHECKPOINTS_DIR / "stage1" / "encoder_pretrained.pth"
        if not encoder_weights_path.exists():
            logger.warning(
                f"Stage 1 weights not found at: {encoder_weights_path}. Will use ImageNet pretrained MobileNetV3 (if requested).")
            encoder_weights_path = None
        else:
            logger.warning("Stage1 encoder weights are for a different backbone (ResNet). "
                           "They cannot be safely loaded into MobileNetV3. Skipping.")

    # Instantiate model: only supports pretrained flag in this implementation
    model = DeepLabV3Plus(
        num_classes=NUM_CLASSES,
        pretrained=(STAGE2_CONFIG.get('encoder_weights', 'imagenet') == 'imagenet')
    )
    model = model.to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # If freeze phase specified, freeze encoder parameters before first fit
    freeze_epochs = STAGE2_CONFIG.get('freeze_encoder_epochs', 0)
    if freeze_epochs > 0:
        freeze_encoder_weights(model)
    else:
        logger.info("No encoder freezing requested (freeze_encoder_epochs <= 0).")

    # Optimizer & Scheduler
    logger.info("Creating optimizer & scheduler...")
    optimizer = create_optimizer(model, STAGE2_CONFIG)
    scheduler = create_scheduler(optimizer, STAGE2_CONFIG)

    # Loss
    criterion = create_loss_function(
        loss_type=STAGE2_CONFIG.get('loss_type', 'ce'),
        num_classes=NUM_CLASSES,
        class_weights=None,
        ce_weight=STAGE2_CONFIG.get('ce_weight', 1.0),
        dice_weight=STAGE2_CONFIG.get('dice_weight', 0.0),
        focal_alpha=STAGE2_CONFIG.get('focal_alpha', 0.25),
        focal_gamma=STAGE2_CONFIG.get('focal_gamma', 2.0),
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
        mixed_precision=STAGE2_CONFIG.get('mixed_precision', False),
        gradient_accumulation_steps=STAGE2_CONFIG.get('gradient_accumulation_steps', 1),
        checkpoint_dir=ckpt_dir,
        task="segmentation"
    )

    # ============================
    # Phase 2.1 — Freeze encoder
    # ============================
    if freeze_epochs > 0:
        logger.info("\n" + "=" * 70)
        logger.info(f"Phase 2.1: Training decoder with frozen encoder ({freeze_epochs} epochs)")
        logger.info("=" * 70)

        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=freeze_epochs,
            metrics_class=SegmentationMetrics,
            num_classes=NUM_CLASSES,
            class_names=CLASS_NAMES,
            log_interval=STAGE2_CONFIG.get('log_interval', 50),
            save_best_only=True,
            checkpoint_metric=STAGE2_CONFIG.get('checkpoint_metric', 'loss')
        )

        # Unfreeze encoder for fine-tuning
        logger.info("Unfreezing encoder for fine-tuning...")
        unfreeze_encoder_weights(model)

        # Re-create optimizer & scheduler to pick up newly trainable params
        optimizer = create_optimizer(model, STAGE2_CONFIG)
        scheduler = create_scheduler(optimizer, STAGE2_CONFIG)
        trainer.optimizer = optimizer
        trainer.scheduler = scheduler

    # ============================
    # Phase 2.2 — Fine-tune
    # ============================
    logger.info("\n" + "=" * 70)
    logger.info("Phase 2.2: Fine-tuning entire DeepLabV3+")
    logger.info("=" * 70)

    ft_epochs = STAGE2_CONFIG.get('num_epochs', 100) - freeze_epochs
    ft_epochs = max(0, ft_epochs)

    if ft_epochs > 0:
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=ft_epochs,
            metrics_class=SegmentationMetrics,
            num_classes=NUM_CLASSES,
            class_names=CLASS_NAMES,
            log_interval=STAGE2_CONFIG.get('log_interval', 50),
            save_best_only=STAGE2_CONFIG.get('save_best_only', True),
            early_stopping_patience=STAGE2_CONFIG.get('early_stopping_patience', None),
            checkpoint_metric=STAGE2_CONFIG.get('checkpoint_metric', 'loss')
        )
    else:
        logger.info("Fine-tuning epochs <= 0, skipping Phase 2.2.")

    # ============================
    # Final Test Evaluation
    # ============================
    logger.info("\n" + "=" * 70)
    logger.info("Evaluating on test set...")
    logger.info("=" * 70)

    best_ckpt = ckpt_dir / "best_model.pth"
    if best_ckpt.exists():
        trainer.load_checkpoint(best_ckpt, load_optimizer=False)
    else:
        logger.warning("Best checkpoint not found; evaluating current model state.")

    test_metric_obj = SegmentationMetrics(NUM_CLASSES, CLASS_NAMES)
    test_metrics = trainer.validate(test_loader, test_metric_obj)

    logger.info(f"  Test Loss: {test_metrics.get('loss', 0):.4f}")
    logger.info(f"  Test mIoU: {test_metrics.get('mean_iou', 0):.4f}")
    logger.info(f"  Test Pixel Accuracy: {test_metrics.get('pixel_accuracy', 0):.4f}")

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
    logger.info("=" * 70)


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

    # Override config if provided
    if args.batch_size:
        STAGE2_CONFIG['batch_size'] = args.batch_size
    if args.epochs:
        STAGE2_CONFIG['num_epochs'] = args.epochs
    if args.encoder_lr is not None:
        STAGE2_CONFIG['encoder_lr'] = args.encoder_lr
    if args.decoder_lr is not None:
        STAGE2_CONFIG['decoder_lr'] = args.decoder_lr
    if args.freeze_epochs is not None:
        STAGE2_CONFIG['freeze_encoder_epochs'] = args.freeze_epochs

    main(args)
