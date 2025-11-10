"""
Stage 1: Pre-train Encoder with EuroSAT RGB Dataset

This script trains a classification model on EuroSAT to pre-train the encoder,
which will later be used as the backbone for U-Net in Stage 2.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
import argparse

from config import (
    EUROSAT_PATH, CHECKPOINTS_DIR, LOGS_DIR, VISUALIZATIONS_DIR,
    STAGE1_CONFIG, DATALOADER_CONFIG, CLASS_NAMES, NUM_CLASSES,
    DEVICE, SEED
)
from src.data import (
    create_eurosat_dataloaders,
    get_classification_train_augmentation,
    get_val_augmentation
)
from src.models import EncoderClassifier
from src.utils import (
    setup_logger,
    ClassificationMetrics,
    Trainer,
    plot_training_curves,
    plot_confusion_matrix
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
    """Create optimizer based on configuration."""
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    return optimizer


def create_scheduler(optimizer, config, steps_per_epoch):
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
            step_size=30,
            gamma=0.1
        )
    elif config['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=config['min_lr']
        )
    else:
        scheduler = None
    
    return scheduler


def main(args):
    """Main training function."""
    
    # Setup logging
    log_file = LOGS_DIR / 'stage1_training.log'
    logger = setup_logger('stage1', log_file, 'INFO')
    
    logger.info("="*70)
    logger.info("Stage 1: Encoder Pre-training with EuroSAT")
    logger.info("="*70)
    
    # Set random seed
    set_seed(SEED)
    logger.info(f"Random seed set to {SEED}")
    
    # Check if EuroSAT dataset exists
    if not EUROSAT_PATH.exists():
        logger.error(f"EuroSAT dataset not found at {EUROSAT_PATH}")
        logger.error("Please download and extract the EuroSAT RGB dataset")
        return
    
    # Create augmentation pipelines
    logger.info("\nCreating data augmentation pipelines...")
    train_transform = get_classification_train_augmentation(
        input_size=STAGE1_CONFIG['input_size'],
        strength=STAGE1_CONFIG['augmentation_strength']
    )
    val_transform = get_val_augmentation(input_size=STAGE1_CONFIG['input_size'])
    
    # Create dataloaders
    logger.info("Loading EuroSAT dataset...")
    try:
        train_loader, val_loader, test_loader = create_eurosat_dataloaders(
            root_dir=EUROSAT_PATH,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=STAGE1_CONFIG['batch_size'],
            num_workers=DATALOADER_CONFIG['num_workers'],
            pin_memory=DATALOADER_CONFIG['pin_memory']
        )
    except Exception as e:
        logger.error(f"Error loading EuroSAT dataset: {e}")
        return
    
    # Create model
    logger.info(f"\nCreating model: {STAGE1_CONFIG['model_name']}")
    model = EncoderClassifier(
        encoder_name=STAGE1_CONFIG['model_name'],
        num_classes=NUM_CLASSES,
        pretrained=STAGE1_CONFIG['pretrained'],
        dropout=0.5
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    logger.info("\nCreating optimizer and scheduler...")
    optimizer = create_optimizer(model, STAGE1_CONFIG)
    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(optimizer, STAGE1_CONFIG, steps_per_epoch)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create checkpoint directory
    stage1_checkpoint_dir = CHECKPOINTS_DIR / 'stage1'
    stage1_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        scheduler=scheduler,
        mixed_precision=False,  # Usually not needed for classification
        gradient_accumulation_steps=1,
        checkpoint_dir=stage1_checkpoint_dir,
        task='classification'
    )
    
    # Train
    logger.info("\nStarting training...\n")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=STAGE1_CONFIG['num_epochs'],
        metrics_class=ClassificationMetrics,
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
        log_interval=50,
        save_best_only=STAGE1_CONFIG['save_best_only'],
        early_stopping_patience=STAGE1_CONFIG['early_stopping_patience'],
        checkpoint_metric='accuracy'
    )
    
    # Save encoder weights separately
    logger.info("\nSaving encoder weights...")
    encoder_weights_path = stage1_checkpoint_dir / 'encoder_pretrained.pth'
    model.save_encoder_weights(encoder_weights_path)
    logger.info(f"Encoder weights saved to {encoder_weights_path}")
    
    # Evaluate on test set
    logger.info("\n" + "="*70)
    logger.info("Evaluating on test set...")
    logger.info("="*70)
    
    # Load best model
    best_checkpoint = stage1_checkpoint_dir / 'best_model.pth'
    if best_checkpoint.exists():
        trainer.load_checkpoint(best_checkpoint, load_optimizer=False)
    
    # Test
    test_metrics_tracker = ClassificationMetrics(NUM_CLASSES, CLASS_NAMES)
    test_metrics = trainer.validate(test_loader, test_metrics_tracker)
    
    logger.info("\nTest Results:")
    logger.info(f"  Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test F1: {test_metrics['weighted_f1']:.4f}")
    
    # Print detailed metrics
    test_metrics_tracker.print_metrics()
    
    # Plot confusion matrix
    logger.info("\nGenerating visualizations...")
    cm = test_metrics_tracker.get_confusion_matrix()
    cm_path = VISUALIZATIONS_DIR / 'stage1_confusion_matrix.png'
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)
    
    # Plot training curves
    curves_path = VISUALIZATIONS_DIR / 'stage1_training_curves.png'
    plot_training_curves(trainer.history, curves_path)
    
    logger.info("\nStage 1 training completed successfully!")
    logger.info(f"Best model saved at: {best_checkpoint}")
    logger.info(f"Encoder weights saved at: {encoder_weights_path}")
    logger.info("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1: Pre-train Encoder with EuroSAT')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--model', type=str, help='Model name (resnet18, resnet50, etc.)')
    
    args = parser.parse_args()
    
    # Override config with command-line arguments
    if args.batch_size:
        STAGE1_CONFIG['batch_size'] = args.batch_size
    if args.epochs:
        STAGE1_CONFIG['num_epochs'] = args.epochs
    if args.lr:
        STAGE1_CONFIG['learning_rate'] = args.lr
    if args.model:
        STAGE1_CONFIG['model_name'] = args.model
    
    main(args)

