# Land Cover Segmentation from Satellite Imagery using U-Net

A comprehensive deep learning project for semantic segmentation of land cover types from Sentinel-2 satellite imagery using U-Net architecture with transfer learning.

## 🎯 Project Overview

This project implements a **two-stage transfer learning approach** for semantic segmentation:

### Stage 1: Encoder Pre-training
- **Dataset**: EuroSAT RGB (~27,000 images)
- **Task**: Image classification (10 classes)
- **Goal**: Pre-train encoder to learn robust feature representations from satellite imagery

### Stage 2: U-Net Training
- **Dataset**: BigEarthNet (~590,000 patches)
- **Task**: Semantic segmentation
- **Architecture**: U-Net with pre-trained encoder from Stage 1
- **Benefits**: Faster convergence, better performance

## 📊 Target Classes (10 Classes)

```
0. AnnualCrop          - Annual crops
1. Forest              - Forest areas
2. HerbaceousVegetation - Herbaceous vegetation
3. Highway             - Roads and transportation
4. Industrial          - Industrial areas
5. Pasture             - Pasture lands
6. PermanentCrop       - Permanent crops
7. Residential         - Residential areas
8. River               - Rivers
9. SeaLake             - Seas and lakes
```

## 🏗️ Project Structure

```
.
├── config.py                  # Configuration and hyperparameters
├── train_stage1.py           # Stage 1 training script
├── train_stage2.py           # Stage 2 training script
├── requirements.txt          # Python dependencies
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── augmentations.py       # Data augmentation pipelines
│   │   ├── bigearthnet_dataset.py # BigEarthNet dataset
│   │   ├── eurosat_dataset.py     # EuroSAT dataset
│   │   └── utils.py               # Data loading utilities
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py        # Encoder architectures
│   │   ├── unet.py           # U-Net implementation
│   │   └── losses.py         # Loss functions
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py        # Evaluation metrics
│       ├── visualization.py  # Visualization utilities
│       ├── logger.py         # Logging setup
│       └── trainer.py        # Training utilities
│
├── notebooks/
│   └── dataloader.ipynb      # Data exploration notebook
│
├── data/                     # Data directory (not in repo)
│   ├── metadata.parquet
│   ├── Reference_Maps/
│   ├── BigEarthNet-S2/
│   └── rgbeurosat/
│
├── outputs/                  # Output directory (created during training)
│   ├── checkpoints/
│   ├── logs/
│   └── visualizations/
│
└── references/
    └── project-summary-doc.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Final_exam
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Setup

#### 1. Download Datasets

**EuroSAT RGB** (for Stage 1):
- Download from: https://github.com/phelber/eurosat
- Extract to: `data/rgbeurosat/RBG/`
- Structure should be: `data/rgbeurosat/RBG/train/`, `val/`, `test/`

**BigEarthNet** (for Stage 2):
- Download from: https://bigearth.net/ or Kaggle
- Required files:
  - `metadata.parquet` → `data/metadata.parquet`
  - `Reference_Maps.tar.zst` → Extract to `data/Reference_Maps/`
  - BigEarthNet image folders → `data/BigEarthNet-S2/`

#### 2. Extract Reference Maps

```bash
# Install zstd if not available
# Ubuntu/Debian: sudo apt-get install zstd
# macOS: brew install zstd
# Windows: Download from https://github.com/facebook/zstd/releases

# Extract reference maps
tar -I zstd -xf data/Reference_Maps.tar.zst -C data/
```

#### 3. Verify Data Structure

```
data/
├── metadata.parquet
├── Reference_Maps/
│   └── Reference_Maps/
│       └── [Tile folders]/
├── BigEarthNet-S2/
│   └── [Tile folders]/
└── rgbeurosat/
    └── RBG/
        ├── train/
        ├── val/
        └── test/
```

## 📖 Usage

### Configuration

Edit `config.py` to adjust:
- Dataset paths
- Model architecture
- Hyperparameters
- Training settings

### Stage 1: Pre-train Encoder

Train the encoder on EuroSAT for classification:

```bash
python train_stage1.py
```

**Optional arguments:**
```bash
python train_stage1.py \
    --model resnet50 \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001
```

**Output:**
- Model checkpoint: `outputs/checkpoints/stage1/best_model.pth`
- Encoder weights: `outputs/checkpoints/stage1/encoder_pretrained.pth`
- Training logs: `outputs/logs/stage1_training.log`
- Visualizations: `outputs/visualizations/stage1_*.png`

### Stage 2: Train U-Net

Train U-Net on BigEarthNet for segmentation:

```bash
python train_stage2.py
```

**Optional arguments:**
```bash
python train_stage2.py \
    --batch-size 16 \
    --epochs 50 \
    --encoder-lr 0.00001 \
    --decoder-lr 0.001 \
    --freeze-epochs 10
```

**Output:**
- Model checkpoint: `outputs/checkpoints/stage2/best_model.pth`
- Training logs: `outputs/logs/stage2_training.log`
- Visualizations: `outputs/visualizations/stage2_*.png`

### Training Strategies

**Two-Phase Training (Recommended):**
1. **Phase 2.1**: Freeze encoder, train decoder only (5-10 epochs)
2. **Phase 2.2**: Unfreeze encoder, fine-tune entire model (remaining epochs)

This approach:
- Prevents catastrophic forgetting of pre-trained features
- Allows decoder to learn good initialization
- Leads to better final performance

## 🎨 Features

### Data Augmentation
- **Classification** (Stage 1): Rotation, flip, brightness/contrast, color jitter
- **Segmentation** (Stage 2): Same as above + elastic transform, grid distortion

### Loss Functions
- **Cross-Entropy Loss**: Standard classification/segmentation loss
- **Dice Loss**: Addresses class imbalance, focuses on overlap
- **Focal Loss**: Handles class imbalance with hard example mining
- **Combined Loss**: CE + Dice for best results

### Model Architectures
**Supported Encoders:**
- ResNet (18, 34, 50)
- EfficientNet (B0, B1)
- MobileNetV2

**U-Net Features:**
- Skip connections for feature preservation
- Pre-trained encoder initialization
- Flexible decoder design
- Dropout for regularization

### Evaluation Metrics
- **Pixel Accuracy**: Overall correctness
- **Mean IoU (mIoU)**: Primary segmentation metric
- **Per-class IoU**: Class-wise performance
- **Precision, Recall, F1**: Detailed analysis
- **Confusion Matrix**: Misclassification patterns

### Optimizations
- **Mixed Precision Training** (FP16): Faster training, lower memory
- **Gradient Accumulation**: Simulate larger batch sizes
- **Learning Rate Scheduling**: Cosine annealing, plateau
- **Early Stopping**: Prevent overfitting
- **Gradient Clipping**: Training stability

## 📊 Expected Results

### Stage 1 (EuroSAT Classification)
- **Accuracy**: 92-96%
- **F1-Score**: 0.90-0.95

### Stage 2 (BigEarthNet Segmentation)

**With Random Initialization:**
- Mean IoU: 30-40%
- Pixel Accuracy: 50-60%

**With Pre-trained Encoder (Stage 1):**
- Mean IoU: 50-65%
- Pixel Accuracy: 70-80%
- Training time: 30-50% faster convergence

**Per-class Performance:**
- **High** (>70% IoU): Forest, AnnualCrop, Residential
- **Medium** (50-70% IoU): Pasture, PermanentCrop, HerbaceousVegetation
- **Low** (<50% IoU): Highway, Industrial, River (limited data)

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# In config.py
STAGE2_CONFIG['batch_size'] = 8  # Reduce batch size
STAGE2_CONFIG['gradient_accumulation_steps'] = 4  # Increase accumulation
STAGE2_CONFIG['mixed_precision'] = True  # Enable FP16
```

**2. Data Loading Errors**
- Verify all data paths in `config.py`
- Check if Reference_Maps are extracted correctly
- Ensure metadata.parquet is accessible

**3. Slow Training**
- Reduce `num_workers` if CPU-bound
- Enable `pin_memory` for faster GPU transfer
- Use mixed precision training

**4. Poor Performance**
- Increase training epochs
- Adjust learning rates
- Try different augmentation strengths
- Check class distribution and balancing

## 📚 References

### Datasets
- **EuroSAT**: [GitHub](https://github.com/phelber/eurosat)
- **BigEarthNet**: [Website](https://bigearth.net/)
- **CORINE Land Cover**: [Copernicus](https://land.copernicus.eu/)

### Papers
- U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- BigEarthNet: Sumbul et al., "BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding"
- EuroSAT: Helber et al., "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification"

### Technical Resources
- [Sentinel-2 Bands](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions)
- [PyTorch Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch)
- [Albumentations Documentation](https://albumentations.ai/)

## 📝 License

This project is for educational purposes. Please cite the original datasets and papers when using this code.

## 🙏 Acknowledgments

- EuroSAT and BigEarthNet dataset creators
- PyTorch and torchvision teams
- Albumentations library contributors

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Happy Training! 🚀**

