# Setting Up on Kaggle

This guide will help you run the Land Cover Segmentation project on Kaggle.

## 📋 Prerequisites

You need to add the following datasets to your Kaggle notebook:

### Required Datasets

1. **BigEarthNet v2.0 (S2)** - Split into 6 parts:
   - `bigearthnetv2-s2-0`
   - `bigearthnetv2-s2-1`
   - `bigearthnetv2-s2-2`
   - `bigearthnetv2-s2-3`
   - `bigearthnetv2-s2-4`
   - `bigearthnetv2-s2-5`

2. **BigEarthNet S2 Metadata**:
   - `bigearthnet-s2-metadata`

3. **BigEarthNet S2 Reference Maps**:
   - `bigearthnet-s2-referencesmap`

4. **EuroSAT RGB**:
   - `rgbeurosat` or `eurosat`

## 🚀 Quick Start

### Step 1: Create a New Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Select "Notebook" type
4. Enable GPU: Settings → Accelerator → GPU

### Step 2: Add Datasets

In the right panel, click "Add data" and search for each dataset mentioned above. Add them all to your notebook.

### Step 3: Clone the Repository

In the first cell of your notebook:

```python
!git clone https://github.com/yuh-tech/ADCV_2025 /kaggle/working/Final_exam
%cd /kaggle/working/Final_exam
```

### Step 4: Install Dependencies

```python
!pip install -q -r requirements.txt
```

### Step 5: Setup Environment

Run the setup script to extract reference maps and verify datasets:

```python
!python setup_kaggle.py
```

This will:
- ✅ Check if all required datasets are present
- ✅ Extract reference maps to `/kaggle/working/data/`
- ✅ Verify Python packages
- ✅ Show setup status

## 📊 Training

### Stage 1: Pre-train Encoder (EuroSAT)

```python
!python train_stage1.py --epochs 50 --batch-size 64
```

**Expected time**: ~1-2 hours on Kaggle GPU

**Outputs**:
- Model: `/kaggle/working/outputs/checkpoints/stage1/best_model.pth`
- Encoder: `/kaggle/working/outputs/checkpoints/stage1/encoder_pretrained.pth`
- Logs: `/kaggle/working/outputs/logs/stage1_training.log`

### Stage 2: Train U-Net (BigEarthNet)

```python
!python train_stage2.py --epochs 50 --batch-size 16
```

**Expected time**: ~4-6 hours on Kaggle GPU (depending on dataset size)

**Outputs**:
- Model: `/kaggle/working/outputs/checkpoints/stage2/best_model.pth`
- Logs: `/kaggle/working/outputs/logs/stage2_training.log`
- Visualizations: `/kaggle/working/outputs/visualizations/`

## 📁 Data Paths on Kaggle

The configuration automatically detects Kaggle environment and uses these paths:

```
/kaggle/input/                                    # Read-only datasets
├── bigearthnetv2-s2-0/BigEarthNet-S2/
├── bigearthnetv2-s2-1/BigEarthNet-S2/
├── bigearthnetv2-s2-2/BigEarthNet-S2/
├── bigearthnetv2-s2-3/BigEarthNet-S2/
├── bigearthnetv2-s2-4/BigEarthNet-S2/
├── bigearthnetv2-s2-5/BigEarthNet-S2/
├── bigearthnet-s2-metadata/metadata.parquet
├── bigearthnet-s2-referencesmap/Reference_Maps.tar.zst
└── rgbeurosat/RBG/

/kaggle/working/                                  # Writable workspace
├── Final_exam/                                   # Your cloned repo
├── data/                                         # Extracted data
│   └── Reference_Maps/Reference_Maps/            # Extracted reference maps
└── outputs/                                      # Training outputs
    ├── checkpoints/
    ├── logs/
    └── visualizations/
```

## 💡 Tips for Kaggle

### 1. Memory Management

If you run into memory issues:

```python
# In a notebook cell before training
import config
config.STAGE2_CONFIG['batch_size'] = 8
config.STAGE2_CONFIG['num_workers'] = 2
config.DATALOADER_CONFIG['num_workers'] = 2
```

### 2. Save Outputs

Kaggle notebooks automatically save everything in `/kaggle/working/` when you commit. To download:

1. Click "Save Version"
2. After version is completed, go to "Output" tab
3. Download the output files

### 3. Monitor Training

View training progress in real-time:

```python
# In a separate cell
!tail -f /kaggle/working/outputs/logs/stage1_training.log
```

Press Stop to exit.

### 4. Visualize Results

```python
from IPython.display import Image, display
import matplotlib.pyplot as plt

# Show training curves
display(Image('/kaggle/working/outputs/visualizations/stage1_training_curves.png'))

# Show confusion matrix
display(Image('/kaggle/working/outputs/visualizations/stage1_confusion_matrix.png'))
```

### 5. Resume Training

If your session times out, you can resume from the last checkpoint:

```python
# This is already handled in the training scripts
# The trainer automatically loads from 'last_model.pth' if available
```

## 🔧 Troubleshooting

### Problem: "Dataset not found"

**Solution**: Make sure you've added all required datasets to your notebook using the "Add data" button.

### Problem: "Out of memory"

**Solution**: Reduce batch size:
```python
!python train_stage2.py --batch-size 8 --epochs 50
```

### Problem: "Reference maps extraction failed"

**Solution**: The extraction requires `zstd`. Install it:
```python
!apt-get install -y zstd
!python setup_kaggle.py
```

### Problem: "Kernel appears to be dead"

**Solution**: This usually means you ran out of memory. Reduce `batch_size` and `num_workers`.

## 📊 Expected Results on Kaggle

### Stage 1 (with Kaggle GPU)
- Training time: ~1-2 hours (50 epochs)
- Expected accuracy: 92-96%
- GPU memory usage: ~2-3 GB

### Stage 2 (with Kaggle GPU)  
- Training time: ~4-6 hours (50 epochs)
- Expected mIoU: 50-65%
- GPU memory usage: ~8-12 GB (batch_size=16)

## 🌟 Complete Kaggle Notebook Example

```python
# Cell 1: Setup
!git clone https://github.com/YOUR_REPO.git /kaggle/working/Final_exam
%cd /kaggle/working/Final_exam
!pip install -q -r requirements.txt

# Cell 2: Verify setup
!python setup_kaggle.py

# Cell 3: Train Stage 1
!python train_stage1.py --epochs 50 --batch-size 64

# Cell 4: View Stage 1 results
from IPython.display import Image, display
display(Image('/kaggle/working/outputs/visualizations/stage1_confusion_matrix.png'))

# Cell 5: Train Stage 2
!python train_stage2.py --epochs 50 --batch-size 16

# Cell 6: View Stage 2 results
display(Image('/kaggle/working/outputs/visualizations/stage2_predictions.png'))
display(Image('/kaggle/working/outputs/visualizations/stage2_confusion_matrix.png'))
```

## 📥 Download Results

After training completes:

1. **Save Version**: Click "Save Version" → "Save & Run All"
2. **Wait for Completion**: The notebook will re-run completely
3. **Download Output**:
   - Go to "Output" tab of your notebook
   - Download the output ZIP file
   - It will contain everything in `/kaggle/working/outputs/`

## 🔗 Useful Links

- [Kaggle Notebooks Documentation](https://www.kaggle.com/docs/notebooks)
- [BigEarthNet Dataset](https://bigearth.net/)
- [EuroSAT Dataset](https://github.com/phelber/eurosat)

---

**Happy Training on Kaggle! 🚀**

