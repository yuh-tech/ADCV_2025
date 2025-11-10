# Bài toán: Phân loại Loại hình Sử dụng Đất từ Ảnh Vệ tinh sử dụng U-Net

## 1. MÔ TẢ BÀI TOÁN

### 1.1. Tổng quan
**Bài toán**: Image Semantic Segmentation cho ảnh vệ tinh
- **Đầu vào**: Ảnh RGB từ vệ tinh Sentinel-2
- **Đầu ra**: Segmentation mask phân loại từng pixel theo loại hình sử dụng đất
- **Số classes**: 10 classes (theo EuroSAT dataset)
- **Kiến trúc**: U-Net với transfer learning

### 1.2. Phương pháp tiếp cận
**Chiến lược Transfer Learning 2-giai đoạn**:

**Giai đoạn 1 - Pre-training Encoder**:
- Sử dụng tập **EuroSAT RGB** để pre-train phần encoder của U-Net
- Task: Classification với 10 classes
- Mục tiêu: Học các feature representations tốt cho ảnh vệ tinh

**Giai đoạn 2 - Fine-tuning toàn bộ U-Net**:
- Sử dụng tập **BigEarthNet** để train toàn bộ U-Net
- Task: Semantic Segmentation
- Encoder được khởi tạo từ weights đã pre-train
- Decoder được train từ đầu
- Lợi ích: Hội tụ nhanh hơn, performance tốt hơn

### 1.3. Classes mục tiêu (EuroSAT)
```
0. AnnualCrop          - Cây trồng hàng năm
1. Forest              - Rừng
2. HerbaceousVegetation - Thảm thực vật cỏ
3. Highway             - Đường cao tốc/giao thông
4. Industrial          - Khu công nghiệp
5. Pasture             - Đồng cỏ chăn thả
6. PermanentCrop       - Cây trồng lâu năm
7. Residential         - Khu dân cư
8. River               - Sông
9. SeaLake             - Biển/Hồ
```

---

## 2. MÔ TẢ TẬP DỮ LIỆU

### 2.1. EuroSAT RGB Dataset

**Thông tin chung**:
- Nguồn: Ảnh vệ tinh Sentinel-2
- Kích thước ảnh: 64x64 pixels
- Format: RGB (3 channels)
- Số lượng: ~27,000 ảnh
- Task: Image Classification

**Cấu trúc thư mục**:
```
rgbeurosat/
└── RBG/
    ├── test/
    │   ├── AnnualCrop/
    │   ├── Forest/
    │   ├── HerbaceousVegetation/
    │   ├── Highway/
    │   ├── Industrial/
    │   ├── Pasture/
    │   ├── PermanentCrop/
    │   ├── Residential/
    │   ├── River/
    │   └── SeaLake/
    ├── train/
    │   └── [same structure as test]
    └── val/
        └── [same structure as test]
```

**Đặc điểm**:
- Mỗi ảnh thuộc duy nhất 1 class (single-label classification)
- Ảnh đã được cắt và gán nhãn sẵn
- Phân chia train/val/test có sẵn
- Chất lượng cao, ít noise

**Mục đích sử dụng**:
- Pre-training encoder của U-Net
- Học feature representations cho 10 loại land cover

---

### 2.2. BigEarthNet Dataset

**Thông tin chung**:
- Nguồn: Ảnh vệ tinh Sentinel-2 Level-2A
- Coverage: 10 quốc gia châu Âu
- Số lượng patches: ~590,000 patches
- Kích thước mỗi patch: 120x120 pixels
- Bands: 12 spectral bands của Sentinel-2
- Label system: CORINE Land Cover (19 classes, multi-label)

**Cấu trúc dữ liệu trên Kaggle**:

#### 2.2.1. Metadata
**File**: `bigearthnet-s2-metadata/metadata.parquet`

**Cấu trúc**:
```python
Columns:
- patch_id: str          # ID duy nhất của patch (VD: S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57)
- labels: list[str]      # Danh sách các labels CORINE (multi-label)
- split: str             # 'train', 'validation', hoặc 'test'
- country: str           # Quốc gia (Austria, Serbia, etc.)
```

**Ví dụ**:
```
patch_id: S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57
labels: ['Arable land', 'Broad-leaved forest', 'Mixed forest']
split: test
country: Austria
```

#### 2.2.2. Reference Maps (Segmentation Masks)
**File**: `bigearthnet-s2-referencesmap/Reference_Maps.tar.zst`

**Cấu trúc sau khi extract**:
```
Reference_Maps/
├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP/
│   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_reference_map.tif
│   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_58_reference_map.tif
│   └── ...
├── S2A_MSIL2A_20170617T133321_N9999_R081_T23WMT/
│   └── ...
└── ...
```

**Đặc điểm reference maps**:
- Format: GeoTIFF (.tif)
- Kích thước: 120x120 pixels
- Data type: uint16
- Giá trị pixel: Mã CORINE Land Cover (VD: 211, 231, 311, 313)
- Mỗi pixel có 1 class duy nhất (semantic segmentation)

**Ví dụ giá trị trong reference map**:
```python
Shape: (120, 120)
Data type: uint16
Unique values: [211, 231, 311, 313]
# 211 = Non-irrigated arable land
# 231 = Pastures
# 311 = Broad-leaved forest
# 313 = Mixed forest
```

#### 2.2.3. Image Data
**Folders**: `bigearthnetv2-s2-0/` đến `bigearthnetv2-s2-5/` (dataset được chia nhỏ)

**Cấu trúc**:
```
bigearthnetv2-s2-0/
└── BigEarthNet-S2/
    ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP/    # Tile folder (chứa nhiều patches)
    │   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57/    # Patch folder
    │   │   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_B01.tif
    │   │   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_B02.tif  # Blue
    │   │   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_B03.tif  # Green
    │   │   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_B04.tif  # Red
    │   │   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_B05.tif
    │   │   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_B06.tif
    │   │   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_B07.tif
    │   │   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_B08.tif
    │   │   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_B09.tif
    │   │   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_B11.tif
    │   │   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_B12.tif
    │   │   └── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_B8A.tif
    │   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_58/
    │   └── ...
    └── ...
```

**Sentinel-2 Bands**:
```
B01: Coastal aerosol (60m resolution)
B02: Blue (10m) - SỬ DỤNG CHO RGB
B03: Green (10m) - SỬ DỤNG CHO RGB
B04: Red (10m) - SỬ DỤNG CHO RGB
B05: Vegetation Red Edge (20m)
B06: Vegetation Red Edge (20m)
B07: Vegetation Red Edge (20m)
B08: NIR (10m)
B8A: Vegetation Red Edge (20m)
B09: Water vapour (60m)
B11: SWIR (20m)
B12: SWIR (20m)
```

**Đặc điểm ảnh**:
- Format: GeoTIFF (.tif)
- Kích thước: 120x120 pixels (các bands 10m), 60x60 (bands 20m), 20x20 (bands 60m)
- Data type: uint16
- Giá trị: 0-10000 (reflectance values)
- Cần normalize về [0, 1] bằng cách chia cho 10000

**Tạo RGB Image**:
```python
RGB = [B04 (Red), B03 (Green), B02 (Blue)]
# Normalize: RGB_normalized = RGB / 10000.0
```

---

## 3. VẤN ĐỀ VÀ GIẢI PHÁP

### 3.1. Vấn đề chính

**Vấn đề 1: Mismatch về label schema**
- BigEarthNet: 19 classes CORINE (multi-label)
- EuroSAT: 10 classes (single-label)
- **Cần mapping từ CORINE → EuroSAT**

**Vấn đề 2: Kích thước ảnh khác nhau**
- EuroSAT: 64x64 pixels
- BigEarthNet: 120x120 pixels
- **Giải pháp**: Resize về cùng kích thước hoặc sử dụng adaptive pooling

**Vấn đề 3: Chất lượng dữ liệu**
- BigEarthNet có thể có patches bị cloud cover
- Một số patches có nhiều classes overlap
- **Giải pháp**: Filter corrupted samples, error handling

### 3.2. Giải pháp: CORINE to EuroSAT Mapping

#### 3.2.1. Mapping Table

```python
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
    
    # Forest and semi-natural areas → Forest (1)
    311: 1,  # Broad-leaved forest
    312: 1,  # Coniferous forest
    313: 1,  # Mixed forest
    
    # Shrub and/or herbaceous vegetation → HerbaceousVegetation (2)
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
    122: 4,  # Road and rail networks and associated land
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
```

#### 3.2.2. Lưu ý về Mapping

**Các trường hợp đặc biệt**:

1. **Multi-label trong BigEarthNet**:
   - Một patch có thể có nhiều labels CORINE
   - Reference map chỉ có 1 class/pixel → Đã được resolved tự động
   - Chọn class chiếm diện tích lớn nhất trong mask

2. **Các CORINE classes không phổ biến**:
   - Một số classes CORINE hiếm (VD: 335 - Glaciers)
   - Có thể gộp vào class gần nhất hoặc loại bỏ

3. **Ambiguous mappings**:
   - 122 (Road and rail networks) → Industrial (có thể là Highway)
   - 141, 142 (Green urban, Sport facilities) → Highway (có thể là Residential)
   - Có thể điều chỉnh dựa trên phân tích dữ liệu

### 3.3. Pipeline Xử lý Dữ liệu

#### Bước 1: Load và Parse Metadata
```python
import pandas as pd

# Load metadata
metadata_df = pd.read_parquet('metadata.parquet')

# Filter by split
train_df = metadata_df[metadata_df['split'] == 'train']
val_df = metadata_df[metadata_df['split'] == 'validation']
test_df = metadata_df[metadata_df['split'] == 'test']
```

#### Bước 2: Load RGB Image từ Bands
```python
import rasterio
import numpy as np

def load_rgb(patch_folder):
    """Load RGB from B04, B03, B02"""
    with rasterio.open(f'{patch_folder}_B04.tif') as src:
        red = src.read(1)
    with rasterio.open(f'{patch_folder}_B03.tif') as src:
        green = src.read(1)
    with rasterio.open(f'{patch_folder}_B02.tif') as src:
        blue = src.read(1)
    
    rgb = np.stack([red, green, blue], axis=-1)
    rgb = rgb / 10000.0  # Normalize to [0, 1]
    return rgb
```

#### Bước 3: Load và Convert Reference Map
```python
def load_mask(reference_map_path, corine_to_eurosat):
    """Load reference map and convert CORINE to EuroSAT"""
    with rasterio.open(reference_map_path) as src:
        corine_mask = src.read(1)
    
    # Convert to EuroSAT classes
    eurosat_mask = np.zeros_like(corine_mask, dtype=np.int64)
    for corine_code, eurosat_idx in corine_to_eurosat.items():
        eurosat_mask[corine_mask == corine_code] = eurosat_idx
    
    return eurosat_mask
```

#### Bước 4: Tạo PyTorch Dataset
```python
from torch.utils.data import Dataset

class BigEarthNetDataset(Dataset):
    def __init__(self, metadata_df, data_folders, ref_maps_folder, transform=None):
        self.metadata_df = metadata_df
        self.data_folders = data_folders
        self.ref_maps_folder = ref_maps_folder
        self.transform = transform
    
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        patch_id = self.metadata_df.iloc[idx]['patch_id']
        
        # Load RGB image
        rgb = load_rgb(self._find_patch(patch_id))
        
        # Load mask
        mask = load_mask(self._find_ref_map(patch_id), CORINE_TO_EUROSAT)
        
        if self.transform:
            transformed = self.transform(image=rgb, mask=mask)
            rgb = transformed['image']
            mask = transformed['mask']
        
        return {'image': rgb, 'mask': mask, 'patch_id': patch_id}
```

---

## 4. KIẾN TRÚC VÀ TRAINING STRATEGY

### 4.1. Giai đoạn 1: Pre-train Encoder với EuroSAT

**Mục tiêu**: Học feature representations tốt cho ảnh vệ tinh

**Architecture**:
```
Input (64x64x3)
    ↓
Encoder (ResNet/EfficientNet backbone)
    ↓
Global Average Pooling
    ↓
Fully Connected Layer
    ↓
Output (10 classes)
```

**Training**:
- Loss: Cross-Entropy Loss
- Optimizer: Adam/AdamW
- Learning rate: 1e-3 → 1e-5 (cosine decay)
- Batch size: 32-64
- Epochs: 50-100
- Data augmentation: Random flip, rotation, color jitter

**Output**: Encoder weights được lưu lại để khởi tạo U-Net

### 4.2. Giai đoạn 2: Train U-Net với BigEarthNet

**Mục tiêu**: Fine-tune toàn bộ U-Net cho semantic segmentation

**Architecture**:
```
Input (120x120x3)
    ↓
Encoder (Pre-trained từ Giai đoạn 1)
    ├── Block 1 → Skip connection 1
    ├── Block 2 → Skip connection 2
    ├── Block 3 → Skip connection 3
    └── Block 4 → Skip connection 4
    ↓
Bottleneck
    ↓
Decoder
    ├── UpBlock 1 ← Skip connection 4
    ├── UpBlock 2 ← Skip connection 3
    ├── UpBlock 3 ← Skip connection 2
    └── UpBlock 4 ← Skip connection 1
    ↓
Output Conv (1x1)
    ↓
Output (120x120x10)
```

**Training**:
- Loss: Cross-Entropy + Dice Loss (combined)
- Optimizer: Adam/AdamW
- Learning rate:
  - Encoder: 1e-5 (frozen ban đầu, sau đó fine-tune)
  - Decoder: 1e-3
- Batch size: 8-16 (do memory constraints)
- Epochs: 30-50
- Data augmentation: Random flip, rotation, elastic transform

**Training Strategy**:
1. **Phase 2.1**: Freeze encoder, train decoder only (5-10 epochs)
2. **Phase 2.2**: Unfreeze encoder, fine-tune toàn bộ model

---

## 5. ĐÁNH GIÁ MODEL

### 5.1. Metrics

**Per-class metrics**:
- Precision, Recall, F1-score cho từng class
- IoU (Intersection over Union) cho từng class

**Overall metrics**:
- Mean IoU (mIoU)
- Pixel Accuracy
- Mean F1-score

**Class-weighted metrics** (do imbalanced classes):
- Weighted mIoU
- Weighted F1-score

### 5.2. Visualization

- Confusion matrix
- Segmentation predictions overlay trên ảnh gốc
- Per-class performance analysis
- Error analysis (misclassified regions)

---

## 6. IMPLEMENTATION CHECKLIST

### Phase 1: Data Preparation
- [ ] Extract Reference_Maps.tar.zst
- [ ] Load và explore metadata
- [ ] Implement CORINE → EuroSAT mapping
- [ ] Create BigEarthNetDataset class
- [ ] Verify data loading với visualization
- [ ] Setup data augmentation pipeline

### Phase 2: EuroSAT Pre-training
- [ ] Load EuroSAT dataset
- [ ] Implement encoder classification model
- [ ] Train encoder với EuroSAT
- [ ] Validate và save encoder weights
- [ ] Visualize learned features

### Phase 3: U-Net Training
- [ ] Implement U-Net architecture
- [ ] Load pre-trained encoder weights
- [ ] Implement combined loss function
- [ ] Train decoder (encoder frozen)
- [ ] Fine-tune toàn bộ U-Net
- [ ] Monitor validation metrics

### Phase 4: Evaluation
- [ ] Evaluate trên test set
- [ ] Calculate all metrics
- [ ] Generate visualization
- [ ] Error analysis
- [ ] Write final report

---

## 7. EXPECTED CHALLENGES & SOLUTIONS

### Challenge 1: Class Imbalance
**Problem**: Một số classes chiếm diện tích nhỏ (VD: Highway, Industrial)
**Solutions**:
- Weighted loss function
- Focal loss
- Oversampling minority classes
- Class-balanced batch sampling

### Challenge 2: Memory Constraints
**Problem**: BigEarthNet có kích thước lớn (120x120), batch size nhỏ
**Solutions**:
- Gradient accumulation
- Mixed precision training (FP16)
- Smaller backbone architecture
- Patch-based training

### Challenge 3: Cloud Cover & Noise
**Problem**: Một số patches có cloud cover, ảnh hưởng chất lượng
**Solutions**:
- Filter corrupted samples based on metadata
- Data augmentation để robust hơn
- Ensemble predictions

### Challenge 4: Mapping Ambiguity
**Problem**: Một số CORINE classes khó map sang EuroSAT
**Solutions**:
- Analyze class distribution trong data
- Điều chỉnh mapping dựa trên validation performance
- Có thể merge một số classes gần nhau

---

## 8. EXPECTED RESULTS

### Baseline Performance (Random initialization)
- mIoU: ~30-40%
- Pixel Accuracy: ~50-60%

### With Pre-trained Encoder
- mIoU: ~50-65%
- Pixel Accuracy: ~70-80%
- Training time: Giảm 30-50%

### Per-class Expected Performance
- **High performance** (>70% IoU): Forest, AnnualCrop, Residential
- **Medium performance** (50-70% IoU): Pasture, PermanentCrop, HerbaceousVegetation
- **Low performance** (<50% IoU): Highway, Industrial, River (do ít data)

---

## 9. REFERENCES

### Datasets
- **EuroSAT**: https://github.com/phelber/eurosat
- **BigEarthNet**: https://bigearth.net/
- **CORINE Land Cover**: https://land.copernicus.eu/

### Papers
- U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- BigEarthNet: Sumbul et al., "BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding"
- EuroSAT: Helber et al., "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification"

### Technical Resources
- Sentinel-2 Bands: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions
- PyTorch Segmentation: https://github.com/qubvel/segmentation_models.pytorch