# Brain Tumor Datasets for Testing YOLOv10 + TransUNet Pipeline

## 🎯 **FOR RESEARCH COMPARISON: See [BENCHMARK_DATASETS_FOR_RESEARCH.md](BENCHMARK_DATASETS_FOR_RESEARCH.md)**

**📊 For research publication and comparison with existing literature, use these standard benchmarks:**
- **BraTS 2021/2023**: Primary benchmark (1,000+ citations)
- **TCGA-LGG**: Clinical validation (2,000+ papers)
- **Standard metrics**: Dice, mAP, Hausdorff Distance

---

## 🧠 Available Brain Tumor Datasets

### 1. **Brain Tumor Segmentation (BraTS) Dataset** ⭐ **RESEARCH STANDARD**
- **Research Importance**: **MOST CITED** brain tumor dataset in literature
- **Citations**: 10,000+ research papers use BraTS data
- **Standard Metrics**: Dice, Hausdorff Distance, Sensitivity, Specificity
- **Comparison**: Direct performance comparison with 100+ published methods
- **Description**: The most comprehensive brain tumor dataset
- **Size**: 1,251 cases (2020 dataset)
- **Modalities**: T1, T1ce, T2, FLAIR MRI sequences
- **Annotations**: Pixel-level segmentation masks
- **Format**: NIfTI (.nii.gz)
- **Classes**: Enhancing tumor, tumor core, whole tumor
- **Download**: https://www.med.upenn.edu/cbica/brats2020/data.html

#### BraTS Dataset Structure:
```
BraTS2020/
├── MICCAI_BraTS2020_TrainingData/
│   ├── BraTS20_Training_001/
│   │   ├── BraTS20_Training_001_flair.nii.gz
│   │   ├── BraTS20_Training_001_t1.nii.gz
│   │   ├── BraTS20_Training_001_t1ce.nii.gz
│   │   ├── BraTS20_Training_001_t2.nii.gz
│   │   └── BraTS20_Training_001_seg.nii.gz  # Ground truth
│   └── ...
└── MICCAI_BraTS2020_ValidationData/
```

### 2. **Brain Tumor Classification (MRI) Dataset** 🚀 **EASY START**
- **Description**: Kaggle dataset for classification
- **Size**: 3,264 images
- **Classes**: Glioma, Meningioma, Pituitary, No tumor
- **Format**: JPEG images (already preprocessed)
- **Download**: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

#### Structure:
```
brain-tumor-classification-mri/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

### 3. **Brain Tumor Dataset (Figshare)**
- **Description**: 3,064 T1-weighted CE-MRI images
- **Classes**: Meningioma, Glioma, Pituitary tumor
- **Format**: MATLAB (.mat) files
- **Download**: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

### 4. **LGG Segmentation Dataset** 📊 **RESEARCH BENCHMARK**
- **Description**: Lower Grade Glioma segmentation
- **Research Importance**: **Subset of TCGA-LGG** - widely cited in medical AI
- **Citations**: 2,000+ papers use TCGA-LGG derived data
- **Size**: 110 patients, **3,929 total images**
- **Tumor Distribution**: 1,373 images with tumors (35%), 2,556 normal (65%)
- **Original Source**: TCGA-LGG (The Cancer Genome Atlas)
- **Format**: TIFF images with masks
- **Download**: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
- **Research Value**: Enables comparison with segmentation literature

#### Structure:
```
lgg-mri-segmentation/
├── TCGA_CS_4941_19960909/
│   ├── TCGA_CS_4941_19960909_1.tif    # MRI slice
│   ├── TCGA_CS_4941_19960909_1_mask.tif  # Segmentation mask
│   └── ...
└── ...
```

### 5. **Brain Tumor Detection Dataset (Roboflow)** 🎯 **YOLO READY**
- **Description**: YOLO format annotations ready
- **Size**: ~500 images
- **Format**: JPEG + YOLO annotations
- **Download**: https://universe.roboflow.com/brain-tumor-detection

## 🛠️ Quick Setup Instructions

### Option 1: Use Kaggle Brain Tumor Classification Dataset (Easiest)

1. **Download the dataset**:
```bash
# Install Kaggle API
pip install kaggle

# Configure Kaggle credentials (get from kaggle.com/account)
mkdir -p ~/.kaggle
echo '{"username":"your_username","key":"your_key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
cd /home/osama/PHD/MRISegmentation/MRIClassification/data
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri
unzip brain-tumor-classification-mri.zip
```

2. **Convert to our pipeline format**:
```bash
python src/utils/data_preparation.py prepare_classification_to_detection \
    --input_dir data/brain-tumor-classification-mri/Training \
    --output_dir data/prepared_dataset
```

### Option 2: Use LGG Segmentation Dataset (Best for Segmentation)

1. **Download**:
```bash
cd /home/osama/PHD/MRISegmentation/MRIClassification/data
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
unzip lgg-mri-segmentation.zip
```

2. **Convert to pipeline format**:
```bash
python src/utils/data_preparation.py convert_lgg_dataset \
    --input_dir data/lgg-mri-segmentation \
    --output_dir data/lgg_prepared
```

### Option 3: Use BraTS Dataset (Most Comprehensive)

1. **Register and download** from BraTS website
2. **Convert NIfTI to images**:
```bash
python src/utils/data_preparation.py convert_brats_dataset \
    --input_dir data/BraTS2020 \
    --output_dir data/brats_prepared \
    --modality t1ce  # or t1, t2, flair
```

## 📁 Data Preparation Scripts

Let me create specific conversion scripts for these datasets:

### Script 1: Classification to Detection Converter
```python
# src/utils/classification_to_detection.py
def convert_classification_to_detection(input_dir, output_dir):
    """Convert classification dataset to detection format"""
    # Create bounding boxes around entire brain region
    # Useful for initial testing
```

### Script 2: LGG Dataset Converter
```python
# src/utils/lgg_converter.py
def convert_lgg_to_pipeline_format(input_dir, output_dir):
    """Convert LGG TIFF images and masks to our format"""
    # Convert TIFF to JPEG/PNG
    # Create YOLO annotations from masks
```

### Script 3: BraTS Dataset Converter
```python
# src/utils/brats_converter.py
def convert_brats_to_2d(input_dir, output_dir, modality='t1ce'):
    """Convert BraTS 3D NIfTI to 2D slices"""
    # Extract 2D slices from 3D volumes
    # Convert segmentation masks
```

## 🚀 Quick Test Dataset (Synthetic)

If you want to test immediately, I can create a synthetic dataset:

```bash
python src/utils/create_synthetic_dataset.py \
    --num_images 100 \
    --output_dir data/synthetic_brain_tumors \
    --include_masks
```

## 📊 Recommended Testing Strategy

### Phase 1: Quick Validation (30 minutes)
1. **Use synthetic dataset** to verify pipeline works
2. **Test all components** (YOLO + TransUNet)
3. **Check output formats** and visualizations

### Phase 2: Real Data Testing (2-3 hours)
1. **Download Kaggle classification dataset** (easiest)
2. **Convert to detection format**
3. **Train on small subset** (50 images)
4. **Evaluate performance**

### Phase 3: Comprehensive Testing (1-2 days)
1. **Download LGG segmentation dataset**
2. **Full pipeline training** (detection + segmentation)
3. **Compare with BraTS dataset** if needed
4. **Performance benchmarking**

## 🔧 Dataset Conversion Tools

I'll create the conversion utilities you need. Which dataset would you like to start with?

### Quick Commands:

**For immediate testing** (synthetic data):
```bash
cd /home/osama/PHD/MRISegmentation/MRIClassification
python src/utils/create_synthetic_dataset.py --num_images 50
python src/main_yolo_transunet.py train --data_dir data/synthetic
```

**For Kaggle dataset**:
```bash
# Download and convert
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri
python src/utils/convert_kaggle_dataset.py
python src/main_yolo_transunet.py train --data_dir data/kaggle_converted
```

**For LGG dataset**:
```bash
# Download and convert
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
python src/utils/convert_lgg_dataset.py
python src/main_yolo_transunet.py train --data_dir data/lgg_converted
```

## 🎯 Which Dataset Should You Choose?

### For Quick Testing:
- **Synthetic Dataset** - Immediate testing
- **Kaggle Classification** - Real data, easy setup

### For Serious Development:
- **LGG Segmentation** - Good balance of size and quality
- **BraTS** - Most comprehensive, research standard

### For Production:
- **BraTS** - Industry standard
- **Custom collected data** - Domain specific

Let me know which dataset you'd like to start with, and I'll create the specific conversion scripts for it!
