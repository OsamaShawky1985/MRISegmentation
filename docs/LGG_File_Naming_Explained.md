# Kaggle LGG Dataset File Naming Convention Explained

## 📁 **File Naming Pattern**

### **Example File Structure:**
```
TCGA_CS_4938_19970222/
├── TCGA_CS_4938_19970222_1.tif        # MRI slice #1
├── TCGA_CS_4938_19970222_1_mask.tif   # Tumor mask for slice #1
├── TCGA_CS_4938_19970222_2.tif        # MRI slice #2
├── TCGA_CS_4938_19970222_2_mask.tif   # Tumor mask for slice #2
├── TCGA_CS_4938_19970222_3.tif        # MRI slice #3
├── TCGA_CS_4938_19970222_3_mask.tif   # Tumor mask for slice #3
└── ... (continues for all slices)
```

## 🔍 **Breaking Down the Names**

### **MRI Image Files: `TCGA_CS_4938_19970222_1.tif`**
```
TCGA_CS_4938_19970222_1.tif
    │      │       │      │
    │      │       │      └── Slice Number (1, 2, 3, ...)
    │      │       └───────── Date (19970222 = Feb 22, 1997)
    │      └───────────────── Patient ID (CS_4938)
    └──────────────────────── TCGA Project Identifier
```

### **Mask Files: `TCGA_CS_4938_19970222_1_mask.tif`**
```
TCGA_CS_4938_19970222_1_mask.tif
    │      │       │   │   │
    │      │       │   │   └── File Extension (.tif)
    │      │       │   └───── Mask Identifier
    │      │       └───────── Slice Number (matches MRI slice)
    │      └───────────────── Patient ID
    └──────────────────────── TCGA Project
```

## 🧠 **What Each Component Means**

### **1. Patient Identifier: `TCGA_CS_4938_19970222`**
- **TCGA**: The Cancer Genome Atlas project
- **CS**: Site code (Clinical Site)
- **4938**: Unique patient number
- **19970222**: Scan date (February 22, 1997)

### **2. Slice Number: `_1`, `_2`, `_3`...**
- **Sequential numbering** of MRI slices from the same patient
- Each number represents a **different brain slice**
- Usually goes from **inferior** (bottom) to **superior** (top) of brain
- Typical range: 1 to 30-40 slices per patient

### **3. File Types:**
- **`.tif`**: Original MRI slice (grayscale brain image)
- **`_mask.tif`**: Corresponding tumor segmentation mask

## 🔬 **Medical Context**

### **Brain MRI Slicing:**
```
Brain (3D Volume)
      ↓
Multiple 2D Slices
      ↓
_1.tif = Bottom slice
_2.tif = Next slice up
_3.tif = Next slice up
...
_N.tif = Top slice
```

### **Each Slice Represents:**
- **Axial view** of the brain (looking from top/bottom)
- **Thickness**: ~5mm per slice typically
- **Coverage**: Complete brain volume
- **Sequence**: From skull base to vertex

## 📊 **Practical Examples**

### **Example 1: Patient with Tumor**
```
TCGA_DU_6404_19850629_12.tif      # MRI slice #12
TCGA_DU_6404_19850629_12_mask.tif # Mask shows tumor in slice #12

TCGA_DU_6404_19850629_13.tif      # MRI slice #13  
TCGA_DU_6404_19850629_13_mask.tif # Mask shows tumor in slice #13
```

### **Example 2: Patient without Tumor**
```
TCGA_DU_7010_19860307_15.tif      # MRI slice #15
TCGA_DU_7010_19860307_15_mask.tif # Mask is empty (all black)
```

## 🎯 **Key Insights**

### **1. Paired Files:**
- Every `.tif` has a corresponding `_mask.tif`
- **Perfect 1:1 correspondence**
- Same pixel dimensions and alignment

### **2. Slice Continuity:**
- Numbers indicate **anatomical sequence**
- Consecutive slices are **spatially adjacent**
- Missing numbers mean slices were excluded during processing

### **3. Tumor Distribution:**
- **Not all slices** contain tumor
- Tumor may appear in slices 8-15 but not in 1-7 or 16-25
- **Mask files show exactly which pixels are tumor**

## 🖼️ **Visual Understanding**

### **MRI Image (`_1.tif`):**
- **Grayscale**: Brain tissue appears in different gray levels
- **Background**: Black (air/skull)
- **Brain tissue**: Various gray intensities
- **Resolution**: Typically 256×256 pixels

### **Mask Image (`1_mask.tif`):**
- **Binary**: Only black (0) and white (255)
- **Black (0)**: Normal tissue/background
- **White (255)**: Tumor tissue
- **Same dimensions** as corresponding MRI slice

## 🛠️ **For Your Pipeline**

### **Processing Strategy:**
```python
# Load image-mask pairs
mri_image = cv2.imread('TCGA_CS_4938_19970222_1.tif', cv2.IMREAD_GRAYSCALE)
tumor_mask = cv2.imread('TCGA_CS_4938_19970222_1_mask.tif', cv2.IMREAD_GRAYSCALE)

# Check if tumor is present
has_tumor = np.max(tumor_mask) > 0  # True if white pixels exist

# Extract bounding box for YOLO
if has_tumor:
    # Find tumor contours for bounding box
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Convert to YOLO format...
```

### **Data Organization:**
```python
# Group files by patient
patient_files = {
    'TCGA_CS_4938_19970222': {
        'images': ['_1.tif', '_2.tif', '_3.tif', ...],
        'masks': ['1_mask.tif', '2_mask.tif', '3_mask.tif', ...]
    }
}
```

## 📋 **Summary**

| Component | Meaning | Example |
|-----------|---------|---------|
| **Patient ID** | Unique identifier from TCGA | `TCGA_CS_4938_19970222` |
| **Slice Number** | Brain slice sequence | `_1`, `_2`, `_3` |
| **Image File** | MRI brain scan | `_1.tif` |
| **Mask File** | Tumor segmentation | `1_mask.tif` |
| **Correspondence** | 1:1 image-mask pairs | Perfect alignment |

**Bottom Line**: The numbers represent **sequential brain slices** from the same patient, with each slice having its corresponding tumor mask. Perfect for training both detection (YOLO) and segmentation (TransUNet) models! 🧠🔬
