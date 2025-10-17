# Quick Answer: LGG Dataset File Naming

## 🔍 **Your Question: What do `_1.tif` and `1_mask.tif` mean?**

## ✅ **Answer: Sequential Brain Slices + Tumor Masks**

### **File Naming Pattern:**
```
TCGA_CS_4938_19970222_1.tif        ← Brain MRI slice #1
TCGA_CS_4938_19970222_1_mask.tif   ← Tumor mask for slice #1
TCGA_CS_4938_19970222_2.tif        ← Brain MRI slice #2  
TCGA_CS_4938_19970222_2_mask.tif   ← Tumor mask for slice #2
```

### **What Each Part Means:**

| Component | Meaning | Example |
|-----------|---------|---------|
| **TCGA_CS_4938_19970222** | Patient ID from TCGA database | Unique patient |
| **_1, _2, _3** | **Brain slice number** | Sequential slices |
| **.tif** | MRI brain image | Grayscale brain scan |
| **_mask.tif** | Tumor segmentation | Binary mask (black/white) |

## 🧠 **Medical Context**

### **Brain Slicing:**
```
3D Brain Volume
      ↓
Multiple 2D Slices
      ↓  
_1.tif = Bottom slice
_2.tif = Next slice up
_3.tif = Next slice up
...
_30.tif = Top slice
```

### **Key Points:**
- ✅ **Each number = different brain slice** (like CT scan layers)
- ✅ **Sequential numbering** (anatomically adjacent)
- ✅ **Perfect pairs** (every image has corresponding mask)
- ✅ **~35 slices per patient** on average

## 🎯 **For Your Pipeline**

### **Data Processing:**
```python
# Each patient folder contains:
patient_folder/
├── patient_1.tif + patient_1_mask.tif    # Slice 1 pair
├── patient_2.tif + patient_2_mask.tif    # Slice 2 pair
├── patient_3.tif + patient_3_mask.tif    # Slice 3 pair
└── ... (30-40 slice pairs per patient)
```

### **Training Data:**
- **3,929 image-mask pairs** total
- **1,373 pairs with tumors** (masks have white pixels)
- **2,556 pairs without tumors** (masks are all black)

## 📋 **Bottom Line**

The numbers represent **sequential brain slices** from MRI scans, with each slice having its corresponding tumor mask. Perfect for training your YOLOv10 + TransUNet pipeline!

---

**See `docs/LGG_File_Naming_Explained.md` for detailed explanation** 🧠📊
