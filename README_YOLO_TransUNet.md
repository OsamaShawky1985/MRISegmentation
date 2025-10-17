# Brain Tumor Detection and Segmentation Pipeline

A comprehensive deep learning pipeline that combines **YOLOv10** for brain tumor detection/localization and **TransUNet** for precise segmentation. This pipeline provides end-to-end functionality for training, inference, and evaluation on brain MRI images.

## 🏗️ Architecture Overview

```
Input MRI Image
       ↓
   YOLOv10 Detector
   (Detection & Localization)
       ↓
   Tumor Regions Extracted
       ↓
   TransUNet Segmentation
   (Pixel-level Segmentation)
       ↓
   Final Result with Masks
```

### Key Components

1. **YOLOv10 Detector**: State-of-the-art object detection for tumor localization
2. **TransUNet**: Vision Transformer + U-Net hybrid for precise segmentation
3. **Integrated Pipeline**: Seamless integration of detection and segmentation
4. **Data Preparation**: Comprehensive utilities for dataset preparation

## 🚀 Features

- **Dual-Stage Approach**: Detection followed by segmentation for improved accuracy
- **Modern Architectures**: YOLOv10 + TransUNet (Vision Transformer + CNN)
- **Comprehensive Training**: Separate training pipelines for detection and segmentation
- **Data Augmentation**: Advanced augmentation techniques for robust training
- **Evaluation Metrics**: Comprehensive evaluation with multiple metrics
- **Visualization**: Rich visualization of results and training progress
- **Batch Processing**: Support for batch inference on multiple images
- **Flexible Configuration**: YAML-based configuration system

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM (16GB+ recommended)
- 4GB+ GPU memory (for training)

### Python Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.0+
- OpenCV 4.8+
- Ultralytics (YOLOv10)
- Albumentations
- NumPy, scikit-learn, matplotlib

## 📁 Project Structure

```
MRIClassification/
├── configs/
│   ├── config.yaml                      # Original config
│   └── yolo_transunet_config.yaml       # Pipeline config
├── src/
│   ├── models/
│   │   ├── YOLOv10Detector.py           # YOLO detection model
│   │   └── TransUNet.py                 # TransUNet segmentation model
│   ├── utils/
│   │   └── data_preparation.py          # Data preparation utilities
│   ├── YOLOTransUNetPipeline.py         # Main pipeline class
│   └── main_yolo_transunet.py           # Main execution script
├── data/
│   ├── raw_images/                      # Original MRI images
│   ├── processed_data/                  # Processed images
│   └── results/                         # Output results
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd MRIClassification
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python src/models/TransUNet.py
   python src/models/YOLOv10Detector.py
   ```

## 📊 Data Preparation

### 1. Dataset Structure

Organize your data as follows:
```
your_dataset/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── masks/
    ├── image001.png    # Segmentation mask
    ├── image002.png
    └── ...
```

### 2. Data Preparation Commands

**Convert masks to YOLO detection format**:
```bash
python src/utils/data_preparation.py masks_to_yolo \
    --images_dir /path/to/images \
    --masks_dir /path/to/masks \
    --output_dir /path/to/yolo_annotations
```

**Create YOLO dataset structure**:
```bash
python src/utils/data_preparation.py create_yolo_dataset \
    --images_dir /path/to/images \
    --annotations_dir /path/to/yolo_annotations \
    --output_dir /path/to/yolo_dataset
```

**Prepare segmentation dataset**:
```bash
python src/utils/data_preparation.py prepare_segmentation \
    --images_dir /path/to/images \
    --masks_dir /path/to/masks \
    --output_dir /path/to/segmentation_dataset
```

**Augment dataset**:
```bash
python src/utils/data_preparation.py augment \
    --images_dir /path/to/images \
    --masks_dir /path/to/masks \
    --output_dir /path/to/augmented_data
```

## 🏃 Usage

### Configuration

Edit `configs/yolo_transunet_config.yaml` to customize:

```yaml
# Key configuration sections
data:
  input_images_path: "data/raw_images"
  output_path: "data/results"
  input_size: [640, 640]        # YOLO input size
  segmentation_size: [224, 224] # TransUNet input size

yolo:
  model_size: "yolov10n"        # yolov10n, yolov10s, yolov10m, yolov10l, yolov10x
  confidence_threshold: 0.25
  epochs: 100
  batch_size: 16

transunet:
  img_size: 224
  patch_size: 16
  num_classes: 2                # Background + tumor
  epochs: 200
  batch_size: 8
```

### Training

**Train the complete pipeline**:
```bash
python src/main_yolo_transunet.py train \
    --config configs/yolo_transunet_config.yaml \
    --data_dir /path/to/training/data \
    --output_dir results/training
```

### Inference

**Single image inference**:
```bash
python src/main_yolo_transunet.py inference \
    --config configs/yolo_transunet_config.yaml \
    --model_dir results/training \
    --image_path /path/to/test/image.jpg \
    --output_dir results/inference
```

**Batch inference**:
```bash
python src/main_yolo_transunet.py batch_inference \
    --config configs/yolo_transunet_config.yaml \
    --model_dir results/training \
    --input_dir /path/to/test/images \
    --output_dir results/batch_inference
```

### Evaluation

**Evaluate trained models**:
```bash
python src/main_yolo_transunet.py evaluate \
    --config configs/yolo_transunet_config.yaml \
    --model_dir results/training \
    --data_dir /path/to/test/data
```

## 📈 Model Performance

### Metrics Tracked

**Detection (YOLO)**:
- Precision, Recall, F1-score
- mAP@0.5, mAP@0.5:0.95
- Detection accuracy per class

**Segmentation (TransUNet)**:
- Pixel accuracy
- Dice coefficient
- IoU (Intersection over Union)
- Precision, Recall, F1-score

### Training Monitoring

The pipeline automatically saves:
- Training/validation loss curves
- Model checkpoints (best and latest)
- Evaluation metrics
- Visualization of results

## 🔧 Advanced Usage

### Custom Model Configuration

**Modify TransUNet architecture**:
```python
# In configs/yolo_transunet_config.yaml
transunet:
  img_size: 384          # Larger input size
  patch_size: 16         # Patch size for ViT
  embed_dim: 1024        # Larger embedding dimension
  depth: 24              # More transformer layers
  num_heads: 16          # More attention heads
```

**Customize YOLO model**:
```python
# In configs/yolo_transunet_config.yaml
yolo:
  model_size: "yolov10x"  # Larger model
  input_size: [1280, 1280]  # Higher resolution
```

### Data Augmentation

Customize augmentation in the config:
```yaml
augmentation:
  rotation_range: 15
  shift_range: 0.1
  zoom_range: 0.1
  horizontal_flip: true
  brightness_range: [0.8, 1.2]
  contrast_range: [0.8, 1.2]
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size in config
   - Use gradient accumulation
   - Enable mixed precision training

2. **Model not converging**:
   - Check learning rate
   - Verify data quality
   - Increase training epochs

3. **Poor detection results**:
   - Check annotation quality
   - Adjust confidence threshold
   - Increase training data

### Debug Mode

Run with verbose logging:
```bash
python src/main_yolo_transunet.py inference \
    --log_level DEBUG \
    --config configs/yolo_transunet_config.yaml \
    --model_dir results/training \
    --image_path test_image.jpg
```

## 📝 Notes

### Data Requirements

- **Images**: High-quality brain MRI scans
- **Annotations**: 
  - Bounding boxes for detection (YOLO format)
  - Pixel-level masks for segmentation
- **Format**: JPEG/PNG for images, PNG for masks
- **Size**: Minimum 256x256 pixels recommended

### Training Tips

1. **Pre-training**: Use pretrained YOLOv10 weights
2. **Data Balance**: Ensure balanced tumor/no-tumor samples
3. **Validation**: Use proper train/val/test splits
4. **Monitoring**: Watch for overfitting with validation metrics

### Performance Optimization

- Use mixed precision training (`mixed_precision: true`)
- Enable gradient clipping for stable training
- Use appropriate batch sizes for your GPU memory
- Consider multi-GPU training for large datasets

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration options

## 🙏 Acknowledgments

- **YOLOv10**: Ultralytics team for the YOLO implementation
- **TransUNet**: Original authors for the TransUNet architecture
- **PyTorch**: For the deep learning framework
- **OpenCV**: For image processing utilities

---

**Happy Training! 🧠🔬**
