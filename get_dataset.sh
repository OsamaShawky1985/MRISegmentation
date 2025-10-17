#!/bin/bash

# Quick Dataset Setup Script for YOLOv10 + TransUNet Pipeline
# This script helps you quickly get a dataset for testing

echo "🧠 Brain Tumor Dataset Setup for YOLOv10 + TransUNet Pipeline"
echo "============================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

# Create data directory
mkdir -p "$DATA_DIR"

echo "📋 Available dataset options:"
echo "1. Synthetic dataset (immediate testing)"
echo "2. Kaggle Brain Tumor Classification (real data, easy setup)"
echo "3. LGG Segmentation Dataset (good for segmentation)"
echo "4. Manual dataset setup instructions"

read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo "🎨 Creating synthetic brain tumor dataset..."
        cd "$PROJECT_ROOT"
        python src/utils/create_synthetic_dataset.py \
            --num_images 100 \
            --output_dir data/synthetic_brain_tumors \
            --include_masks

        echo "✅ Synthetic dataset created!"
        echo "📁 Location: data/synthetic_brain_tumors"
        echo ""
        echo "🚀 To train with synthetic data:"
        echo "python src/main_yolo_transunet.py train \\"
        echo "    --config configs/yolo_transunet_config.yaml \\"
        echo "    --data_dir data/synthetic_brain_tumors/yolo_dataset \\"
        echo "    --output_dir results/synthetic_training"
        ;;
        
    2)
        echo "📥 Setting up Kaggle Brain Tumor Classification dataset..."
        
        # Check if kaggle is installed
        if ! command_exists kaggle; then
            echo "📦 Installing Kaggle API..."
            pip install kaggle
        fi
        
        # Check for Kaggle credentials
        if [ ! -f ~/.kaggle/kaggle.json ]; then
            echo "🔑 Kaggle API credentials not found."
            echo "Please:"
            echo "1. Go to https://www.kaggle.com/account"
            echo "2. Create new API token"
            echo "3. Download kaggle.json"
            echo "4. Run: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/"
            echo "5. Run: chmod 600 ~/.kaggle/kaggle.json"
            echo "6. Then run this script again"
            exit 1
        fi
        
        echo "📥 Downloading Kaggle dataset..."
        cd "$DATA_DIR"
        kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri
        unzip -o brain-tumor-classification-mri.zip
        
        echo "🔄 Converting to pipeline format..."
        cd "$PROJECT_ROOT"
        python src/utils/convert_kaggle_dataset.py \
            --input_dir data/Training \
            --output_dir data/kaggle_converted \
            --target_size 640 640 \
            --seg_size 224 224
        
        echo "✅ Kaggle dataset converted!"
        echo "📁 Location: data/kaggle_converted"
        echo ""
        echo "🚀 To train with Kaggle data:"
        echo "python src/main_yolo_transunet.py train \\"
        echo "    --config configs/yolo_transunet_config.yaml \\"
        echo "    --data_dir data/kaggle_converted/yolo_dataset \\"
        echo "    --output_dir results/kaggle_training"
        ;;
        
    3)
        echo "📥 Setting up LGG Segmentation dataset..."
        echo "ℹ️  Note: This dataset is a processed subset of TCGA-LGG"
        echo "   (The Cancer Genome Atlas - Lower Grade Glioma)"
        
        # Check if kaggle is installed
        if ! command_exists kaggle; then
            echo "📦 Installing Kaggle API..."
            pip install kaggle
        fi
        
        # Check for Kaggle credentials
        if [ ! -f ~/.kaggle/kaggle.json ]; then
            echo "🔑 Kaggle API credentials needed (see option 2 instructions)"
            exit 1
        fi
        
        echo "📥 Downloading LGG dataset..."
        cd "$DATA_DIR"
        kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
        unzip -o lgg-mri-segmentation.zip
        
        echo "🔄 Converting to pipeline format..."
        cd "$PROJECT_ROOT"
        python src/utils/convert_lgg_dataset.py \
            --input_dir data/lgg-mri-segmentation \
            --output_dir data/lgg_converted
        
        echo "✅ LGG dataset converted!"
        echo "📁 Location: data/lgg_converted"
        echo ""
        echo "🚀 To train with LGG data:"
        echo "python src/main_yolo_transunet.py train \\"
        echo "    --config configs/yolo_transunet_config.yaml \\"
        echo "    --data_dir data/lgg_converted/yolo_dataset \\"
        echo "    --output_dir results/lgg_training"
        ;;
        
    4)
        echo "📖 Manual Dataset Setup Instructions"
        echo "===================================="
        echo ""
        echo "🏥 For Medical Datasets:"
        echo "1. BraTS Challenge: https://www.med.upenn.edu/cbica/brats2020/"
        echo "2. TCIA Collections: https://www.cancerimagingarchive.net/"
        echo "3. OpenNeuro: https://openneuro.org/"
        echo ""
        echo "📊 Public Datasets:"
        echo "1. Kaggle Brain Tumor: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri"
        echo "2. LGG Segmentation: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation"
        echo "3. Brain Tumor Dataset: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427"
        echo ""
        echo "📁 Expected Dataset Structure:"
        echo "your_dataset/"
        echo "├── images/"
        echo "│   ├── image001.jpg"
        echo "│   └── ..."
        echo "└── masks/"
        echo "    ├── image001.png"
        echo "    └── ..."
        echo ""
        echo "🔧 Data Preparation Commands:"
        echo ""
        echo "# Convert masks to YOLO format:"
        echo "python src/utils/data_preparation.py masks_to_yolo \\"
        echo "    --images_dir /path/to/images \\"
        echo "    --masks_dir /path/to/masks \\"
        echo "    --output_dir /path/to/yolo_annotations"
        echo ""
        echo "# Create YOLO dataset structure:"
        echo "python src/utils/data_preparation.py create_yolo_dataset \\"
        echo "    --images_dir /path/to/images \\"
        echo "    --annotations_dir /path/to/yolo_annotations \\"
        echo "    --output_dir /path/to/yolo_dataset"
        echo ""
        echo "# Prepare segmentation dataset:"
        echo "python src/utils/data_preparation.py prepare_segmentation \\"
        echo "    --images_dir /path/to/images \\"
        echo "    --masks_dir /path/to/masks \\"
        echo "    --output_dir /path/to/segmentation_dataset"
        ;;
        
    *)
        echo "❌ Invalid option. Please choose 1-4."
        exit 1
        ;;
esac

echo ""
echo "🎯 Next Steps:"
echo "1. Check the generated dataset in the data/ directory"
echo "2. Review the configuration in configs/yolo_transunet_config.yaml"
echo "3. Start training with the provided command"
echo "4. Monitor training progress in the output directory"
echo ""
echo "📖 For detailed instructions, see:"
echo "• README_YOLO_TransUNet.md"
echo "• docs/DATASETS_FOR_TESTING.md"
echo ""
echo "🎉 Dataset setup completed! Happy training! 🧠🔬"
