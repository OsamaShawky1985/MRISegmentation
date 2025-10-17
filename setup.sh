#!/bin/bash

# YOLOv10 + TransUNet Brain Tumor Detection Pipeline Setup Script
# This script sets up the environment and dependencies

echo "🧠 Setting up YOLOv10 + TransUNet Brain Tumor Detection Pipeline"
echo "=================================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
echo "📋 Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✅ Python $PYTHON_VERSION found"
else
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Creating one..."
    python3 -m venv venv
    echo "🔄 Please activate the virtual environment:"
    echo "   source venv/bin/activate  # On Linux/Mac"
    echo "   venv\\Scripts\\activate     # On Windows"
    echo "Then run this script again."
    exit 0
fi

# Install/upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (with CUDA support if available)
echo "🔥 Installing PyTorch..."
if command_exists nvidia-smi; then
    echo "🚀 NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/raw_images
mkdir -p data/processed_data
mkdir -p data/results
mkdir -p results/training
mkdir -p results/inference

# Download YOLOv10 model (this will be done automatically on first use)
echo "🤖 YOLOv10 models will be downloaded automatically on first use."

# Test installations
echo "🧪 Testing installations..."

echo "Testing PyTorch..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"

echo "Testing OpenCV..."
python3 -c "import cv2; print(f'OpenCV {cv2.__version__} installed successfully')"

echo "Testing Ultralytics (YOLOv10)..."
python3 -c "from ultralytics import YOLO; print('Ultralytics installed successfully')"

echo "Testing other dependencies..."
python3 -c "import numpy, sklearn, matplotlib, yaml, tqdm; print('Other dependencies installed successfully')"

# Test model initialization
echo "🔬 Testing model initialization..."
cd src
python3 -c "
try:
    from models.TransUNet import TransUNet
    model = TransUNet(img_size=224, in_chans=1, num_classes=2)
    print('✅ TransUNet model initialized successfully')
except Exception as e:
    print(f'❌ TransUNet initialization failed: {e}')

try:
    from models.YOLOv10Detector import YOLOv10Detector
    config = {'yolo': {'model_size': 'yolov10n', 'confidence_threshold': 0.25, 'iou_threshold': 0.45, 'max_detections': 100, 'num_classes': 1, 'epochs': 100, 'batch_size': 16, 'learning_rate': 0.001, 'weight_decay': 0.0005}, 'data': {'input_size': [640, 640]}, 'training': {'early_stopping_patience': 20, 'mixed_precision': True}}
    detector = YOLOv10Detector(config)
    print('✅ YOLOv10Detector initialized successfully')
except Exception as e:
    print(f'❌ YOLOv10Detector initialization failed: {e}')
"
cd ..

# Run example demonstration
echo "🚀 Running example demonstration..."
python3 src/example_usage.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. Prepare your brain MRI data in the data/ directory"
echo "2. Use data preparation utilities to format your data:"
echo "   python src/utils/data_preparation.py --help"
echo "3. Train the models:"
echo "   python src/main_yolo_transunet.py train --help"
echo "4. Run inference:"
echo "   python src/main_yolo_transunet.py inference --help"
echo ""
echo "📖 For detailed instructions, see README_YOLO_TransUNet.md"
echo ""
echo "🔧 Configuration files:"
echo "• configs/yolo_transunet_config.yaml - Main pipeline configuration"
echo "• requirements.txt - Python dependencies"
echo ""
echo "🆘 If you encounter issues:"
echo "• Check the troubleshooting section in the README"
echo "• Ensure your GPU drivers are up to date for CUDA support"
echo "• Verify your data format matches the expected structure"
