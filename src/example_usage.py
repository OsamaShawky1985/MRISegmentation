#!/usr/bin/env python3
"""
Example usage of YOLOv10 + TransUNet Brain Tumor Detection and Segmentation Pipeline

This script demonstrates how to:
1. Initialize the pipeline
2. Load or train models
3. Run inference on sample images
4. Visualize results

Author: Your Name
Date: 2025
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Import our pipeline
from YOLOTransUNetPipeline import YOLOTransUNetPipeline


def setup_example_logging():
    """Setup logging for the example"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_example_config():
    """Load configuration for the example"""
    config = {
        'data': {
            'input_images_path': 'data/raw_images',
            'output_path': 'data/results',
            'input_size': [640, 640],
            'segmentation_size': [224, 224],
            'val_split': 0.2
        },
        'yolo': {
            'model_size': 'yolov10n',
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45,
            'max_detections': 100,
            'num_classes': 1,
            'epochs': 50,  # Reduced for example
            'batch_size': 8,   # Reduced for example
            'learning_rate': 0.001,
            'weight_decay': 0.0005
        },
        'transunet': {
            'img_size': 224,
            'patch_size': 16,
            'num_classes': 2,
            'embed_dim': 384,  # Reduced for example
            'depth': 6,        # Reduced for example
            'num_heads': 6,    # Reduced for example
            'mlp_ratio': 4,
            'qkv_bias': True,
            'drop_rate': 0.1,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.1,
            'epochs': 50,      # Reduced for example
            'batch_size': 4,   # Reduced for example
            'learning_rate': 0.0001,
            'weight_decay': 0.01
        },
        'preprocessing': {
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': [8, 8],
            'denoise_h': 10,
            'edge_sigma': 2.0,
            'normalize': True,
            'resize_method': 'bilinear'
        },
        'augmentation': {
            'rotation_range': 15,
            'shift_range': 0.1,
            'zoom_range': 0.1,
            'horizontal_flip': True,
            'vertical_flip': False,
            'brightness_range': [0.8, 1.2],
            'contrast_range': [0.8, 1.2],
            'noise_sigma': 0.02
        },
        'training': {
            'device': 'cuda',
            'num_workers': 2,  # Reduced for example
            'pin_memory': True,
            'mixed_precision': True,
            'gradient_clipping': 1.0,
            'early_stopping_patience': 10,
            'save_best_only': True
        },
        'evaluation': {
            'metrics': ['precision', 'recall', 'f1', 'iou', 'dice'],
            'save_predictions': True,
            'visualize_results': True
        }
    }
    return config


def create_synthetic_brain_image(size=(640, 640)):
    """
    Create a synthetic brain MRI image for demonstration
    
    Args:
        size: Image size (width, height)
    
    Returns:
        Synthetic brain image
    """
    # Create a circular brain-like structure
    image = np.zeros(size, dtype=np.uint8)
    center = (size[0] // 2, size[1] // 2)
    
    # Main brain area
    cv2.circle(image, center, size[0] // 3, 120, -1)
    
    # Add some brain-like texture
    for i in range(50):
        x = np.random.randint(center[0] - size[0] // 4, center[0] + size[0] // 4)
        y = np.random.randint(center[1] - size[1] // 4, center[1] + size[1] // 4)
        radius = np.random.randint(5, 20)
        intensity = np.random.randint(80, 160)
        cv2.circle(image, (x, y), radius, intensity, -1)
    
    # Add a tumor-like bright spot
    tumor_x = center[0] + size[0] // 6
    tumor_y = center[1] - size[1] // 8
    cv2.circle(image, (tumor_x, tumor_y), 25, 255, -1)
    cv2.circle(image, (tumor_x, tumor_y), 15, 200, -1)
    
    # Add noise
    noise = np.random.normal(0, 10, size).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


def create_synthetic_mask(size=(224, 224)):
    """
    Create a synthetic tumor mask for demonstration
    
    Args:
        size: Mask size (width, height)
    
    Returns:
        Binary tumor mask
    """
    mask = np.zeros(size, dtype=np.uint8)
    center = (size[0] // 2 + size[0] // 6, size[1] // 2 - size[1] // 8)
    
    # Scale tumor size to match the resized image
    tumor_radius = int(25 * size[0] / 640)  # Scale based on original size
    cv2.circle(mask, center, tumor_radius, 1, -1)
    
    return mask


def demonstrate_pipeline():
    """Main demonstration function"""
    logger = logging.getLogger(__name__)
    logger.info("Starting YOLOv10 + TransUNet Pipeline Demonstration")
    
    # Setup directories
    output_dir = Path('example_results')
    output_dir.mkdir(exist_ok=True)
    
    # Load configuration
    config = load_example_config()
    config['data']['output_path'] = str(output_dir)
    
    try:
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = YOLOTransUNetPipeline(config)
        
        # Create synthetic data for demonstration
        logger.info("Creating synthetic brain image for demonstration...")
        synthetic_image = create_synthetic_brain_image((640, 640))
        synthetic_mask = create_synthetic_mask((224, 224))
        
        # Save synthetic image
        image_path = output_dir / 'synthetic_brain.jpg'
        cv2.imwrite(str(image_path), synthetic_image)
        
        # Save synthetic mask for reference
        mask_path = output_dir / 'synthetic_mask.png'
        cv2.imwrite(str(mask_path), synthetic_mask * 255)
        
        logger.info(f"Synthetic data saved to {output_dir}")
        
        # Load pretrained YOLO model (this will download if not available)
        logger.info("Loading YOLO model...")
        pipeline.yolo_detector.load_model(pretrained=True)
        
        # Note: For a real demonstration, you would train the models:
        # 1. Train YOLO: pipeline.train_yolo(train_data_path, val_data_path)
        # 2. Train TransUNet: pipeline.train_transunet(train_images, train_masks, val_images, val_masks)
        
        # Run inference on synthetic image
        logger.info("Running inference on synthetic image...")
        results = pipeline.predict(str(image_path), save_results=True)
        
        # Display results
        logger.info("Inference Results:")
        logger.info(f"Number of detections: {len(results['detections'])}")
        
        for i, detection in enumerate(results['detections']):
            logger.info(f"  Detection {i+1}:")
            logger.info(f"    Confidence: {detection['confidence']:.3f}")
            logger.info(f"    Bounding box: {detection['bbox']}")
            logger.info(f"    Area: {detection['area']} pixels")
        
        # Create visualization
        create_results_visualization(results, output_dir)
        
        logger.info(f"Results saved to {output_dir}")
        logger.info("Demonstration completed successfully!")
        
        # Print usage instructions
        print_usage_instructions()
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        logger.info("This is expected if PyTorch/dependencies are not installed.")
        logger.info("Please install requirements: pip install -r requirements.txt")


def create_results_visualization(results, output_dir):
    """Create a visualization of the results"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        original = results['original_image']
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Detection results
        detection_vis = original.copy()
        for detection in results['detections']:
            bbox = detection['bbox']
            confidence = detection['confidence']
            # Note: matplotlib uses RGB, opencv uses BGR
            from matplotlib.patches import Rectangle
            rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                           linewidth=2, edgecolor='red', facecolor='none')
            axes[1].add_patch(rect)
            axes[1].text(bbox[0], bbox[1]-10, f'Tumor: {confidence:.2f}', 
                        color='red', fontsize=10, weight='bold')
        
        axes[1].imshow(detection_vis, cmap='gray')
        axes[1].set_title('YOLO Detections')
        axes[1].axis('off')
        
        # Segmentation results
        if results['segmentations']:
            seg_vis = original.copy()
            for seg_result in results['segmentations']:
                mask = seg_result['segmentation_mask']
                bbox = seg_result['bbox']
                
                # Create colored mask overlay
                mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                mask_colored[mask == 1] = [255, 0, 0]  # Red for tumor
                
                # This is a simplified overlay - in practice you'd need proper scaling
                axes[2].imshow(mask_colored, alpha=0.5)
        
        axes[2].imshow(original, cmap='gray')
        axes[2].set_title('TransUNet Segmentation')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'results_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("Matplotlib not available for visualization")
    except Exception as e:
        print(f"Error creating visualization: {e}")


def print_usage_instructions():
    """Print usage instructions for the pipeline"""
    print("\n" + "="*60)
    print("🧠 YOLOv10 + TransUNet Pipeline Usage Instructions")
    print("="*60)
    
    print("\n📋 Quick Start:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Prepare your data using data_preparation.py")
    print("3. Train models: python main_yolo_transunet.py train")
    print("4. Run inference: python main_yolo_transunet.py inference")
    
    print("\n📊 Data Preparation:")
    print("python src/utils/data_preparation.py masks_to_yolo \\")
    print("    --images_dir /path/to/images \\")
    print("    --masks_dir /path/to/masks \\")
    print("    --output_dir /path/to/yolo_annotations")
    
    print("\n🏃 Training:")
    print("python src/main_yolo_transunet.py train \\")
    print("    --config configs/yolo_transunet_config.yaml \\")
    print("    --data_dir /path/to/training/data \\")
    print("    --output_dir results/training")
    
    print("\n🔍 Inference:")
    print("python src/main_yolo_transunet.py inference \\")
    print("    --config configs/yolo_transunet_config.yaml \\")
    print("    --model_dir results/training \\")
    print("    --image_path /path/to/test/image.jpg \\")
    print("    --output_dir results/inference")
    
    print("\n📈 Key Features:")
    print("• YOLOv10 for fast and accurate tumor detection")
    print("• TransUNet for precise pixel-level segmentation")
    print("• Comprehensive data augmentation")
    print("• Multiple evaluation metrics")
    print("• Batch processing support")
    print("• Rich visualization of results")
    
    print("\n🔧 Configuration:")
    print("Edit configs/yolo_transunet_config.yaml to customize:")
    print("• Model architectures and hyperparameters")
    print("• Data augmentation settings")
    print("• Training parameters")
    print("• Evaluation metrics")
    
    print("\n📁 Expected Data Structure:")
    print("your_dataset/")
    print("├── images/")
    print("│   ├── brain_001.jpg")
    print("│   └── brain_002.jpg")
    print("└── masks/")
    print("    ├── brain_001.png")
    print("    └── brain_002.png")
    
    print("\n🆘 Support:")
    print("• Check README_YOLO_TransUNet.md for detailed documentation")
    print("• Review troubleshooting section for common issues")
    print("• Ensure CUDA is available for GPU acceleration")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    setup_example_logging()
    demonstrate_pipeline()
