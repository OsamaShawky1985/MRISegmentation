#!/usr/bin/env python3
"""
Brain Tumor Detection and Segmentation Pipeline
Using YOLOv10 for detection/localization and TransUNet for segmentation

Author: Your Name
Date: 2025
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import cv2

from YOLOTransUNetPipeline import YOLOTransUNetPipeline


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('brain_tumor_pipeline.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> Dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_image_paths(data_dir: str, extensions: List[str] = None) -> List[str]:
    """Get all image paths from directory"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    
    data_path = Path(data_dir)
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(list(data_path.rglob(f'*{ext}')))
        image_paths.extend(list(data_path.rglob(f'*{ext.upper()}')))
    
    return [str(p) for p in image_paths]


def prepare_data_splits(image_paths: List[str], val_split: float = 0.2, 
                       test_split: float = 0.1) -> Dict[str, List[str]]:
    """Split data into train/val/test sets"""
    np.random.seed(42)
    indices = np.random.permutation(len(image_paths))
    
    n_test = int(len(image_paths) * test_split)
    n_val = int(len(image_paths) * val_split)
    n_train = len(image_paths) - n_test - n_val
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return {
        'train': [image_paths[i] for i in train_indices],
        'val': [image_paths[i] for i in val_indices],
        'test': [image_paths[i] for i in test_indices]
    }


def train_pipeline(config_path: str, data_dir: str, output_dir: str):
    """Train the complete pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("Starting pipeline training...")
    
    # Load configuration
    config = load_config(config_path)
    config['data']['output_path'] = output_dir
    
    # Initialize pipeline
    pipeline = YOLOTransUNetPipeline(config)
    
    # Get image paths
    image_paths = get_image_paths(data_dir)
    if not image_paths:
        raise ValueError(f"No images found in {data_dir}")
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Split data
    data_splits = prepare_data_splits(image_paths, config['data']['val_split'])
    
    logger.info(f"Data splits: Train={len(data_splits['train'])}, "
                f"Val={len(data_splits['val'])}, Test={len(data_splits['test'])}")
    
    # Train YOLO (if you have detection annotations)
    # Note: You'll need to prepare YOLO format annotations
    # pipeline.train_yolo(data_dir)
    
    # Train TransUNet (if you have segmentation masks)
    # Note: You'll need corresponding mask files
    # mask_paths = [p.replace('images', 'masks').replace('.jpg', '.png') for p in image_paths]
    # pipeline.train_transunet(data_splits['train'], mask_paths_train, 
    #                         data_splits['val'], mask_paths_val)
    
    logger.info("Pipeline training completed!")


def inference_pipeline(config_path: str, model_dir: str, image_path: str, 
                      output_dir: str):
    """Run inference on a single image"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running inference on {image_path}")
    
    # Load configuration
    config = load_config(config_path)
    config['data']['output_path'] = output_dir
    
    # Initialize pipeline
    pipeline = YOLOTransUNetPipeline(config)
    
    # Load trained models
    yolo_model_path = Path(model_dir) / 'yolo_best.pt'
    transunet_model_path = Path(model_dir) / 'transunet_best.pth'
    
    if yolo_model_path.exists():
        pipeline.yolo_detector.load_model(str(yolo_model_path))
        logger.info("Loaded YOLO model")
    else:
        logger.warning("YOLO model not found, using pretrained")
        pipeline.yolo_detector.load_model(pretrained=True)
    
    if transunet_model_path.exists():
        import torch
        pipeline.transunet.load_state_dict(
            torch.load(transunet_model_path, map_location=pipeline.device)
        )
        logger.info("Loaded TransUNet model")
    else:
        logger.warning("TransUNet model not found, using randomly initialized weights")
    
    # Run inference
    results = pipeline.predict(image_path, save_results=True)
    
    # Print results summary
    logger.info(f"Detection results: {len(results['detections'])} tumors detected")
    for i, detection in enumerate(results['detections']):
        logger.info(f"  Tumor {i+1}: Confidence={detection['confidence']:.3f}, "
                   f"Area={detection['area']} pixels")
    
    logger.info("Inference completed!")
    return results


def batch_inference(config_path: str, model_dir: str, input_dir: str, 
                   output_dir: str):
    """Run inference on multiple images"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running batch inference on {input_dir}")
    
    # Get all image paths
    image_paths = get_image_paths(input_dir)
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")
    
    logger.info(f"Found {len(image_paths)} images for inference")
    
    # Process each image
    all_results = {}
    for image_path in image_paths:
        try:
            results = inference_pipeline(config_path, model_dir, image_path, output_dir)
            all_results[image_path] = results
            logger.info(f"Processed {Path(image_path).name}")
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue
    
    logger.info(f"Batch inference completed! Processed {len(all_results)} images")
    return all_results


def evaluate_pipeline(config_path: str, model_dir: str, test_data_dir: str):
    """Evaluate the trained pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("Starting pipeline evaluation...")
    
    # Load configuration
    config = load_config(config_path)
    
    # Initialize pipeline
    pipeline = YOLOTransUNetPipeline(config)
    
    # Load trained models
    yolo_model_path = Path(model_dir) / 'yolo_best.pt'
    transunet_model_path = Path(model_dir) / 'transunet_best.pth'
    
    if yolo_model_path.exists():
        pipeline.yolo_detector.load_model(str(yolo_model_path))
    
    if transunet_model_path.exists():
        import torch
        pipeline.transunet.load_state_dict(
            torch.load(transunet_model_path, map_location=pipeline.device)
        )
    
    # Get test images and masks
    test_images = get_image_paths(test_data_dir)
    # Note: You'll need corresponding mask files for evaluation
    # test_masks = [p.replace('images', 'masks').replace('.jpg', '.png') for p in test_images]
    
    # Evaluate segmentation performance
    # metrics = pipeline.evaluate_segmentation(test_images, test_masks)
    
    logger.info("Pipeline evaluation completed!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Brain Tumor Detection and Segmentation Pipeline"
    )
    parser.add_argument(
        'mode', 
        choices=['train', 'inference', 'batch_inference', 'evaluate'],
        help='Pipeline mode'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/yolo_transunet_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data_dir', 
        type=str,
        help='Directory containing training data'
    )
    parser.add_argument(
        '--model_dir', 
        type=str,
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--image_path', 
        type=str,
        help='Path to input image for inference'
    )
    parser.add_argument(
        '--input_dir', 
        type=str,
        help='Directory containing input images for batch inference'
    )
    parser.add_argument(
        '--output_dir', 
        type=str,
        default='results',
        help='Output directory'
    )
    parser.add_argument(
        '--log_level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        if args.mode == 'train':
            if not args.data_dir:
                raise ValueError("--data_dir is required for training")
            train_pipeline(args.config, args.data_dir, args.output_dir)
            
        elif args.mode == 'inference':
            if not args.image_path or not args.model_dir:
                raise ValueError("--image_path and --model_dir are required for inference")
            inference_pipeline(args.config, args.model_dir, args.image_path, args.output_dir)
            
        elif args.mode == 'batch_inference':
            if not args.input_dir or not args.model_dir:
                raise ValueError("--input_dir and --model_dir are required for batch inference")
            batch_inference(args.config, args.model_dir, args.input_dir, args.output_dir)
            
        elif args.mode == 'evaluate':
            if not args.data_dir or not args.model_dir:
                raise ValueError("--data_dir and --model_dir are required for evaluation")
            evaluate_pipeline(args.config, args.model_dir, args.data_dir)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
