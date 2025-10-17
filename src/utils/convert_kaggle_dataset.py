#!/usr/bin/env python3
"""
Kaggle Brain Tumor Classification Dataset Converter
Converts Kaggle brain tumor classification dataset to detection and segmentation format

Dataset: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
Classes: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
import shutil


class KaggleDatasetConverter:
    """Convert Kaggle brain tumor classification dataset to our pipeline format"""
    
    def __init__(self, target_size=(640, 640), seg_size=(224, 224)):
        self.target_size = target_size
        self.seg_size = seg_size
        self.logger = logging.getLogger(__name__)
        
        # Class mapping
        self.class_mapping = {
            'glioma_tumor': 0,
            'meningioma_tumor': 0,
            'pituitary_tumor': 0,
            'no_tumor': -1  # No tumor class, will be excluded from detection
        }
    
    def detect_brain_region(self, image):
        """Detect brain region in the image using simple thresholding"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to separate brain from background
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Return entire image if no contours found
            h, w = gray.shape
            return [0, 0, w, h]
        
        # Find largest contour (brain region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add some padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2 * padding)
        h = min(gray.shape[0] - y, h + 2 * padding)
        
        return [x, y, x + w, y + h]
    
    def create_tumor_mask(self, image, bbox, tumor_type):
        """Create synthetic tumor mask for classification images"""
        h, w = image.shape[:2] if len(image.shape) == 3 else image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if tumor_type == 'no_tumor':
            return mask
        
        x1, y1, x2, y2 = bbox
        brain_center_x = (x1 + x2) // 2
        brain_center_y = (y1 + y2) // 2
        brain_width = x2 - x1
        brain_height = y2 - y1
        
        # Create tumor region based on type
        if tumor_type == 'glioma_tumor':
            # Gliomas are usually larger and irregular
            tumor_size = min(brain_width, brain_height) // 6
            tumor_x = brain_center_x + np.random.randint(-brain_width//8, brain_width//8)
            tumor_y = brain_center_y + np.random.randint(-brain_height//8, brain_height//8)
        elif tumor_type == 'meningioma_tumor':
            # Meningiomas are usually at brain surface
            tumor_size = min(brain_width, brain_height) // 8
            # Place near edge of brain
            edge_offset = brain_width // 6
            tumor_x = brain_center_x + np.random.choice([-edge_offset, edge_offset])
            tumor_y = brain_center_y + np.random.randint(-brain_height//8, brain_height//8)
        else:  # pituitary_tumor
            # Pituitary tumors are smaller and central
            tumor_size = min(brain_width, brain_height) // 10
            tumor_x = brain_center_x
            tumor_y = brain_center_y + brain_height // 8
        
        # Create elliptical tumor shape
        cv2.ellipse(mask, (tumor_x, tumor_y), (tumor_size, tumor_size), 0, 0, 360, 1, -1)
        
        return mask
    
    def convert_to_yolo_format(self, bbox, image_size):
        """Convert bounding box to YOLO format"""
        x1, y1, x2, y2 = bbox
        
        # Calculate center and dimensions
        center_x = (x1 + x2) / 2 / image_size[1]
        center_y = (y1 + y2) / 2 / image_size[0]
        width = (x2 - x1) / image_size[1]
        height = (y2 - y1) / image_size[0]
        
        return f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
    
    def convert_dataset(self, input_dir, output_dir, create_synthetic_masks=True):
        """Convert Kaggle dataset to our pipeline format"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directories
        (output_path / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / 'yolo_annotations').mkdir(parents=True, exist_ok=True)
        if create_synthetic_masks:
            (output_path / 'masks').mkdir(parents=True, exist_ok=True)
        
        dataset_stats = {
            'total_images': 0,
            'tumor_images': 0,
            'no_tumor_images': 0,
            'class_distribution': {},
            'conversion_settings': {
                'target_size': self.target_size,
                'segmentation_size': self.seg_size,
                'synthetic_masks': create_synthetic_masks
            }
        }
        
        # Process each class directory
        for class_dir in input_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            if class_name not in self.class_mapping:
                self.logger.warning(f"Unknown class: {class_name}")
                continue
            
            class_id = self.class_mapping[class_name]
            dataset_stats['class_distribution'][class_name] = 0
            
            self.logger.info(f"Processing class: {class_name}")
            
            # Process images in this class
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            for img_file in tqdm(image_files, desc=f"Converting {class_name}"):
                try:
                    # Load image
                    image = cv2.imread(str(img_file))
                    if image is None:
                        self.logger.warning(f"Could not load {img_file}")
                        continue
                    
                    # Resize image
                    image_resized = cv2.resize(image, self.target_size)
                    
                    # Generate output filename
                    output_name = f"{class_name}_{img_file.stem}_{dataset_stats['total_images']:04d}"
                    
                    # Save image
                    cv2.imwrite(
                        str(output_path / 'images' / f"{output_name}.jpg"),
                        image_resized
                    )
                    
                    # Create annotation for tumor classes
                    annotation_content = ""
                    if class_id != -1:  # Has tumor
                        # Detect brain region
                        bbox = self.detect_brain_region(image_resized)
                        
                        # Convert to YOLO format
                        annotation_content = self.convert_to_yolo_format(bbox, self.target_size)
                        
                        dataset_stats['tumor_images'] += 1
                        
                        # Create synthetic segmentation mask
                        if create_synthetic_masks:
                            mask = self.create_tumor_mask(image_resized, bbox, class_name)
                            mask_resized = cv2.resize(mask, self.seg_size, interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(
                                str(output_path / 'masks' / f"{output_name}.png"),
                                mask_resized * 255
                            )
                    else:
                        dataset_stats['no_tumor_images'] += 1
                        
                        # Create empty mask for no tumor
                        if create_synthetic_masks:
                            empty_mask = np.zeros(self.seg_size, dtype=np.uint8)
                            cv2.imwrite(
                                str(output_path / 'masks' / f"{output_name}.png"),
                                empty_mask
                            )
                    
                    # Save YOLO annotation
                    with open(output_path / 'yolo_annotations' / f"{output_name}.txt", 'w') as f:
                        if annotation_content:
                            f.write(annotation_content)
                    
                    dataset_stats['total_images'] += 1
                    dataset_stats['class_distribution'][class_name] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing {img_file}: {e}")
                    continue
        
        # Create YOLO dataset structure
        self.create_yolo_structure(output_path)
        
        # Save dataset statistics
        with open(output_path / 'conversion_stats.json', 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        self.logger.info("Dataset conversion completed!")
        self.logger.info(f"Total images: {dataset_stats['total_images']}")
        self.logger.info(f"Tumor images: {dataset_stats['tumor_images']}")
        self.logger.info(f"No tumor images: {dataset_stats['no_tumor_images']}")
        
        return output_path
    
    def create_yolo_structure(self, base_path):
        """Create YOLO-compatible dataset structure"""
        image_files = list((base_path / 'images').glob('*.jpg'))
        np.random.shuffle(image_files)
        
        # Split data: 70% train, 20% val, 10% test
        n_total = len(image_files)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.2)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Create YOLO directory structure
        yolo_path = base_path / 'yolo_dataset'
        for split in ['train', 'val', 'test']:
            (yolo_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (yolo_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Copy files to appropriate splits
        splits = [('train', train_files), ('val', val_files), ('test', test_files)]
        
        for split_name, files in splits:
            for img_file in files:
                # Copy image
                shutil.copy2(img_file, yolo_path / 'images' / split_name / img_file.name)
                
                # Copy annotation
                ann_file = base_path / 'yolo_annotations' / f"{img_file.stem}.txt"
                if ann_file.exists():
                    shutil.copy2(ann_file, yolo_path / 'labels' / split_name / f"{img_file.stem}.txt")
        
        # Create data.yaml
        data_config = {
            'path': str(yolo_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,
            'names': ['brain_tumor']
        }
        
        import yaml
        with open(yolo_path / 'data.yaml', 'w') as f:
            yaml.dump(data_config, f)
        
        self.logger.info(f"YOLO dataset structure created at {yolo_path}")
        self.logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")


def download_kaggle_dataset(output_dir):
    """Download Kaggle dataset using kaggle API"""
    try:
        import kaggle
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'sartajbhuvaji/brain-tumor-classification-mri',
            path=output_dir,
            unzip=True
        )
        
        return Path(output_dir) / 'Training'
        
    except ImportError:
        print("Kaggle API not installed. Install with: pip install kaggle")
        print("Then configure credentials and run again.")
        return None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please download manually from:")
        print("https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert Kaggle brain tumor dataset")
    parser.add_argument('--input_dir', type=str, 
                       help='Input directory containing Training folder')
    parser.add_argument('--output_dir', type=str, default='data/kaggle_converted',
                       help='Output directory')
    parser.add_argument('--download', action='store_true',
                       help='Download dataset from Kaggle')
    parser.add_argument('--target_size', type=int, nargs=2, default=[640, 640],
                       help='Target image size [width height]')
    parser.add_argument('--seg_size', type=int, nargs=2, default=[224, 224],
                       help='Segmentation mask size [width height]')
    parser.add_argument('--no_synthetic_masks', action='store_true',
                       help='Do not create synthetic segmentation masks')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Download dataset if requested
    if args.download:
        print("Downloading Kaggle dataset...")
        input_dir = download_kaggle_dataset(args.output_dir)
        if input_dir is None:
            return
    else:
        if not args.input_dir:
            print("Error: --input_dir is required when not downloading")
            return
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory {input_dir} does not exist")
            return
    
    # Create converter
    converter = KaggleDatasetConverter(
        target_size=tuple(args.target_size),
        seg_size=tuple(args.seg_size)
    )
    
    # Convert dataset
    output_path = converter.convert_dataset(
        input_dir=input_dir,
        output_dir=args.output_dir,
        create_synthetic_masks=not args.no_synthetic_masks
    )
    
    print(f"\n🎉 Dataset conversion completed!")
    print(f"📁 Output directory: {output_path}")
    print(f"📊 YOLO dataset: {output_path}/yolo_dataset")
    
    print(f"\n🚀 To train the pipeline:")
    print(f"python src/main_yolo_transunet.py train \\")
    print(f"    --config configs/yolo_transunet_config.yaml \\")
    print(f"    --data_dir {output_path}/yolo_dataset \\")
    print(f"    --output_dir results/kaggle_training")


if __name__ == "__main__":
    main()
