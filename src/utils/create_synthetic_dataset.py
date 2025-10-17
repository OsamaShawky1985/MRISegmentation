#!/usr/bin/env python3
"""
Synthetic Brain Tumor Dataset Generator
Creates synthetic brain MRI images with tumors for testing the YOLOv10 + TransUNet pipeline

This is useful for:
1. Testing pipeline functionality without downloading large datasets
2. Debugging and development
3. Quick validation of the complete workflow
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import logging


class SyntheticBrainDatasetGenerator:
    """Generate synthetic brain MRI images with tumors"""
    
    def __init__(self, image_size=(640, 640), segmentation_size=(224, 224)):
        self.image_size = image_size
        self.seg_size = segmentation_size
        self.logger = logging.getLogger(__name__)
        
    def create_brain_background(self, size):
        """Create realistic brain-like background"""
        image = np.zeros(size, dtype=np.uint8)
        center = (size[0] // 2, size[1] // 2)
        
        # Main brain oval
        cv2.ellipse(image, center, (size[0]//3, size[1]//3), 0, 0, 360, 120, -1)
        
        # Add brain tissue variation
        for _ in range(30):
            x = np.random.randint(center[0] - size[0]//4, center[0] + size[0]//4)
            y = np.random.randint(center[1] - size[1]//4, center[1] + size[1]//4)
            radius = np.random.randint(8, 25)
            intensity = np.random.randint(90, 140)
            cv2.circle(image, (x, y), radius, intensity, -1)
        
        # Add ventricles (darker regions)
        ventricle_centers = [
            (center[0] - 30, center[1] - 20),
            (center[0] + 30, center[1] - 20)
        ]
        for vc in ventricle_centers:
            cv2.ellipse(image, vc, (15, 25), 0, 0, 360, 60, -1)
        
        # Add cerebral cortex texture
        for _ in range(100):
            x = np.random.randint(20, size[0] - 20)
            y = np.random.randint(20, size[1] - 20)
            if image[y, x] > 50:  # Only add texture to brain tissue
                radius = np.random.randint(2, 8)
                intensity = int(image[y, x] * np.random.uniform(0.8, 1.2))
                intensity = np.clip(intensity, 0, 255)
                cv2.circle(image, (x, y), radius, intensity, -1)
        
        # Add noise for realism
        noise = np.random.normal(0, 8, size).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def create_tumor(self, image, tumor_type='enhancing'):
        """Add tumor to brain image"""
        h, w = image.shape
        
        # Tumor parameters based on type
        if tumor_type == 'enhancing':
            intensity_range = (200, 255)
            size_range = (15, 35)
        elif tumor_type == 'core':
            intensity_range = (150, 200)
            size_range = (20, 45)
        else:  # whole tumor
            intensity_range = (140, 180)
            size_range = (25, 55)
        
        # Random tumor location (avoid edges and ventricles)
        center_x = np.random.randint(w//4, 3*w//4)
        center_y = np.random.randint(h//4, 3*h//4)
        
        # Tumor size
        radius = np.random.randint(*size_range)
        
        # Create irregular tumor shape
        tumor_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Main tumor body
        cv2.circle(tumor_mask, (center_x, center_y), radius, 1, -1)
        
        # Add irregular edges
        for _ in range(np.random.randint(3, 8)):
            offset_x = np.random.randint(-radius//2, radius//2)
            offset_y = np.random.randint(-radius//2, radius//2)
            small_radius = np.random.randint(radius//3, radius//2)
            cv2.circle(tumor_mask, 
                      (center_x + offset_x, center_y + offset_y), 
                      small_radius, 1, -1)
        
        # Apply morphological operations for more realistic shape
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel)
        
        # Add tumor intensity to image
        tumor_intensity = np.random.randint(*intensity_range)
        image[tumor_mask == 1] = tumor_intensity
        
        # Add tumor enhancement effects
        enhancement_mask = cv2.dilate(tumor_mask, kernel, iterations=1)
        edge_mask = enhancement_mask - tumor_mask
        if np.any(edge_mask):
            enhancement_intensity = int(tumor_intensity * 0.7)
            image[edge_mask == 1] = enhancement_intensity
        
        # Calculate bounding box
        coords = np.where(tumor_mask == 1)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            bbox = [x_min, y_min, x_max, y_max]
        else:
            bbox = [center_x-radius, center_y-radius, center_x+radius, center_y+radius]
        
        return image, tumor_mask, bbox
    
    def create_yolo_annotation(self, bbox, image_size):
        """Convert bounding box to YOLO format"""
        x_min, y_min, x_max, y_max = bbox
        
        # Convert to YOLO format (normalized center_x, center_y, width, height)
        center_x = (x_min + x_max) / 2 / image_size[1]
        center_y = (y_min + y_max) / 2 / image_size[0]
        width = (x_max - x_min) / image_size[1]
        height = (y_max - y_min) / image_size[0]
        
        return f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
    
    def generate_dataset(self, num_images, output_dir, include_masks=True, 
                        tumor_probability=0.8):
        """Generate complete synthetic dataset"""
        output_path = Path(output_dir)
        
        # Create directory structure
        (output_path / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / 'yolo_annotations').mkdir(parents=True, exist_ok=True)
        if include_masks:
            (output_path / 'masks').mkdir(parents=True, exist_ok=True)
        
        dataset_info = {
            'total_images': num_images,
            'images_with_tumors': 0,
            'images_without_tumors': 0,
            'tumor_types': [],
            'image_size': self.image_size,
            'segmentation_size': self.seg_size
        }
        
        self.logger.info(f"Generating {num_images} synthetic brain images...")
        
        for i in tqdm(range(num_images), desc="Generating images"):
            # Create brain background
            brain_image = self.create_brain_background(self.image_size)
            
            # Decide if this image should have a tumor
            has_tumor = np.random.random() < tumor_probability
            
            yolo_annotation = ""
            segmentation_mask = np.zeros(self.seg_size, dtype=np.uint8)
            
            if has_tumor:
                # Choose tumor type
                tumor_type = np.random.choice(['enhancing', 'core', 'whole'])
                
                # Add tumor
                brain_image, tumor_mask, bbox = self.create_tumor(brain_image, tumor_type)
                
                # Create YOLO annotation
                yolo_annotation = self.create_yolo_annotation(bbox, self.image_size)
                
                # Create segmentation mask
                if include_masks:
                    tumor_mask_resized = cv2.resize(
                        tumor_mask, self.seg_size, 
                        interpolation=cv2.INTER_NEAREST
                    )
                    segmentation_mask = tumor_mask_resized
                
                dataset_info['images_with_tumors'] += 1
                dataset_info['tumor_types'].append(tumor_type)
            else:
                dataset_info['images_without_tumors'] += 1
            
            # Save files
            image_name = f"synthetic_brain_{i:04d}"
            
            # Save image
            cv2.imwrite(
                str(output_path / 'images' / f"{image_name}.jpg"), 
                brain_image
            )
            
            # Save YOLO annotation
            with open(output_path / 'yolo_annotations' / f"{image_name}.txt", 'w') as f:
                if yolo_annotation:
                    f.write(yolo_annotation)
            
            # Save segmentation mask
            if include_masks:
                cv2.imwrite(
                    str(output_path / 'masks' / f"{image_name}.png"),
                    segmentation_mask * 255  # Convert to 0-255 range
                )
        
        # Save dataset info
        with open(output_path / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Create YOLO dataset structure
        self.create_yolo_dataset_structure(output_path)
        
        self.logger.info(f"Dataset generated successfully!")
        self.logger.info(f"Images with tumors: {dataset_info['images_with_tumors']}")
        self.logger.info(f"Images without tumors: {dataset_info['images_without_tumors']}")
        
        return output_path
    
    def create_yolo_dataset_structure(self, base_path):
        """Create YOLO-compatible dataset structure"""
        # Create train/val split
        image_files = list((base_path / 'images').glob('*.jpg'))
        np.random.shuffle(image_files)
        
        split_idx = int(len(image_files) * 0.8)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Create YOLO structure
        yolo_path = base_path / 'yolo_dataset'
        for split in ['train', 'val']:
            (yolo_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (yolo_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Copy files
        import shutil
        
        for split, files in [('train', train_files), ('val', val_files)]:
            for img_file in files:
                # Copy image
                shutil.copy2(img_file, yolo_path / 'images' / split / img_file.name)
                
                # Copy annotation
                ann_file = base_path / 'yolo_annotations' / f"{img_file.stem}.txt"
                if ann_file.exists():
                    shutil.copy2(ann_file, yolo_path / 'labels' / split / f"{img_file.stem}.txt")
        
        # Create data.yaml
        data_config = {
            'path': str(yolo_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['brain_tumor']
        }
        
        import yaml
        with open(yolo_path / 'data.yaml', 'w') as f:
            yaml.dump(data_config, f)
        
        self.logger.info(f"YOLO dataset structure created at {yolo_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic brain tumor dataset")
    parser.add_argument('--num_images', type=int, default=100, 
                       help='Number of images to generate')
    parser.add_argument('--output_dir', type=str, default='data/synthetic_brain_tumors',
                       help='Output directory')
    parser.add_argument('--image_size', type=int, nargs=2, default=[640, 640],
                       help='Image size [width height]')
    parser.add_argument('--segmentation_size', type=int, nargs=2, default=[224, 224],
                       help='Segmentation mask size [width height]')
    parser.add_argument('--tumor_probability', type=float, default=0.8,
                       help='Probability of tumor in each image')
    parser.add_argument('--include_masks', action='store_true', default=True,
                       help='Generate segmentation masks')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    # Create generator
    generator = SyntheticBrainDatasetGenerator(
        image_size=tuple(args.image_size),
        segmentation_size=tuple(args.segmentation_size)
    )
    
    # Generate dataset
    output_path = generator.generate_dataset(
        num_images=args.num_images,
        output_dir=args.output_dir,
        include_masks=args.include_masks,
        tumor_probability=args.tumor_probability
    )
    
    print(f"\n🎉 Synthetic dataset generated successfully!")
    print(f"📁 Output directory: {output_path}")
    print(f"📊 Total images: {args.num_images}")
    print(f"🧠 Tumor probability: {args.tumor_probability}")
    
    print(f"\n🚀 To test with the pipeline:")
    print(f"python src/main_yolo_transunet.py train \\")
    print(f"    --config configs/yolo_transunet_config.yaml \\")
    print(f"    --data_dir {output_path}/yolo_dataset \\")
    print(f"    --output_dir results/synthetic_training")


if __name__ == "__main__":
    main()
