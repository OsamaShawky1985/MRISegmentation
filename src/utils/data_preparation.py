"""
Data preparation utilities for YOLO and TransUNet training
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import logging
from tqdm import tqdm
import yaml


class DataPreparator:
    """Prepare data for YOLO detection and TransUNet segmentation training"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def convert_masks_to_yolo_annotations(self, image_dir: str, mask_dir: str, 
                                        output_dir: str, class_id: int = 0):
        """
        Convert segmentation masks to YOLO detection format
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing segmentation masks
            output_dir: Output directory for YOLO annotations
            class_id: Class ID for brain tumor (default: 0)
        """
        image_path = Path(image_dir)
        mask_path = Path(mask_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png'))
        
        for img_file in tqdm(image_files, desc="Converting masks to YOLO format"):
            # Find corresponding mask
            mask_file = mask_path / f"{img_file.stem}.png"
            if not mask_file.exists():
                mask_file = mask_path / f"{img_file.stem}.jpg"
            
            if not mask_file.exists():
                self.logger.warning(f"No mask found for {img_file.name}")
                continue
            
            # Load image and mask
            image = cv2.imread(str(img_file))
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                self.logger.warning(f"Could not load {img_file.name} or its mask")
                continue
            
            h, w = image.shape[:2]
            
            # Find bounding boxes from mask
            bboxes = self.extract_bboxes_from_mask(mask)
            
            # Convert to YOLO format and save
            yolo_annotations = []
            for bbox in bboxes:
                x_center = (bbox[0] + bbox[2]) / 2 / w
                y_center = (bbox[1] + bbox[3]) / 2 / h
                width = (bbox[2] - bbox[0]) / w
                height = (bbox[3] - bbox[1]) / h
                
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save annotation file
            if yolo_annotations:
                annotation_file = output_path / f"{img_file.stem}.txt"
                with open(annotation_file, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
    
    def extract_bboxes_from_mask(self, mask: np.ndarray, min_area: int = 100) -> List[List[int]]:
        """
        Extract bounding boxes from segmentation mask
        
        Args:
            mask: Binary mask
            min_area: Minimum area threshold for valid detections
            
        Returns:
            List of bounding boxes [x1, y1, x2, y2]
        """
        # Threshold mask
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append([x, y, x + w, y + h])
        
        return bboxes
    
    def create_yolo_dataset_structure(self, images_dir: str, annotations_dir: str, 
                                    output_dir: str, train_split: float = 0.7, 
                                    val_split: float = 0.2):
        """
        Create YOLO dataset structure with train/val splits
        
        Args:
            images_dir: Directory containing images
            annotations_dir: Directory containing YOLO annotations
            output_dir: Output directory for YOLO dataset
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
        """
        images_path = Path(images_dir)
        annotations_path = Path(annotations_dir)
        output_path = Path(output_dir)
        
        # Create directory structure
        (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (output_path / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        image_files = [f for f in image_files if (annotations_path / f"{f.stem}.txt").exists()]
        
        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(image_files)
        
        n_total = len(image_files)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Copy files to appropriate directories
        for split_name, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            for img_file in tqdm(split_files, desc=f"Copying {split_name} files"):
                # Copy image
                img_dst = output_path / 'images' / split_name / img_file.name
                import shutil
                shutil.copy2(img_file, img_dst)
                
                # Copy annotation
                ann_file = annotations_path / f"{img_file.stem}.txt"
                ann_dst = output_path / 'labels' / split_name / f"{img_file.stem}.txt"
                if ann_file.exists():
                    shutil.copy2(ann_file, ann_dst)
        
        # Create data.yaml
        data_config = {
            'path': str(output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,
            'names': ['brain_tumor']
        }
        
        with open(output_path / 'data.yaml', 'w') as f:
            yaml.dump(data_config, f)
        
        self.logger.info(f"YOLO dataset created with {len(train_files)} train, "
                        f"{len(val_files)} val, {len(test_files)} test images")
    
    def prepare_segmentation_dataset(self, images_dir: str, masks_dir: str, 
                                   output_dir: str, target_size: Tuple[int, int] = (224, 224)):
        """
        Prepare segmentation dataset with proper train/val/test splits
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing masks
            output_dir: Output directory
            target_size: Target image size (width, height)
        """
        images_path = Path(images_dir)
        masks_path = Path(masks_dir)
        output_path = Path(output_dir)
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'masks' / split).mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        image_files = [f for f in image_files if self.find_corresponding_mask(f, masks_path)]
        
        # Split data
        np.random.seed(42)
        np.random.shuffle(image_files)
        
        n_total = len(image_files)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.2)
        
        splits = {
            'train': image_files[:n_train],
            'val': image_files[n_train:n_train + n_val],
            'test': image_files[n_train + n_val:]
        }
        
        # Process and save files
        for split_name, split_files in splits.items():
            for img_file in tqdm(split_files, desc=f"Processing {split_name} files"):
                # Load and resize image
                image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                image_resized = cv2.resize(image, target_size)
                
                # Load and resize mask
                mask_file = self.find_corresponding_mask(img_file, masks_path)
                if mask_file:
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    mask_resized = cv2.resize(mask, target_size)
                    # Convert to binary
                    mask_resized = (mask_resized > 127).astype(np.uint8) * 255
                else:
                    mask_resized = np.zeros(target_size, dtype=np.uint8)
                
                # Save processed files
                img_dst = output_path / 'images' / split_name / f"{img_file.stem}.png"
                mask_dst = output_path / 'masks' / split_name / f"{img_file.stem}.png"
                
                cv2.imwrite(str(img_dst), image_resized)
                cv2.imwrite(str(mask_dst), mask_resized)
        
        self.logger.info(f"Segmentation dataset prepared: "
                        f"{len(splits['train'])} train, {len(splits['val'])} val, "
                        f"{len(splits['test'])} test images")
    
    def find_corresponding_mask(self, image_file: Path, masks_dir: Path) -> Path:
        """Find corresponding mask file for an image"""
        possible_names = [
            f"{image_file.stem}.png",
            f"{image_file.stem}.jpg",
            f"{image_file.stem}_mask.png",
            f"{image_file.stem}_mask.jpg"
        ]
        
        for name in possible_names:
            mask_file = masks_dir / name
            if mask_file.exists():
                return mask_file
        return None
    
    def augment_dataset(self, images_dir: str, masks_dir: str, output_dir: str, 
                       augmentation_factor: int = 3):
        """
        Apply data augmentation to increase dataset size
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing masks
            output_dir: Output directory for augmented data
            augmentation_factor: Number of augmented versions per original image
        """
        try:
            import albumentations as A
        except ImportError:
            self.logger.error("albumentations not installed. Run: pip install albumentations")
            return
        
        images_path = Path(images_dir)
        masks_path = Path(masks_dir)
        output_path = Path(output_dir)
        
        (output_path / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / 'masks').mkdir(parents=True, exist_ok=True)
        
        # Define augmentation pipeline
        transform = A.Compose([
            A.Rotate(limit=20, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
            A.GridDistortion(p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.3)
        ])
        
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        
        for img_file in tqdm(image_files, desc="Augmenting dataset"):
            # Load image and mask
            image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            mask_file = self.find_corresponding_mask(img_file, masks_path)
            
            if image is None or mask_file is None:
                continue
            
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            # Save original
            cv2.imwrite(str(output_path / 'images' / img_file.name), image)
            cv2.imwrite(str(output_path / 'masks' / f"{img_file.stem}.png"), mask)
            
            # Generate augmented versions
            for i in range(augmentation_factor):
                try:
                    augmented = transform(image=image, mask=mask)
                    aug_image = augmented['image']
                    aug_mask = augmented['mask']
                    
                    # Save augmented versions
                    aug_img_name = f"{img_file.stem}_aug_{i}.png"
                    aug_mask_name = f"{img_file.stem}_aug_{i}.png"
                    
                    cv2.imwrite(str(output_path / 'images' / aug_img_name), aug_image)
                    cv2.imwrite(str(output_path / 'masks' / aug_mask_name), aug_mask)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to augment {img_file.name}: {e}")
                    continue
        
        self.logger.info(f"Dataset augmentation completed. Check {output_path}")


def main():
    """Main function for data preparation"""
    parser = argparse.ArgumentParser(description="Data preparation for YOLO-TransUNet pipeline")
    parser.add_argument('command', choices=[
        'masks_to_yolo', 'create_yolo_dataset', 'prepare_segmentation', 'augment'
    ], help='Data preparation command')
    parser.add_argument('--images_dir', type=str, required=True, help='Images directory')
    parser.add_argument('--masks_dir', type=str, help='Masks directory')
    parser.add_argument('--annotations_dir', type=str, help='Annotations directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--config', type=str, default='configs/yolo_transunet_config.yaml',
                       help='Configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    preparator = DataPreparator(config)
    
    if args.command == 'masks_to_yolo':
        if not args.masks_dir:
            raise ValueError("--masks_dir is required for masks_to_yolo command")
        preparator.convert_masks_to_yolo_annotations(
            args.images_dir, args.masks_dir, args.output_dir
        )
    
    elif args.command == 'create_yolo_dataset':
        if not args.annotations_dir:
            raise ValueError("--annotations_dir is required for create_yolo_dataset command")
        preparator.create_yolo_dataset_structure(
            args.images_dir, args.annotations_dir, args.output_dir
        )
    
    elif args.command == 'prepare_segmentation':
        if not args.masks_dir:
            raise ValueError("--masks_dir is required for prepare_segmentation command")
        preparator.prepare_segmentation_dataset(
            args.images_dir, args.masks_dir, args.output_dir
        )
    
    elif args.command == 'augment':
        if not args.masks_dir:
            raise ValueError("--masks_dir is required for augment command")
        preparator.augment_dataset(
            args.images_dir, args.masks_dir, args.output_dir
        )
    
    print(f"Data preparation completed! Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
