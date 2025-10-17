import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.YOLOv10Detector import YOLOv10Detector
from models.TransUNet import TransUNet


class BrainTumorDataset(Dataset):
    """Dataset for brain tumor segmentation"""
    
    def __init__(self, image_paths: List[str], mask_paths: List[str] = None, 
                 transform=None, img_size: int = 224):
        self.image_paths = image_paths
        self.mask_paths = mask_paths if mask_paths else [None] * len(image_paths)
        self.transform = transform
        self.img_size = img_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {self.image_paths[idx]}")
        
        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0
        
        # Load mask if available
        if self.mask_paths[idx] is not None:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            mask = (mask > 127).astype(np.float32)  # Binary mask
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert to tensor and add channel dimension
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        
        return image, mask


class YOLOTransUNetPipeline:
    """
    Integrated pipeline using YOLOv10 for detection and TransUNet for segmentation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.yolo_detector = YOLOv10Detector(config)
        self.transunet = TransUNet(
            img_size=config['transunet']['img_size'],
            patch_size=config['transunet']['patch_size'],
            in_chans=1,  # Grayscale
            num_classes=config['transunet']['num_classes'],
            embed_dim=config['transunet']['embed_dim'],
            depth=config['transunet']['depth'],
            num_heads=config['transunet']['num_heads'],
            mlp_ratio=config['transunet']['mlp_ratio'],
            qkv_bias=config['transunet']['qkv_bias'],
            drop_rate=config['transunet']['drop_rate'],
            attn_drop_rate=config['transunet']['attn_drop_rate'],
            drop_path_rate=config['transunet']['drop_path_rate']
        ).to(self.device)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir = Path(config['data']['output_path'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_transforms(self, is_training: bool = True):
        """Setup data augmentation transforms"""
        if is_training:
            return A.Compose([
                A.Rotate(limit=self.config['augmentation']['rotation_range'], p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=self.config['augmentation']['shift_range'],
                    scale_limit=self.config['augmentation']['zoom_range'],
                    rotate_limit=0, p=0.5
                ),
                A.HorizontalFlip(p=0.5 if self.config['augmentation']['horizontal_flip'] else 0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.Normalize(mean=0.0, std=1.0),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Normalize(mean=0.0, std=1.0),
                ToTensorV2()
            ])
    
    def train_yolo(self, train_data_path: str, val_data_path: str = None):
        """Train YOLOv10 detection model"""
        self.logger.info("Starting YOLOv10 training...")
        
        # Prepare data in YOLO format
        yolo_data_config = self.yolo_detector.prepare_training_data(
            train_data_path, 
            str(self.output_dir / 'yolo_data')
        )
        
        # Train the model
        results = self.yolo_detector.train(yolo_data_config)
        
        # Save model
        model_save_path = self.output_dir / 'yolo_best.pt'
        self.yolo_detector.model.save(str(model_save_path))
        
        self.logger.info(f"YOLOv10 training completed. Model saved to {model_save_path}")
        return results
    
    def train_transunet(self, train_images: List[str], train_masks: List[str],
                       val_images: List[str] = None, val_masks: List[str] = None):
        """Train TransUNet segmentation model"""
        self.logger.info("Starting TransUNet training...")
        
        # Setup transforms
        train_transform = self.setup_transforms(is_training=True)
        val_transform = self.setup_transforms(is_training=False)
        
        # Create datasets
        train_dataset = BrainTumorDataset(
            train_images, train_masks, train_transform, 
            self.config['transunet']['img_size']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['transunet']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=self.config['training']['pin_memory']
        )
        
        val_loader = None
        if val_images and val_masks:
            val_dataset = BrainTumorDataset(
                val_images, val_masks, val_transform,
                self.config['transunet']['img_size']
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['transunet']['batch_size'],
                shuffle=False,
                num_workers=self.config['training']['num_workers'],
                pin_memory=self.config['training']['pin_memory']
            )
        
        # Setup optimizer and loss
        optimizer = optim.AdamW(
            self.transunet.parameters(),
            lr=self.config['transunet']['learning_rate'],
            weight_decay=self.config['transunet']['weight_decay']
        )
        
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['transunet']['epochs']
        )
        
        # Training loop
        best_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['transunet']['epochs']):
            # Training phase
            self.transunet.train()
            epoch_train_loss = 0.0
            
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["transunet"]["epochs"]}') as pbar:
                for batch_idx, (images, masks) in enumerate(pbar):
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.transunet(images)
                    loss = criterion(outputs, masks)
                    
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config['training']['gradient_clipping'] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.transunet.parameters(),
                            self.config['training']['gradient_clipping']
                        )
                    
                    optimizer.step()
                    
                    epoch_train_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                self.transunet.eval()
                epoch_val_loss = 0.0
                
                with torch.no_grad():
                    for images, masks in val_loader:
                        images = images.to(self.device)
                        masks = masks.to(self.device)
                        
                        outputs = self.transunet(images)
                        loss = criterion(outputs, masks)
                        epoch_val_loss += loss.item()
                
                avg_val_loss = epoch_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # Save best model
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(
                        self.transunet.state_dict(),
                        self.output_dir / 'transunet_best.pth'
                    )
                
                self.logger.info(
                    f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, '
                    f'Val Loss: {avg_val_loss:.4f}'
                )
            else:
                self.logger.info(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}')
            
            scheduler.step()
        
        # Plot training history
        self.plot_training_history(train_losses, val_losses)
        
        self.logger.info("TransUNet training completed!")
        return train_losses, val_losses
    
    def plot_training_history(self, train_losses: List[float], val_losses: List[float] = None):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('TransUNet Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'training_history.png')
        plt.close()
    
    def predict(self, image_path: str, save_results: bool = True):
        """
        Complete pipeline: detect with YOLO, segment with TransUNet
        
        Args:
            image_path: Path to input image
            save_results: Whether to save results
            
        Returns:
            Dictionary containing detection and segmentation results
        """
        self.logger.info(f"Processing image: {image_path}")
        
        # Step 1: YOLO Detection
        yolo_results = self.yolo_detector.detect(image_path, save_results=False)
        tumor_regions = self.yolo_detector.extract_tumor_regions(image_path, yolo_results)
        
        if not tumor_regions:
            self.logger.warning("No tumors detected by YOLO")
            return {
                'detections': [],
                'segmentations': [],
                'original_image': cv2.imread(image_path)
            }
        
        # Step 2: TransUNet Segmentation for each detected region
        segmentation_results = []
        original_image = cv2.imread(image_path)
        
        for i, region in enumerate(tumor_regions):
            # Prepare crop for segmentation
            crop = region['crop']
            if len(crop.shape) == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Resize and normalize
            crop_resized = cv2.resize(crop, (
                self.config['transunet']['img_size'],
                self.config['transunet']['img_size']
            ))
            crop_normalized = crop_resized.astype(np.float32) / 255.0
            
            # Convert to tensor
            crop_tensor = torch.from_numpy(crop_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Segment with TransUNet
            self.transunet.eval()
            with torch.no_grad():
                segmentation_logits = self.transunet(crop_tensor)
                segmentation_pred = torch.argmax(segmentation_logits, dim=1)
                segmentation_mask = segmentation_pred.squeeze().cpu().numpy()
            
            # Resize segmentation back to original crop size
            original_height, original_width = crop.shape[:2]
            segmentation_resized = cv2.resize(
                segmentation_mask.astype(np.uint8),
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST
            )
            
            segmentation_results.append({
                'region_index': i,
                'bbox': region['bbox'],
                'confidence': region['confidence'],
                'segmentation_mask': segmentation_resized,
                'original_crop': crop
            })
        
        # Combine results
        results = {
            'detections': tumor_regions,
            'segmentations': segmentation_results,
            'original_image': original_image
        }
        
        # Save results if requested
        if save_results:
            self.save_results(image_path, results)
        
        return results
    
    def save_results(self, image_path: str, results: Dict):
        """Save detection and segmentation results"""
        image_name = Path(image_path).stem
        
        # Create visualization
        vis_image = results['original_image'].copy()
        
        for seg_result in results['segmentations']:
            bbox = seg_result['bbox']
            confidence = seg_result['confidence']
            mask = seg_result['segmentation_mask']
            
            # Draw bounding box
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(
                vis_image, f'Tumor: {confidence:.2f}',
                (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
            
            # Overlay segmentation mask
            x1, y1, x2, y2 = bbox
            mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            mask_colored[mask == 1] = [0, 0, 255]  # Red for tumor
            
            # Resize mask to match bbox
            mask_resized = cv2.resize(mask_colored, (x2 - x1, y2 - y1))
            
            # Overlay on original image
            roi = vis_image[y1:y2, x1:x2]
            overlay = cv2.addWeighted(roi, 0.7, mask_resized, 0.3, 0)
            vis_image[y1:y2, x1:x2] = overlay
        
        # Save visualization
        output_path = self.output_dir / f'{image_name}_result.jpg'
        cv2.imwrite(str(output_path), vis_image)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def evaluate_segmentation(self, test_images: List[str], test_masks: List[str]):
        """Evaluate TransUNet segmentation performance"""
        self.logger.info("Evaluating segmentation performance...")
        
        # Load best model
        model_path = self.output_dir / 'transunet_best.pth'
        if model_path.exists():
            self.transunet.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.transunet.eval()
        
        all_predictions = []
        all_targets = []
        dice_scores = []
        
        with torch.no_grad():
            for img_path, mask_path in tqdm(zip(test_images, test_masks), total=len(test_images)):
                # Load and preprocess image
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (self.config['transunet']['img_size'], 
                                         self.config['transunet']['img_size']))
                image = image.astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Load ground truth mask
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (self.config['transunet']['img_size'], 
                                       self.config['transunet']['img_size']))
                mask = (mask > 127).astype(np.int64)
                
                # Predict
                outputs = self.transunet(image_tensor)
                pred = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
                
                # Calculate metrics
                all_predictions.extend(pred.flatten())
                all_targets.extend(mask.flatten())
                
                # Calculate Dice score
                dice = self.calculate_dice_score(pred, mask)
                dice_scores.append(dice)
        
        # Calculate overall metrics
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        mean_dice = np.mean(dice_scores)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'dice_score': mean_dice,
            'individual_dice_scores': dice_scores
        }
        
        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1 Score: {f1:.4f}")
        self.logger.info(f"Mean Dice Score: {mean_dice:.4f}")
        
        return metrics
    
    def calculate_dice_score(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate Dice coefficient"""
        pred_binary = (pred == 1).astype(np.float32)
        target_binary = (target == 1).astype(np.float32)
        
        intersection = np.sum(pred_binary * target_binary)
        union = np.sum(pred_binary) + np.sum(target_binary)
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        dice = (2.0 * intersection) / union
        return dice


if __name__ == "__main__":
    # Test the pipeline
    with open('configs/yolo_transunet_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    pipeline = YOLOTransUNetPipeline(config)
    print("YOLOv10-TransUNet pipeline initialized successfully!")
