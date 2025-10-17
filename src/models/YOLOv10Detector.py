import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import yaml
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import logging


class YOLOv10Detector:
    """
    YOLOv10 Brain Tumor Detection and Localization
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_size = config['yolo']['model_size']
        self.confidence_threshold = config['yolo']['confidence_threshold']
        self.iou_threshold = config['yolo']['iou_threshold']
        self.max_detections = config['yolo']['max_detections']
        self.input_size = config['data']['input_size']
        self.num_classes = config['yolo']['num_classes']
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger = logging.getLogger(__name__)
        
    def load_model(self, model_path: Optional[str] = None, pretrained: bool = True):
        """
        Load YOLOv10 model
        
        Args:
            model_path: Path to trained model weights
            pretrained: Whether to use pretrained weights
        """
        if model_path and Path(model_path).exists():
            self.logger.info(f"Loading trained model from {model_path}")
            self.model = YOLO(model_path)
        elif pretrained:
            self.logger.info(f"Loading pretrained {self.model_size} model")
            self.model = YOLO(f'{self.model_size}.pt')
        else:
            self.logger.info(f"Initializing {self.model_size} model from scratch")
            self.model = YOLO(f'{self.model_size}.yaml')
            
        self.model.to(self.device)
        
    def prepare_training_data(self, data_dir: str, output_dir: str):
        """
        Prepare data in YOLO format
        
        Args:
            data_dir: Directory containing images and annotations
            output_dir: Output directory for YOLO format data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        
        # Create YOLO data config file
        data_config = {
            'path': str(output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': self.num_classes,
            'names': ['brain_tumor']
        }
        
        with open(output_path / 'data.yaml', 'w') as f:
            yaml.dump(data_config, f)
            
        self.logger.info(f"YOLO data structure created at {output_path}")
        return str(output_path / 'data.yaml')
    
    def train(self, data_config_path: str, epochs: int = None, batch_size: int = None):
        """
        Train YOLOv10 model
        
        Args:
            data_config_path: Path to YOLO data configuration file
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        if self.model is None:
            self.load_model(pretrained=True)
            
        epochs = epochs or self.config['yolo']['epochs']
        batch_size = batch_size or self.config['yolo']['batch_size']
        
        # Training parameters
        train_params = {
            'data': data_config_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': self.input_size[0],
            'lr0': self.config['yolo']['learning_rate'],
            'weight_decay': self.config['yolo']['weight_decay'],
            'device': self.device,
            'patience': self.config['training'].get('early_stopping_patience', 20),
            'save_period': 10,
            'val': True,
            'plots': True,
            'cache': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': False,
            'amp': self.config['training'].get('mixed_precision', True),
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': True,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val_fraction': 1.0,
            'verbose': True
        }
        
        self.logger.info("Starting YOLOv10 training...")
        results = self.model.train(**train_params)
        
        self.logger.info("Training completed!")
        return results
    
    def detect(self, image_path: str, save_results: bool = True, output_dir: str = None):
        """
        Detect brain tumors in an image
        
        Args:
            image_path: Path to input image
            save_results: Whether to save detection results
            output_dir: Directory to save results
            
        Returns:
            Detection results with bounding boxes and confidence scores
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Run inference
        results = self.model(
            image_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            imgsz=self.input_size[0],
            device=self.device,
            half=False,
            dnn=False,
            vid_stride=1,
            stream_buffer=False,
            visualize=False,
            augment=False,
            agnostic_nms=False,
            retina_masks=False,
            embed=None,
            show=False,
            save=save_results,
            save_frames=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            show_labels=True,
            show_conf=True,
            show_boxes=True,
            line_width=None,
        )
        
        return results
    
    def extract_tumor_regions(self, image_path: str, results) -> List[Dict]:
        """
        Extract tumor regions based on detection results
        
        Args:
            image_path: Path to original image
            results: YOLO detection results
            
        Returns:
            List of dictionaries containing tumor region information
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        tumor_regions = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Extract tumor region
                    tumor_crop = image[y1:y2, x1:x2]
                    
                    # Calculate center point for segmentation
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    tumor_info = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'center': [center_x, center_y],
                        'crop': tumor_crop,
                        'area': (x2 - x1) * (y2 - y1)
                    }
                    
                    tumor_regions.append(tumor_info)
        
        return tumor_regions
    
    def validate(self, data_config_path: str):
        """
        Validate the trained model
        
        Args:
            data_config_path: Path to YOLO data configuration file
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        results = self.model.val(
            data=data_config_path,
            imgsz=self.input_size[0],
            batch=1,
            device=self.device,
            plots=True,
            save_json=True,
            save_hybrid=False,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            half=False,
            dnn=False,
            split='val',
            rect=False,
            verbose=True
        )
        
        return results
    
    def export_model(self, format: str = 'onnx', output_path: str = None):
        """
        Export trained model to different formats
        
        Args:
            format: Export format (onnx, torchscript, etc.)
            output_path: Path to save exported model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        export_path = self.model.export(
            format=format,
            imgsz=self.input_size[0],
            keras=False,
            optimize=False,
            half=False,
            int8=False,
            dynamic=False,
            simplify=False,
            opset=None,
            workspace=4,
            nms=False,
            lr=0.01,
            decay=0.0005,
            batch=1,
            device=self.device,
            verbose=True
        )
        
        if output_path:
            import shutil
            shutil.move(export_path, output_path)
            return output_path
        
        return export_path


if __name__ == "__main__":
    # Test the detector
    config = {
        'yolo': {
            'model_size': 'yolov10n',
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45,
            'max_detections': 100,
            'num_classes': 1,
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'weight_decay': 0.0005
        },
        'data': {
            'input_size': [640, 640]
        },
        'training': {
            'early_stopping_patience': 20,
            'mixed_precision': True
        }
    }
    
    detector = YOLOv10Detector(config)
    detector.load_model(pretrained=True)
    print("YOLOv10 Detector initialized successfully!")
