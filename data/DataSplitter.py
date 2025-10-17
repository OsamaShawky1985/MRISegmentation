import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from preprocessing.ImageEnhancer import MRIEnhancer

class DataSplitter:
    def __init__(self, config):
        self.config = config
        self.enhancer = MRIEnhancer(config)
        
    def prepare_data(self, raw_data_path, test_size=0.2):
        """
        Process and split data into train and test sets
        """
        # Create directories if they don't exist
        os.makedirs('data/processed_data', exist_ok=True)
        os.makedirs('data/train', exist_ok=True)
        os.makedirs('data/test', exist_ok=True)
        
        # Get all image paths and labels
        images = []
        labels = []
        
        # Assuming data is organized in folders by class
        for class_name in os.listdir(raw_data_path):
            class_path = os.path.join(raw_data_path, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.png', '.nii.gz')):
                        img_path = os.path.join(class_path, img_name)
                        images.append(img_path)
                        labels.append(class_name)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, 
            labels, 
            test_size=test_size, 
            stratify=labels, 
            random_state=42
        )
        
        # Process and save train data
        self._process_and_save_set(X_train, y_train, 'train')
        
        # Process and save test data
        self._process_and_save_set(X_test, y_test, 'test')
        
        return len(X_train), len(X_test)
    
    def _process_and_save_set(self, image_paths, labels, set_type):
        """Process and save images to appropriate directory"""
        for img_path, label in zip(image_paths, labels):
            # Create class directory if it doesn't exist
            class_dir = os.path.join(f'data/{set_type}', label)
            os.makedirs(class_dir, exist_ok=True)
            
            # Read and preprocess image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            processed_img = self.enhancer.process(img)
            
            # Save processed image
            filename = os.path.basename(img_path)
            save_path = os.path.join(class_dir, filename)
            cv2.imwrite(save_path, processed_img)