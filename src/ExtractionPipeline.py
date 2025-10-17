from preprocessing.ImageEnhancer import ImageEnhancer
from preprocessing.DataAugmentor import DataAugmentor
from CNNFeatureExtractor import CNNFeatureExtractor
import numpy as np
import os
from tqdm import tqdm
import cv2
from PIL import Image

class ExtractionPipeline:
    def __init__(self, config):
        self.config = config
        self.enhancer = ImageEnhancer(config)
        self.augmentor = DataAugmentor(config)
        self.feature_extractor = CNNFeatureExtractor(config)
    
    def process_directory(self, input_dir, output_dir):
        """Process entire directory of images"""
        os.makedirs(output_dir, exist_ok=True)
        print("DEBUG: Start ExtractionPipeline.process_directory")
        print(f"Processing directory: {input_dir}")
        all_features = []
        all_labels = []
        
        # Process each class directory or images directly if no subdirectories
        entries = os.listdir(input_dir)
        subdirs = [d for d in entries if os.path.isdir(os.path.join(input_dir, d))]
        if subdirs:
            for class_name in subdirs:
                class_path = os.path.join(input_dir, class_name)
                print(f"DEBUG: Processing class path: {class_path}")
                if os.path.isdir(class_path):
                    print(f"\nProcessing class: {class_name}")
                    for img_name in tqdm(os.listdir(class_path)):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                            features = self._process_single_image(
                                os.path.join(class_path, img_name)
                            )
                            all_features.extend(features)
                            all_labels.extend([class_name] * len(features))
                            print("Distinct classes:", set(all_labels))
                            print("Number of distinct classes:", len(set(all_labels)))
        else:
            # No subdirectories, process images directly
            print("DEBUG: No subdirectories found, processing images directly in input_dir")
            for img_name in tqdm(entries):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    features = self._process_single_image(
                        os.path.join(input_dir, img_name)
                    )
                    all_features.extend(features)
                    all_labels.extend(['unknown'] * len(features))
                    print("Distinct classes:", set(all_labels))
                    print("Number of distinct classes:", len(set(all_labels)))                  
        
        # Save processed features
        print("\nSaving features...")
        np.save(os.path.join(output_dir, 'features.npy'), np.array(all_features))
        np.save(os.path.join(output_dir, 'labels.npy'), np.array(all_labels))

    def _process_single_image(self, image_path):
        """Process single image through pipeline"""
        # Read and enhance
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        enhanced = self.enhancer.enhance(image)
        base_name = os.path.basename(image_path)
        output_path = os.path.join("data/processed_data", f"enhanced_{base_name}")
        #enhanced.save(output_path)
        Image.fromarray(enhanced).save(output_path)
        # Generate augmentations
        augmented = self.augmentor.generate_augmentations(enhanced)
        
        # Extract features
        features = []
        # Original image features
        features.append(self.feature_extractor.extract_features(enhanced))
        # Augmented images features
        for aug_img in augmented:
            features.append(self.feature_extractor.extract_features(aug_img))
        
        return features