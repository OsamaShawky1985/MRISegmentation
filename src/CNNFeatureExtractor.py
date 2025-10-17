import tensorflow as tf
from tensorflow.keras.applications import VGG16
import numpy as np
import os
from tqdm import tqdm

class CNNFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.model = self._build_feature_extractor()
        
    def _build_feature_extractor(self):
        """Initialize VGG16 model for feature extraction"""
        base_model = VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
            pooling='avg'
        )
        
        # Freeze the layers
        for layer in base_model.layers:
            layer.trainable = False
            
        return base_model
    
    def extract_features(self, image):
        """Extract features from a single image"""
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
            
        # Resize if needed
        if image.shape[:2] != (224, 224):
            image = tf.image.resize(image, (224, 224))
            
        # Preprocess
        x = tf.keras.applications.vgg16.preprocess_input(image)
        x = np.expand_dims(x, axis=0)
        
        # Extract features
        features = self.model.predict(x, verbose=0)
        return features.flatten()
    
    def extract_features_batch(self, images):
        """Extract features from a batch of images (numpy array or list)"""
        # Convert list to numpy array if needed
        images = np.array(images)
        # Convert grayscale to RGB if needed
        if images.ndim == 3:  # (batch, H, W)
            images = np.stack([images] * 3, axis=-1)
        elif images.shape[-1] == 1:
            images = np.repeat(images, 3, axis=-1)
        # Resize all images to (224, 224)
        images_resized = np.array([tf.image.resize(img, (224, 224)).numpy() if img.shape[:2] != (224, 224) else img for img in images])
        # Preprocess
        x = tf.keras.applications.vgg16.preprocess_input(images_resized)
        # Extract features in batch
        features = self.model.predict(x, verbose=0)
        return features