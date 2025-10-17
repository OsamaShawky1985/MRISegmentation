import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import rotate, zoom, shift
from skimage.transform import elastic_transform

class MedicalImageAugmentor:
    def __init__(self, rotation_range=20, shift_range=0.1, zoom_range=0.1):
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        
    def add_gaussian_noise(self, image):
        """Add controlled Gaussian noise"""
        mean = 0
        sigma = 0.03 * np.mean(image)
        noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 1)
    
    def apply_elastic_deformation(self, image):
        """Apply elastic deformation - simulates tissue variability"""
        sigma = 4
        alpha = 50
        return elastic_transform(image, alpha=alpha, sigma=sigma)
    
    def simulate_intensity_variation(self, image):
        """Simulate MRI intensity variations"""
        gamma = np.random.uniform(0.8, 1.2)
        return np.power(image, gamma)
    
    def create_generator(self):
        """Create specialized medical image data generator"""
        return ImageDataGenerator(
            preprocessing_function=self.preprocessing_pipeline,
            rotation_range=self.rotation_range,
            width_shift_range=self.shift_range,
            height_shift_range=self.shift_range,
            zoom_range=self.zoom_range,
            fill_mode='reflect',
            horizontal_flip=True,
            vertical_flip=True
        )
    
    def preprocessing_pipeline(self, image):
        """Complete augmentation pipeline"""
        # Apply transformations with probability
        if np.random.random() > 0.5:
            image = self.add_gaussian_noise(image)
        if np.random.random() > 0.7:
            image = self.apply_elastic_deformation(image)
        if np.random.random() > 0.6:
            image = self.simulate_intensity_variation(image)
        return image