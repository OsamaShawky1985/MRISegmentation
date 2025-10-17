import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import rotate, shift, zoom

class DataAugmentor:
    def __init__(self, config):
        self.config = config
        self.generator = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect'
        )
    
    def apply_gaussian_noise(self, image):
        """Add Gaussian noise to image"""
        mean = 0
        sigma = 0.03
        noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def generate_augmentations(self, image, num_augmentations=5):
        """Generate multiple augmented versions of an image"""
        augmented_images = []
        image = image.reshape((1,) + image.shape + (1,))
        
        # Generate augmented images
        for i, _ in enumerate(self.generator.flow(image, batch_size=1)):
            if i >= num_augmentations:
                break
            aug_img = _.squeeze()
            # Add noise to some images
            if np.random.random() > 0.5:
                aug_img = self.apply_gaussian_noise(aug_img)
            augmented_images.append(aug_img)
        
        return augmented_images