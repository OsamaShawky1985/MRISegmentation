import cv2
import numpy as np
from skimage import exposure, filters

class ImageEnhancer:
    def __init__(self, config):
        self.config = config
        self.clahe = cv2.createCLAHE(
            clipLimit=config['preprocessing']['clahe_clip_limit'],
            tileGridSize=tuple(config['preprocessing']['clahe_grid_size'])
        )
    
    def enhance(self, image):
        """Apply enhancement pipeline to image"""
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Apply CLAHE
        enhanced = self.clahe.apply(image)
        
        # Denoise
        enhanced = cv2.fastNlMeansDenoising(
            enhanced, 
            h=self.config['preprocessing']['denoise_h']
        )
        
        # Edge enhancement using unsharp masking
        gaussian = cv2.GaussianBlur(enhanced, (5,5), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # Contrast stretching
        p2, p98 = np.percentile(enhanced, (2, 98))
        enhanced = exposure.rescale_intensity(enhanced, in_range=(p2, p98))
        
        return enhanced.astype(np.uint8)