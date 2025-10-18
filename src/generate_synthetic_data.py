import os
import cv2
import numpy as np
from pathlib import Path

def generate_synthetic_data(image_dir, output_dir, class_id=0):
    """
    Generate synthetic YOLO labels and segmentation masks for a binary dataset.
    
    Args:
        image_dir (str): Path to the directory containing "yes" and "no" folders.
        output_dir (str): Path to save the generated labels and masks.
        class_id (int): Class ID for the objects (default: 0 for tumors).
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_dir = output_dir / "yolo_labels"
    mask_dir = output_dir / "masks"
    label_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for class_folder in ["yes", "no"]:
        class_path = image_dir / class_folder
        for image_path in class_path.glob("*.jpg"):
            image = cv2.imread(str(image_path))
            height, width, _ = image.shape

            # Generate YOLO labels and masks
            label_path = label_dir / f"{image_path.stem}.txt"
            mask_path = mask_dir / f"{image_path.stem}.png"

            if class_folder == "yes":
                # Create a synthetic bounding box (centered, 50% of image size)
                x_center, y_center = 0.5, 0.5
                box_width, box_height = 0.5, 0.5
                with open(label_path, "w") as f:
                    f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

                # Create a synthetic mask (white rectangle in the center)
                mask = np.zeros((height, width), dtype=np.uint8)
                x1, y1 = int(width * 0.25), int(height * 0.25)
                x2, y2 = int(width * 0.75), int(height * 0.75)
                mask[y1:y2, x1:x2] = 255
                cv2.imwrite(str(mask_path), mask)
            else:
                # No bounding box for "no" class
                open(label_path, "w").close()

                # Create an empty mask
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.imwrite(str(mask_path), mask)

            print(f"Processed: {image_path.name}")

if __name__ == "__main__":
    # Paths to input and output directories
    image_dir = "data/sample_dataset"
    output_dir = "data/processed_dataset"

    # Generate synthetic data
    generate_synthetic_data(image_dir, output_dir)