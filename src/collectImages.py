import os
import shutil
from pathlib import Path

def consolidate_folders(source_folders, destination_folder, class_name):
    """
    Consolidate images from multiple folders into a single class folder
    
    Args:
        source_folders: List of source folder paths
        destination_folder: Path to destination folder
        class_name: Name of the class (e.g., 'tumor' or 'normal')
    """
    # Create destination folder
    dest_path = Path(destination_folder) / class_name
    dest_path.mkdir(parents=True, exist_ok=True)
    
    total_files = 0
    
    for folder in source_folders:
        if not os.path.exists(folder):
            print(f"Warning: Source folder {folder} does not exist")
            continue
            
        # Copy all image files
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_file = Path(root) / file
                    # Create unique filename to avoid overwrites
                    dest_file = dest_path / f"{Path(root).name}_{file}"
                    shutil.copy2(src_file, dest_file)
                    total_files += 1
    
    print(f"Consolidated {total_files} images into {dest_path}")

# Example usage
if __name__ == "__main__":
    # List your source folders
    tumor_folders = [
        "data/NINS_Dataset/"
        # Add more folders as needed
    ]
    
    # Set destination folder
    destination = "data/NINS_DataSet_Collected"
    
    # Consolidate tumor images
    consolidate_folders(tumor_folders, destination, "tumor")