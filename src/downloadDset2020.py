import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract_kaggle_dataset(dataset, download_path):
    os.makedirs(download_path, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    zip_path = os.path.join(download_path, 'brian-tumor-dataset.zip')
    print(f"Downloading {dataset} to {zip_path}...")
    api.dataset_download_files(dataset, path=download_path, quiet=False)
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    os.remove(zip_path)
    print("Download and extraction complete.")

if __name__ == "__main__":
    download_and_extract_kaggle_dataset(
        dataset="awsaf49/brats20-dataset-training-validation",
        download_path="data/brian-tumor-dataset"
    )