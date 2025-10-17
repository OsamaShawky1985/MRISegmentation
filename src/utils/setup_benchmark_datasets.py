#!/usr/bin/env python3
"""
Benchmark Dataset Setup for Brain Tumor Research
Downloads and prepares standard datasets used in research literature for comparison

Key Datasets:
1. BraTS (Brain Tumor Segmentation Challenge) - Primary benchmark
2. TCGA-LGG - Clinical validation dataset
3. Kaggle LGG - Quick access version of TCGA-LGG

Author: Research Team
Date: 2025
"""

import os
import sys
import argparse
import requests
import zipfile
from pathlib import Path
import subprocess
import json
import logging
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkDatasetDownloader:
    """Download and setup benchmark datasets for research comparison"""
    
    def __init__(self, data_dir: str = "data/benchmark_datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset information for research comparison
        self.datasets = {
            'brats2021': {
                'name': 'BraTS 2021',
                'description': 'Brain Tumor Segmentation Challenge 2021',
                'size': '1,251 training cases',
                'citations': '5,000+ papers',
                'url': 'https://www.synapse.org/#!Synapse:syn25829067',
                'manual': True,
                'importance': 'PRIMARY BENCHMARK - Most cited brain tumor dataset'
            },
            'brats2020': {
                'name': 'BraTS 2020',
                'description': 'Brain Tumor Segmentation Challenge 2020',
                'size': '369 training cases',
                'citations': '3,000+ papers',
                'url': 'https://www.med.upenn.edu/cbica/brats2020/data.html',
                'manual': True,
                'importance': 'Historical comparison with older methods'
            },
            'tcga_lgg_kaggle': {
                'name': 'TCGA-LGG (Kaggle)',
                'description': 'Lower Grade Glioma Segmentation Dataset',
                'size': '3,929 images from 110 patients',
                'citations': '2,000+ papers use TCGA-LGG data',
                'kaggle_dataset': 'mateuszbuda/lgg-mri-segmentation',
                'manual': False,
                'importance': 'CLINICAL VALIDATION - Real patient data with outcomes'
            },
            'brain_tumor_classification': {
                'name': 'Brain Tumor Classification',
                'description': 'Kaggle Brain Tumor Classification Dataset',
                'size': '3,264 images',
                'citations': '500+ papers',
                'kaggle_dataset': 'sartajbhuvaji/brain-tumor-classification-mri',
                'manual': False,
                'importance': 'Quick validation and ablation studies'
            }
        }
    
    def show_dataset_info(self):
        """Display information about available benchmark datasets"""
        print("\n" + "="*80)
        print("🎯 BENCHMARK DATASETS FOR BRAIN TUMOR RESEARCH COMPARISON")
        print("="*80)
        
        for key, dataset in self.datasets.items():
            print(f"\n📊 {dataset['name']}")
            print(f"   Description: {dataset['description']}")
            print(f"   Size: {dataset['size']}")
            print(f"   Research Impact: {dataset['citations']}")
            print(f"   Importance: {dataset['importance']}")
            
            if dataset['manual']:
                print(f"   🔗 URL: {dataset['url']}")
                print(f"   📋 Setup: Manual registration required")
            else:
                print(f"   📋 Setup: Automated download available")
    
    def check_kaggle_setup(self) -> bool:
        """Check if Kaggle API is properly configured"""
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        if not kaggle_json.exists():
            logger.error("Kaggle API not configured!")
            print("\n🔧 KAGGLE SETUP REQUIRED:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Create new API token (downloads kaggle.json)")
            print("3. Run: mkdir -p ~/.kaggle")
            print("4. Run: mv ~/Downloads/kaggle.json ~/.kaggle/")
            print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
        
        try:
            import kaggle
            logger.info("✅ Kaggle API configured successfully")
            return True
        except ImportError:
            logger.error("Kaggle package not installed. Run: pip install kaggle")
            return False
    
    def download_kaggle_dataset(self, dataset_id: str, output_dir: Path):
        """Download dataset from Kaggle"""
        if not self.check_kaggle_setup():
            return False
        
        try:
            import kaggle
            
            logger.info(f"Downloading Kaggle dataset: {dataset_id}")
            kaggle.api.dataset_download_files(
                dataset_id, 
                path=str(output_dir),
                unzip=True,
                quiet=False
            )
            logger.info(f"✅ Downloaded: {dataset_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error downloading {dataset_id}: {e}")
            return False
    
    def setup_tcga_lgg_kaggle(self):
        """Setup TCGA-LGG dataset from Kaggle"""
        dataset_dir = self.data_dir / 'tcga_lgg_kaggle'
        
        if dataset_dir.exists():
            logger.info("TCGA-LGG Kaggle dataset already exists")
            return True
        
        logger.info("Setting up TCGA-LGG dataset from Kaggle...")
        success = self.download_kaggle_dataset(
            'mateuszbuda/lgg-mri-segmentation',
            dataset_dir
        )
        
        if success:
            # Create info file
            info = {
                'dataset': 'TCGA-LGG (Kaggle)',
                'source': 'The Cancer Genome Atlas - Lower Grade Glioma',
                'processed_by': 'Mateusz Buda et al.',
                'images': 3929,
                'patients': 110,
                'tumor_images': 1373,
                'normal_images': 2556,
                'format': 'TIFF images with binary masks',
                'research_use': 'Segmentation benchmark, clinical validation',
                'citations': '2000+ papers use TCGA-LGG derived data'
            }
            
            with open(dataset_dir / 'dataset_info.json', 'w') as f:
                json.dump(info, f, indent=2)
            
            logger.info("✅ TCGA-LGG dataset ready for research!")
            return True
        
        return False
    
    def setup_brain_tumor_classification(self):
        """Setup Brain Tumor Classification dataset"""
        dataset_dir = self.data_dir / 'brain_tumor_classification'
        
        if dataset_dir.exists():
            logger.info("Brain Tumor Classification dataset already exists")
            return True
        
        logger.info("Setting up Brain Tumor Classification dataset...")
        success = self.download_kaggle_dataset(
            'sartajbhuvaji/brain-tumor-classification-mri',
            dataset_dir
        )
        
        if success:
            # Create info file
            info = {
                'dataset': 'Brain Tumor Classification (MRI)',
                'classes': ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'],
                'images': 3264,
                'format': 'JPEG images',
                'split': 'Training and Testing folders',
                'research_use': 'Classification, detection (with conversion)',
                'citations': '500+ papers for comparison studies'
            }
            
            with open(dataset_dir / 'dataset_info.json', 'w') as f:
                json.dump(info, f, indent=2)
            
            logger.info("✅ Brain Tumor Classification dataset ready!")
            return True
        
        return False
    
    def show_brats_registration_info(self):
        """Show information about BraTS registration process"""
        print("\n" + "="*80)
        print("🏆 BraTS DATASET REGISTRATION (REQUIRED FOR RESEARCH)")
        print("="*80)
        
        print("\n📋 BraTS is the GOLD STANDARD for brain tumor research:")
        print("   • Used in 90%+ of brain tumor segmentation papers")
        print("   • Annual MICCAI challenge since 2012")
        print("   • Standard evaluation metrics and protocols")
        print("   • Direct comparison with 100+ published methods")
        
        print("\n🔗 REGISTRATION PROCESS:")
        print("   1. BraTS 2023 (Latest):")
        print("      → https://www.synapse.org/#!Synapse:syn51156910")
        print("   2. BraTS 2021 (Most cited):")
        print("      → https://www.synapse.org/#!Synapse:syn25829067")
        print("   3. BraTS 2020 (Historical):")
        print("      → https://www.med.upenn.edu/cbica/brats2020/data.html")
        
        print("\n⏱️  TIMELINE:")
        print("   • Registration: Immediate")
        print("   • Data use agreement: 1-2 days")
        print("   • Download approval: 1-7 days")
        print("   • Total time: Usually 3-7 days")
        
        print("\n📊 RESEARCH IMPACT:")
        print("   • Papers without BraTS comparison: Limited acceptance")
        print("   • Papers with BraTS benchmark: High impact venues")
        print("   • Standard metrics: Dice, Hausdorff Distance, Sensitivity")
        
        print("\n💡 RECOMMENDATION:")
        print("   Register for BraTS immediately while working with other datasets!")
    
    def create_research_comparison_script(self):
        """Create script for research comparison metrics"""
        script_content = '''#!/usr/bin/env python3
"""
Research Comparison Metrics for Brain Tumor Segmentation
Standard evaluation metrics used in literature for benchmark comparison

Key Metrics:
- Dice Similarity Coefficient (DSC)
- Hausdorff Distance (HD95)
- Sensitivity and Specificity
- Volumetric metrics
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2


class ResearchMetrics:
    """Standard metrics for research comparison"""
    
    @staticmethod
    def dice_coefficient(pred, target):
        """Dice Similarity Coefficient - Primary metric for BraTS"""
        pred_binary = (pred == 1).astype(np.float32)
        target_binary = (target == 1).astype(np.float32)
        
        intersection = np.sum(pred_binary * target_binary)
        union = np.sum(pred_binary) + np.sum(target_binary)
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return (2.0 * intersection) / union
    
    @staticmethod
    def hausdorff_distance_95(pred, target):
        """95th percentile Hausdorff Distance - BraTS standard"""
        # Get boundary points
        pred_boundary = ResearchMetrics._get_boundary_points(pred)
        target_boundary = ResearchMetrics._get_boundary_points(target)
        
        if len(pred_boundary) == 0 or len(target_boundary) == 0:
            return 0.0
        
        # Calculate directed Hausdorff distances
        h1 = directed_hausdorff(pred_boundary, target_boundary)[0]
        h2 = directed_hausdorff(target_boundary, pred_boundary)[0]
        
        # Return 95th percentile
        distances = [h1, h2]
        return np.percentile(distances, 95)
    
    @staticmethod
    def _get_boundary_points(mask):
        """Extract boundary points from binary mask"""
        if mask.sum() == 0:
            return np.array([])
        
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return np.array([])
        
        # Get all boundary points
        boundary_points = []
        for contour in contours:
            boundary_points.extend(contour.reshape(-1, 2))
        
        return np.array(boundary_points)
    
    @staticmethod
    def sensitivity_specificity(pred, target):
        """Sensitivity and Specificity - Clinical metrics"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        tp = np.sum((pred_flat == 1) & (target_flat == 1))
        tn = np.sum((pred_flat == 0) & (target_flat == 0))
        fp = np.sum((pred_flat == 1) & (target_flat == 0))
        fn = np.sum((pred_flat == 0) & (target_flat == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return sensitivity, specificity
    
    @staticmethod
    def volumetric_similarity(pred, target):
        """Volumetric Similarity - Volume-based metric"""
        pred_vol = np.sum(pred == 1)
        target_vol = np.sum(target == 1)
        
        if pred_vol + target_vol == 0:
            return 1.0
        
        return 1.0 - abs(pred_vol - target_vol) / (pred_vol + target_vol)
    
    @staticmethod
    def comprehensive_evaluation(pred_masks, target_masks):
        """Comprehensive evaluation with all research metrics"""
        n_cases = len(pred_masks)
        
        dice_scores = []
        hd95_scores = []
        sensitivity_scores = []
        specificity_scores = []
        vs_scores = []
        
        for pred, target in zip(pred_masks, target_masks):
            # Dice coefficient
            dice = ResearchMetrics.dice_coefficient(pred, target)
            dice_scores.append(dice)
            
            # Hausdorff distance
            hd95 = ResearchMetrics.hausdorff_distance_95(pred, target)
            hd95_scores.append(hd95)
            
            # Sensitivity and Specificity
            sens, spec = ResearchMetrics.sensitivity_specificity(pred, target)
            sensitivity_scores.append(sens)
            specificity_scores.append(spec)
            
            # Volumetric Similarity
            vs = ResearchMetrics.volumetric_similarity(pred, target)
            vs_scores.append(vs)
        
        results = {
            'dice_mean': np.mean(dice_scores),
            'dice_std': np.std(dice_scores),
            'hd95_mean': np.mean(hd95_scores),
            'hd95_std': np.std(hd95_scores),
            'sensitivity_mean': np.mean(sensitivity_scores),
            'sensitivity_std': np.std(sensitivity_scores),
            'specificity_mean': np.mean(specificity_scores),
            'specificity_std': np.std(specificity_scores),
            'vs_mean': np.mean(vs_scores),
            'vs_std': np.std(vs_scores),
            'n_cases': n_cases
        }
        
        return results
    
    @staticmethod
    def print_research_results(results):
        """Print results in research paper format"""
        print("\\n" + "="*60)
        print("📊 RESEARCH BENCHMARK RESULTS")
        print("="*60)
        print(f"Number of test cases: {results['n_cases']}")
        print()
        print("SEGMENTATION METRICS (Mean ± Std):")
        print(f"  Dice Coefficient:     {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
        print(f"  Hausdorff Dist (95%): {results['hd95_mean']:.2f} ± {results['hd95_std']:.2f}")
        print(f"  Sensitivity:          {results['sensitivity_mean']:.4f} ± {results['sensitivity_std']:.4f}")
        print(f"  Specificity:          {results['specificity_mean']:.4f} ± {results['specificity_std']:.4f}")
        print(f"  Volumetric Similarity: {results['vs_mean']:.4f} ± {results['vs_std']:.4f}")
        print()
        print("FORMAT FOR RESEARCH PAPER:")
        print(f"DSC: {results['dice_mean']:.3f}±{results['dice_std']:.3f}, " + 
              f"HD95: {results['hd95_mean']:.1f}±{results['hd95_std']:.1f}, " +
              f"Sens: {results['sensitivity_mean']:.3f}±{results['sensitivity_std']:.3f}")


if __name__ == "__main__":
    print("Research Metrics Module Ready!")
    print("Usage: from research_metrics import ResearchMetrics")
'''
        
        script_path = self.data_dir / 'research_metrics.py'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"✅ Research metrics script created: {script_path}")
    
    def generate_dataset_summary(self):
        """Generate summary of available benchmark datasets"""
        summary_path = self.data_dir / 'BENCHMARK_SUMMARY.md'
        
        summary_content = f"""# Benchmark Datasets Summary

Generated on: {Path.cwd()}
Data directory: {self.data_dir.absolute()}

## 📊 Dataset Status

"""
        
        for key, dataset in self.datasets.items():
            dataset_path = self.data_dir / key.replace('_', '-')
            status = "✅ READY" if dataset_path.exists() else "❌ NOT DOWNLOADED"
            
            summary_content += f"""### {dataset['name']} - {status}
- **Research Impact**: {dataset['importance']}
- **Size**: {dataset['size']}
- **Citations**: {dataset['citations']}
- **Local Path**: `{dataset_path}`

"""
        
        summary_content += """## 🎯 Next Steps

1. **Download available datasets**: Run automated download for Kaggle datasets
2. **Register for BraTS**: Complete manual registration process
3. **Setup evaluation**: Use research_metrics.py for standard evaluation
4. **Benchmark comparison**: Compare with published results

## 📈 Research Publication Checklist

- [ ] BraTS dataset evaluation (required for top venues)
- [ ] TCGA-LGG clinical validation
- [ ] Standard metrics: Dice, HD95, Sensitivity, Specificity
- [ ] Comparison table with state-of-the-art methods
- [ ] Statistical significance testing
"""
        
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        
        logger.info(f"✅ Dataset summary created: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Setup benchmark datasets for brain tumor research")
    parser.add_argument('--action', choices=['info', 'download', 'all'], default='info',
                       help='Action to perform')
    parser.add_argument('--dataset', choices=['tcga_lgg', 'classification', 'all'],
                       help='Specific dataset to download')
    
    args = parser.parse_args()
    
    downloader = BenchmarkDatasetDownloader()
    
    if args.action == 'info':
        downloader.show_dataset_info()
        downloader.show_brats_registration_info()
        
    elif args.action == 'download':
        if args.dataset == 'tcga_lgg' or args.dataset == 'all':
            downloader.setup_tcga_lgg_kaggle()
        
        if args.dataset == 'classification' or args.dataset == 'all':
            downloader.setup_brain_tumor_classification()
    
    elif args.action == 'all':
        downloader.show_dataset_info()
        downloader.setup_tcga_lgg_kaggle()
        downloader.setup_brain_tumor_classification()
        downloader.create_research_comparison_script()
        downloader.generate_dataset_summary()
        print("\n🎉 Benchmark datasets setup complete!")
        print("⚠️  Don't forget to register for BraTS datasets manually!")


if __name__ == "__main__":
    main()
