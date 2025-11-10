"""
Utility script for common tasks like extracting archives, checking data, etc.
"""

import argparse
import subprocess
from pathlib import Path
import sys

from config import (
    DATA_DIR, REFERENCE_MAPS_ARCHIVE, REFERENCE_MAPS_FOLDER,
    METADATA_PATH, BIGEARTHNET_FOLDERS, EUROSAT_PATH
)


def extract_reference_maps():
    """Extract reference maps archive."""
    print("\n" + "="*60)
    print("Extracting Reference Maps")
    print("="*60)
    
    if not REFERENCE_MAPS_ARCHIVE.exists():
        print(f"✗ Archive not found: {REFERENCE_MAPS_ARCHIVE}")
        return False
    
    if REFERENCE_MAPS_FOLDER.exists():
        print(f"✓ Reference maps already extracted at: {REFERENCE_MAPS_FOLDER}")
        return True
    
    print(f"Extracting {REFERENCE_MAPS_ARCHIVE}...")
    print("This may take several minutes...")
    
    try:
        # Use tar with zstd compression
        subprocess.run([
            'tar',
            '-I', 'zstd',
            '-xf', str(REFERENCE_MAPS_ARCHIVE),
            '-C', str(DATA_DIR)
        ], check=True)
        
        print(f"✓ Successfully extracted to: {REFERENCE_MAPS_FOLDER}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error extracting archive: {e}")
        print("\nMake sure 'zstd' is installed:")
        print("  Ubuntu/Debian: sudo apt-get install zstd")
        print("  macOS: brew install zstd")
        print("  Windows: Download from https://github.com/facebook/zstd/releases")
        return False
    except FileNotFoundError:
        print("✗ 'tar' or 'zstd' command not found")
        return False


def check_data():
    """Check if all required data is available."""
    print("\n" + "="*60)
    print("Checking Data Availability")
    print("="*60)
    
    all_good = True
    
    # Check EuroSAT
    print("\n1. EuroSAT Dataset:")
    if EUROSAT_PATH.exists():
        train_dir = EUROSAT_PATH / 'train'
        val_dir = EUROSAT_PATH / 'val'
        test_dir = EUROSAT_PATH / 'test'
        
        if train_dir.exists() and val_dir.exists() and test_dir.exists():
            print(f"   ✓ Found at: {EUROSAT_PATH}")
            
            # Count samples
            num_train = sum(1 for _ in train_dir.rglob('*.jpg'))
            num_val = sum(1 for _ in val_dir.rglob('*.jpg'))
            num_test = sum(1 for _ in test_dir.rglob('*.jpg'))
            
            print(f"   ✓ Train: {num_train} images")
            print(f"   ✓ Val: {num_val} images")
            print(f"   ✓ Test: {num_test} images")
        else:
            print(f"   ✗ Incomplete structure at: {EUROSAT_PATH}")
            all_good = False
    else:
        print(f"   ✗ Not found at: {EUROSAT_PATH}")
        print(f"   → Download from: https://github.com/phelber/eurosat")
        all_good = False
    
    # Check BigEarthNet metadata
    print("\n2. BigEarthNet Metadata:")
    if METADATA_PATH.exists():
        import pandas as pd
        try:
            df = pd.read_parquet(METADATA_PATH)
            print(f"   ✓ Found at: {METADATA_PATH}")
            print(f"   ✓ Total patches: {len(df):,}")
        except Exception as e:
            print(f"   ✗ Error reading metadata: {e}")
            all_good = False
    else:
        print(f"   ✗ Not found at: {METADATA_PATH}")
        all_good = False
    
    # Check Reference Maps
    print("\n3. Reference Maps:")
    if REFERENCE_MAPS_FOLDER.exists():
        # Count reference map files
        num_maps = sum(1 for _ in REFERENCE_MAPS_FOLDER.rglob('*_reference_map.tif'))
        print(f"   ✓ Found at: {REFERENCE_MAPS_FOLDER}")
        print(f"   ✓ Total maps: {num_maps:,}")
    else:
        print(f"   ✗ Not found at: {REFERENCE_MAPS_FOLDER}")
        if REFERENCE_MAPS_ARCHIVE.exists():
            print(f"   → Run: python utils.py extract")
        else:
            print(f"   → Download Reference_Maps.tar.zst")
        all_good = False
    
    # Check BigEarthNet folders
    print("\n4. BigEarthNet Image Folders:")
    found_folders = 0
    for i, folder in enumerate(BIGEARTHNET_FOLDERS):
        if folder.exists():
            num_patches = sum(1 for _ in folder.rglob('*_B02.tif'))
            print(f"   ✓ Folder {i}: {folder} ({num_patches:,} patches)")
            found_folders += 1
        else:
            print(f"   ✗ Folder {i}: {folder}")
    
    if found_folders == 0:
        print(f"   ✗ No BigEarthNet folders found")
        all_good = False
    elif found_folders < len(BIGEARTHNET_FOLDERS):
        print(f"   ⚠ Only {found_folders}/{len(BIGEARTHNET_FOLDERS)} folders found")
        print(f"   → You can still train with available data")
    
    # Summary
    print("\n" + "="*60)
    if all_good:
        print("✓ All data is available! You're ready to train.")
    else:
        print("✗ Some data is missing. Please download and prepare the datasets.")
    print("="*60 + "\n")
    
    return all_good


def check_environment():
    """Check Python environment and dependencies."""
    print("\n" + "="*60)
    print("Checking Environment")
    print("="*60)
    
    # Python version
    print(f"\nPython version: {sys.version}")
    
    # Check imports
    packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 'rasterio',
        'albumentations', 'matplotlib', 'seaborn', 'sklearn', 'tqdm'
    ]
    
    print("\nRequired packages:")
    all_installed = True
    
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {package:20s} {version}")
        except ImportError:
            print(f"  ✗ {package:20s} NOT INSTALLED")
            all_installed = False
    
    # Check CUDA
    try:
        import torch
        print(f"\nCUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - {torch.cuda.get_device_name(i)}")
    except:
        pass
    
    print("\n" + "="*60)
    if all_installed:
        print("✓ All required packages are installed!")
    else:
        print("✗ Some packages are missing. Run: pip install -r requirements.txt")
    print("="*60 + "\n")
    
    return all_installed


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Utility script for data preparation and environment checks'
    )
    parser.add_argument(
        'command',
        choices=['extract', 'check-data', 'check-env', 'check-all'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        extract_reference_maps()
    
    elif args.command == 'check-data':
        check_data()
    
    elif args.command == 'check-env':
        check_environment()
    
    elif args.command == 'check-all':
        env_ok = check_environment()
        data_ok = check_data()
        
        if env_ok and data_ok:
            print("\n🎉 Everything is ready! You can start training now.")
            print("\nNext steps:")
            print("  1. python train_stage1.py  # Pre-train encoder")
            print("  2. python train_stage2.py  # Train U-Net")
        else:
            print("\n⚠ Please fix the issues above before training.")


if __name__ == '__main__':
    main()

