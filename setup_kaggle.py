"""
Setup script for Kaggle environment
Run this after cloning the repo to Kaggle to extract and prepare data
"""

import subprocess
from pathlib import Path
import sys
def ensure_zstd_installed():
    """Ensure zstd is installed on Kaggle for reference maps extraction."""
    import subprocess
    import shutil
    
    # Check if zstd is already available
    if shutil.which('zstd'):
        print("✓ zstd is already installed")
        return True
    
    print("⚙ Installing zstd...")
    try:
        # Update apt
        subprocess.run(['apt-get', 'update', '-qq'], 
                      check=True, 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        
        # Install zstd
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'zstd'], 
                      check=True,
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        
        print("✓ zstd installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install zstd: {e}")
        return False
    except Exception as e:
        print(f"✗ Error installing zstd: {e}")
        return False

def extract_reference_maps():
    """Extract reference maps on Kaggle."""
    print("\n" + "="*60)
    print("Extracting Reference Maps on Kaggle")
    print("="*60)
    
    archive_path = Path("/kaggle/input/bigearthnet-s2-referencesmap/Reference_Maps.tar.zst")
    extract_dir = Path("/kaggle/working/data")
    
    if not archive_path.exists():
        print(f"✗ Archive not found: {archive_path}")
        print("  Make sure you've added 'bigearthnet-s2-referencesmap' dataset to your Kaggle notebook")
        return False
    
    # After extraction, reference maps will be at: /kaggle/working/data/Reference_Maps/
    output_path = extract_dir / "Reference_Maps"
    if output_path.exists() and any(output_path.iterdir()):
        print(f"✓ Reference maps already extracted at: {output_path}")
        return True
    
    print(f"Extracting to {extract_dir}...")
    print("This may take several minutes...")
    
    try:
        # Create extract directory
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract using tar with zstd
        subprocess.run([
            'tar',
            '-I', 'zstd',
            '-xf', str(archive_path),
            '-C', str(extract_dir)
        ], check=True)
        
        print(f"✓ Successfully extracted to: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error extracting archive: {e}")
        return False
    except FileNotFoundError:
        print("✗ 'tar' or 'zstd' command not found")
        # Try installing zstd
        print("  Attempting to install zstd...")
        try:
            subprocess.run(['apt-get', 'install', '-y', 'zstd'], check=True)
            print("  ✓ zstd installed, retrying extraction...")
            return extract_reference_maps()  # Retry
        except:
            print("  ✗ Could not install zstd")
            return False


def check_kaggle_datasets():
    """Check if all required Kaggle datasets are available."""
    print("\n" + "="*60)
    print("Checking Kaggle Datasets")
    print("="*60)
    
    required_datasets = {
        'BigEarthNet S2 (parts 0-5)': [
            '/kaggle/input/bigearthnetv2-s2-0',
            '/kaggle/input/bigearthnetv2-s2-1',
            '/kaggle/input/bigearthnetv2-s2-2',
            '/kaggle/input/bigearthnetv2-s2-3',
            '/kaggle/input/bigearthnetv2-s2-4',
            '/kaggle/input/bigearthnetv2-s2-5',
        ],
        'BigEarthNet Metadata': ['/kaggle/input/bigearthnet-s2-metadata'],
        'BigEarthNet Reference Maps': ['/kaggle/input/bigearthnet-s2-referencesmap'],
        'EuroSAT RGB': ['/kaggle/input/rgbeurosat', '/kaggle/input/eurosat'],
    }
    
    all_good = True
    
    for name, paths in required_datasets.items():
        found = False
        for path in paths:
            if Path(path).exists():
                print(f"  ✓ {name}: {path}")
                found = True
                break
        
        if not found:
            print(f"  ✗ {name}: Not found")
            print(f"     Expected paths: {paths}")
            all_good = False
    
    print("\n" + "="*60)
    if all_good:
        print("✓ All datasets are available!")
    else:
        print("✗ Some datasets are missing.")
        print("\nTo add datasets to your Kaggle notebook:")
        print("  1. Click 'Add data' in the right panel")
        print("  2. Search and add:")
        print("     - BigEarthNet v2.0 (S2) parts 0-5")
        print("     - BigEarthNet S2 Metadata")
        print("     - BigEarthNet S2 Reference Maps")
        print("     - EuroSAT RGB")
    print("="*60)
    
    return all_good


def setup_environment():
    """Setup Kaggle environment."""
    print("\n" + "="*70)
    print("Setting up Kaggle Environment for Land Cover Segmentation")
    print("="*70)
    
    # Check if running on Kaggle
    if not Path('/kaggle/input').exists():
        print("✗ This script should only be run on Kaggle!")
        print("  For local setup, use utils.py instead.")
        sys.exit(1)
    
    print("✓ Running on Kaggle")
    
    # Ensure zstd is installed
    zstd_ok = ensure_zstd_installed()
    if not zstd_ok:
        print("\n⚠ Failed to install zstd. Reference maps extraction will fail.")
        sys.exit(1)
    
    # Check datasets
    datasets_ok = check_kaggle_datasets()
    
    if not datasets_ok:
        print("\n⚠ Please add missing datasets before continuing.")
        sys.exit(1)
    
    # Extract reference maps
    print("\nExtracting reference maps...")
    maps_ok = extract_reference_maps()
    
    if not maps_ok:
        print("\n⚠ Failed to extract reference maps. Training may fail.")
    
    # Check Python packages
    print("\nChecking Python packages...")
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 'rasterio',
        'albumentations', 'matplotlib', 'seaborn', 'sklearn', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("  Installing missing packages...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--quiet'
        ] + missing_packages)
    
    # Summary
    print("\n" + "="*70)
    print("Setup Complete!")
    print("="*70)
    print("\nYou can now run training:")
    print("  Stage 1 (EuroSAT): !python train_stage1.py --epochs 50")
    print("  Stage 2 (U-Net):   !python train_stage2.py --epochs 50")
    print("\nOutputs will be saved to /kaggle/working/outputs/")
    print("="*70 + "\n")


if __name__ == '__main__':
    setup_environment()

