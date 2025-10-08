"""
Automated Installation Script for Melanoma DIP Engine
Installs dependencies in the correct order to avoid conflicts.
"""

import subprocess
import sys
import platform
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"SUCCESS: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {description}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and 8 <= version.minor < 13:
        print(f"OK: Python {version.major}.{version.minor} is compatible")
        return True
    else:
        print(f"ERROR: Python {version.major}.{version.minor} is not compatible")
        print("  Required: Python 3.8 - 3.12")
        return False

def check_cuda():
    """Check if CUDA is available."""
    try:
        result = subprocess.run(
            "nvcc --version",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("OK: CUDA is available")
            print(result.stdout)
            return True
        else:
            print("WARNING: CUDA not found - will install CPU-only PyTorch")
            return False
    except:
        print("WARNING: CUDA not found - will install CPU-only PyTorch")
        return False

def main():
    """Main installation process."""
    print("\n" + "="*80)
    print("MELANOMA DIP ENGINE - DEPENDENCY INSTALLATION")
    print("="*80)
    
    # Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        sys.exit(1)
    
    # Check CUDA
    print("\n2. Checking CUDA availability...")
    has_cuda = check_cuda()
    
    # Upgrade pip
    print("\n3. Upgrading pip...")
    if not run_command(
        f"{sys.executable} -m pip install --upgrade pip setuptools wheel",
        "Upgrading pip, setuptools, and wheel"
    ):
        print("WARNING: pip upgrade failed, continuing anyway...")
    
    # Install PyTorch
    print("\n4. Installing PyTorch...")
    if has_cuda:
        pytorch_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        description = "Installing PyTorch with CUDA 11.8 support"
    else:
        pytorch_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio"
        description = "Installing PyTorch (CPU only)"
    
    if not run_command(pytorch_cmd, description):
        print("ERROR: PyTorch installation failed!")
        sys.exit(1)
    
    # Install Detectron2 dependencies first
    print("\n5. Installing Detectron2 dependencies...")
    detectron2_deps = [
        "fvcore",
        "iopath",
        "omegaconf",
        "hydra-core",
        "pycocotools"
    ]
    
    for dep in detectron2_deps:
        if not run_command(
            f"{sys.executable} -m pip install {dep}",
            f"Installing {dep}"
        ):
            print(f"WARNING: {dep} installation failed, continuing anyway...")
    
    # Install Detectron2
    print("\n6. Installing Detectron2...")
    
    # Different installation methods based on platform
    system = platform.system()
    
    if system == "Windows":
        print("Detected Windows - using source installation")
        detectron2_cmd = f"{sys.executable} -m pip install git+https://github.com/facebookresearch/detectron2.git"
    elif system == "Linux":
        print("Detected Linux - attempting pre-built wheel")
        detectron2_cmd = f"{sys.executable} -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html"
    else:  # macOS or others
        print(f"Detected {system} - using source installation")
        detectron2_cmd = f"{sys.executable} -m pip install git+https://github.com/facebookresearch/detectron2.git"
    
    if not run_command(detectron2_cmd, "Installing Detectron2"):
        print("\n" + "="*80)
        print("WARNING: DETECTRON2 INSTALLATION FAILED")
        print("="*80)
        print("\nPossible solutions:")
        print("1. Install Visual Studio Build Tools (Windows)")
        print("2. Install gcc/g++ compiler (Linux/Mac)")
        print("3. Try manual installation:")
        print("   git clone https://github.com/facebookresearch/detectron2.git")
        print("   cd detectron2")
        print("   pip install -e .")
        print("\nContinuing with other dependencies...")
    
    # Install remaining requirements
    print("\n7. Installing remaining dependencies...")
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing remaining requirements"
    ):
        print("WARNING: Some dependencies may have failed")
    
    # Verify installation
    print("\n8. Verifying installation...")
    print("\nChecking critical packages:")
    
    packages_to_check = [
        "torch",
        "torchvision", 
        "cv2",
        "numpy",
        "transformers",
        "timm",
        "albumentations",
        "matplotlib"
    ]
    
    all_ok = True
    for package in packages_to_check:
        try:
            if package == "cv2":
                __import__("cv2")
            else:
                __import__(package)
            print(f"  OK: {package}")
        except ImportError:
            print(f"  ERROR: {package} - NOT FOUND")
            all_ok = False
    
    # Check Detectron2 separately
    try:
        import detectron2
        print(f"  OK: detectron2 (version: {detectron2.__version__})")
    except ImportError:
        print(f"  ERROR: detectron2 - NOT FOUND (may need manual installation)")
        all_ok = False
    
    # Final summary
    print("\n" + "="*80)
    if all_ok:
        print("SUCCESS: INSTALLATION COMPLETE!")
        print("="*80)
        print("\nYou can now:")
        print("1. Open train_segmentation_model.ipynb")
        print("2. Run the training pipeline")
        print("3. Start training your model!")
    else:
        print("WARNING: INSTALLATION COMPLETED WITH WARNINGS")
        print("="*80)
        print("\nSome packages failed to install.")
        print("Please check the errors above and install missing packages manually.")
        print("\nFor Detectron2 issues, see:")
        print("https://detectron2.readthedocs.io/en/latest/tutorials/install.html")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

