#!/usr/bin/env python3
"""
Environment setup script for Melanoma DIP Engine.
This script helps resolve NumPy/OpenCV compatibility issues.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error during {description}: {e}")
        return False

def check_numpy_version():
    """Check current NumPy version"""
    try:
        import numpy as np
        version = np.__version__
        print(f"📊 Current NumPy version: {version}")
        
        if version.startswith('2.'):
            print("⚠️  NumPy 2.x detected - this may cause compatibility issues with OpenCV")
            print("🔧 Recommend downgrading to NumPy 1.x for compatibility")
            return False
        elif version.startswith('1.'):
            print("✅ NumPy 1.x detected - compatible with OpenCV")
            return True
        else:
            print(f"❓ Unknown NumPy version: {version}")
            return False
    except ImportError:
        print("❌ NumPy not installed")
        return False

def fix_environment():
    """Fix NumPy/OpenCV compatibility issues"""
    print("🔧 Fixing NumPy/OpenCV compatibility issues...")
    
    # Check current NumPy version
    numpy_ok = check_numpy_version()
    
    if not numpy_ok:
        print("\n🔄 Attempting to fix compatibility issues...")
        
        # Uninstall current packages
        commands = [
            ("pip uninstall -y numpy opencv-python opencv-contrib-python", "Uninstalling incompatible packages"),
            ("pip install 'numpy>=1.21.0,<2.0.0'", "Installing compatible NumPy version"),
            ("pip install 'opencv-python>=4.8.0'", "Installing compatible OpenCV version"),
            ("pip install scikit-image scikit-learn matplotlib scipy jupyterlab", "Installing other dependencies")
        ]
        
        for command, description in commands:
            if not run_command(command, description):
                print(f"⚠️  {description} failed - you may need to run this manually")
        
        print("\n🔄 Verifying installation...")
        if check_numpy_version():
            print("✅ Environment setup completed successfully!")
            return True
        else:
            print("❌ Environment setup failed - manual intervention required")
            return False
    else:
        print("✅ Environment is already compatible!")
        return True

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🧪 Testing module imports...")
    
    modules = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('skimage', 'Scikit-image'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('scipy', 'SciPy')
    ]
    
    all_ok = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            all_ok = False
    
    return all_ok

def main():
    """Main setup function"""
    print("🔬 Melanoma DIP Engine - Environment Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('config.py'):
        print("❌ Please run this script from the melanoma_dip_engine directory")
        return
    
    # Fix environment
    if fix_environment():
        # Test imports
        if test_imports():
            print("\n🎉 Environment setup completed successfully!")
            print("✅ You can now run the PH2 pipeline test:")
            print("   python test_ph2_pipeline.py")
            print("\n✅ Or use the Jupyter notebook:")
            print("   jupyter lab test_and_visualize.ipynb")
        else:
            print("\n❌ Some modules failed to import - check error messages above")
    else:
        print("\n❌ Environment setup failed")
        print("\n🔧 Manual fix instructions:")
        print("1. pip uninstall -y numpy opencv-python")
        print("2. pip install 'numpy>=1.21.0,<2.0.0'")
        print("3. pip install 'opencv-python>=4.8.0'")
        print("4. pip install -r requirements.txt")

if __name__ == "__main__":
    main()
