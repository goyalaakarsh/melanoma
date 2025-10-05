@echo off
echo 🔬 Melanoma DIP Engine - Environment Fix
echo ========================================

echo 🔄 Fixing NumPy/OpenCV compatibility issues...

echo.
echo 📦 Uninstalling incompatible packages...
pip uninstall -y numpy opencv-python opencv-contrib-python

echo.
echo 📦 Installing compatible NumPy version...
pip install "numpy>=1.21.0,<2.0.0"

echo.
echo 📦 Installing compatible OpenCV version...
pip install "opencv-python>=4.8.0"

echo.
echo 📦 Installing other dependencies...
pip install scikit-image scikit-learn matplotlib scipy jupyterlab

echo.
echo 🧪 Testing imports...
python -c "import cv2, numpy, skimage, sklearn, matplotlib, scipy; print('✅ All modules imported successfully!')"

if %errorlevel% == 0 (
    echo.
    echo 🎉 Environment fixed successfully!
    echo ✅ You can now run: python test_ph2_pipeline.py
) else (
    echo.
    echo ❌ Some modules failed to import
    echo 🔧 Please check the error messages above
)

pause
