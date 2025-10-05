# 🧬 PH2 Dataset Integration Guide

## Overview
This guide explains how to use the Melanoma DIP Engine with PH2 dataset images. The PH2 dataset is a well-known dermatology dataset containing dermoscopic images and their corresponding lesion masks.

## Your PH2 Dataset Images
- **Dermoscopic Image**: `IMD427.bmp` - High-resolution skin lesion image
- **Ground Truth Mask**: `IMD427_lesion.bmp` - Binary mask showing lesion region

## Quick Start

### Prerequisites - Fix Environment Issues
If you encounter NumPy/OpenCV compatibility errors, run:
```bash
cd melanoma_dip_engine
python setup_environment.py
```
Or on Windows, double-click `fix_environment.bat`

### Option 1: Run Test Script (Recommended)
```bash
cd melanoma_dip_engine
python test_ph2_pipeline.py
```
This will test the complete pipeline with your PH2 images and show results.

### Option 2: Use Jupyter Notebook
1. Open `test_and_visualize.ipynb` in Jupyter
2. Run all cells sequentially
3. The notebook is already configured with your PH2 image paths

## What the Pipeline Does

### 1. **Image Loading & Preprocessing**
- Loads your PH2 dermoscopic image (BMP format)
- Converts from BGR to RGB color space
- Resizes to 512x512 pixels for consistent processing
- Applies CLAHE contrast enhancement

### 2. **Hair Removal (DullRazor Technique)**
- Detects hair artifacts using morphological operations
- Removes hair using TELEA inpainting algorithm
- Provides quality metrics on hair removal effectiveness

### 3. **Lesion Segmentation**
- Uses CIELab color space for robust segmentation
- Applies Otsu's thresholding on the 'a' channel
- Refines with morphological operations
- Selects largest contour as main lesion

### 4. **Feature Extraction (ABCD Rule)**
- **A - Asymmetry**: Measures shape symmetry using major/minor axis ratio
- **B - Border Irregularity**: Calculates compactness index
- **C - Color Variation**: Counts distinct color regions in lesion
- **D - Diameter**: Calculates multiple diameter measurements
- **T - Texture**: GLCM contrast and homogeneity features

### 5. **Ground Truth Validation**
- Loads PH2 ground truth mask
- Handles potential mask inversion automatically
- Calculates Dice coefficient and IoU metrics
- Provides visual comparison

## Expected Results

### Segmentation Quality Metrics
- **Dice Coefficient**: >0.7 is good, >0.8 is excellent
- **IoU Score**: >0.6 is good, >0.7 is excellent
- **Area Difference**: Should be minimal between our segmentation and ground truth

### Feature Measurements
- **Asymmetry**: 0.0 (perfect circle) to 1.0 (highly asymmetric)
- **Border Irregularity**: >1.0 (1.0 = perfect circle)
- **Color Variation**: Number of distinct colors (≥1)
- **Diameter**: Multiple measurements in pixels and mm
- **Texture**: Contrast and homogeneity values

## Troubleshooting

### Common Issues

1. **NumPy/OpenCV Compatibility Error**
   ```
   ImportError: numpy.core.multiarray failed to import
   ```
   **Solution**: Run the environment fix:
   ```bash
   python setup_environment.py
   ```
   Or use the batch file on Windows: `fix_environment.bat`

2. **File Not Found Error**
   - Verify the file paths in the notebook/script
   - Ensure the PH2 dataset is extracted properly

3. **Poor Segmentation Results**
   - PH2 images may need different preprocessing parameters
   - Try adjusting CLAHE settings in `config.py`

4. **Mask Inversion Issues**
   - The script automatically detects and handles mask inversion
   - Check the console output for inversion messages

### Configuration Adjustments

You can modify parameters in `config.py` for better results:

```python
# For better segmentation on PH2 images
SEGMENTATION_KERNEL_SIZE: Tuple[int, int] = (5, 5)  # Smaller kernel
MIN_LESION_AREA: int = 100  # Lower minimum area

# For better contrast enhancement
CLAHE_CLIP_LIMIT: float = 3.0  # Higher contrast
CLAHE_TILE_SIZE: Tuple[int, int] = (4, 4)  # Smaller tiles
```

## Output Files

The pipeline generates:
- **Visualizations**: Complete pipeline results with comparisons
- **Feature Report**: Detailed ABCD/T feature measurements
- **Quality Metrics**: Segmentation confidence and validation scores

## Research Use Only

⚠️ **Important**: This tool is for research and educational purposes only. Do not use for medical diagnosis or clinical decisions.

## Next Steps

1. **Run the test script** to verify everything works
2. **Try different PH2 images** from the dataset
3. **Experiment with parameters** in `config.py`
4. **Compare results** across different lesion types
5. **Use features for machine learning** or further analysis

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify file paths are correct
3. Ensure all dependencies are installed
4. Try with different PH2 images from the dataset

Happy analyzing! 🔬
