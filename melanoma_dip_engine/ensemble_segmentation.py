#!/usr/bin/env python3
"""
Ensemble Segmentation Approach for PH2 Dataset
Combines multiple segmentation methods for optimal performance
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def ensemble_segment_lesion(image, lab_image):
    """
    Ensemble segmentation combining multiple proven methods
    
    Args:
        image: RGB image array
        lab_image: CIELab image array
        
    Returns:
        final_mask: Binary segmentation mask
        main_contour: Primary contour
        confidence_score: Segmentation confidence
    """
    
    print("🎯 ENSEMBLE SEGMENTATION APPROACH")
    print("=" * 50)
    
    # Convert to different color spaces
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Method 1: HSV Color-based (Dice 0.218 from diagnostic)
    hsv_lower = np.array([0, 30, 30])
    hsv_upper = np.array([30, 255, 255])
    hsv_mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
    
    # Apply intensity refinement
    _, intensity_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    method1_mask = cv2.bitwise_and(hsv_mask, intensity_mask)
    method1_pixels = np.sum(method1_mask > 0)
    
    print(f"🔍 Method 1 (HSV+Intensity): {method1_pixels} pixels ({method1_pixels/method1_mask.size*100:.2f}%)")
    
    # Method 2: Adaptive Thresholding (Dice 0.726 from diagnostic)
    adaptive_mask = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Intelligent inversion for adaptive thresholding
    adaptive_pixels = np.sum(adaptive_mask > 0)
    if adaptive_pixels > adaptive_mask.size * 0.7:
        adaptive_mask = 255 - adaptive_mask
        print(f"🔄 Inverted adaptive thresholding")
    
    method2_pixels = np.sum(adaptive_mask > 0)
    print(f"🔍 Method 2 (Adaptive): {method2_pixels} pixels ({method2_pixels/adaptive_mask.size*100:.2f}%)")
    
    # Method 3: CIELab a-channel (baseline)
    a_channel = lab_image[:, :, 1]
    _, a_mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    method3_pixels = np.sum(a_mask > 0)
    
    print(f"🔍 Method 3 (CIELab a): {method3_pixels} pixels ({method3_pixels/a_mask.size*100:.2f}%)")
    
    # Ensemble voting with performance-based weights
    # Based on diagnostic results: Adaptive (0.726), HSV (0.218), CIELab (0.081)
    weights = {
        'adaptive': 0.6,    # Best performer
        'hsv': 0.3,         # Good performer
        'cielab': 0.1       # Baseline
    }
    
    # Create weighted ensemble mask
    ensemble_mask = np.zeros_like(adaptive_mask, dtype=np.float32)
    ensemble_mask += weights['adaptive'] * (adaptive_mask > 128).astype(np.float32)
    ensemble_mask += weights['hsv'] * (method1_mask > 128).astype(np.float32)
    ensemble_mask += weights['cielab'] * (a_mask > 128).astype(np.float32)
    
    # Threshold ensemble (at least 40% agreement required)
    final_mask = (ensemble_mask >= 0.4).astype(np.uint8) * 255
    ensemble_pixels = np.sum(final_mask > 0)
    
    print(f"🔍 Ensemble result: {ensemble_pixels} pixels ({ensemble_pixels/final_mask.size*100:.2f}%)")
    
    # Conservative morphological operations
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Find contours and select best one
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_mask = np.zeros_like(closed_mask)
    main_contour = None
    confidence_score = 0.0
    
    if contours:
        # Advanced contour selection
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 <= area <= 200000:  # Reasonable area range
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
                solidity = area / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(cv2.convexHull(contour)) > 0 else 0
                
                if aspect_ratio < 10 and solidity > 0.3:
                    valid_contours.append(contour)
        
        if valid_contours:
            # Select largest valid contour
            main_contour = max(valid_contours, key=cv2.contourArea)
            cv2.fillPoly(final_mask, [main_contour], 255)
            
            area = cv2.contourArea(main_contour)
            confidence_score = min(1.0, area / 100000)  # Normalize by expected lesion size
            print(f"✅ Ensemble segmentation completed: {np.sum(final_mask > 0)} pixels, confidence={confidence_score:.3f}")
    
    return final_mask, main_contour, confidence_score

def test_ensemble_approach():
    """Test the ensemble approach on PH2 dataset"""
    
    # PH2 dataset paths
    sample_image_path = r"C:\Users\Aakarsh Goyal\Downloads\archive\PH2Dataset\PH2 Dataset images\IMD427\IMD427_Dermoscopic_Image\IMD427.bmp"
    ground_truth_mask_path = r"C:\Users\Aakarsh Goyal\Downloads\archive\PH2Dataset\PH2 Dataset images\IMD427\IMD427_lesion\IMD427_lesion.bmp"
    
    print("🧪 TESTING ENSEMBLE SEGMENTATION")
    print("=" * 50)
    
    # Load and preprocess images
    rgb_image = cv2.imread(sample_image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, (512, 512), interpolation=cv2.INTER_AREA)
    
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    
    # Load ground truth
    gt_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is not None:
        lesion_pixels = np.sum(gt_mask > 128)
        background_pixels = np.sum(gt_mask <= 128)
        if background_pixels > lesion_pixels:
            gt_mask = 255 - gt_mask
        gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Run ensemble segmentation
    binary_mask, main_contour, confidence = ensemble_segment_lesion(rgb_image, lab_image)
    
    # Calculate metrics
    if gt_mask is not None:
        intersection = np.sum((gt_mask > 0) & (binary_mask > 0))
        union = np.sum((gt_mask > 0) | (binary_mask > 0))
        dice = (2 * intersection) / (np.sum(gt_mask > 0) + np.sum(binary_mask > 0)) if (np.sum(gt_mask > 0) + np.sum(binary_mask > 0)) > 0 else 0
        iou = intersection / union if union > 0 else 0
        
        print(f"\n📊 ENSEMBLE RESULTS:")
        print(f"Dice coefficient: {dice:.3f}")
        print(f"IoU score: {iou:.3f}")
        print(f"Our coverage: {np.sum(binary_mask > 0)/binary_mask.size*100:.1f}%")
        print(f"GT coverage: {np.sum(gt_mask > 0)/gt_mask.size*100:.1f}%")
        
        return dice, iou
    else:
        print("❌ Could not load ground truth")
        return 0, 0

if __name__ == "__main__":
    dice, iou = test_ensemble_approach()
    print(f"\n🎉 Final ensemble performance: Dice={dice:.3f}, IoU={iou:.3f}")
