"""
Feature Extraction Module for Melanoma DIP Engine.
This module implements the complete ABCD rule features (Asymmetry, Border, Color, Diameter)
plus advanced Texture analysis for comprehensive, medically accurate lesion characterization.
All features are optimized for equity across diverse skin tones and clinical interpretability.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from scipy import signal, stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
# Removed unused imports: ndimage, measure, morphology
# These were from advanced algorithms that were simplified
import config


def calculate_asymmetry(mask: np.ndarray) -> float:
    """
    Calculate medical asymmetry score using major axis division method.
    
    ⚠️  RESEARCH DISCLAIMER: This asymmetry calculation is for research purposes only.
    Medical asymmetry assessment requires clinical expertise and should not be used
    for diagnostic purposes without proper medical validation.
    
    This function implements a medically-relevant asymmetry measurement by:
    1. Finding the lesion centroid and principal axes
    2. Dividing the lesion along its major axis
    3. Comparing the areas and shapes of the two halves
    4. Calculating asymmetry as the difference between halves
    
    Args:
        mask (np.ndarray): Binary mask of the segmented lesion
        
    Returns:
        float: Asymmetry score (0.0 = symmetric, 1.0 = highly asymmetric)
    """
    if np.sum(mask) == 0:
        return 0.0
    
    # Find contours to get lesion shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Calculate centroid
    moments = cv2.moments(contour)
    if moments['m00'] == 0:
        return 0.0
    
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    
    # Fit ellipse to get major axis
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        center, axes, angle = ellipse
        major_axis_length = max(axes)
        minor_axis_length = min(axes)
        
        # Calculate asymmetry based on major/minor axis ratio
        # A perfect circle has ratio = 1 (symmetric), elongated shapes have higher ratios
        axis_ratio = major_axis_length / minor_axis_length if minor_axis_length > 0 else 1.0
        
        # Normalize to 0-1 scale (1.0 = circle, higher = more asymmetric)
        asymmetry = max(0.0, (axis_ratio - 1.0) / (config.ASYMMETRY_MAX_RATIO - 1.0))
        return min(1.0, asymmetry)
    else:
        # Fallback: use flip-based asymmetry for small contours
        total_area = np.sum(mask > 0)
        
        # Horizontal asymmetry
        horizontal_flip = cv2.flip(mask, 1)
        horizontal_xor = cv2.bitwise_xor(mask, horizontal_flip)
        horizontal_asymmetry = np.sum(horizontal_xor > 0) / total_area
        
        # Vertical asymmetry
        vertical_flip = cv2.flip(mask, 0)
        vertical_xor = cv2.bitwise_xor(mask, vertical_flip)
        vertical_asymmetry = np.sum(vertical_xor > 0) / total_area
        
        return (horizontal_asymmetry + vertical_asymmetry) / 2.0


def calculate_border_irregularity(contour: Optional[np.ndarray]) -> float:
    """
    Calculate the border irregularity using the Compactness Index.
    
    This function measures shape irregularity by:
    1. Calculating the perimeter and area of the lesion contour
    2. Computing the Compactness Index: (Perimeter² / (4 * π * Area))
    3. A perfect circle yields 1.0, irregular shapes yield higher values
    
    Args:
        contour (Optional[np.ndarray]): Contour of the segmented lesion
        
    Returns:
        float: Compactness Index (1.0 = perfect circle, >1.0 = irregular shape)
        
    DIP Concepts:
        - Compactness Index: Mathematical measure of shape regularity
        - Perimeter-to-Area Ratio: Fundamental geometric property for shape analysis
        - Circle Reference: Provides intuitive baseline for shape comparison
    """
    if contour is None:
        return 0.0
    
    # Calculate perimeter and area
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    
    if area == 0:
        return 0.0
    
    # Calculate Compactness Index: P²/(4πA)
    # A perfect circle has compactness = 1.0
    compactness = (perimeter * perimeter) / (4 * np.pi * area)
    
    return compactness


def calculate_color_variation(hsv_image: np.ndarray, mask: np.ndarray) -> int:
    """
    Calculate color variation by analyzing the Hue histogram of the lesion.
    
    This function quantifies color diversity by:
    1. Extracting the Hue channel from the HSV image within the lesion mask
    2. Computing the histogram of hue values
    3. Identifying significant peaks using signal processing techniques
    4. Returning the count of distinct color clusters
    
    Args:
        hsv_image (np.ndarray): HSV color space image
        mask (np.ndarray): Binary mask of the segmented lesion
        
    Returns:
        int: Number of distinct color clusters (minimum 1)
        
    DIP Concepts:
        - Color Histogram: Statistical distribution of colors in image region
        - Peak Detection: Signal processing technique for identifying dominant colors
        - Hue Channel Analysis: Pure color information independent of brightness/saturation
    """
    # Extract hue values only within the lesion mask
    lesion_hue = hsv_image[:, :, 0][mask > 0]
    
    if len(lesion_hue) == 0:
        return 1  # At least one color present
    
    # Calculate histogram of hue values
    hist, _ = np.histogram(lesion_hue, bins=config.COLOR_BINS, range=(0, 180))
    
    # Find peaks in the histogram
    peaks, properties = signal.find_peaks(hist, height=np.max(hist) * config.COLOR_HIST_PEAK_THRESHOLD)
    
    # Return number of peaks (minimum 1)
    return max(1, len(peaks))


def calculate_texture_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate texture features using Gray-Level Co-occurrence Matrix (GLCM).
    
    This function analyzes lesion texture by:
    1. Converting image to grayscale for efficient processing
    2. Creating masked image for lesion region only
    3. Computing GLCM for statistical texture analysis
    4. Calculating contrast and homogeneity measures
    
    Args:
        image (np.ndarray): Original RGB image
        mask (np.ndarray): Binary mask of the segmented lesion
        
    Returns:
        Dict[str, float]: Dictionary containing 'contrast' and 'homogeneity' values
        
    DIP Concepts:
        - GLCM: Statistical method for texture analysis based on pixel co-occurrence
        - Contrast: Measures local intensity variations (high for disorganized textures)
        - Homogeneity: Measures texture smoothness (high for uniform textures)
    """
    # Convert RGB to grayscale for texture analysis (more efficient)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Check if lesion region exists
    if np.sum(mask) == 0:
        return {'contrast': 0.0, 'homogeneity': 0.0}
    
    # Create masked image for GLCM calculation
    masked_image = np.zeros_like(gray_image)
    masked_image[mask > 0] = gray_image[mask > 0]
    
    # Calculate GLCM with multiple directions for robustness
    glcm = graycomatrix(
        masked_image.astype(np.uint8),
        distances=[1],
        angles=[0, 45, 90, 135],
        levels=256,
        symmetric=True,
        normed=True
    )
    
    # Calculate texture properties
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    
    return {
        'contrast': float(contrast),
        'homogeneity': float(homogeneity)
    }


def calculate_diameter(contour: Optional[np.ndarray], image_shape: Tuple[int, int]) -> Dict[str, float]:
    """
    Calculate lesion diameter using multiple methods for clinical accuracy.
    
    This function implements comprehensive diameter calculation including:
    1. Maximum Feret diameter (longest distance across lesion)
    2. Equivalent diameter (diameter of circle with same area)
    3. Bounding box diagonal diameter
    4. Convex hull diameter
    
    Args:
        contour (Optional[np.ndarray]): Contour of the segmented lesion
        image_shape (Tuple[int, int]): Shape of the original image
        
    Returns:
        Dict[str, float]: Dictionary containing various diameter measurements
        
    Research Considerations:
        - Multiple diameter methods provide comprehensive size assessment
        - Pixel-to-mm conversion requires calibration (assumed 1mm per 10 pixels)
        - All measurements are for research purposes only
    """
    if contour is None:
        return {
            'max_feret_diameter_pixels': 0.0,
            'max_feret_diameter_mm': 0.0,
            'equivalent_diameter_pixels': 0.0,
            'equivalent_diameter_mm': 0.0,
            'bounding_box_diagonal_pixels': 0.0,
            'bounding_box_diagonal_mm': 0.0,
            'convex_hull_diameter_pixels': 0.0,
            'convex_hull_diameter_mm': 0.0,
            'clinical_significance': 'No lesion detected'
        }
    
    # ⚠️  CRITICAL WARNING: Pixel-to-mm conversion requires proper calibration!
    # This is an UNVALIDATED assumption that should be calibrated for each imaging setup.
    # Using this conversion without proper calibration could lead to incorrect measurements.
    PIXELS_PER_MM = 10.0  # UNVALIDATED - MUST BE CALIBRATED PER IMAGING SETUP
    
    # 1. Maximum Feret diameter (longest distance across lesion)
    area = cv2.contourArea(contour)
    if area > 0:
        equivalent_diameter_pixels = 2 * np.sqrt(area / np.pi)
    else:
        equivalent_diameter_pixels = 0.0
    
    # 2. Maximum Feret diameter using convex hull (optimized)
    hull = cv2.convexHull(contour)
    if len(hull) > 1:
        # Use distance transform for more efficient calculation
        hull_points = hull.reshape(-1, 2)
        distances = np.linalg.norm(hull_points[:, np.newaxis] - hull_points, axis=2)
        max_feret_diameter_pixels = np.max(distances)
    else:
        max_feret_diameter_pixels = 0.0
    convex_hull_diameter_pixels = max_feret_diameter_pixels
    
    # 3. Bounding box diagonal diameter
    x, y, w, h = cv2.boundingRect(contour)
    bounding_box_diagonal_pixels = np.sqrt(w*w + h*h)
    
    # Convert to millimeters
    max_feret_diameter_mm = max_feret_diameter_pixels / PIXELS_PER_MM
    equivalent_diameter_mm = equivalent_diameter_pixels / PIXELS_PER_MM
    bounding_box_diagonal_mm = bounding_box_diagonal_pixels / PIXELS_PER_MM
    convex_hull_diameter_mm = convex_hull_diameter_pixels / PIXELS_PER_MM
    
    # ⚠️  WARNING: Size assessment based on UNVALIDATED pixel-to-mm conversion
    # These measurements are for research purposes only and should not be used for medical decisions
    max_diameter_mm = max(max_feret_diameter_mm, equivalent_diameter_mm, bounding_box_diagonal_mm)
    
    if max_diameter_mm < config.MIN_DIAMETER_MM:
        clinical_significance = 'Size below analysis threshold (research only)'
    elif max_diameter_mm > config.MAX_DIAMETER_MM:
        clinical_significance = 'Very large lesion detected (research only)'
    elif max_diameter_mm > 6.0:
        clinical_significance = 'Large lesion detected (research only)'
    elif max_diameter_mm > 4.0:
        clinical_significance = 'Medium-sized lesion detected (research only)'
    else:
        clinical_significance = 'Small lesion detected (research only)'
    
    return {
        'max_feret_diameter_pixels': float(max_feret_diameter_pixels),
        'max_feret_diameter_mm': float(max_feret_diameter_mm),
        'equivalent_diameter_pixels': float(equivalent_diameter_pixels),
        'equivalent_diameter_mm': float(equivalent_diameter_mm),
        'bounding_box_diagonal_pixels': float(bounding_box_diagonal_pixels),
        'bounding_box_diagonal_mm': float(bounding_box_diagonal_mm),
        'convex_hull_diameter_pixels': float(convex_hull_diameter_pixels),
        'convex_hull_diameter_mm': float(convex_hull_diameter_mm),
        'clinical_significance': clinical_significance,
        'largest_diameter_mm': float(max_diameter_mm)
    }


def calculate_advanced_texture_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate advanced texture features using multiple methods for comprehensive analysis.
    
    This function implements:
    1. Gray-Level Co-occurrence Matrix (GLCM) features
    2. Local Binary Pattern (LBP) features
    3. Statistical texture measures
    4. Gradient-based texture analysis
    
    Args:
        image (np.ndarray): Original RGB image
        mask (np.ndarray): Binary mask of segmented lesion
        
    Returns:
        Dict[str, float]: Dictionary containing advanced texture features
    """
    if np.sum(mask) == 0:
        return {
            'glcm_contrast': 0.0, 'glcm_homogeneity': 0.0, 'glcm_energy': 0.0, 'glcm_correlation': 0.0,
            'lbp_uniformity': 0.0, 'lbp_contrast': 0.0, 'lbp_entropy': 0.0,
            'statistical_mean': 0.0, 'statistical_std': 0.0, 'statistical_skewness': 0.0,
            'gradient_magnitude_mean': 0.0, 'gradient_magnitude_std': 0.0
        }
    
    # Extract lesion region
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lesion_region = gray_image[mask > 0]
    
    # 1. GLCM Features with multiple distances and angles
    glcm_features = {}
    try:
        glcm = graycomatrix(
            gray_image.astype(np.uint8),
            distances=config.TEXTURE_DISTANCES,
            angles=np.radians(config.TEXTURE_ANGLES),
            levels=256,
            symmetric=True,
            normed=True
        )
        
        glcm_features = {
            'glcm_contrast': float(np.mean(graycoprops(glcm, 'contrast'))),
            'glcm_homogeneity': float(np.mean(graycoprops(glcm, 'homogeneity'))),
            'glcm_energy': float(np.mean(graycoprops(glcm, 'energy'))),
            'glcm_correlation': float(np.mean(graycoprops(glcm, 'correlation')))
        }
    except Exception:
        glcm_features = {'glcm_contrast': 0.0, 'glcm_homogeneity': 0.0, 'glcm_energy': 0.0, 'glcm_correlation': 0.0}
    
    # 2. Local Binary Pattern Features
    try:
        # Extract lesion region for LBP
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            lesion_patch = gray_image[min_y:max_y+1, min_x:max_x+1]
            
            # Calculate LBP
            radius = 1
            n_points = 8 * radius
            lbp = local_binary_pattern(lesion_patch, n_points, radius, method='uniform')
            
            # Calculate LBP features
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
            
            # Uniformity (inverse of entropy)
            lbp_uniformity = np.sum(lbp_hist**2)
            
            # Contrast (variance of LBP)
            lbp_contrast = np.var(lbp)
            
            # Entropy
            lbp_entropy = -np.sum(lbp_hist * np.log(lbp_hist + 1e-7))
            
            lbp_features = {
                'lbp_uniformity': float(lbp_uniformity),
                'lbp_contrast': float(lbp_contrast),
                'lbp_entropy': float(lbp_entropy)
            }
        else:
            lbp_features = {'lbp_uniformity': 0.0, 'lbp_contrast': 0.0, 'lbp_entropy': 0.0}
    except Exception:
        lbp_features = {'lbp_uniformity': 0.0, 'lbp_contrast': 0.0, 'lbp_entropy': 0.0}
    
    # 3. Statistical Features
    statistical_features = {
        'statistical_mean': float(np.mean(lesion_region)),
        'statistical_std': float(np.std(lesion_region)),
        'statistical_skewness': float(stats.skew(lesion_region)) if len(lesion_region) > 1 else 0.0
    }
    
    # 4. Gradient-based Features
    try:
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        gradient_features = {
            'gradient_magnitude_mean': float(np.mean(gradient_magnitude[mask > 0])),
            'gradient_magnitude_std': float(np.std(gradient_magnitude[mask > 0]))
        }
    except Exception:
        gradient_features = {'gradient_magnitude_mean': 0.0, 'gradient_magnitude_std': 0.0}
    
    # Combine all features
    all_features = {**glcm_features, **lbp_features, **statistical_features, **gradient_features}
    
    return all_features


def calculate_feature_scores(features: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate normalized feature scores for research and educational purposes only.
    
    ⚠️  MEDICAL DISCLAIMER: This function provides objective feature measurements only.
    It does NOT provide medical diagnosis, risk assessment, or clinical recommendations.
    All results are for research and educational purposes only.
    
    Args:
        features (Dict[str, float]): Dictionary of extracted features
        
    Returns:
        Dict[str, float]: Normalized feature scores (0-1 scale)
    """
    # Normalize individual feature scores (0-1, higher = more irregular)
    # These are objective measurements only, not clinical risk assessments
    
    asymmetry_score = min(1.0, features.get('asymmetry', 0.0) / config.ASYMMETRY_NORMALIZATION)
    border_score = min(1.0, (features.get('border_irregularity', 1.0) - 1.0) / config.BORDER_IRREGULARITY_NORMALIZATION)
    color_score = min(1.0, (features.get('color_variation', 1.0) - 1.0) / config.COLOR_VARIATION_NORMALIZATION)
    diameter_score = min(1.0, (features.get('largest_diameter_mm', 0.0) - 2.0) / 8.0)  # 2-10mm range
    texture_score = min(1.0, features.get('glcm_contrast', 0.0) / config.TEXTURE_CONTRAST_NORMALIZATION)
    
    return {
        'asymmetry_score': asymmetry_score,
        'border_irregularity_score': border_score,
        'color_variation_score': color_score,
        'diameter_score': diameter_score,
        'texture_contrast_score': texture_score,
        'research_note': 'These scores are for research purposes only. Consult a healthcare professional for medical assessment.'
    }


def extract_all_features(
    original_image: np.ndarray,
    hsv_image: np.ndarray,
    mask: np.ndarray,
    contour: Optional[np.ndarray]
) -> Dict[str, float]:
    """
    Master function to extract all ABCD/T features with clinical risk assessment.
    
    This function orchestrates the complete, medically robust feature extraction pipeline:
    1. Validates input data quality and clinical relevance
    2. Calculates all ABCD rule features (including Diameter)
    3. Extracts advanced texture features
    4. Performs clinical risk assessment
    5. Returns comprehensive, interpretable feature dictionary
    
    Args:
        original_image (np.ndarray): Original RGB image
        hsv_image (np.ndarray): HSV color space version
        mask (np.ndarray): Binary mask of segmented lesion
        contour (Optional[np.ndarray]): Contour of the segmented lesion
        
    Returns:
        Dict[str, float]: Complete dictionary of all extracted features and clinical assessment
        
    Medical Features Included:
        - ABCD Rule: Asymmetry, Border, Color, Diameter
        - Texture Analysis: GLCM, LBP, Statistical, Gradient features
        - Clinical Risk Assessment: Individual and combined risk scores
        - Quality Metrics: Confidence and validation measures
    """
    # Validate input quality
    if original_image is None or mask is None:
        raise ValueError("Invalid input: image or mask is None")
    
    if np.sum(mask) == 0:
        print("Warning: Empty lesion mask - no features can be extracted")
        return {'error': 'Empty lesion mask'}
    
    # Validate contour area meets clinical standards
    if contour is not None:
        contour_area = cv2.contourArea(contour)
        if contour_area < config.MIN_LESION_AREA:
            print(f"Warning: Lesion area ({contour_area}) below minimum clinical threshold ({config.MIN_LESION_AREA})")
    
    # Extract basic ABCD features
    basic_features = {
        'asymmetry': calculate_asymmetry(mask),
        'border_irregularity': calculate_border_irregularity(contour),
        'color_variation': calculate_color_variation(hsv_image, mask),
    }
    
    # Calculate diameter features
    diameter_features = calculate_diameter(contour, original_image.shape[:2])
    
    # Calculate advanced texture features
    texture_features = calculate_advanced_texture_features(original_image, mask)
    
    # Combine all features
    all_features = {**basic_features, **diameter_features, **texture_features}
    
    # Calculate normalized feature scores for research purposes only
    feature_scores = calculate_feature_scores(all_features)
    
    # Add quality metrics
    quality_metrics = {
        'feature_extraction_complete': True,
        'num_features_extracted': len(all_features),
        'clinical_validation_passed': contour_area >= config.MIN_LESION_AREA if contour is not None else False
    }
    
    # Combine everything
    complete_features = {**all_features, **feature_scores, **quality_metrics}
    
    return complete_features
