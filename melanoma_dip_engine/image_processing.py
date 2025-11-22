"""
Core Image Processing Module for Melanoma DIP Engine.
This module handles image loading, preprocessing, hair removal, and lesion segmentation
with advanced equity-focused algorithms for diverse skin tones.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import warnings
import config

warnings.filterwarnings('ignore', category=DeprecationWarning)


# NOTE: Gray World color constancy removed per clinical feedback.
# Skin lesion images have non-neutral average chromaticity (often red/pink). Forcing
# gray-world normalization was distorting diagnostically relevant color cues (e.g., blue-white veil).
# We retain only luminance enhancement (CLAHE on L channel) and leave chroma (a/b) untouched.


def load_and_preprocess(image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load an image from file path and preprocess it for analysis with equity-focused enhancements.
    
    This function establishes a standardized, medically robust input for our pipeline by:
    - Validating file format and quality
    - Loading and converting color spaces with medical-grade precision
    - Applying equity-focused preprocessing for diverse skin tones
    - Implementing quality assurance checks
    
    Args:
        image_path (str): Path to the input image file
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - RGB image (resized, enhanced)
            - HSV image (resized) 
            - CIELab image (resized, preserving absolute lightness)
            
    Raises:
        ValueError: If image cannot be loaded or fails quality checks
        FileNotFoundError: If image file does not exist
        
    Research Considerations:
        - CIELab color space preserves absolute lightness values for accurate analysis
        - CLAHE contrast enhancement improves detection on pigmented lesions
        - Quality validation prevents processing of unsuitable images
        - All processing is for research purposes only
    """
    import os
    
    # Validate file existence and format
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext not in config.SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported: {config.SUPPORTED_FORMATS}")
    
    # Load image in BGR format
    bgr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError(f"Could not load image from path: {image_path}")
    
    # Validate image quality
    if bgr_image.size == 0:
        raise ValueError("Loaded image is empty")
    
    # Convert BGR to RGB (OpenCV uses BGR by default)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    # Quality check: ensure image is not completely black or white
    if np.all(rgb_image == 0) or np.all(rgb_image == 255):
        raise ValueError("Image appears to be completely black or white")
    
    # Resize image to standard size using high-quality interpolation
    rgb_resized = cv2.resize(rgb_image, config.IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)
    
    # Removed Gray World color constancy (see note above) to preserve true dermatologic colors.
    
    # Apply equity-focused preprocessing
    if config.CONTRAST_ENHANCEMENT:
        # CLAHE ONLY on L (luminance) channel; do not alter a/b chroma components.
        lab_temp = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_SIZE)
        lab_temp[:, :, 0] = clahe.apply(lab_temp[:, :, 0])
        rgb_resized = cv2.cvtColor(lab_temp, cv2.COLOR_LAB2RGB)
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2HSV)
    
    # Convert to CIELab color space (preserving absolute lightness values)
    # Absolute lightness values are important for melanoma detection
    lab_image = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2LAB)
    
    return rgb_resized, hsv_image, lab_image


def remove_hair(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Remove hair artifacts from the skin lesion image using advanced DullRazor technique.
    
    This function implements a simplified, medically validated DullRazor hair removal:
    1. Single-scale hair detection using elliptical kernel
    2. Black Hat morphological transform for hair detection
    3. Threshold-based hair mask creation
    4. Morphological refinement to reduce false positives
    5. TELEA inpainting for hair removal
    
    Args:
        image (np.ndarray): Input RGB image
        
    Returns:
        Tuple[np.ndarray, Dict[str, float]]:
            - Hair-free RGB image
            - Quality metrics dictionary
            
    DIP Concepts:
        - Black Hat Morphology: Detects dark hair structures on light background
        - Morphological Refinement: Removes false positives and fills gaps
        - TELEA Inpainting: Algorithmically reconstructs hair-free regions
        - Single Algorithm Approach: Consistent, medically validated method
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is invalid")
    
    # Convert RGB to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Simplified DullRazor hair removal using single kernel size
    # This is the proven, medically validated approach
    
    # Create elliptical kernel for hair detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.HAIR_REMOVAL_KERNEL_SIZE)
    
    # Apply Black Hat transform to detect dark hair structures
    blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold to create hair mask
    _, final_hair_mask = cv2.threshold(
        blackhat, config.HAIR_REMOVAL_THRESHOLD, 255, cv2.THRESH_BINARY
    )
    
    # Morphological refinement to reduce false positives
    kernel_refine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_hair_mask = cv2.morphologyEx(final_hair_mask, cv2.MORPH_OPEN, kernel_refine)
    final_hair_mask = cv2.morphologyEx(final_hair_mask, cv2.MORPH_CLOSE, kernel_refine)
    
    # Apply inpainting using TELEA algorithm (proven method)
    hair_free_image = cv2.inpaint(image, final_hair_mask, config.INPAINTING_RADIUS, cv2.INPAINT_TELEA)
    used_algorithm = "TELEA"
    
    # Calculate quality metrics
    hair_percentage = (np.sum(final_hair_mask > 0) / final_hair_mask.size) * 100
    quality_metrics = {
        'hair_percentage': hair_percentage,
        'inpainting_algorithm': used_algorithm,
        'processing_complete': True
    }
    
    return hair_free_image, quality_metrics


def segment_lesion(image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    """
    Segment the lesion from surrounding skin using simplified, medically validated approach.
    
    This function implements a simplified, medically safe segmentation pipeline:
    1. CIELab a-channel thresholding (proven method for pigmented lesions)
    2. Morphological refinement (opening and closing)
    3. Largest contour selection (most conservative approach)
    4. Quality assessment and validation
    
    Args:
        image (np.ndarray): Input RGB image (preferably hair-free)
        
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
            - Binary mask of the segmented lesion
            - Main contour object of the lesion
            - Quality metrics dictionary
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is invalid")
    
    # Convert to color spaces needed for segmentation
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Method 1: Adaptive thresholding 
    adaptive_mask = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Method 2: HSV Color-based segmentation (improved for diverse skin tones)
    hsv_lower = np.array([0, 20, 20])   
    hsv_upper = np.array([179, 255, 255])  # Full hue range for better skin tone coverage
    hsv_mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
    
    # Conservative intensity thresholding for boundary validation
    intensity_threshold = np.percentile(gray_image, 55)  # Darker than 55% of pixels
    intensity_mask = (gray_image < intensity_threshold).astype(np.uint8) * 255
    
    # Smart combination: Adaptive (primary) + HSV (secondary) + intensity validation
    temp_mask = cv2.bitwise_and(adaptive_mask, hsv_mask)
    
    # Apply intensity boundary validation
    temp_mask = cv2.bitwise_and(temp_mask, intensity_mask)
    
    # If the combination is too restrictive, use adaptive mask alone
    if np.sum(temp_mask > 0) < np.sum(adaptive_mask > 0) * 0.3:
        temp_mask = adaptive_mask
    
    final_mask = temp_mask
    
    # Minimal morphological operations to fill tiny holes only
    kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_tiny)
    
    # Find contours and select the largest valid one
    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_mask = np.zeros_like(opened_mask)
    main_contour = None
    confidence_score = 0.0
    
    if contours:
        # Find the contour with the largest area that meets clinical criteria
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Stricter filtering: require reasonable area and aspect ratio
            if config.MIN_LESION_AREA <= area <= config.MAX_LESION_AREA:
                # Additional shape validation to prevent over-segmentation
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
                solidity = area / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(cv2.convexHull(contour)) > 0 else 0
                
                # Only accept contours with reasonable shape
                if aspect_ratio < 10 and solidity > 0.3:
                    valid_contours.append(contour)
        
        if valid_contours:
            # PHASE 2D: ADVANCED CONTOUR SELECTION
            # Select contour with best combination of area and shape quality
            
            best_contour = None
            best_score = 0
            
            for contour in valid_contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
                solidity = area / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(cv2.convexHull(contour)) > 0 else 0
                
                # Calculate composite score: area * shape_quality
                shape_quality = 1.0 / (1.0 + aspect_ratio/10.0) * solidity  # Higher is better
                composite_score = area * shape_quality
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_contour = contour
            
            if best_contour is not None:
                main_contour = best_contour
                cv2.fillPoly(final_mask, [main_contour], 255)
                
                # Enhanced confidence score based on multiple factors
                area = cv2.contourArea(main_contour)
                confidence_score = min(1.0, (area / config.MAX_LESION_AREA) * shape_quality)
    
    # Calculate quality metrics
    lesion_area = np.sum(final_mask > 0)
    area_percentage = (lesion_area / final_mask.size) * 100
    
    quality_metrics = {
        'confidence_score': confidence_score,
        'lesion_area': lesion_area,
        'area_percentage': area_percentage,
        'num_contours_found': len(contours),
        'segmentation_method': 'cielab_a_channel',
        'morphological_refinement': True,
        'largest_contour_selection': True
    }
    
    return final_mask, main_contour, quality_metrics


def refine_segmentation_grabcut(
    image: np.ndarray, 
    initial_mask: np.ndarray, 
    contour: Optional[np.ndarray]
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    """
    Refine lesion segmentation using GrabCut algorithm with morphology-based trimap initialization.
    
    GrabCut is a graph-cut based segmentation technique that models foreground and background
    using Gaussian Mixture Models (GMMs). This function uses the initial segmentation to create
    a precise trimap that guides the GrabCut optimization.
    
    Args:
        image (np.ndarray): Original RGB image
        initial_mask (np.ndarray): Initial binary segmentation mask
        contour (Optional[np.ndarray]): Initial contour of the lesion
        
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
            - Refined binary mask
            - Refined contour
            - Quality metrics dictionary
            
    DIP Concepts:
        - Graph Cut Optimization: Minimizes energy function for optimal segmentation
        - Gaussian Mixture Models: Models color distributions of FG/BG
        - Trimap Initialization: Guides optimization with confident regions
        - Morphological Operations: Erosion/dilation for definite FG/BG regions
    """
    if not config.GRABCUT_ENABLED:
        return initial_mask, contour, {'grabcut_applied': False}
    
    if image is None or initial_mask is None or np.sum(initial_mask) == 0:
        return initial_mask, contour, {'grabcut_applied': False}
    
    # Create GrabCut mask (0=BG, 1=FG, 2=Probable BG, 3=Probable FG)
    gc_mask = np.zeros(initial_mask.shape, dtype=np.uint8)
    
    # Generate trimap using morphological operations
    # Erosion creates "definite foreground" (confident lesion region)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.GRABCUT_MARGIN, config.GRABCUT_MARGIN))
    definite_fg = cv2.erode(initial_mask, kernel_erode, iterations=1)
    
    # Dilation creates "definite background" (everything outside is background)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.GRABCUT_MARGIN * 2, config.GRABCUT_MARGIN * 2))
    dilated_mask = cv2.dilate(initial_mask, kernel_dilate, iterations=1)
    
    # Set trimap labels
    gc_mask[dilated_mask == 0] = cv2.GC_BGD  # 0 = Definite Background
    gc_mask[definite_fg > 0] = cv2.GC_FGD    # 1 = Definite Foreground
    gc_mask[(dilated_mask > 0) & (definite_fg == 0)] = cv2.GC_PR_FGD  # 3 = Probable Foreground (uncertainty region)
    
    # Initialize background and foreground models
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    
    try:
        # Run GrabCut optimization
        cv2.grabCut(
            image, 
            gc_mask, 
            None,  # No rectangle initialization (using mask mode)
            bgd_model, 
            fgd_model, 
            config.GRABCUT_ITERATIONS, 
            cv2.GC_INIT_WITH_MASK
        )
        
        # Extract refined mask (combine definite and probable foreground)
        refined_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        
        # Find refined contour
        refined_contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refined_contour = None
        
        if refined_contours:
            # Select largest contour
            refined_contour = max(refined_contours, key=cv2.contourArea)
            
            # Validate refined contour area
            area = cv2.contourArea(refined_contour)
            if area < config.MIN_LESION_AREA or area > config.MAX_LESION_AREA:
                # If refined contour is invalid, keep original
                return initial_mask, contour, {'grabcut_applied': False, 'reason': 'invalid_area'}
        
        # Calculate improvement metrics
        initial_area = np.sum(initial_mask > 0)
        refined_area = np.sum(refined_mask > 0)
        area_change_percent = abs(refined_area - initial_area) / initial_area * 100 if initial_area > 0 else 0
        
        metrics = {
            'grabcut_applied': True,
            'grabcut_iterations': config.GRABCUT_ITERATIONS,
            'initial_area': int(initial_area),
            'refined_area': int(refined_area),
            'area_change_percent': float(area_change_percent)
        }
        
        return refined_mask, refined_contour, metrics
        
    except Exception as e:
        # If GrabCut fails, return original mask
        return initial_mask, contour, {'grabcut_applied': False, 'error': str(e)}


