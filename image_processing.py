"""
🌟Overall Purpose
	•	Preprocessing + segmentation stage of melanoma pipeline.
	•	Goals:
	•	Input images ko standardize karna.
	•	Artifacts remove karna (especially hair).
	•	Accurate lesion mask banana for feature extraction.
	
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
    1️⃣ load_and_preprocess — Standardization Stage

Functions & Steps
	•	Input file validation (image sahi hai ya nahi).
	•	Image resize → 512×512 using Lanczos interpolation
(high-quality resizing).
	•	CLAHE apply hota hai:
	•	Sirf Luminance channel (L) par.
	•	CIELab color space ke andar.

Why?
	•	Contrast improve hota hai.
	•	Chrominance preserve hoti hai (e.g., blue-white veil jaisa melanoma indicator distort nahi hota).

Outputs
	•	Synchronized:
	•	RGB
	•	HSV
	•	CIELab
	•	Next stages me consistent processing ensure hoti hai.


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

2️⃣ remove_hair — Artifact Removal (Modified DullRazor)

Steps
	•	Image → Grayscale conversion.
	•	Black Hat transform (elliptical kernel)
→ dark hair detect karne ke liye.
	•	Morphological opening + closing
→ false positives kam hote hain.
	•	TELEA inpainting
→ hair regions ko smooth, natural skin se fill karta hai.

Why?
	•	Hair segmentation aur feature extraction ko
majorly disturb karta hai — so removal is crucial.
    
    
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
    3️⃣ segment_lesion — Main Segmentation Engine

Multi-Method Fusion Strategy
	•	Adaptive Gaussian thresholding
→ variable illumination handle.
	•	HSV color segmentation
→ wide hue range → diverse skin tones support.
	•	Intensity validation
→ percentile-based thresholds.

Mask Handling
	•	Saare masks bitwise combine kiye jaate hain.
	•	Fusion restrictive ho toh fallback = pure adaptive thresholding.

Contour Selection
	•	Contours identify kiye jaate hain.
	•	Filter based on:
	•	Area bounds
	•	Aspect ratio
	•	Solidity (shape quality)
	•	Best contour choose using composite scoring
(area + shape quality balance).
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

4️⃣ refine_segmentation_grabcut — Optional Boundary Refinement

Trimap Creation
	•	Eroded region → Definite foreground (pakka lesion)
	•	Dilated region → Definite background (pakka skin)
	•	Middle region → Probable foreground (confusing area)

GrabCut Refinement
	•	Gaussian Mixture Models (GMMs) use hota hai.
	•	Multiple iterations me boundary optimize karta hai.
	•	Leakage fix karta hai — background include ho jaye toh.
.
    
    
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









Result
	•	Lesion boundary becomes much cleaner and clinically reliable.





"""

⭐ 1. Lanczos Interpolation

Simple meaning:
High-quality image resizing technique.


⭐ 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)

Simple meaning:
Image ka contrast ache se improve karta hai without over-brightening.



⭐ 3. Luminance Channel

Simple meaning:
Image ka “brightness” wala part.



⭐ 4. CIELab Color Space

Simple meaning:
Ek color model jisme brightness aur colors clearly separate hote hain.

Breakdown:
	•	L = lightness (brightness)
	•	a = red–green axis
	•	b = yellow–blue axis



⭐ 5. HSV Color Space

Simple meaning:
Color model based on how humans see colors.

Breakdown:
	•	H = Hue (actual color)
	•	S = Saturation (color kitna pure hai)
	•	V = Value (brightness / lightness)



⭐ 6. Black Hat Transform

Simple meaning:
Image me dark lines (jaise baal/hair) highlight karne ka trick.



⭐ 7. Morphological Opening

Simple meaning:
Choti choti noise remove karta hai.



⭐ 8. Morphological Closing

Simple meaning:
Gaps fill karta hai.


⭐ 9. TELEA Inpainting

Simple meaning:
Hair hataane ke baad jo black holes ban jaate hain, unhe natural skin texture se fill kar deta hai.



⭐ 10. Contour Selection

Simple meaning:
Image me jo shape milti hai (outline) — uski boundary ko detect karna.



⭐ 11. Trimap

Simple meaning:
Image ko 3 regions me divide karna segmentation refine karne ke liye.

Three parts:
	1.	Definite foreground (pukka lesion)
	2.	Definite background (pukka skin)
	3.	Probable foreground (confusing area)



⭐ 12. GrabCut

Simple meaning:
Advanced algorithm jo background aur foreground ko perfect boundary ke saath separate karta hai.



⭐ 13. GMM (Gaussian Mixture Models)

Simple meaning:
GrabCut ke andar use hota hai to learn color patterns.

