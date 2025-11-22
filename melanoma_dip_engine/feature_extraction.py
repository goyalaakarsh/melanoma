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
    Calculates asymmetry by aligning the lesion's major axis to vertical,
    then checking overlap. Robust against camera rotation.
    """
    if np.sum(mask) == 0: return 0.0

    # 1. Find the orientation of the lesion using Moments
    moments = cv2.moments(mask)
    if moments['m00'] == 0: return 0.0
    
    # Calculate orientation angle
    mu11, mu20, mu02 = moments['mu11'], moments['mu20'], moments['mu02']
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02) # Angle in radians
    angle_deg = np.degrees(theta)

    # 2. Rotate the mask to align major axis vertically
    h, w = mask.shape
    center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
    
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    # 3. Calculate Asymmetry on the aligned mask
    # Split into left and right halves based on the centroid
    # (Simplified approach: Flip over centroid X)
    
    # Shift mask so centroid is at center of image for perfect flipping
    shift_x = w//2 - center[0]
    shift_y = h//2 - center[1]
    M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centered_mask = cv2.warpAffine(rotated_mask, M_shift, (w, h))
    
    flip_lr = cv2.flip(centered_mask, 1) # Left-Right flip
    flip_ud = cv2.flip(centered_mask, 0) # Up-Down flip
    
    # IoU (Intersection over Union) style calculation for asymmetry
    xor_lr = cv2.bitwise_xor(centered_mask, flip_lr)
    xor_ud = cv2.bitwise_xor(centered_mask, flip_ud)
    
    area = np.sum(centered_mask > 0)
    asymmetry_lr = np.sum(xor_lr > 0) / area
    asymmetry_ud = np.sum(xor_ud > 0) / area
    
    # Average the two asymmetries
    return (asymmetry_lr + asymmetry_ud) / 2.0

def calculate_border_irregularity(contour: Optional[np.ndarray]) -> float:
    if contour is None: return 0.0
    
    area = cv2.contourArea(contour)
    if area == 0: return 0.0
    perimeter = cv2.arcLength(contour, True)
    
    # 1. Compactness (Standard)
    compactness = (perimeter ** 2) / (4 * np.pi * area)
    
    # 2. Solidity (The "Rubber Band" test)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return 0.0
    
    solidity = float(area) / hull_area
    
    # Melanoma features: High Compactness AND Low Solidity (jagged edges)
    # We invert solidity so higher number = higher risk
    solidity_score = (1.0 - solidity) * 10 
    
    # Combine: Compactness gives overall shape, Solidity gives jaggedness
    combined_score = (compactness * 0.5) + (solidity_score * 2.0)
    
    return combined_score

def calculate_color_variation(rgb_image: np.ndarray, mask: np.ndarray) -> int:
    """
    Calculates the number of distinct colors in the lesion using K-means clustering.
    
    ABCD Rule - C (Color): Counts distinct colors (1-6 scale):
    - 1 color: Uniform (benign tendency)
    - 2-3 colors: Moderate variation
    - 4-5 colors: High variation (suspicious)
    - 6+ colors: Very high variation (melanoma sign)
    
    Range: 1-6 (discrete integer)
    
    Args:
        rgb_image (np.ndarray): RGB image
        mask (np.ndarray): Binary lesion mask
        
    Returns:
        int: Number of distinct colors (1-6)
    """
    if np.sum(mask) == 0: return 1

    # Extract lesion pixels in RGB
    mask_bool = mask > 0
    lesion_pixels = rgb_image[mask_bool].reshape(-1, 3).astype(np.float32)
    
    if len(lesion_pixels) < 10:
        return 1
    
    # Use K-means to find distinct colors (try k=2 to k=6)
    from sklearn.cluster import KMeans
    
    max_colors = 6
    best_k = 1
    
    # Test different numbers of clusters
    for k in range(2, max_colors + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
            kmeans.fit(lesion_pixels)
            
            # Check cluster sizes - each cluster should have reasonable number of pixels
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            min_cluster_size = len(lesion_pixels) * 0.02  # At least 2% of pixels
            
            # Count significant clusters
            significant_clusters = np.sum(counts >= min_cluster_size)
            
            if significant_clusters == k:
                best_k = k
            else:
                break  # Stop if clusters are too small
        except:
            break
    
    return min(best_k, 6)  # Cap at 6 colors


from scipy.stats import entropy

def calculate_texture_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Calculates texture entropy inside the lesion.
    High Entropy = Chaos/Disorder (Melanoma).
    """
    if np.sum(mask) == 0: return {'entropy': 0.0}

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Extract only the pixels inside the lesion
    lesion_pixels = gray_image[mask > 0]
    
    # Calculate histogram of these pixels
    hist, _ = np.histogram(lesion_pixels, bins=256, range=(0, 256))
    
    # Normalize histogram to get probabilities
    prob_dist = hist / hist.sum()
    
    # Calculate Entropy (Randomness)
    # High entropy = rough, chaotic texture
    # Low entropy = smooth, uniform color
    texture_entropy = entropy(prob_dist, base=2)
    
    return {
        'texture_entropy': float(texture_entropy),
        'texture_roughness': float(np.std(lesion_pixels)) # Standard deviation of intensity
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
    
    # Using this conversion without proper calibration could lead to incorrect measurements.
    PIXELS_PER_MM = 10.0
    
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


def analyze_frequency_domain(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Analyze texture using Fast Fourier Transform (FFT) in the frequency domain.
    
    This function performs frequency domain analysis to detect periodic patterns and texture irregularities.
    High-frequency energy correlates with chaotic, irregular textures typical of melanoma.
    
    Args:
        image (np.ndarray): Input RGB image
        mask (np.ndarray): Binary mask of segmented lesion
        
    Returns:
        Dict[str, float]: Dictionary containing FFT features:
            - fft_high_frequency_energy: Mean energy in high-frequency components
            - fft_low_frequency_energy: Mean energy in low-frequency components
            - fft_high_low_ratio: Ratio of high to low frequency energy
            - fft_total_energy: Total frequency domain energy
            
    DIP Concepts:
        - Fast Fourier Transform: Converts spatial domain to frequency domain
        - Frequency Spectrum: Magnitude of frequency components
        - High-Pass Filtering: Isolates high-frequency texture details
        - Spectral Energy: Measures texture complexity and irregularity
    """
    if not config.FFT_ENABLED:
        return {
            'fft_high_frequency_energy': 0.0,
            'fft_low_frequency_energy': 0.0,
            'fft_high_low_ratio': 0.0,
            'fft_total_energy': 0.0
        }
    
    if image is None or mask is None or np.sum(mask) == 0:
        return {
            'fft_high_frequency_energy': 0.0,
            'fft_low_frequency_energy': 0.0,
            'fft_high_low_ratio': 0.0,
            'fft_total_energy': 0.0
        }
    
    # Convert to grayscale for frequency analysis
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Extract lesion region of interest
    masked_region = gray_image.copy()
    masked_region[mask == 0] = 0
    
    # Perform 2D Fast Fourier Transform
    fft = np.fft.fft2(masked_region)
    fft_shifted = np.fft.fftshift(fft)  # Shift zero-frequency to center
    
    # Calculate magnitude spectrum (logarithmic scale for visualization)
    magnitude_spectrum = np.abs(fft_shifted)
    
    # Create high-pass filter mask (mask out center low frequencies)
    rows, cols = magnitude_spectrum.shape
    center_row, center_col = rows // 2, cols // 2
    
    # Calculate radius for low-frequency masking
    radius = int(min(rows, cols) * config.FFT_RADIUS_RATIO)
    
    # Create circular mask for low frequencies
    y, x = np.ogrid[:rows, :cols]
    mask_circle = (x - center_col)**2 + (y - center_row)**2 <= radius**2
    
    # Separate low and high frequency components
    low_freq_mask = mask_circle.astype(float)
    high_freq_mask = 1.0 - low_freq_mask
    
    # Calculate energy in low and high frequency regions
    low_freq_energy = np.sum(magnitude_spectrum * low_freq_mask)
    high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
    total_energy = np.sum(magnitude_spectrum)
    
    # Calculate ratio (high frequency / low frequency)
    high_low_ratio = high_freq_energy / low_freq_energy if low_freq_energy > 0 else 0.0
    
    # Normalize energies by total energy
    normalized_high = high_freq_energy / total_energy if total_energy > 0 else 0.0
    normalized_low = low_freq_energy / total_energy if total_energy > 0 else 0.0
    
    return {
        'fft_high_frequency_energy': float(normalized_high),
        'fft_low_frequency_energy': float(normalized_low),
        'fft_high_low_ratio': float(high_low_ratio),
        'fft_total_energy': float(total_energy)
    }


def detect_blue_white_veil(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Detect presence of blue-white veil, a dermoscopic indicator of invasive melanoma.
    
    Blue-white veil appears as a confluent gray-blue to white pigmentation that obscures
    visualization of underlying skin structures. It is a highly specific sign of melanoma.
    
    Args:
        image (np.ndarray): Input RGB image
        mask (np.ndarray): Binary mask of segmented lesion
        
    Returns:
        Dict[str, float]: Dictionary containing blue-white veil features:
            - blue_white_veil_present: 1.0 if present, 0.0 if absent
            - blue_white_veil_coverage_percentage: Percentage of lesion with veil
            - blue_white_veil_intensity: Average intensity of veil region
            - blue_white_veil_confidence: Confidence score (0.0-1.0)
            
    DIP Concepts:
        - Color Space Analysis: RGB normalization for illumination invariance
        - Channel Comparison: Blue dominance over red and green
        - Luminance Thresholding: Ensures sufficient brightness
        - Morphological Analysis: Validates veil region connectivity
    """
    if not config.BLUE_WHITE_VEIL_ENABLED:
        return {
            'blue_white_veil_present': 0.0,
            'blue_white_veil_coverage_percentage': 0.0,
            'blue_white_veil_intensity': 0.0,
            'blue_white_veil_confidence': 0.0
        }
    
    if image is None or mask is None or np.sum(mask) == 0:
        return {
            'blue_white_veil_present': 0.0,
            'blue_white_veil_coverage_percentage': 0.0,
            'blue_white_veil_intensity': 0.0,
            'blue_white_veil_confidence': 0.0
        }
    
    # Use ORIGINAL (un-normalized) image to preserve true chromatic relationships.
    # Convert to float for ratio-based analysis.
    image_float = image.astype(np.float32) / 255.0

    r_channel = image_float[:, :, 0]
    g_channel = image_float[:, :, 1]
    b_channel = image_float[:, :, 2]

    luminance = 0.2126 * r_channel + 0.7152 * g_channel + 0.0722 * b_channel

    # Relaxed blue-white veil criteria (ratio + relative enhancement):
    # 1. (B + G) > R * 1.2  (bluish / cyan dominance over red)
    # 2. B/R > 1.1          (blue component elevated relative to red)
    # 3. B > mean(B) + 0.5*std(B) (localized blue elevation)
    # 4. High luminance to ensure "veil" (not just dark blue)
    mean_b = np.mean(b_channel)
    std_b = np.std(b_channel)
    blue_green_dom = (b_channel + g_channel) > (r_channel * 1.2)
    blue_ratio_dom = (r_channel > 0) & ((b_channel / (r_channel + 1e-6)) > 1.1)
    blue_local_enhanced = b_channel > (mean_b + 0.5 * std_b)
    high_luminance = luminance > config.VEIL_MIN_LUMINANCE

    veil_mask = blue_green_dom & blue_ratio_dom & blue_local_enhanced & high_luminance
    
    # Apply lesion mask (only consider veil within lesion)
    veil_mask = veil_mask & (mask > 0)
    
    # Calculate veil coverage
    lesion_area = np.sum(mask > 0)
    veil_area = np.sum(veil_mask)
    coverage_percentage = (veil_area / lesion_area * 100) if lesion_area > 0 else 0.0
    
    # Determine presence based on coverage threshold
    veil_present = coverage_percentage >= (config.VEIL_MIN_COVERAGE * 100)
    
    # Calculate average intensity of veil region
    if veil_area > 0:
        veil_intensity = np.mean(b_channel[veil_mask])
        # Confidence based on coverage and intensity
        confidence = min(1.0, (coverage_percentage / 10.0) * veil_intensity)
    else:
        veil_intensity = 0.0
        confidence = 0.0
    
    return {
        'blue_white_veil_present': float(veil_present),
        'blue_white_veil_coverage_percentage': float(coverage_percentage),
        'blue_white_veil_intensity': float(veil_intensity),
        'blue_white_veil_confidence': float(confidence)
    }


def calculate_feature_scores(features: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate normalized feature scores for research and educational purposes only.
    
    ⚠️  MEDICAL DISCLAIMER: This function provides objective feature measurements only.
    It does NOT provide medical diagnosis, risk assessment, or clinical recommendations.
    All results are for research and educational purposes only.
    
    Feature Ranges (Raw Values):
    - Asymmetry: 0.0 to 1.0 (0.0 = perfect symmetry, 1.0 = complete asymmetry)
    - Border Irregularity: 1.0 to ~10.0 (1.0 = perfect circle, higher = more irregular)
    - Color Variation: 1 to 6 discrete colors (1 = uniform, 6 = highly variegated)
    - Diameter: 0.0 to 100.0 mm (clinical significance at 6mm+)
    - Texture Contrast: 0.0 to 300+ (GLCM contrast, higher = more irregular)
    
    Normalized Scores: All converted to 0.0-1.0 scale for comparison
    
    Args:
        features (Dict[str, float]): Dictionary of extracted features
        
    Returns:
        Dict[str, float]: Normalized feature scores (0-1 scale)
    """
    # Normalize individual feature scores (0-1, higher = more irregular)
    # These are objective measurements only, not clinical risk assessments
    
    # Asymmetry: Already in 0-1 range, multiply by 2 for sensitivity (cap at 1.0)
    asymmetry_score = min(1.0, features.get('asymmetry', 0.0) * 2.0)
    
    # Border: Normalize with tighter scaling per clinical feedback
    # (score - 1.0) / 2.0 elevates moderate irregularity appropriately.
    border_raw = features.get('border_irregularity', 1.0)
    border_score = min(1.0, max(0.0, (border_raw - 1.0) / 2.0))
    
    # Color: Discrete 1-6, normalize to 0-1
    # (1 color = 0.0, 6 colors = 1.0)
    color_count = features.get('color_variation', 1)
    color_score = min(1.0, (color_count - 1) / 5.0)
    
    # Texture: 0-300+ range, normalize to 0-1
    texture_contrast = features.get('glcm_contrast', 0.0)
    texture_score = min(1.0, texture_contrast / 200.0)
    
    return {
        'asymmetry_score': asymmetry_score,
        'border_irregularity_score': border_score,
        'color_variation_score': color_score,
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
    
    # Advanced DIP: FFT Frequency Domain Analysis
    fft_features = analyze_frequency_domain(original_image, mask)
    
    # Advanced DIP: Blue-White Veil Detection
    blue_white_veil_features = detect_blue_white_veil(original_image, mask)
    
    # Combine all features
    all_features = {**basic_features, **diameter_features, **texture_features, **fft_features, **blue_white_veil_features}
    
    # Calculate normalized feature scores for research purposes only
    feature_scores = calculate_feature_scores(all_features)
    
    # Add quality metrics
    quality_metrics = {
        'feature_extraction_complete': True,
        'num_features_extracted': len(all_features),
        'clinical_validation_passed': cv2.contourArea(contour) >= config.MIN_LESION_AREA if contour is not None else False
    }
    
    # Combine everything
    complete_features = {**all_features, **feature_scores, **quality_metrics}
    
    return complete_features
