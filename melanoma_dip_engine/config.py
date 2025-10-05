"""
Configuration File for the Melanoma DIP Engine.
All tunable parameters, constants, and settings are defined here to facilitate
easy experimentation and maintenance. Parameters are optimized for medical accuracy
and equity across diverse skin tones.
"""
from typing import Tuple, Dict, List

# --- General Settings ---
IMAGE_SIZE: Tuple[int, int] = (512, 512)  # Increased for better detail
SUPPORTED_FORMATS: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# --- Medical Standards & Clinical Thresholds ---
# Based on clinical guidelines and research literature
# ⚠️  Note: These values are mathematically consistent assuming 10 pixels/mm
# MIN_LESION_AREA corresponds to ~3.14 mm² (2mm diameter circle)
MIN_LESION_AREA: int = 100  # Lower threshold for PH2 dataset compatibility
MAX_LESION_AREA: int = 200000  # Maximum to avoid processing errors
MIN_DIAMETER_MM: float = 2.0  # Minimum 2mm diameter for clinical relevance
MAX_DIAMETER_MM: float = 100.0  # Maximum 100mm (10cm) diameter

# --- Segmentation Pipeline Settings (Optimized for Equity) ---
HAIR_REMOVAL_KERNEL_SIZE: Tuple[int, int] = (21, 21)  # Increased for better hair detection
HAIR_REMOVAL_THRESHOLD: int = 15  # Adaptive threshold for hair detection
SEGMENTATION_KERNEL_SIZE: Tuple[int, int] = (7, 7)  # Larger kernel for better morphology
ADAPTIVE_THRESHOLD_BLOCK_SIZE: int = 11  # For adaptive thresholding
ADAPTIVE_THRESHOLD_C: int = 2  # Constant subtracted from mean

# --- Feature Extraction Settings (Research Only) ---
COLOR_HIST_PEAK_THRESHOLD: float = 0.08  # More sensitive for subtle color variations
COLOR_BINS: int = 256  # Higher resolution for color analysis
TEXTURE_DISTANCES: List[int] = [1, 2, 3]  # Multiple distances for GLCM
TEXTURE_ANGLES: List[float] = [0, 45, 90, 135]  # Multiple angles for robustness

# --- Preprocessing Parameters ---
CONTRAST_ENHANCEMENT: bool = True  # Enable CLAHE contrast enhancement
CLAHE_CLIP_LIMIT: float = 2.0  # CLAHE clipping limit
CLAHE_TILE_SIZE: Tuple[int, int] = (8, 8)  # CLAHE tile grid size

# --- Hair Removal Parameters ---
INPAINTING_RADIUS: int = 3  # Radius for inpainting algorithm

# --- Quality Assurance Parameters ---
MIN_CONFIDENCE_THRESHOLD: float = 0.7  # Minimum confidence for segmentation
EDGE_DETECTION_THRESHOLD_LOW: int = 50  # Canny edge detection parameters
EDGE_DETECTION_THRESHOLD_HIGH: int = 150

# --- Feature Normalization Thresholds (Research Only) ---
# ⚠️  WARNING: These thresholds are for research normalization only.
# They do NOT represent validated clinical risk thresholds.
# Medical assessment requires consultation with healthcare professionals.
ASYMMETRY_NORMALIZATION: float = 0.5  # For normalizing asymmetry scores to 0-1 range
ASYMMETRY_MAX_RATIO: float = 4.0  # Maximum expected axis ratio for normalization
BORDER_IRREGULARITY_NORMALIZATION: float = 2.0  # For normalizing compactness scores
COLOR_VARIATION_NORMALIZATION: int = 5  # For normalizing color count scores
TEXTURE_CONTRAST_NORMALIZATION: float = 100.0  # For normalizing texture contrast scores

# --- Visualization Settings ---
OVERLAY_ALPHA: float = 0.4  # Transparency for overlay visualization
FIGURE_SIZE: Tuple[int, int] = (15, 12)  # Larger figures for better detail
DPI: int = 300  # High DPI for publication-quality figures
