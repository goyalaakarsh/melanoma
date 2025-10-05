"""
Utility Functions for Melanoma DIP Engine.
This module provides helper functions for visualization, metrics, and debugging.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Tuple


def calculate_dice_coefficient(true_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Calculate the Dice Coefficient for segmentation accuracy assessment.
    
    The Dice Coefficient is a measure of overlap between two binary masks:
    Dice = (2 * |A ∩ B|) / (|A| + |B|)
    
    Args:
        true_mask (np.ndarray): Ground truth binary mask
        pred_mask (np.ndarray): Predicted binary mask
        
    Returns:
        float: Dice coefficient (0.0 = no overlap, 1.0 = perfect overlap)
        
    DIP Concepts:
        - Dice Coefficient: Standard metric for binary segmentation evaluation
        - Set Theory: Measures intersection over union of two sets
        - Robustness: Handles cases where masks have different sizes or shapes
    """
    # Ensure masks are binary
    true_binary = (true_mask > 0).astype(np.uint8)
    pred_binary = (pred_mask > 0).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.sum(true_binary * pred_binary)
    union = np.sum(true_binary) + np.sum(pred_binary)
    
    # Handle edge case of empty masks
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection) / union
    
    return float(dice)


def visualize_steps(
    original_img: np.ndarray,
    hair_free_img: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
    title: str = "DIP Pipeline Visualization"
) -> None:
    """
    Create a comprehensive visualization of the DIP pipeline steps.
    
    This function creates a 2x2 grid showing:
    1. Original input image
    2. Hair-free preprocessed image
    3. Binary segmentation mask
    4. Final result with lesion overlay
    
    Args:
        original_img (np.ndarray): Original RGB input image
        hair_free_img (np.ndarray): Image after hair removal
        mask (np.ndarray): Binary segmentation mask
        overlay (np.ndarray): Original image with segmented lesion overlay
        title (str): Title for the visualization
        
    DIP Concepts:
        - Pipeline Visualization: Essential for debugging and validation
        - Multi-step Display: Shows intermediate results for quality assessment
        - Color Space Consistency: Maintains proper RGB display format
    """
    # Create figure with 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=config.FIGURE_SIZE)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('1. Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Hair-free image
    axes[0, 1].imshow(hair_free_img)
    axes[0, 1].set_title('2. Hair-Free Image', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Binary mask
    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title('3. Segmentation Mask', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Overlay result
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('4. Final Segmentation', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def create_overlay_image(original_img: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Create an overlay image showing the segmented lesion on the original image.
    
    Args:
        original_img (np.ndarray): Original RGB image
        mask (np.ndarray): Binary segmentation mask
        alpha (float): Transparency factor for overlay (0.0-1.0)
        
    Returns:
        np.ndarray: RGB image with colored lesion overlay
    """
    # Create colored mask (red for demonstration)
    colored_mask = np.zeros_like(original_img)
    colored_mask[mask > 0] = [255, 0, 0]  # Red color
    
    # Blend original image with colored mask
    overlay = cv2.addWeighted(original_img, 1-alpha, colored_mask, alpha, 0)
    
    return overlay


def print_feature_summary(features: dict) -> None:
    """
    Print a comprehensive, medical-grade summary of extracted features.
    
    Args:
        features (dict): Dictionary of extracted features including clinical assessment
    """
    print("\n" + "="*80)
    print("🏥 MELANOMA DIP ENGINE - CLINICAL FEATURE ANALYSIS REPORT")
    print("="*80)
    
    # Check for errors first
    if 'error' in features:
        print(f"❌ ERROR: {features['error']}")
        print("="*80)
        return
    
    # ABCD Rule Features
    print("\n📊 ABCD RULE ASSESSMENT:")
    print("-" * 40)
    
    # Asymmetry
    asymmetry = features.get('asymmetry', 0.0)
    print(f"🔸 Asymmetry Score: {asymmetry:.3f}")
    print("   📊 Research measurement only - no clinical interpretation")
    
    # Border Irregularity
    border_irr = features.get('border_irregularity', 1.0)
    print(f"\n🔸 Border Irregularity: {border_irr:.3f}")
    print("   📊 Compactness index - research measurement only")
    
    # Color Variation
    color_var = features.get('color_variation', 1)
    print(f"\n🔸 Color Variation: {color_var} distinct colors")
    print("   📊 Color count - research measurement only")
    
    # Diameter
    diameter_mm = features.get('largest_diameter_mm', 0.0)
    print(f"\n🔸 Largest Diameter: {diameter_mm:.1f} mm")
    clinical_sig = features.get('clinical_significance', 'Unknown')
    print(f"   📏 Clinical Significance: {clinical_sig}")
    
    # Advanced Texture Features
    print("\n🧬 ADVANCED TEXTURE ANALYSIS:")
    print("-" * 40)
    
    glcm_contrast = features.get('glcm_contrast', 0.0)
    print(f"🔸 Texture Contrast: {glcm_contrast:.3f}")
    if glcm_contrast < 20:
        print("   ✅ SMOOTH TEXTURE - Low concern")
    elif glcm_contrast < 50:
        print("   ⚠️  MODERATE TEXTURE - Moderate concern")
    else:
        print("   🚨 DISORGANIZED TEXTURE - High concern")
    
    glcm_homogeneity = features.get('glcm_homogeneity', 0.0)
    print(f"\n🔸 Texture Homogeneity: {glcm_homogeneity:.3f}")
    print("   (Higher values indicate more uniform texture)")
    
    # Clinical Risk Assessment
    print("\n🚨 CLINICAL RISK ASSESSMENT:")
    print("-" * 40)
    
    risk_level = features.get('risk_level', 'UNKNOWN')
    combined_risk = features.get('combined_risk_score', 0.0)
    recommendation = features.get('clinical_recommendation', 'No recommendation available')
    
    print(f"🔸 Overall Risk Level: {risk_level}")
    print(f"🔸 Combined Risk Score: {combined_risk:.3f} (0.0 = Low, 1.0 = Very High)")
    print(f"🔸 Clinical Recommendation: {recommendation}")
    
    # Individual Risk Scores
    print("\n📈 INDIVIDUAL RISK SCORES:")
    print("-" * 40)
    print(f"   • Asymmetry Risk: {features.get('asymmetry_risk_score', 0.0):.3f}")
    print(f"   • Border Risk: {features.get('border_risk_score', 0.0):.3f}")
    print(f"   • Color Risk: {features.get('color_risk_score', 0.0):.3f}")
    print(f"   • Diameter Risk: {features.get('diameter_risk_score', 0.0):.3f}")
    print(f"   • Texture Risk: {features.get('texture_risk_score', 0.0):.3f}")
    
    # Quality Metrics
    print("\n🔍 QUALITY ASSURANCE:")
    print("-" * 40)
    confidence_level = features.get('confidence_level', 'UNKNOWN')
    clinical_validation = features.get('clinical_validation_passed', False)
    num_features = features.get('num_features_extracted', 0)
    
    print(f"🔸 Analysis Confidence: {confidence_level}")
    print(f"🔸 Clinical Validation: {'✅ PASSED' if clinical_validation else '❌ FAILED'}")
    print(f"🔸 Features Extracted: {num_features}")
    
    # Disclaimer
    print("\n" + "="*80)
    print("⚠️  MEDICAL DISCLAIMER:")
    print("This analysis is for research and educational purposes only.")
    print("It does not constitute medical advice or diagnosis.")
    print("Always consult with a qualified healthcare professional.")
    print("="*80)


def create_clinical_visualization(
    original_img: np.ndarray,
    hair_free_img: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
    features: dict,
    title: str = "Research Analysis Report"
) -> None:
    """
    Create a research-focused visualization with feature analysis.
    
    ⚠️  DISCLAIMER: This visualization is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.
    It does NOT provide medical diagnosis, risk assessment, or clinical recommendations.
    
    Args:
        original_img (np.ndarray): Original RGB input image
        hair_free_img (np.ndarray): Image after hair removal
        mask (np.ndarray): Binary segmentation mask
        overlay (np.ndarray): Original image with segmented lesion overlay
        features (dict): Extracted features dictionary
        title (str): Title for the visualization
    """
    # Create figure with subplots
    fig = plt.figure(figsize=config.FIGURE_SIZE, dpi=config.DPI)
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle(f"🔬 {title}", fontsize=20, fontweight='bold', y=0.95)
    
    # Image processing pipeline visualization
    axes = []
    axes.append(fig.add_subplot(gs[0, 0]))
    axes.append(fig.add_subplot(gs[0, 1]))
    axes.append(fig.add_subplot(gs[0, 2]))
    axes.append(fig.add_subplot(gs[0, 3]))
    
    axes[0].imshow(original_img)
    axes[0].set_title('1. Original Image', fontweight='bold', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(hair_free_img)
    axes[1].set_title('2. Hair-Free Image', fontweight='bold', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('3. Segmentation Mask', fontweight='bold', fontsize=12)
    axes[2].axis('off')
    
    axes[3].imshow(overlay)
    axes[3].set_title('4. Final Overlay', fontweight='bold', fontsize=12)
    axes[3].axis('off')
    
    # ABCD Features visualization
    abcd_ax = fig.add_subplot(gs[1, :2])
    abcd_features = ['asymmetry', 'border_irregularity', 'color_variation']
    abcd_values = [features.get(f, 0) for f in abcd_features]
    abcd_labels = ['Asymmetry', 'Border\nIrregularity', 'Color\nVariation']
    
    bars = abcd_ax.bar(abcd_labels, abcd_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    abcd_ax.set_title('ABCD Rule Assessment', fontweight='bold', fontsize=14)
    abcd_ax.set_ylabel('Feature Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, abcd_values):
        height = bar.get_height()
        abcd_ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Feature scores visualization (research only)
    scores_ax = fig.add_subplot(gs[1, 2:])
    feature_scores = ['asymmetry_score', 'border_irregularity_score', 'color_variation_score', 'diameter_score']
    score_values = [features.get(f, 0) for f in feature_scores]
    score_labels = ['Asymmetry', 'Border\nIrregularity', 'Color\nVariation', 'Diameter']
    
    bars = scores_ax.bar(score_labels, score_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    scores_ax.set_title('Feature Scores (Research Only)', fontweight='bold', fontsize=14)
    scores_ax.set_ylabel('Normalized Score (0-1)')
    scores_ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, score_values):
        height = bar.get_height()
        scores_ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Research disclaimer
    rec_ax = fig.add_subplot(gs[2, :])
    diameter_mm = features.get('largest_diameter_mm', 0.0)
    clinical_sig = features.get('clinical_significance', 'Unknown')
    
    disclaimer_text = f"""📋 RESEARCH SUMMARY:
    
    🔸 Lesion Diameter: {diameter_mm:.1f} mm ({clinical_sig})
    🔸 Feature Analysis: ABCD rule measurements extracted
    🔸 Processing Status: Complete
    
    ⚠️  IMPORTANT DISCLAIMER:
    This analysis is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.
    It does NOT provide medical diagnosis, risk assessment, or clinical recommendations.
    Always consult a qualified healthcare professional for medical assessment."""
    
    rec_ax.text(0.05, 0.95, disclaimer_text, transform=rec_ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    rec_ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def validate_image_quality(image: np.ndarray, min_size: Tuple[int, int] = (100, 100)) -> bool:
    """
    Validate comprehensive image quality requirements for medical analysis.
    
    Args:
        image (np.ndarray): Input image to validate
        min_size (Tuple[int, int]): Minimum acceptable dimensions
        
    Returns:
        bool: True if image meets quality requirements
    """
    validation_results = []
    
    # Check if image exists
    if image is None:
        print("❌ Error: Image is None")
        return False
    
    # Check image dimensions
    if len(image.shape) != 3 or image.shape[2] != 3:
        print("❌ Error: Image must be RGB (3 channels)")
        validation_results.append(False)
    else:
        validation_results.append(True)
    
    # Check image size
    if image.shape[0] < min_size[0] or image.shape[1] < min_size[1]:
        print(f"❌ Error: Image too small. Minimum size: {min_size}")
        validation_results.append(False)
    else:
        validation_results.append(True)
    
    # Check for completely black or white images
    if np.all(image == 0):
        print("❌ Error: Image is completely black")
        validation_results.append(False)
    elif np.all(image == 255):
        print("❌ Error: Image is completely white")
        validation_results.append(False)
    else:
        validation_results.append(True)
    
    # Check for sufficient contrast
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    contrast = np.std(gray)
    if contrast < 10:
        print(f"⚠️  Warning: Low contrast detected (std: {contrast:.2f})")
        validation_results.append(False)
    else:
        validation_results.append(True)
    
    # Check for image corruption
    if np.any(np.isnan(image)) or np.any(np.isinf(image)):
        print("❌ Error: Image contains invalid values (NaN or Inf)")
        validation_results.append(False)
    else:
        validation_results.append(True)
    
    # Overall validation result
    all_passed = all(validation_results)
    
    if all_passed:
        print("✅ Image quality validation passed")
    else:
        print(f"❌ Image quality validation failed ({sum(validation_results)}/{len(validation_results)} checks passed)")
    
    return all_passed


def create_error_report(error: Exception, context: str = "") -> None:
    """
    Create a comprehensive error report for debugging and user feedback.
    
    Args:
        error (Exception): The exception that occurred
        context (str): Additional context about where the error occurred
    """
    print("\n" + "="*60)
    print("🚨 ERROR REPORT - Melanoma DIP Engine")
    print("="*60)
    
    print(f"📍 Context: {context}")
    print(f"🔍 Error Type: {type(error).__name__}")
    print(f"💬 Error Message: {str(error)}")
    
    # Provide specific guidance based on error type
    if isinstance(error, FileNotFoundError):
        print("\n💡 Suggested Solutions:")
        print("   • Check if the image file path is correct")
        print("   • Ensure the file exists and is accessible")
        print("   • Verify file permissions")
    elif isinstance(error, ValueError):
        print("\n💡 Suggested Solutions:")
        print("   • Check image format (supported: JPG, PNG, BMP, TIFF)")
        print("   • Ensure image is not corrupted")
        print("   • Verify image dimensions are adequate")
    elif isinstance(error, cv2.error):
        print("\n💡 Suggested Solutions:")
        print("   • Check OpenCV installation")
        print("   • Verify image format compatibility")
        print("   • Ensure sufficient system memory")
    
    print("\n📞 For technical support, please provide:")
    print("   • Full error message")
    print("   • Image file details (format, size)")
    print("   • System specifications")
    print("="*60)


def save_analysis_report(features: dict, output_path: str) -> None:
    """
    Save a comprehensive analysis report to a text file.
    
    Args:
        features (dict): Extracted features dictionary
        output_path (str): Path to save the report
    """
    import datetime
    
    with open(output_path, 'w') as f:
        f.write("MELANOMA DIP ENGINE - CLINICAL ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # ABCD Assessment
        f.write("ABCD RULE ASSESSMENT:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Asymmetry Score: {features.get('asymmetry', 0.0):.3f}\n")
        f.write(f"Border Irregularity: {features.get('border_irregularity', 1.0):.3f}\n")
        f.write(f"Color Variation: {features.get('color_variation', 1)}\n")
        f.write(f"Diameter: {features.get('largest_diameter_mm', 0.0):.1f} mm\n\n")
        
        # Risk Assessment
        f.write("CLINICAL RISK ASSESSMENT:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Risk Level: {features.get('risk_level', 'UNKNOWN')}\n")
        f.write(f"Combined Risk Score: {features.get('combined_risk_score', 0.0):.3f}\n")
        f.write(f"Recommendation: {features.get('clinical_recommendation', 'N/A')}\n\n")
        
        # All Features
        f.write("ALL EXTRACTED FEATURES:\n")
        f.write("-" * 30 + "\n")
        for key, value in features.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        # Disclaimer
        f.write("\n" + "=" * 60 + "\n")
        f.write("MEDICAL DISCLAIMER:\n")
        f.write("This analysis is for research and educational purposes only.\n")
        f.write("It does not constitute medical advice or diagnosis.\n")
        f.write("Always consult with a qualified healthcare professional.\n")
    
    print(f"✅ Analysis report saved to: {output_path}")
