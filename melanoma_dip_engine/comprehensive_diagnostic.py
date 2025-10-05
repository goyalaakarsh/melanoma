#!/usr/bin/env python3
"""
Comprehensive diagnostic script for PH2 dataset segmentation analysis.
Provides 9 logical variants for deep insight into texture, spread, depth, and color variation.
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import ndimage

import image_processing as ip
import config

def create_comprehensive_diagnostic():
    """Create comprehensive 9-panel diagnostic visualization"""
    
    # PH2 dataset paths
    sample_image_path = r"C:\Users\Aakarsh Goyal\Downloads\archive\PH2Dataset\PH2 Dataset images\IMD427\IMD427_Dermoscopic_Image\IMD427.bmp"
    ground_truth_mask_path = r"C:\Users\Aakarsh Goyal\Downloads\archive\PH2Dataset\PH2 Dataset images\IMD427\IMD427_lesion\IMD427_lesion.bmp"
    
    print("🔬 COMPREHENSIVE SEGMENTATION DIAGNOSTIC")
    print("=" * 60)
    
    # Load images
    print("📥 Loading and preprocessing images...")
    rgb_image, hsv_image, lab_image = ip.load_and_preprocess(sample_image_path)
    hair_free_image, _ = ip.remove_hair(rgb_image)
    
    # Load ground truth
    gt_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is not None:
        lesion_pixels = np.sum(gt_mask > 128)
        background_pixels = np.sum(gt_mask <= 128)
        if background_pixels > lesion_pixels:
            gt_mask = 255 - gt_mask
        gt_mask_resized = cv2.resize(gt_mask, config.IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    else:
        print("❌ Could not load ground truth")
        return
    
    # Our current segmentation
    binary_mask, main_contour, seg_metrics = ip.segment_lesion(hair_free_image)
    
    # Create enhanced diagnostic images
    print("🎨 Creating comprehensive diagnostic visualizations...")
    
    # 1. Multi-channel analysis
    a_channel = lab_image[:, :, 1]
    b_channel = lab_image[:, :, 2]
    l_channel = lab_image[:, :, 0]
    h_channel = hsv_image[:, :, 0]
    s_channel = hsv_image[:, :, 1]
    v_channel = hsv_image[:, :, 2]
    
    # 2. Enhanced segmentation attempts
    # Method 1: Multi-channel voting
    _, a_mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, s_mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, v_mask = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine masks with voting
    multi_channel_mask = np.zeros_like(a_mask)
    vote_threshold = 2  # At least 2 channels must agree
    for i in range(a_mask.shape[0]):
        for j in range(a_mask.shape[1]):
            votes = sum([
                a_mask[i, j] > 128,
                s_mask[i, j] > 128, 
                v_mask[i, j] > 128
            ])
            if votes >= vote_threshold:
                multi_channel_mask[i, j] = 255
    
    # Method 2: Region growing from high-confidence seeds
    seed_threshold = np.percentile(a_channel, 90)  # Top 10% as seeds
    seed_mask = (a_channel > seed_threshold).astype(np.uint8) * 255
    
    # Method 3: Adaptive thresholding
    adaptive_mask = cv2.adaptiveThreshold(
        cv2.cvtColor(hair_free_image, cv2.COLOR_RGB2GRAY),
        255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Method 4: Edge-based segmentation
    gray = cv2.cvtColor(hair_free_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Method 5: Color-based segmentation (HSV)
    hsv_lower = np.array([0, 30, 30])  # Lower HSV threshold for brown/dark colors
    hsv_upper = np.array([30, 255, 255])  # Upper HSV threshold
    hsv_mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
    
    # 3. Texture analysis
    # GLCM texture features
    from skimage.feature import graycomatrix, graycoprops
    gray_image = cv2.cvtColor(hair_free_image, cv2.COLOR_RGB2GRAY)
    
    # Create texture map
    texture_contrast = np.zeros_like(gray_image, dtype=np.float32)
    texture_homogeneity = np.zeros_like(gray_image, dtype=np.float32)
    
    # Calculate texture in sliding windows
    window_size = 15
    for i in range(0, gray_image.shape[0] - window_size, window_size):
        for j in range(0, gray_image.shape[1] - window_size, window_size):
            window = gray_image[i:i+window_size, j:j+window_size]
            if window.std() > 10:  # Only for textured regions
                glcm = graycomatrix(window, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                
                texture_contrast[i:i+window_size, j:j+window_size] = contrast
                texture_homogeneity[i:i+window_size, j:j+window_size] = homogeneity
    
    # 4. Depth/Intensity analysis
    intensity_gradient = np.gradient(gray_image.astype(np.float32))
    depth_map = np.sqrt(intensity_gradient[0]**2 + intensity_gradient[1]**2)
    
    # 5. Color variation analysis
    # Calculate local color variation
    color_variation = np.zeros_like(gray_image, dtype=np.float32)
    for i in range(5, gray_image.shape[0] - 5):
        for j in range(5, gray_image.shape[1] - 5):
            local_region = hair_free_image[i-5:i+5, j-5:j+5]
            color_std = np.std(local_region.reshape(-1, 3), axis=0).mean()
            color_variation[i, j] = color_std
    
    # Create comprehensive 3x3 visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    fig.suptitle('Comprehensive PH2 Lesion Analysis - 9 Logical Variants', fontsize=16, fontweight='bold')
    
    # Row 1: Original and Ground Truth Comparison
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title('1. Original PH2 Image\n(Reference)', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_mask_resized, cmap='gray')
    axes[0, 1].set_title(f'2. Ground Truth Mask\n({np.sum(gt_mask_resized > 0)} pixels)', fontweight='bold', fontsize=12)
    axes[0, 1].axis('off')
    
    # Overlay comparison
    overlay_gt = rgb_image.copy()
    overlay_gt[gt_mask_resized > 0] = [0, 255, 0]  # Green for ground truth
    axes[0, 2].imshow(overlay_gt)
    axes[0, 2].set_title('3. Ground Truth Overlay\n(Green = Lesion)', fontweight='bold', fontsize=12)
    axes[0, 2].axis('off')
    
    # Row 2: Our Segmentation Analysis
    axes[1, 0].imshow(binary_mask, cmap='gray')
    axes[1, 0].set_title(f'4. Our Current Segmentation\n({np.sum(binary_mask > 0)} pixels, Dice: {seg_metrics.get("dice_score", 0.010):.3f})', fontweight='bold', fontsize=12)
    axes[1, 0].axis('off')
    
    # Our segmentation overlay
    overlay_ours = rgb_image.copy()
    overlay_ours[binary_mask > 0] = [255, 0, 0]  # Red for our segmentation
    axes[1, 1].imshow(overlay_ours)
    axes[1, 1].set_title('5. Our Segmentation Overlay\n(Red = Detected)', fontweight='bold', fontsize=12)
    axes[1, 1].axis('off')
    
    # Combined comparison
    overlay_combined = rgb_image.copy()
    overlay_combined[gt_mask_resized > 0] = [0, 255, 0]  # Green for ground truth
    overlay_combined[binary_mask > 0] = [255, 0, 0]  # Red for our segmentation
    # Yellow where both agree
    intersection = (gt_mask_resized > 0) & (binary_mask > 0)
    overlay_combined[intersection] = [255, 255, 0]
    axes[1, 2].imshow(overlay_combined)
    axes[1, 2].set_title('6. Combined Analysis\n(Green=GT, Red=Ours, Yellow=Agreement)', fontweight='bold', fontsize=12)
    axes[1, 2].axis('off')
    
    # Row 3: Advanced Analysis
    # Multi-channel segmentation
    axes[2, 0].imshow(multi_channel_mask, cmap='gray')
    axes[2, 0].set_title(f'7. Multi-Channel Voting\n({np.sum(multi_channel_mask > 0)} pixels)', fontweight='bold', fontsize=12)
    axes[2, 0].axis('off')
    
    # Texture analysis
    axes[2, 1].imshow(texture_contrast, cmap='hot')
    axes[2, 1].set_title('8. Texture Contrast Map\n(Hot = High Texture Variation)', fontweight='bold', fontsize=12)
    axes[2, 1].axis('off')
    
    # Color variation analysis
    axes[2, 2].imshow(color_variation, cmap='viridis')
    axes[2, 2].set_title('9. Color Variation Map\n(Bright = High Color Variation)', fontweight='bold', fontsize=12)
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('comprehensive_diagnostic.png', dpi=150, bbox_inches='tight')
    print("✅ Comprehensive diagnostic saved as 'comprehensive_diagnostic.png'")
    
    # Calculate metrics for all methods
    print("\n📊 SEGMENTATION METHOD COMPARISON:")
    print("-" * 50)
    
    methods = {
        'Current Method': binary_mask,
        'Multi-Channel Voting': multi_channel_mask,
        'Seed-Based': seed_mask,
        'Adaptive Threshold': adaptive_mask,
        'HSV Color-Based': hsv_mask
    }
    
    for method_name, mask in methods.items():
        if mask is not None:
            intersection = np.sum((gt_mask_resized > 0) & (mask > 0))
            union = np.sum((gt_mask_resized > 0) | (mask > 0))
            dice = (2 * intersection) / (np.sum(gt_mask_resized > 0) + np.sum(mask > 0)) if (np.sum(gt_mask_resized > 0) + np.sum(mask > 0)) > 0 else 0
            iou = intersection / union if union > 0 else 0
            
            print(f"{method_name:20}: Dice={dice:.3f}, IoU={iou:.3f}, Pixels={np.sum(mask > 0)}")
    
    # Analysis insights
    print("\n🔍 KEY INSIGHTS:")
    print("-" * 30)
    print(f"• Ground Truth Coverage: {np.sum(gt_mask_resized > 0)/gt_mask_resized.size*100:.1f}% of image")
    print(f"• Our Coverage: {np.sum(binary_mask > 0)/binary_mask.size*100:.1f}% of image")
    print(f"• Coverage Gap: {np.sum(gt_mask_resized > 0)/gt_mask_resized.size*100 - np.sum(binary_mask > 0)/binary_mask.size*100:.1f}%")
    print(f"• Texture Variation: Max={texture_contrast.max():.1f}, Mean={texture_contrast.mean():.1f}")
    print(f"• Color Variation: Max={color_variation.max():.1f}, Mean={color_variation.mean():.1f}")
    
    return True

if __name__ == "__main__":
    success = create_comprehensive_diagnostic()
    if success:
        print("\n🎉 Comprehensive diagnostic completed successfully!")
        print("📁 Check 'comprehensive_diagnostic.png' for detailed analysis")
    else:
        print("\n❌ Diagnostic failed")
