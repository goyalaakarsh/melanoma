#!/usr/bin/env python3
"""
Diagnostic script to identify segmentation issues with PH2 dataset.
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import image_processing as ip
import config

def diagnose_segmentation():
    """Diagnose segmentation issues"""
    
    # PH2 dataset paths
    sample_image_path = r"C:\Users\Aakarsh Goyal\Downloads\archive\PH2Dataset\PH2 Dataset images\IMD427\IMD427_Dermoscopic_Image\IMD427.bmp"
    ground_truth_mask_path = r"C:\Users\Aakarsh Goyal\Downloads\archive\PH2Dataset\PH2 Dataset images\IMD427\IMD427_lesion\IMD427_lesion.bmp"
    
    print("🔍 SEGMENTATION DIAGNOSTIC ANALYSIS")
    print("=" * 50)
    
    # Load images
    print("📥 Loading images...")
    rgb_image, hsv_image, lab_image = ip.load_and_preprocess(sample_image_path)
    hair_free_image, _ = ip.remove_hair(rgb_image)
    
    # Load ground truth
    gt_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is not None:
        # Handle inversion
        lesion_pixels = np.sum(gt_mask > 128)
        background_pixels = np.sum(gt_mask <= 128)
        if background_pixels > lesion_pixels:
            gt_mask = 255 - gt_mask
            print("🔄 Inverted ground truth mask")
        
        gt_mask_resized = cv2.resize(gt_mask, config.IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    else:
        print("❌ Could not load ground truth")
        return
    
    # Our segmentation
    print("🎯 Running our segmentation...")
    binary_mask, main_contour, seg_metrics = ip.segment_lesion(hair_free_image)
    
    # Analysis
    print("\n📊 SEGMENTATION ANALYSIS:")
    print("-" * 30)
    
    # Our mask analysis
    our_white = np.sum(binary_mask > 0)
    our_black = np.sum(binary_mask == 0)
    print(f"Our mask - White pixels: {our_white} ({our_white/binary_mask.size*100:.2f}%)")
    print(f"Our mask - Black pixels: {our_black} ({our_black/binary_mask.size*100:.2f}%)")
    
    # Ground truth analysis
    gt_white = np.sum(gt_mask_resized > 0)
    gt_black = np.sum(gt_mask_resized == 0)
    print(f"GT mask - White pixels: {gt_white} ({gt_white/gt_mask_resized.size*100:.2f}%)")
    print(f"GT mask - Black pixels: {gt_black} ({gt_black/gt_mask_resized.size*100:.2f}%)")
    
    # Check for inversion
    print(f"\n🔄 INVERSION CHECK:")
    if our_white > gt_white * 10:  # Our mask has way more white pixels
        print("❌ Our mask has too many white pixels - might be inverted")
        print("💡 Try inverting our mask and recalculating Dice")
        
        # Test inversion
        binary_mask_inverted = 255 - binary_mask
        intersection_inv = np.sum((gt_mask_resized > 0) & (binary_mask_inverted > 0))
        union_inv = np.sum((gt_mask_resized > 0) | (binary_mask_inverted > 0))
        dice_inv = (2 * intersection_inv) / (np.sum(gt_mask_resized > 0) + np.sum(binary_mask_inverted > 0)) if (np.sum(gt_mask_resized > 0) + np.sum(binary_mask_inverted > 0)) > 0 else 0
        
        print(f"✅ Inverted mask Dice coefficient: {dice_inv:.3f}")
        
        if dice_inv > 0.1:  # Significant improvement
            print("🎉 INVERSION FIXES THE ISSUE!")
            return "invert_mask"
    
    # Original Dice calculation
    intersection = np.sum((gt_mask_resized > 0) & (binary_mask > 0))
    union = np.sum((gt_mask_resized > 0) | (binary_mask > 0))
    dice = (2 * intersection) / (np.sum(gt_mask_resized > 0) + np.sum(binary_mask > 0)) if (np.sum(gt_mask_resized > 0) + np.sum(binary_mask > 0)) > 0 else 0
    iou = intersection / union if union > 0 else 0
    
    print(f"\n📈 METRICS:")
    print(f"Dice coefficient: {dice:.3f}")
    print(f"IoU: {iou:.3f}")
    print(f"Intersection: {intersection}")
    print(f"Union: {union}")
    
    # Save diagnostic images
    print("\n🖼️ Saving diagnostic images...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Segmentation Diagnostic Analysis', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title('1. Original PH2 Image')
    axes[0, 0].axis('off')
    
    # Our segmentation
    axes[0, 1].imshow(binary_mask, cmap='gray')
    axes[0, 1].set_title(f'2. Our Segmentation\n({our_white} white pixels)')
    axes[0, 1].axis('off')
    
    # Ground truth
    axes[0, 2].imshow(gt_mask_resized, cmap='gray')
    axes[0, 2].set_title(f'3. Ground Truth\n({gt_white} white pixels)')
    axes[0, 2].axis('off')
    
    # Hair-free image
    axes[1, 0].imshow(hair_free_image)
    axes[1, 0].set_title('4. Hair-Free Image')
    axes[1, 0].axis('off')
    
    # Difference
    difference = cv2.absdiff(gt_mask_resized, binary_mask)
    axes[1, 1].imshow(difference, cmap='hot')
    axes[1, 1].set_title(f'5. Difference\n(Dice: {dice:.3f})')
    axes[1, 1].axis('off')
    
    # Overlay
    overlay = hair_free_image.copy()
    overlay[binary_mask > 0] = [255, 0, 0]  # Red overlay
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('6. Our Segmentation Overlay')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_diagnostic.png', dpi=150, bbox_inches='tight')
    print("✅ Diagnostic image saved as 'segmentation_diagnostic.png'")
    
    return "continue_optimization"

if __name__ == "__main__":
    result = diagnose_segmentation()
    print(f"\n🎯 DIAGNOSTIC RESULT: {result}")
