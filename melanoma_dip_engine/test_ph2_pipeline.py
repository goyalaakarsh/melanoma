#!/usr/bin/env python3
"""
Test script for PH2 dataset integration with Melanoma DIP Engine.
This script tests the complete pipeline with PH2 dataset images.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import image_processing as ip
import feature_extraction as fe
import utils
import config

def test_ph2_pipeline():
    """Test the complete pipeline with PH2 dataset images"""
    
    # PH2 dataset paths
    sample_image_path = r"C:\Users\Aakarsh Goyal\Downloads\archive\PH2Dataset\PH2 Dataset images\IMD427\IMD427_Dermoscopic_Image\IMD427.bmp"
    ground_truth_mask_path = r"C:\Users\Aakarsh Goyal\Downloads\archive\PH2Dataset\PH2 Dataset images\IMD427\IMD427_lesion\IMD427_lesion.bmp"
    
    print("🔬 Testing PH2 Dataset Integration with Melanoma DIP Engine")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(sample_image_path):
        print(f"❌ Dermoscopic image not found: {sample_image_path}")
        return False
    
    if not os.path.exists(ground_truth_mask_path):
        print(f"❌ Ground truth mask not found: {ground_truth_mask_path}")
        return False
    
    print(f"✅ PH2 dermoscopic image found: {os.path.basename(sample_image_path)}")
    print(f"✅ PH2 ground truth mask found: {os.path.basename(ground_truth_mask_path)}")
    
    try:
        # Step 1: Load and preprocess
        print("\n📥 Step 1: Loading and preprocessing image...")
        rgb_image, hsv_image, lab_image = ip.load_and_preprocess(sample_image_path)
        print(f"✅ Image loaded: {rgb_image.shape}")
        
        # Step 2: Hair removal
        print("\n🧹 Step 2: Hair removal...")
        hair_free_image, hair_metrics = ip.remove_hair(rgb_image)
        print(f"✅ Hair removal completed: {hair_metrics['hair_percentage']:.2f}% hair removed")
        
        # Step 3: Segmentation
        print("\n🎯 Step 3: Lesion segmentation...")
        binary_mask, main_contour, seg_metrics = ip.segment_lesion(hair_free_image)
        print(f"✅ Segmentation completed: {seg_metrics['confidence_score']:.3f} confidence")
        
        # Debug segmentation output
        print(f"🔍 Debug - Binary mask shape: {binary_mask.shape}")
        print(f"🔍 Debug - Binary mask unique values: {np.unique(binary_mask)}")
        print(f"🔍 Debug - Binary mask sum: {np.sum(binary_mask)}")
        print(f"🔍 Debug - Binary mask percentage white: {(np.sum(binary_mask > 0) / binary_mask.size) * 100:.2f}%")
        
        # Step 4: Feature extraction
        print("\n📊 Step 4: Feature extraction...")
        if main_contour is not None:
            features = fe.extract_all_features(
                original_image=hair_free_image,
                hsv_image=hsv_image,
                mask=binary_mask,
                contour=main_contour
            )
            print(f"✅ Features extracted: {features.get('num_features_extracted', 0)} features")
        else:
            print("❌ No valid contour found for feature extraction")
            return False
        
        # Step 5: Ground truth validation
        print("\n🎯 Step 5: Ground truth validation...")
        gt_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            # Handle potential mask inversion
            lesion_pixels = np.sum(gt_mask > 128)
            background_pixels = np.sum(gt_mask <= 128)
            if background_pixels > lesion_pixels:
                gt_mask = 255 - gt_mask
                print("🔄 Inverted ground truth mask")
            
            # Resize to match our processing size
            gt_mask_resized = cv2.resize(gt_mask, config.IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
            
            # Calculate Dice coefficient
            dice_score = utils.calculate_dice_coefficient(gt_mask_resized, binary_mask)
            print(f"✅ Dice coefficient: {dice_score:.3f}")
            
            # Calculate IoU
            intersection = np.sum((gt_mask_resized > 0) & (binary_mask > 0))
            union = np.sum((gt_mask_resized > 0) | (binary_mask > 0))
            iou = intersection / union if union > 0 else 0
            print(f"✅ IoU score: {iou:.3f}")
        else:
            print("❌ Could not load ground truth mask")
        
        # Step 6: Results summary
        print("\n📋 Step 6: Results Summary")
        print("-" * 40)
        utils.print_feature_summary(features)
        
        # Step 7: Visualization
        print("\n🖼️ Step 7: Creating visualizations...")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PH2 Dataset - Complete DIP Pipeline Results', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title('1. Original PH2 Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Hair-free image
        axes[0, 1].imshow(hair_free_image)
        axes[0, 1].set_title('2. After Hair Removal', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Our segmentation
        axes[0, 2].imshow(binary_mask, cmap='gray')
        axes[0, 2].set_title('3. Our Segmentation', fontweight='bold')
        axes[0, 2].axis('off')
        
        # Ground truth
        if gt_mask is not None:
            axes[1, 0].imshow(gt_mask_resized, cmap='gray')
            axes[1, 0].set_title('4. PH2 Ground Truth', fontweight='bold')
            axes[1, 0].axis('off')
            
            # Difference
            difference = cv2.absdiff(gt_mask_resized, binary_mask)
            axes[1, 1].imshow(difference, cmap='hot')
            axes[1, 1].set_title(f'5. Difference (Dice: {dice_score:.3f})', fontweight='bold')
            axes[1, 1].axis('off')
        
        # Overlay
        overlay = utils.create_overlay_image(hair_free_image, binary_mask)
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('6. Segmentation Overlay', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\n🎉 PH2 Dataset Integration Test Completed Successfully!")
        print("✅ All pipeline steps executed without errors")
        print("✅ Ready for Jupyter notebook usage")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        utils.create_error_report(e, "PH2 pipeline test")
        return False

if __name__ == "__main__":
    success = test_ph2_pipeline()
    if success:
        print("\n🚀 You can now run the Jupyter notebook with your PH2 dataset images!")
    else:
        print("\n⚠️ Please check the error messages above and fix any issues.")
