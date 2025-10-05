#!/usr/bin/env python3
"""
Comprehensive PH2 Dataset Pipeline for Melanoma DIP Engine
=========================================================

This script implements a complete, production-ready pipeline for melanoma lesion analysis
using the PH2 dataset. It combines all segmentation improvements and provides detailed
logging, comprehensive metrics, and thorough documentation.

Features:
- Smart HSV + Intensity-based segmentation (eliminates false positives)
- Conservative morphological operations (preserves lesion shape)
- Comprehensive feature extraction (ABCD + Texture analysis)
- Detailed performance metrics and validation
- Medical-grade error handling and reporting
- Complete visualization suite

Author: Melanoma DIP Engine Team
Version: 2.0 (Enhanced with Precise Segmentation)
Date: 2024
"""

import os
import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import image_processing as ip
import feature_extraction as fe
import utils
import config

class MelanomaPipelineLogger:
    """Comprehensive logging system for the melanoma analysis pipeline"""
    
    def __init__(self):
        self.start_time = time.time()
        self.steps_completed = []
        self.metrics = {}
        self.errors = []
        
    def log_step(self, step_name, status="INFO", message="", metrics=None):
        """Log a pipeline step with timestamp and status"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        elapsed = time.time() - self.start_time
        
        if status == "SUCCESS":
            icon = "✅"
        elif status == "ERROR":
            icon = "❌"
        elif status == "WARNING":
            icon = "⚠️"
        else:
            icon = "ℹ️"
            
        log_entry = f"[{timestamp}] {icon} {step_name}: {message}"
        print(log_entry)
        
        self.steps_completed.append({
            'timestamp': timestamp,
            'elapsed': elapsed,
            'step': step_name,
            'status': status,
            'message': message,
            'metrics': metrics or {}
        })
        
        if metrics:
            self.metrics.update(metrics)
    
    def log_error(self, step_name, error):
        """Log an error with detailed information"""
        self.errors.append({
            'step': step_name,
            'error': str(error),
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        self.log_step(step_name, "ERROR", f"Failed: {str(error)}")
    
    def generate_summary(self):
        """Generate comprehensive pipeline summary"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE PIPELINE SUMMARY")
        print("="*80)
        
        print(f"⏱️  Total Execution Time: {total_time:.2f} seconds")
        print(f"📋 Steps Completed: {len(self.steps_completed)}")
        print(f"❌ Errors Encountered: {len(self.errors)}")
        
        print(f"\n🎯 KEY METRICS:")
        for key, value in self.metrics.items():
            if isinstance(value, float):
                print(f"   • {key}: {value:.3f}")
            else:
                print(f"   • {key}: {value}")
        
        if self.errors:
            print(f"\n🚨 ERRORS SUMMARY:")
            for error in self.errors:
                print(f"   • {error['step']}: {error['error']}")
        
        print("="*80)


def test_ph2_pipeline():
    """
    Comprehensive PH2 Dataset Pipeline with Enhanced Segmentation
    
    This function implements the complete melanoma analysis pipeline with:
    - Smart HSV + Intensity-based segmentation (eliminates false positives)
    - Conservative morphological operations (preserves lesion shape)
    - Comprehensive feature extraction (ABCD + Texture analysis)
    - Detailed performance metrics and validation
    - Medical-grade error handling and reporting
    
    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    
    # Initialize comprehensive logging
    logger = MelanomaPipelineLogger()
    
    # PH2 dataset paths - Update these paths for your specific dataset location
    sample_image_path = r"C:\Users\Aakarsh Goyal\Downloads\archive\PH2Dataset\PH2 Dataset images\IMD427\IMD427_Dermoscopic_Image\IMD427.bmp"
    ground_truth_mask_path = r"C:\Users\Aakarsh Goyal\Downloads\archive\PH2Dataset\PH2 Dataset images\IMD427\IMD427_lesion\IMD427_lesion.bmp"
    
    logger.log_step("PIPELINE_INIT", "INFO", "Starting comprehensive PH2 melanoma analysis pipeline")
    logger.log_step("CONFIG_LOAD", "INFO", f"Image size: {config.IMAGE_SIZE}, Min lesion area: {config.MIN_LESION_AREA}")
    
    # Step 1: File Validation and Path Verification
    try:
        logger.log_step("FILE_VALIDATION", "INFO", "Validating input file paths")
        
        if not os.path.exists(sample_image_path):
            raise FileNotFoundError(f"Dermoscopic image not found: {sample_image_path}")
        
        if not os.path.exists(ground_truth_mask_path):
            raise FileNotFoundError(f"Ground truth mask not found: {ground_truth_mask_path}")
        
        # Get file sizes for validation
        image_size = os.path.getsize(sample_image_path) / (1024 * 1024)  # MB
        mask_size = os.path.getsize(ground_truth_mask_path) / (1024 * 1024)  # MB
        
        logger.log_step("FILE_VALIDATION", "SUCCESS", 
                       f"Found PH2 image ({os.path.basename(sample_image_path)}, {image_size:.1f}MB) and mask ({os.path.basename(ground_truth_mask_path)}, {mask_size:.1f}MB)")
        
    except Exception as e:
        logger.log_error("FILE_VALIDATION", e)
        return False
    
    try:
        # Step 2: Image Loading and Preprocessing
        logger.log_step("IMAGE_LOADING", "INFO", "Loading and preprocessing PH2 dermoscopic image")
        
        rgb_image, hsv_image, lab_image = ip.load_and_preprocess(sample_image_path)
        
        # Validate image properties
        image_properties = {
            'shape': rgb_image.shape,
            'dtype': str(rgb_image.dtype),
            'memory_size_mb': rgb_image.nbytes / (1024 * 1024),
            'color_range': f"R:[{rgb_image[:,:,0].min()}-{rgb_image[:,:,0].max()}], G:[{rgb_image[:,:,1].min()}-{rgb_image[:,:,1].max()}], B:[{rgb_image[:,:,2].min()}-{rgb_image[:,:,2].max()}]"
        }
        
        logger.log_step("IMAGE_LOADING", "SUCCESS", 
                       f"Loaded {image_properties['shape']} image ({image_properties['memory_size_mb']:.1f}MB)", 
                       image_properties)
        
        # Step 3: Hair Removal Processing
        logger.log_step("HAIR_REMOVAL", "INFO", "Applying DullRazor technique for hair artifact removal")
        
        hair_free_image, hair_metrics = ip.remove_hair(rgb_image)
        
        # Calculate hair removal effectiveness
        hair_removal_metrics = {
            'hair_percentage_removed': hair_metrics.get('hair_percentage', 0),
            'processing_complete': hair_metrics.get('processing_complete', False),
            'image_quality_score': hair_metrics.get('quality_score', 0)
        }
        
        logger.log_step("HAIR_REMOVAL", "SUCCESS", 
                       f"Hair removal completed: {hair_removal_metrics['hair_percentage_removed']:.2f}% hair removed", 
                       hair_removal_metrics)
        
        # Step 4: Enhanced Lesion Segmentation
        logger.log_step("LESION_SEGMENTATION", "INFO", "Applying smart HSV + Intensity-based segmentation")
        
        binary_mask, main_contour, seg_metrics = ip.segment_lesion(hair_free_image)
        
        # Calculate comprehensive segmentation metrics
        lesion_pixels = np.sum(binary_mask > 0)
        total_pixels = binary_mask.size
        coverage_percentage = (lesion_pixels / total_pixels) * 100
        
        segmentation_metrics = {
            'confidence_score': seg_metrics.get('confidence_score', 0),
            'lesion_area_pixels': lesion_pixels,
            'coverage_percentage': coverage_percentage,
            'contour_area': cv2.contourArea(main_contour) if main_contour is not None else 0,
            'shape_quality': seg_metrics.get('shape_quality', 0),
            'binary_mask_shape': binary_mask.shape,
            'unique_values': np.unique(binary_mask).tolist()
        }
        
        logger.log_step("LESION_SEGMENTATION", "SUCCESS", 
                       f"Segmentation completed: {segmentation_metrics['confidence_score']:.3f} confidence, {segmentation_metrics['coverage_percentage']:.1f}% coverage", 
                       segmentation_metrics)
        
        # Step 5: Comprehensive Feature Extraction
        logger.log_step("FEATURE_EXTRACTION", "INFO", "Extracting ABCD + Texture features from segmented lesion")
        
        if main_contour is None:
            raise ValueError("No valid contour found for feature extraction - segmentation failed")
        
        features = fe.extract_all_features(
            original_image=hair_free_image,
            hsv_image=hsv_image,
            mask=binary_mask,
            contour=main_contour
        )
        
        # Validate feature extraction
        num_features = features.get('num_features_extracted', 0)
        feature_metrics = {
            'num_features_extracted': num_features,
            'asymmetry_score': features.get('asymmetry_score', 0),
            'border_irregularity': features.get('border_irregularity', 0),
            'color_variation': features.get('color_variation', 0),
            'diameter_mm': features.get('diameter_mm', 0),
            'texture_contrast': features.get('texture_contrast', 0),
            'texture_homogeneity': features.get('texture_homogeneity', 0)
        }
        
        logger.log_step("FEATURE_EXTRACTION", "SUCCESS", 
                       f"Extracted {num_features} features including ABCD + Texture analysis", 
                       feature_metrics)
        
        # Step 6: Ground Truth Validation and Performance Metrics
        logger.log_step("GROUND_TRUTH_VALIDATION", "INFO", "Loading and validating against PH2 ground truth")
        
        gt_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            raise ValueError("Could not load ground truth mask")
        
        # Handle potential mask inversion (PH2 dataset specific)
        lesion_pixels_gt = np.sum(gt_mask > 128)
        background_pixels_gt = np.sum(gt_mask <= 128)
        
        # PH2 dataset uses BLACK for lesion, WHITE for background
        # We need to invert to match our convention: WHITE for lesion, BLACK for background
        if lesion_pixels_gt > background_pixels_gt:
            # If more pixels are bright (white), then lesion is white - need to invert
            gt_mask = 255 - gt_mask
            logger.log_step("GROUND_TRUTH_VALIDATION", "INFO", "Inverted ground truth mask (PH2 uses black for lesion)")
        
        # Resize to match our processing size
        gt_mask_resized = cv2.resize(gt_mask, config.IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        
        # Ensure both masks use same convention: WHITE = lesion, BLACK = background
        logger.log_step("GROUND_TRUTH_VALIDATION", "INFO", f"Standardized color convention: WHITE=lesion, BLACK=background")
        
        # DEBUG: Verify color conventions are correct
        our_white_pixels = np.sum(binary_mask > 0)
        our_total_pixels = binary_mask.size
        our_white_percentage = (our_white_pixels / our_total_pixels) * 100
        
        gt_white_pixels = np.sum(gt_mask_resized > 0)
        gt_total_pixels = gt_mask_resized.size
        gt_white_percentage = (gt_white_pixels / gt_total_pixels) * 100
        
        logger.log_step("COLOR_CONVENTION_DEBUG", "INFO", 
                       f"Our mask: {our_white_percentage:.1f}% white (lesion) | GT mask: {gt_white_percentage:.1f}% white (lesion)")
        
        # Calculate comprehensive performance metrics
        dice_score = utils.calculate_dice_coefficient(gt_mask_resized, binary_mask)
        
        intersection = np.sum((gt_mask_resized > 0) & (binary_mask > 0))
        union = np.sum((gt_mask_resized > 0) | (binary_mask > 0))
        iou = intersection / union if union > 0 else 0
        
        # Calculate additional metrics
        gt_coverage = (np.sum(gt_mask_resized > 0) / gt_mask_resized.size) * 100
        our_coverage = (np.sum(binary_mask > 0) / binary_mask.size) * 100
        coverage_gap = abs(gt_coverage - our_coverage)
        
        # Calculate precision and recall
        true_positives = intersection
        false_positives = np.sum((binary_mask > 0) & (gt_mask_resized == 0))
        false_negatives = np.sum((gt_mask_resized > 0) & (binary_mask == 0))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        validation_metrics = {
            'dice_coefficient': dice_score,
            'iou_score': iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'gt_coverage_percentage': gt_coverage,
            'our_coverage_percentage': our_coverage,
            'coverage_gap_percentage': coverage_gap,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives)
        }
        
        logger.log_step("GROUND_TRUTH_VALIDATION", "SUCCESS", 
                       f"Validation completed: Dice={dice_score:.3f}, IoU={iou:.3f}, F1={f1_score:.3f}", 
                       validation_metrics)
        
        # Step 7: Clinical Feature Analysis Report
        logger.log_step("CLINICAL_REPORT", "INFO", "Generating comprehensive clinical feature analysis")
        
        utils.print_feature_summary(features)
        
        # Step 8: Comprehensive Visualization Generation
        logger.log_step("VISUALIZATION", "INFO", "Creating comprehensive visualization suite")
        
        # Create enhanced visualization with detailed metrics
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle(f'PH2 Dataset - Enhanced Melanoma DIP Pipeline Results\nDice: {dice_score:.3f} | IoU: {iou:.3f} | F1: {f1_score:.3f} | Coverage: {our_coverage:.1f}%', 
                     fontsize=16, fontweight='bold')
        
        # Original image with metadata
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title(f'1. Original PH2 Image\n{image_properties["shape"]} | {image_properties["memory_size_mb"]:.1f}MB', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Hair-free image with metrics
        axes[0, 1].imshow(hair_free_image)
        axes[0, 1].set_title(f'2. After Hair Removal\n{hair_removal_metrics["hair_percentage_removed"]:.1f}% hair removed', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Our segmentation with detailed metrics
        axes[0, 2].imshow(binary_mask, cmap='gray')
        axes[0, 2].set_title(f'3. Enhanced Segmentation\n{segmentation_metrics["coverage_percentage"]:.1f}% coverage | Confidence: {segmentation_metrics["confidence_score"]:.3f}', fontweight='bold')
        axes[0, 2].axis('off')
        
        # Ground truth with coverage info (now using same color convention)
        axes[1, 0].imshow(gt_mask_resized, cmap='gray')
        axes[1, 0].set_title(f'4. PH2 Ground Truth (Corrected)\n{validation_metrics["gt_coverage_percentage"]:.1f}% coverage (WHITE=lesion)', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Difference with comprehensive metrics
        difference = cv2.absdiff(gt_mask_resized, binary_mask)
        axes[1, 1].imshow(difference, cmap='hot')
        axes[1, 1].set_title(f'5. Segmentation Difference\nPrecision: {precision:.3f} | Recall: {recall:.3f}', fontweight='bold')
        axes[1, 1].axis('off')
        
        # Enhanced overlay with feature information
        overlay = utils.create_overlay_image(hair_free_image, binary_mask)
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title(f'6. Segmentation Overlay\nTP: {validation_metrics["true_positives"]} | FP: {validation_metrics["false_positives"]} | FN: {validation_metrics["false_negatives"]}', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('enhanced_pipeline_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.log_step("VISUALIZATION", "SUCCESS", "Comprehensive visualization saved as 'enhanced_pipeline_results.png'")
        
        # Step 9: Pipeline Completion
        logger.log_step("PIPELINE_COMPLETION", "SUCCESS", "All pipeline steps completed successfully")
        
        # Generate comprehensive summary
        logger.generate_summary()
        
        return True
        
    except Exception as e:
        logger.log_error("PIPELINE_EXECUTION", e)
        logger.generate_summary()
        return False


def main():
    """
    Main execution function with comprehensive error handling and user guidance
    
    This function provides:
    - Clear execution instructions
    - Comprehensive error reporting
    - Success confirmation with next steps
    - Medical disclaimer and usage guidelines
    """
    
    print("="*80)
    print("🏥 MELANOMA DIP ENGINE - PH2 DATASET PIPELINE")
    print("="*80)
    print("📋 This pipeline implements:")
    print("   • Smart HSV + Intensity-based segmentation")
    print("   • Conservative morphological operations")
    print("   • Comprehensive ABCD + Texture feature extraction")
    print("   • Medical-grade validation and reporting")
    print("="*80)
    
    try:
        success = test_ph2_pipeline()
        
        if success:
            print("\n" + "="*80)
            print("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("✅ All processing steps completed without errors")
            print("✅ Enhanced segmentation with precise boundaries")
            print("✅ Comprehensive feature extraction completed")
            print("✅ Performance metrics calculated and validated")
            print("✅ High-quality visualizations generated")
            print("\n📁 Generated Files:")
            print("   • enhanced_pipeline_results.png - Comprehensive visualization")
            print("   • Detailed console logs with metrics")
            print("\n🚀 Next Steps:")
            print("   • Review the generated visualizations")
            print("   • Analyze the performance metrics")
            print("   • Use the Jupyter notebook for interactive analysis")
            print("   • Validate results against clinical standards")
        else:
            print("\n" + "="*80)
            print("❌ PIPELINE EXECUTION FAILED")
            print("="*80)
            print("⚠️  Please review the error messages above")
            print("⚠️  Check file paths and dependencies")
            print("⚠️  Ensure PH2 dataset is properly configured")
            
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    finally:
        print("\n" + "="*80)
        print("⚠️  MEDICAL DISCLAIMER:")
        print("   This tool is for research and educational purposes only.")
        print("   It does not constitute medical advice or diagnosis.")
        print("   Always consult with qualified healthcare professionals.")
        print("="*80)


if __name__ == "__main__":
    main()
