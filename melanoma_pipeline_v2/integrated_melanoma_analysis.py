"""
Integrated Melanoma Classification & Analysis Pipeline
-------------------------------------------------------
1. Classifies all images in train/images folder using YOLO classifier
2. For images detected as MELANOMA, performs complete analysis:
   - Segmentation
   - A (Asymmetry)
   - B (Border irregularity)
   - C (Color variation)
   - T (Texture analysis)
3. Outputs comprehensive metrics in organized CSV format
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from colorama import init, Fore, Style

# Add melanoma_dip_engine to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'melanoma_dip_engine'))

import image_processing as ip
import feature_extraction as fe
import utils
import config

# Initialize colorama
init(autoreset=True)

# Configuration
MODEL_CLS_PATH = r'melanoma_pipeline_v2\models\melanoma_classifier_opt.pt'
IMAGES_FOLDER = r"melanoma_dip_engine\data\train\images"
OUTPUT_CSV = r"melanoma_pipeline_v2\integrated_analysis_results.csv"

print("="*120)
print("INTEGRATED MELANOMA CLASSIFICATION & ANALYSIS PIPELINE")
print("="*120)

# Load classifier
print(f"\n🔧 Loading classifier model: {MODEL_CLS_PATH}")
cls_model = YOLO(MODEL_CLS_PATH)
print("✅ Classifier loaded successfully\n")

# Get all images
image_files = sorted([f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith('.jpg')])
total_images = len(image_files)

print(f"📁 Found {total_images} images in {IMAGES_FOLDER}")
print(f"🎯 Ground Truth: ALL images are MELANOMA\n")
print("="*120)
print("\nSTARTING ANALYSIS...\n")

# Results storage
all_results = []

# Process each image
for idx, img_file in enumerate(image_files, 1):
    image_path = os.path.join(IMAGES_FOLDER, img_file)
    
    print(f"{Fore.CYAN}[{idx}/{total_images}] Processing: {img_file}{Style.RESET_ALL}")
    
    # ============================================================================
    # STEP 1: CLASSIFICATION
    # ============================================================================
    cls_results = cls_model(image_path, verbose=False)[0]
    
    # Get predictions
    probs = cls_results.probs
    top1_idx = probs.top1
    top1_conf = probs.top1conf.item()
    top1_name = cls_results.names[top1_idx]
    
    # Get all class probabilities
    all_probs = probs.data.cpu().numpy()
    
    # Determine which index is melanoma
    melanoma_idx = 0 if cls_results.names[0].upper() == 'MELANOMA' else 1
    not_melanoma_idx = 1 - melanoma_idx
    
    melanoma_conf = all_probs[melanoma_idx]
    not_melanoma_conf = all_probs[not_melanoma_idx]
    
    is_melanoma = top1_name.upper() == 'MELANOMA'
    is_correct = is_melanoma  # Since all ground truth are melanoma
    
    # Initialize result dictionary
    result = {
        'image_name': img_file,
        'predicted_class': top1_name.upper(),
        'classification_confidence': top1_conf,
        'melanoma_probability': melanoma_conf,
        'not_melanoma_probability': not_melanoma_conf,
        'classification_correct': is_correct,
        'decision_margin': abs(melanoma_conf - not_melanoma_conf),
    }
    
    # ============================================================================
    # STEP 2: DETAILED ANALYSIS (Only if detected as MELANOMA)
    # ============================================================================
    if is_melanoma:
        print(f"  {Fore.GREEN}✓ Classified as MELANOMA ({melanoma_conf*100:.1f}%) - Performing detailed analysis...{Style.RESET_ALL}")
        
        try:
            # Load and preprocess
            rgb_image, hsv_image, lab_image = ip.load_and_preprocess(image_path)
            
            # Hair removal
            hair_free_image, quality_metrics = ip.remove_hair(rgb_image)
            
            # Segmentation
            binary_mask, main_contour, seg_metrics = ip.segment_lesion(hair_free_image)
            
            # Feature extraction (ABCT)
            features = fe.extract_all_features(
                original_image=hair_free_image,
                hsv_image=hsv_image,
                mask=binary_mask,
                contour=main_contour
            )
            
            # Extract features
            asymmetry_score = features.get('asymmetry_score', 0)
            border_irregularity = features.get('border_irregularity', 0)
            color_variation = features.get('color_variation', 0)
            texture_contrast = features.get('glcm_contrast', 0)
            texture_homogeneity = features.get('glcm_homogeneity', 0)
            texture_energy = features.get('glcm_energy', 0)
            texture_correlation = features.get('glcm_correlation', 0)
            
            # Calculate risk scores (normalized 0-1)
            asymmetry_risk = min(1.0, asymmetry_score * 10)
            border_risk = min(1.0, (border_irregularity - 1.0) / 5.0)
            color_risk = min(1.0, (color_variation - 1) / 4.0)
            texture_risk = min(1.0, texture_contrast / 200.0)
            
            # Overall risk
            overall_risk = (asymmetry_risk + border_risk + color_risk + texture_risk) / 4.0
            
            # Determine risk level
            if overall_risk < 0.3:
                risk_level = "LOW"
            elif overall_risk < 0.7:
                risk_level = "MODERATE"
            else:
                risk_level = "HIGH"
            
            # Add to results
            result.update({
                'analysis_performed': True,
                'asymmetry_score': asymmetry_score,
                'asymmetry_risk': asymmetry_risk,
                'border_irregularity': border_irregularity,
                'border_risk': border_risk,
                'color_variation': color_variation,
                'color_risk': color_risk,
                'texture_contrast': texture_contrast,
                'texture_homogeneity': texture_homogeneity,
                'texture_energy': texture_energy,
                'texture_correlation': texture_correlation,
                'texture_risk': texture_risk,
                'overall_risk_score': overall_risk,
                'risk_level': risk_level,
                'segmentation_quality': seg_metrics.get('confidence', 0) if seg_metrics else 0,
                'lesion_area_pixels': np.sum(binary_mask > 0) if binary_mask is not None else 0,
            })
            
            print(f"  {Fore.GREEN}✓ Analysis complete: {risk_level} risk (Score: {overall_risk:.3f}){Style.RESET_ALL}")
            print(f"    A={asymmetry_score:.3f}, B={border_irregularity:.3f}, C={color_variation}, T={texture_contrast:.3f}")
            
        except Exception as e:
            print(f"  {Fore.RED}✗ Analysis failed: {str(e)}{Style.RESET_ALL}")
            result.update({
                'analysis_performed': False,
                'analysis_error': str(e),
                'asymmetry_score': None,
                'asymmetry_risk': None,
                'border_irregularity': None,
                'border_risk': None,
                'color_variation': None,
                'color_risk': None,
                'texture_contrast': None,
                'texture_homogeneity': None,
                'texture_energy': None,
                'texture_correlation': None,
                'texture_risk': None,
                'overall_risk_score': None,
                'risk_level': None,
                'segmentation_quality': None,
                'lesion_area_pixels': None,
            })
    else:
        # Not classified as melanoma - no detailed analysis
        print(f"  {Fore.RED}✗ Classified as NOT MELANOMA ({not_melanoma_conf*100:.1f}%) - Skipping analysis{Style.RESET_ALL}")
        result.update({
            'analysis_performed': False,
            'asymmetry_score': None,
            'asymmetry_risk': None,
            'border_irregularity': None,
            'border_risk': None,
            'color_variation': None,
            'color_risk': None,
            'texture_contrast': None,
            'texture_homogeneity': None,
            'texture_energy': None,
            'texture_correlation': None,
            'texture_risk': None,
            'overall_risk_score': None,
            'risk_level': None,
            'segmentation_quality': None,
            'lesion_area_pixels': None,
        })
    
    all_results.append(result)
    print()

# ============================================================================
# FINAL SUMMARY & CSV EXPORT
# ============================================================================
print("="*120)
print("PROCESSING COMPLETE - GENERATING SUMMARY")
print("="*120)

# Create DataFrame
df = pd.DataFrame(all_results)

# Calculate summary statistics
total_processed = len(df)
correctly_classified = len(df[df['classification_correct'] == True])
incorrectly_classified = len(df[df['classification_correct'] == False])
analysis_performed_count = len(df[df['analysis_performed'] == True])

classification_accuracy = correctly_classified / total_processed * 100
false_negative_rate = incorrectly_classified / total_processed * 100

print(f"\n📊 CLASSIFICATION PERFORMANCE:")
print(f"   Total Images:              {total_processed}")
print(f"   Correctly Classified:      {correctly_classified} ({classification_accuracy:.1f}%)")
print(f"   Incorrectly Classified:    {incorrectly_classified} ({false_negative_rate:.1f}%)")
print(f"   False Negative Rate:       {false_negative_rate:.1f}%")

print(f"\n🔬 DETAILED ANALYSIS:")
print(f"   Analysis Performed:        {analysis_performed_count}")
print(f"   Analysis Skipped:          {total_processed - analysis_performed_count}")

# Analysis statistics for melanoma detections
melanoma_df = df[df['analysis_performed'] == True]

if len(melanoma_df) > 0:
    print(f"\n📈 MELANOMA ANALYSIS STATISTICS (n={len(melanoma_df)}):")
    print(f"   Asymmetry Score:           {melanoma_df['asymmetry_score'].mean():.3f} ± {melanoma_df['asymmetry_score'].std():.3f}")
    print(f"   Border Irregularity:       {melanoma_df['border_irregularity'].mean():.3f} ± {melanoma_df['border_irregularity'].std():.3f}")
    print(f"   Color Variation:           {melanoma_df['color_variation'].mean():.1f} ± {melanoma_df['color_variation'].std():.1f}")
    print(f"   Texture Contrast:          {melanoma_df['texture_contrast'].mean():.3f} ± {melanoma_df['texture_contrast'].std():.3f}")
    print(f"   Overall Risk Score:        {melanoma_df['overall_risk_score'].mean():.3f} ± {melanoma_df['overall_risk_score'].std():.3f}")
    
    # Risk level distribution
    risk_counts = melanoma_df['risk_level'].value_counts()
    print(f"\n⚠️  RISK LEVEL DISTRIBUTION:")
    for risk, count in risk_counts.items():
        print(f"   {risk:12}: {count} ({count/len(melanoma_df)*100:.1f}%)")

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Results saved to: {OUTPUT_CSV}")

# Display sample results (first 10 melanoma detections with analysis)
print("\n" + "="*120)
print("SAMPLE RESULTS (First 10 Melanoma Detections with Analysis)")
print("="*120)

sample_df = melanoma_df.head(10)
if len(sample_df) > 0:
    display_cols = [
        'image_name', 
        'melanoma_probability',
        'asymmetry_score',
        'border_irregularity', 
        'color_variation',
        'texture_contrast',
        'overall_risk_score',
        'risk_level'
    ]
    
    print(sample_df[display_cols].to_string(index=False))
else:
    print("No melanoma detections with successful analysis found!")

print("\n" + "="*120)
print("ANALYSIS COMPLETE")
print("="*120)
