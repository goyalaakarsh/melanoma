import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ultralytics import YOLO
import os
import sys
import pandas as pd
from datetime import datetime

# --- IMPORT CUSTOM DIP MODULES ---
try:
    import src.image_processing as ip
    import src.feature_extraction as fe
    import src.utils as utils
    import src.config
except ImportError as e:
    print(f"❌ Critical Error: Could not import DIP modules. {e}")
    sys.exit(1)

# --- CONFIGURATION: LOCAL PATHS ---
MODEL_CLS_PATH = r'melanoma_pipeline_v2\models\melanoma_classifier.pt'
MODEL_DET_PATH = r'melanoma_pipeline_v2\models\melanoma_detection.pt'
MODEL_SEG_PATH = r'melanoma_pipeline_v2\models\melanoma_segmentation.pt'

# --- CONFIGURATION: CONSTANTS ---
PIXELS_PER_MM = 45.0 

# --- HELPER FUNCTIONS ---

def load_model(path, task_name):
    if not os.path.exists(path):
        print(f"❌ Error: {task_name} model not found at {path}")
        sys.exit(1)
    print(f"✅ Loaded {task_name} Model")
    return YOLO(path)

def get_guided_ai_mask(model, image, roi_box=None):
    """
    Runs YOLO Segmentation guided by the Detection Box (ROI).
    This prevents the model from segmenting dark corners or artifacts.
    """
    h, w = image.shape[:2]
    processed_image = image.copy()
    
    # 1. Apply ROI Masking (Crucial Step)
    if roi_box is not None:
        x1, y1, x2, y2 = roi_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        black_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(black_mask, (x1, y1), (x2, y2), 255, -1)
        processed_image = cv2.bitwise_and(image, image, mask=black_mask)

    # 2. Run Inference
    results = model(processed_image, verbose=False, retina_masks=True)
    result = results[0]
    
    final_mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is not None:
        masks_data = result.masks.data.cpu().numpy()
        combined = np.zeros((masks_data.shape[1], masks_data.shape[2]))
        for m in masks_data:
            combined = np.maximum(combined, m)
            
        final_mask = cv2.resize(combined, (w, h))
        final_mask = (final_mask > 0.5).astype(np.uint8) * 255
        
        # 3. Post-Processing: Keep only the largest blob
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            clean_mask = np.zeros_like(final_mask)
            cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            final_mask = clean_mask
            
        # Smooth edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
    return final_mask

def process_single_image(image_path, cls_model, det_model, seg_model):
    """Process a single image and return results dictionary"""
    result = {
        'image_name': os.path.basename(image_path),
        'diagnosis': 'ERROR',
        'confidence': 0.0,
        'detection_success': False,
        'segmentation_success': False,
        'diameter_mm': 0.0,
        'area_px': 0,
        'asymmetry': 0.0,
        'border_irregularity': 0.0,
        'color_variation': 0.0,
        'glcm_contrast': 0.0,
        'risk_level': 'UNKNOWN',
        'status': 'Processing...'
    }
    
    try:
        # 1. PREPROCESSING
        rgb_img, hsv_img, _ = ip.load_and_preprocess(image_path)
        hair_free_img, _ = ip.remove_hair(rgb_img)
        
        # 2. CLASSIFICATION
        cls_results = cls_model(image_path, verbose=False)[0]
        diagnosis = cls_results.names[cls_results.probs.top1].upper()
        confidence = cls_results.probs.top1conf.item()
        
        result['diagnosis'] = diagnosis
        result['confidence'] = confidence
        
        if diagnosis != 'MELANOMA':
            result['risk_level'] = 'LOW'
            result['status'] = 'Benign - No further analysis needed'
            return result
        
        result['risk_level'] = 'HIGH'
        
        # 3. DETECTION
        det_results = det_model(image_path, verbose=False)[0]
        roi_box = None
        
        if len(det_results.boxes) > 0:
            best_box = det_results.boxes[0]
            roi_box = best_box.xyxy[0].cpu().numpy().astype(int)
            result['detection_success'] = True
        else:
            h, w = rgb_img.shape[:2]
            roi_box = [int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8)]
            result['detection_success'] = False
        
        # 4. SEGMENTATION
        ai_mask = get_guided_ai_mask(seg_model, hair_free_img, roi_box)
        
        if np.sum(ai_mask) == 0:
            result['segmentation_success'] = False
            result['status'] = 'Segmentation Failed'
            return result
        
        result['segmentation_success'] = True
        
        # 5. FEATURE EXTRACTION
        contours, _ = cv2.findContours(ai_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea) if contours else None
        hsv_clean = cv2.cvtColor(hair_free_img, cv2.COLOR_RGB2HSV)
        
        features = fe.extract_all_features(
            original_image=hair_free_img,
            hsv_image=hsv_clean,
            mask=ai_mask,
            contour=main_contour
        )
        
        # Calculate metrics
        area_px = np.sum(ai_mask > 0)
        diameter_px = features.get('largest_diameter_pixels', 0)
        if diameter_px == 0:
            diameter_px = np.sqrt(area_px / np.pi) * 2
        
        result['area_px'] = area_px
        result['diameter_mm'] = diameter_px / PIXELS_PER_MM
        result['asymmetry'] = features.get('asymmetry', 0)
        result['border_irregularity'] = features.get('border_irregularity', 0)
        result['color_variation'] = features.get('color_variation', 0)
        result['glcm_contrast'] = features.get('glcm_contrast', 0)
        result['status'] = 'Complete'
        
    except Exception as e:
        result['status'] = f'Error: {str(e)}'
    
    return result

# --- BATCH PROCESSING ---

def run_batch_analysis(images_folder):
    print("="*80)
    print(f"🧬 BATCH MEDICAL AI ANALYSIS")
    print("="*80)
    
    # 1. LOAD MODELS
    print("📦 Loading AI Models...")
    cls_model = load_model(MODEL_CLS_PATH, "Classification")
    det_model = load_model(MODEL_DET_PATH, "Detection")
    seg_model = load_model(MODEL_SEG_PATH, "Segmentation")
    
    # 2. GET ALL IMAGES
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n📁 Found {len(image_files)} images to process\n")
    
    # 3. PROCESS EACH IMAGE
    results = []
    for i, img_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing {img_file}...")
        image_path = os.path.join(images_folder, img_file)
        result = process_single_image(image_path, cls_model, det_model, seg_model)
        results.append(result)
        print(f"    → {result['diagnosis']} ({result['confidence']:.1%}) - {result['status']}\n")
    
    # 4. CREATE DATAFRAME
    df = pd.DataFrame(results)
    
    # 5. SAVE RESULTS
    output_csv = 'melanoma_pipeline_v2/batch_analysis_results.csv'
    df.to_csv(output_csv, index=False)
    print(f"✅ Results saved to: {output_csv}\n")
    
    # 6. DISPLAY TABLE
    print("="*80)
    print("RESULTS SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # 7. CREATE VISUAL DASHBOARD
    create_visual_dashboard(df, results)
    
    return df

def create_visual_dashboard(df, results):
    """Create comprehensive visual dashboard"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('BATCH MELANOMA ANALYSIS DASHBOARD', fontsize=22, fontweight='bold', y=0.98)
    
    # 1. DIAGNOSIS DISTRIBUTION (PIE CHART)
    ax1 = fig.add_subplot(gs[0, 0])
    diagnosis_counts = df['diagnosis'].value_counts()
    colors = ['#e74c3c' if d == 'MELANOMA' else '#27ae60' for d in diagnosis_counts.index]
    ax1.pie(diagnosis_counts.values, labels=diagnosis_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax1.set_title('Diagnosis Distribution', fontweight='bold', fontsize=12)
    
    # 2. CONFIDENCE SCORES (BOX PLOT)
    ax2 = fig.add_subplot(gs[0, 1])
    melanoma_conf = df[df['diagnosis'] == 'MELANOMA']['confidence'].values
    benign_conf = df[df['diagnosis'] != 'MELANOMA']['confidence'].values
    
    box_data = [d for d in [melanoma_conf, benign_conf] if len(d) > 0]
    labels = []
    if len(melanoma_conf) > 0:
        labels.append('Melanoma')
    if len(benign_conf) > 0:
        labels.append('Benign')
    
    if box_data:
        bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#e74c3c', '#27ae60'][:len(labels)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax2.set_ylabel('Confidence Score', fontweight='bold')
    ax2.set_title('Classification Confidence', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 1.1)
    
    # 3. PROCESSING SUCCESS RATES (BAR CHART)
    ax3 = fig.add_subplot(gs[0, 2])
    success_metrics = {
        'Detection': df['detection_success'].sum(),
        'Segmentation': df['segmentation_success'].sum(),
        'Complete': len(df[df['status'] == 'Complete'])
    }
    bars = ax3.bar(success_metrics.keys(), success_metrics.values(), 
                   color=['#3498db', '#9b59b6', '#27ae60'], alpha=0.8)
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('Processing Success Rates', fontweight='bold', fontsize=12)
    ax3.set_ylim(0, len(df) + 1)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 4. LESION SIZE DISTRIBUTION (HISTOGRAM)
    ax4 = fig.add_subplot(gs[1, 0])
    melanoma_sizes = df[df['diagnosis'] == 'MELANOMA']['diameter_mm'].values
    if len(melanoma_sizes) > 0:
        ax4.hist(melanoma_sizes, bins=10, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax4.axvline(6, color='red', linestyle='--', linewidth=2, label='6mm threshold')
        ax4.legend()
    ax4.set_xlabel('Diameter (mm)', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Melanoma Lesion Size Distribution', fontweight='bold', fontsize=12)
    
    # 5. ASYMMETRY vs BORDER IRREGULARITY (SCATTER)
    ax5 = fig.add_subplot(gs[1, 1])
    melanoma_df = df[df['diagnosis'] == 'MELANOMA']
    benign_df = df[df['diagnosis'] != 'MELANOMA']
    
    if len(melanoma_df) > 0:
        ax5.scatter(melanoma_df['asymmetry'], melanoma_df['border_irregularity'], 
                   c='red', s=100, alpha=0.6, label='Melanoma', edgecolors='black')
    if len(benign_df) > 0:
        ax5.scatter(benign_df['asymmetry'], benign_df['border_irregularity'], 
                   c='green', s=100, alpha=0.6, label='Benign', edgecolors='black')
    
    ax5.set_xlabel('Asymmetry Score', fontweight='bold')
    ax5.set_ylabel('Border Irregularity', fontweight='bold')
    ax5.set_title('ABCD Features: Asymmetry vs Border', fontweight='bold', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. COLOR VARIATION (BAR CHART)
    ax6 = fig.add_subplot(gs[1, 2])
    melanoma_df_sorted = df[df['diagnosis'] == 'MELANOMA'].sort_values('color_variation', ascending=False)
    if len(melanoma_df_sorted) > 0:
        colors_bar = ax6.barh(range(len(melanoma_df_sorted)), melanoma_df_sorted['color_variation'].values, 
                              color='#f39c12', alpha=0.8, edgecolor='black')
        ax6.set_yticks(range(len(melanoma_df_sorted)))
        ax6.set_yticklabels(melanoma_df_sorted['image_name'].values, fontsize=8)
        ax6.set_xlabel('Color Variation Score', fontweight='bold')
        ax6.set_title('Melanoma Color Diversity', fontweight='bold', fontsize=12)
    
    # 7. FEATURE HEATMAP (TABLE)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('tight')
    ax7.axis('off')
    
    melanoma_features = df[df['diagnosis'] == 'MELANOMA'][['image_name', 'asymmetry', 'border_irregularity', 
                                                             'color_variation', 'glcm_contrast', 'diameter_mm']]
    
    if len(melanoma_features) > 0:
        table_data = []
        for _, row in melanoma_features.iterrows():
            table_data.append([
                row['image_name'],
                f"{row['asymmetry']:.3f}",
                f"{row['border_irregularity']:.2f}",
                f"{row['color_variation']:.2f}",
                f"{row['glcm_contrast']:.1f}",
                f"{row['diameter_mm']:.2f}"
            ])
        
        table = ax7.table(cellText=table_data,
                         colLabels=['Image', 'Asymmetry', 'Border Irreg.', 'Color Var.', 'Texture', 'Diameter (mm)'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color header
        for i in range(6):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows
        for i in range(1, len(table_data) + 1):
            for j in range(6):
                table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    
    ax7.set_title('Detailed Melanoma Feature Matrix', fontweight='bold', fontsize=14, pad=20)
    
    # 8. RISK SUMMARY (TEXT)
    ax8 = fig.add_subplot(gs[3, 0])
    ax8.axis('off')
    
    total_images = len(df)
    melanoma_count = len(df[df['diagnosis'] == 'MELANOMA'])
    high_risk = len(df[(df['diagnosis'] == 'MELANOMA') & (df['diameter_mm'] > 6)])
    
    summary_text = (
        f"CLINICAL SUMMARY\n"
        f"{'='*30}\n\n"
        f"Total Images Analyzed: {total_images}\n"
        f"Melanoma Detected: {melanoma_count} ({melanoma_count/total_images*100:.1f}%)\n"
        f"High-Risk Cases (>6mm): {high_risk}\n\n"
        f"Detection Success: {df['detection_success'].sum()}/{total_images}\n"
        f"Segmentation Success: {df['segmentation_success'].sum()}/{total_images}\n"
    )
    
    ax8.text(0.1, 0.5, summary_text, fontsize=11, family='monospace', 
             va='center', bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.3))
    
    # 9. RECOMMENDATIONS (TEXT)
    ax9 = fig.add_subplot(gs[3, 1:])
    ax9.axis('off')
    
    recommendations = "CLINICAL RECOMMENDATIONS\n" + "="*50 + "\n\n"
    
    for _, row in df[df['diagnosis'] == 'MELANOMA'].iterrows():
        risk = "🔴 URGENT" if row['diameter_mm'] > 6 else "🟡 MONITOR"
        recommendations += f"{risk} {row['image_name']}: "
        
        if row['diameter_mm'] > 6:
            recommendations += "Immediate dermatoscopy referral (>6mm diameter)\n"
        elif row['asymmetry'] > 0.3:
            recommendations += "High asymmetry - Biopsy recommended\n"
        elif row['border_irregularity'] > 3:
            recommendations += "Irregular borders - Close monitoring\n"
        else:
            recommendations += "Routine follow-up in 3 months\n"
    
    ax9.text(0.05, 0.5, recommendations, fontsize=10, family='monospace', 
             va='center', bbox=dict(boxstyle="round", facecolor='#ffe6e6', alpha=0.5))
    
    plt.savefig('melanoma_pipeline_v2/batch_analysis_dashboard.png', dpi=150, bbox_inches='tight')
    print("✅ Visual dashboard saved to: melanoma_pipeline_v2/batch_analysis_dashboard.png")
    plt.show()

if __name__ == "__main__":
    IMAGES_FOLDER = r"melanoma_dip_engine\data\images"
    
    if os.path.exists(IMAGES_FOLDER):
        df_results = run_batch_analysis(IMAGES_FOLDER)
    else:
        print(f"❌ Images folder not found: {IMAGES_FOLDER}")
