import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ultralytics import YOLO
import os
import sys

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
# Adjust this based on image resolution to get realistic mm sizes
# For 640x640 dermoscopy, ~45 pixels usually equals 1mm
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
    # If detection found a box, we black out everything else.
    if roi_box is not None:
        x1, y1, x2, y2 = roi_box
        # Ensure coords are within bounds
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

def create_master_dashboard(original, hair_free, det_plot, overlay, mask, features, diagnosis, conf):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig)
    
    fig.suptitle(f"DERMATOLOGY AI DIAGNOSTIC REPORT\nDiagnosis: {diagnosis} ({conf:.2%})", 
                 fontsize=20, fontweight='bold', color='#2c3e50', y=0.96)

    # Row 1
    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(original); ax1.set_title("1. Patient Scan", fontweight='bold'); ax1.axis('off')
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(det_plot); ax2.set_title("2. AI Localization", fontweight='bold'); ax2.axis('off')
    ax3 = fig.add_subplot(gs[0, 2]); ax3.imshow(hair_free); ax3.set_title("3. Artifact Removal", fontweight='bold'); ax3.axis('off')
    ax4 = fig.add_subplot(gs[0, 3]); ax4.imshow(overlay); ax4.set_title("4. Morphology Segmentation", fontweight='bold'); ax4.axis('off')

    # Row 2
    ax5 = fig.add_subplot(gs[1, 0]); ax5.imshow(mask, cmap='gray'); ax5.set_title("Binary Mask Analysis", fontweight='bold'); ax5.axis('off')
    
    # Metrics Chart
    ax6 = fig.add_subplot(gs[1, 1:3])
    metrics = ['Asymmetry', 'Border', 'Color Var', 'Texture']
    
    # Safely get values
    val_asym = features.get('asymmetry', 0)
    val_bord = (features.get('border_irregularity', 1) - 1) / 5
    val_col = features.get('color_variation', 1) / 10
    val_tex = features.get('glcm_contrast', 0) / 100
    
    values = [val_asym, val_bord, val_col, val_tex]
    bars = ax6.bar(metrics, values, color=['#e74c3c', '#f39c12', '#3498db', '#9b59b6'], alpha=0.8)
    ax6.set_title("ABCD & Texture Quantitative Metrics", fontweight='bold')
    ax6.set_ylim(0, 1.2)
    
    for bar, val in zip(bars, [val_asym, features.get('border_irregularity', 1), features.get('color_variation', 1), features.get('glcm_contrast', 0)]):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.2f}", ha='center', va='bottom', fontweight='bold')

    # Summary
    ax7 = fig.add_subplot(gs[1, 3]); ax7.axis('off')
    risk_level = "HIGH" if diagnosis == "MELANOMA" else "LOW"
    risk_color = "red" if risk_level == "HIGH" else "green"
    
    # Recalculate Area/Diameter for display if missing
    area_px = np.sum(mask > 0)
    diameter_px = features.get('largest_diameter_pixels', 0)
    if diameter_px == 0: # Estimate if missing
        diameter_px = np.sqrt(area_px / np.pi) * 2
    
    diameter_mm = diameter_px / PIXELS_PER_MM
    
    summary_text = (
        f"CLINICAL SUMMARY\n----------------\n"
        f"Primary Diagnosis: {diagnosis}\n"
        f"Risk Level: {risk_level}\n\n"
        f"Morphology:\n"
        f"• Diameter: {diameter_mm:.1f} mm\n"
        f"• Area: {area_px:,} px\n"
        f"• Asymmetry: {'High' if val_asym > 0.2 else 'Low'}\n\n"
        f"Recommendation:\n"
        f"{'Immediate Dermatoscopy Referral' if risk_level == 'HIGH' else 'Routine Monitoring'}"
    )
    ax7.text(0, 0.5, summary_text, fontsize=12, family='monospace', va='center', bbox=dict(boxstyle="round", facecolor=risk_color, alpha=0.1))

    # Row 3 (Texture Viz)
    ax8 = fig.add_subplot(gs[2, :])
    lbp_viz = utils.visualize_texture(hair_free, mask)
    ax8.imshow(lbp_viz)
    ax8.set_title("Advanced Feature Extraction (LBP Texture Analysis)", fontweight='bold')
    ax8.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

# --- MAIN PIPELINE ---

def run_pipeline(image_path):
    print("="*60)
    print(f"🧬 STARTING MEDICAL AI PIPELINE: {os.path.basename(image_path)}")
    print("="*60)
    
    # 1. LOAD MODELS
    print("📦 Loading AI Models...")
    cls_model = load_model(MODEL_CLS_PATH, "Classification")
    det_model = load_model(MODEL_DET_PATH, "Detection")
    seg_model = load_model(MODEL_SEG_PATH, "Segmentation")
    
    # 2. PREPROCESSING
    print("\n1️⃣  Phase 1: DIP Preprocessing...")
    try:
        rgb_img, hsv_img, _ = ip.load_and_preprocess(image_path)
        hair_free_img, _ = ip.remove_hair(rgb_img)
        print("   ✅ Artifacts removed (DullRazor Algorithm)")
    except Exception as e:
        print(f"   ❌ Error loading image: {e}")
        return

    # 3. CLASSIFICATION
    print("\n2️⃣  Phase 2: AI Screening...")
    cls_results = cls_model(image_path, verbose=False)[0]
    diagnosis = cls_results.names[cls_results.probs.top1].upper()
    confidence = cls_results.probs.top1conf.item()
    print(f"   ✅ Diagnosis: {diagnosis} (Conf: {confidence:.2%})")
    
    if diagnosis != 'MELANOMA':
        print("\n🛑 Lesion classified as BENIGN. Stopping pipeline.")
        plt.imshow(rgb_img); plt.title(f"Diagnosis: {diagnosis} ({confidence:.1%})", color='green'); plt.axis('off'); plt.show()
        return

    # 4. DETECTION & ROI EXTRACTION
    print("\n3️⃣  Phase 3: Lesion Localization...")
    det_results = det_model(image_path, verbose=False)[0]
    
    roi_box = None
    det_plot = rgb_img 
    
    if len(det_results.boxes) > 0:
        # Get the box with highest confidence
        best_box = det_results.boxes[0]
        roi_box = best_box.xyxy[0].cpu().numpy().astype(int) # [x1, y1, x2, y2]
        print(f"   ✅ Detected lesion at coordinates: {roi_box}")
        det_plot = det_results.plot()
        det_plot = cv2.cvtColor(det_plot, cv2.COLOR_BGR2RGB)
    else:
        print("   ⚠️ Classifier saw melanoma, but Detector found no lesion.")
        print("   ⚠️ Fallback: Using center crop guidance.")
        # Fallback: Define a central ROI (middle 60% of image)
        h, w = rgb_img.shape[:2]
        roi_box = [int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8)]

    # 5. SEGMENTATION (GUIDED)
    print("\n4️⃣  Phase 4: Guided Morphological Segmentation...")
    # CRITICAL: Passing the ROI box to restrict segmentation to the lesion area
    ai_mask = get_guided_ai_mask(seg_model, hair_free_img, roi_box)
    
    if np.sum(ai_mask) == 0:
        print("   ❌ Segmentation failed to identify lesion boundaries.")
        return
    print("   ✅ Binary mask generated successfully")

    # 6. FEATURE EXTRACTION
    print("\n5️⃣  Phase 5: Clinical Feature Extraction (ABCD Rule)...")
    contours, _ = cv2.findContours(ai_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea) if contours else None
    hsv_clean = cv2.cvtColor(hair_free_img, cv2.COLOR_RGB2HSV)
    
    features = fe.extract_all_features(
        original_image=hair_free_img,
        hsv_image=hsv_clean,
        mask=ai_mask,
        contour=main_contour
    )
    
    # 7. VISUALIZATION
    print("\n🎨 Generating Dashboard...")
    overlay = utils.create_overlay_image(hair_free_img, ai_mask)
    create_master_dashboard(rgb_img, hair_free_img, det_plot, overlay, ai_mask, features, diagnosis, confidence)
    print("\n✨ Analysis Complete.")

if __name__ == "__main__":
    TEST_IMAGE_PATH = r"melanoma_dip_engine\data\images\IMD004.jpg" 
    
    if os.path.exists(TEST_IMAGE_PATH):
        run_pipeline(TEST_IMAGE_PATH)
    else:
        print(f"❌ Image not found: {TEST_IMAGE_PATH}")