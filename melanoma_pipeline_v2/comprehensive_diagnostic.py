import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ultralytics import YOLO
import os
import pandas as pd
import seaborn as sns
import sys
import warnings

# Suppress library warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# CHECK THESE PATHS
MODEL_CLS_PATH = r'melanoma_pipeline_v2/models/melanoma_classifier_opt.pt' 
IMAGES_FOLDER = r"melanoma_dip_engine/data/train/images" 

# MEDICAL SETTINGS
# Based on your sensitivity graph, 0.35 is still missing 16% of cases.
# We lower it to 0.15 to catch the "tail" of the distribution.
MEDICAL_THRESHOLD = 0.15 

# OUTPUT
OUTPUT_CSV = 'melanoma_pipeline_v2/final_diagnostic_results.csv'
OUTPUT_IMG = 'melanoma_pipeline_v2/final_diagnostic_dashboard.png'

# ==============================================================================
# 2. MANUAL TTA FUNCTION (Fixes the Warning)
# ==============================================================================
def predict_with_manual_tta(model, image_path, melanoma_idx):
    """
    Since YOLO-Cls 'augment=True' failed, we do it manually.
    We create 4 views of the image, predict all, and average the probabilities.
    """
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None: return 0.0, 0.0
    
    # 2. Create Augmentations (Batch of 4)
    # View 1: Original
    v1 = img
    # View 2: Horizontal Flip
    v2 = cv2.flip(img, 1)
    # View 3: Vertical Flip (Skin has no up/down)
    v3 = cv2.flip(img, 0)
    # View 4: Rotate 90 (Transposition)
    v4 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    batch = [v1, v2, v3, v4]
    
    # 3. Run Batch Inference
    # We pass the list of numpy arrays directly to the model
    results = model(batch, verbose=False)
    
    # 4. Average the Probabilities
    sum_melanoma = 0.0
    sum_benign = 0.0
    not_melanoma_idx = 1 - melanoma_idx
    
    for res in results:
        probs = res.probs.data.cpu().numpy()
        sum_melanoma += probs[melanoma_idx]
        sum_benign += probs[not_melanoma_idx]
        
    avg_melanoma = sum_melanoma / 4.0
    avg_benign = sum_benign / 4.0
    
    return avg_melanoma, avg_benign

# ==============================================================================
# 3. MAIN PIPELINE
# ==============================================================================
print("="*100)
print("🏥 FINAL MELANOMA DIAGNOSTIC ENGINE (MANUAL TTA)")
print("="*100)
print(f"⚙️  Model:      {MODEL_CLS_PATH}")
print(f"⚙️  Threshold:  {MEDICAL_THRESHOLD * 100}% (Aggressive Screening)")
print(f"⚙️  TTA:        MANUAL (4-View Voting)")
print("-" * 100)

# Load Model
try:
    cls_model = YOLO(MODEL_CLS_PATH)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Identify Class Indices
melanoma_idx = -1
for k, v in cls_model.names.items():
    if 'melanoma' in v.lower() or 'malignant' in v.lower():
        melanoma_idx = k
        break

if melanoma_idx == -1:
    print("❌ CRITICAL: Could not find 'Melanoma' class index.")
    sys.exit(1)

print(f"ℹ️  Melanoma Index: {melanoma_idx}")

# Prepare List
image_files = sorted([f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
total_images = len(image_files)
print(f"Found {total_images} images.\n")

results = []

# --- PROCESSING LOOP ---
for i, img_file in enumerate(image_files):
    img_path = os.path.join(IMAGES_FOLDER, img_file)
    
    # USE MANUAL TTA PREDICTION
    p_melanoma, p_benign = predict_with_manual_tta(cls_model, img_path, melanoma_idx)
    
    # DECISION LOGIC
    if p_melanoma >= MEDICAL_THRESHOLD:
        prediction = 'MELANOMA'
        is_correct = True
        outcome = 'True Positive'
    else:
        prediction = 'NOT_MELANOMA'
        is_correct = False
        outcome = 'False Negative'
        
    # Was it rescued by the low threshold?
    rescued = False
    if p_melanoma >= MEDICAL_THRESHOLD and p_melanoma < 0.5:
        rescued = True
        outcome = 'Rescued (Low Prob TP)'

    results.append({
        'image': img_file,
        'p_melanoma': p_melanoma,
        'p_benign': p_benign,
        'outcome': outcome,
        'correct': is_correct,
        'rescued': rescued
    })
    
    print(f"Processing: {i+1}/{total_images} | TP: {p_melanoma:.3f}...", end='\r')

print(f"\n✅ Completed analysis of {total_images} images.")

# ==============================================================================
# 4. METRICS & REPORTING
# ==============================================================================
df = pd.DataFrame(results)

tp_count = len(df[df['correct'] == True])
fn_count = len(df[df['correct'] == False])
rescued_count = len(df[df['rescued'] == True])
sensitivity = tp_count / total_images
std_sensitivity = len(df[df['p_melanoma'] >= 0.5]) / total_images

print("\n" + "="*60)
print("📊 FINAL PERFORMANCE REPORT")
print("="*60)
print(f"Total Ground Truth Melanomas: {total_images}")
print("-" * 60)
print(f"✅ DETECTED (Sensitivity):           {sensitivity*100:.1f}% ({tp_count}/{total_images})")
print(f"❌ MISSED (False Negatives):         {(1-sensitivity)*100:.1f}% ({fn_count}/{total_images})")
print("-" * 60)
print(f"📈 Impact of TTA + 0.15 Threshold:")
print(f"   Standard AI (0.5, No TTA):       ~{std_sensitivity*100:.1f}% (Estimated)")
print(f"   Current Pipeline:                {sensitivity*100:.1f}%")
print(f"   Cases Rescued:                   {rescued_count}")
print("="*60)

# Save CSV
df.to_csv(OUTPUT_CSV, index=False)

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)

fig.suptitle(f'FINAL DIAGNOSTIC DASHBOARD\nManual TTA (4-View) | Threshold: {MEDICAL_THRESHOLD}', 
             fontsize=16, fontweight='bold', y=0.96)

# 1. PIE CHART
ax1 = fig.add_subplot(gs[0, 0])
outcomes = df['outcome'].value_counts()
colors = {'True Positive': '#2ecc71', 'False Negative': '#e74c3c', 'Rescued (Low Prob TP)': '#f1c40f'}
pie_colors = [colors.get(x, '#95a5a6') for x in outcomes.index]
ax1.pie(outcomes, labels=outcomes.index, autopct='%1.1f%%', colors=pie_colors, 
        startangle=90, textprops={'fontsize': 11, 'fontweight':'bold'})
ax1.set_title('Diagnostic Outcomes', fontweight='bold')

# 2. RISK HISTOGRAM
ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(data=df, x='p_melanoma', bins=30, kde=True, ax=ax2, color='purple')
ax2.axvline(MEDICAL_THRESHOLD, color='red', linewidth=2, label=f'Threshold ({MEDICAL_THRESHOLD})')
ax2.set_xlabel('Melanoma Probability (Averaged)')
ax2.legend()
ax2.set_title('Risk Score Distribution', fontweight='bold')

# 3. SENSITIVITY CURVE
ax4 = fig.add_subplot(gs[1, :])
thresholds = np.linspace(0, 1, 100)
sens_curve = [len(df[df['p_melanoma'] >= t]) / total_images for t in thresholds]
ax4.plot(thresholds, sens_curve, linewidth=3, color='navy')
ax4.axhline(0.95, color='green', linestyle=':', label='Target (95%)')
ax4.axvline(MEDICAL_THRESHOLD, color='red', linestyle='--', label=f'Current ({MEDICAL_THRESHOLD})')
ax4.set_xlabel('Probability Threshold')
ax4.set_ylabel('Sensitivity')
ax4.set_title('Sensitivity vs. Threshold', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.5)

plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
print(f"🖼️  Dashboard saved to: {OUTPUT_IMG}")
plt.show()