"""
Comprehensive Test Diagnostic for Melanoma Classification
==========================================================
Tests on labeled benign and melanoma images with proper metrics
"""

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
import random
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# Suppress library warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
MODEL_CLS_PATH = r'melanoma_pipeline_v2/models/melanoma_classifier_opt.pt' 
BENIGN_FOLDER = r"malignant_and_benign/test/benign"
MELANOMA_FOLDER = r"malignant_and_benign/test/melanoma"

# Sampling
SAMPLE_SIZE = 100  # Images per class

# Classification threshold
THRESHOLD = 0.5  # Standard threshold

# OUTPUT
OUTPUT_CSV = 'melanoma_pipeline_v2/test_diagnostic_results.csv'
OUTPUT_IMG = 'melanoma_pipeline_v2/test_diagnostic_dashboard.png'

# ==============================================================================
# 2. MANUAL TTA FUNCTION
# ==============================================================================
def predict_with_manual_tta(model, image_path, melanoma_idx):
    """
    Manual Test-Time Augmentation with 4 views
    """
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None: 
        return 0.0, 0.0
    
    # 2. Create Augmentations
    v1 = img
    v2 = cv2.flip(img, 1)  # Horizontal flip
    v3 = cv2.flip(img, 0)  # Vertical flip
    v4 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    batch = [v1, v2, v3, v4]
    
    # 3. Run Batch Inference
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

def predict_ensemble(model_A, model_B, image_path, melanoma_idx):
    """
    Runs inference on TWO models and averages the result.
    """
    # Load Image
    img = cv2.imread(image_path)
    
    # --- Prediction Model A (Medium, 320px) ---
    # Note: YOLO handles resizing internally, just pass the image
    res_A = model_A(img, verbose=False) 
    prob_A = res_A[0].probs.data.cpu().numpy()[melanoma_idx]
    
    # --- Prediction Model B (Large, 512px) ---
    res_B = model_B(img, verbose=False)
    prob_B = res_B[0].probs.data.cpu().numpy()[melanoma_idx]
    
    # --- Ensemble Average ---
    final_prob = (prob_A + prob_B) / 2.0
    
    return final_prob
# ==============================================================================
# 3. MAIN PIPELINE
# ==============================================================================
print("="*100)
print("🧪 MELANOMA TEST DIAGNOSTIC - LABELED DATA EVALUATION")
print("="*100)
print(f"⚙️  Model:           {MODEL_CLS_PATH}")
print(f"⚙️  Benign Folder:   {BENIGN_FOLDER}")
print(f"⚙️  Melanoma Folder: {MELANOMA_FOLDER}")
print(f"⚙️  Sample Size:     {SAMPLE_SIZE} per class")
print(f"⚙️  Threshold:       {THRESHOLD}")
print(f"⚙️  TTA:             MANUAL (4-View Averaging)")
print("-" * 100)

# Load Model
try:
    cls_model = YOLO(MODEL_CLS_PATH)
    print("✅ Model loaded successfully")
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

print(f"ℹ️  Melanoma Index: {melanoma_idx}\n")

# ==============================================================================
# 4. LOAD AND SAMPLE IMAGES
# ==============================================================================
print("📁 Loading and sampling images...")

# Get all images
benign_files = [f for f in os.listdir(BENIGN_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
melanoma_files = [f for f in os.listdir(MELANOMA_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"   Total benign images available: {len(benign_files)}")
print(f"   Total melanoma images available: {len(melanoma_files)}")

# Random sampling
random.seed(42)  # For reproducibility
benign_sample = random.sample(benign_files, min(SAMPLE_SIZE, len(benign_files)))
melanoma_sample = random.sample(melanoma_files, min(SAMPLE_SIZE, len(melanoma_files)))

print(f"   Sampled benign: {len(benign_sample)}")
print(f"   Sampled melanoma: {len(melanoma_sample)}")

# Create test dataset
test_data = []

# Add benign samples
for img_file in benign_sample:
    test_data.append({
        'image': img_file,
        'path': os.path.join(BENIGN_FOLDER, img_file),
        'true_label': 'BENIGN',
        'true_class': 0
    })

# Add melanoma samples
for img_file in melanoma_sample:
    test_data.append({
        'image': img_file,
        'path': os.path.join(MELANOMA_FOLDER, img_file),
        'true_label': 'MELANOMA',
        'true_class': 1
    })

# Shuffle
random.shuffle(test_data)
total_images = len(test_data)

print(f"\n🔬 Processing {total_images} test images...\n")

# ==============================================================================
# 5. INFERENCE
# ==============================================================================
results = []

for i, item in enumerate(test_data):
    img_path = item['path']
    true_label = item['true_label']
    true_class = item['true_class']
    
    # Predict with TTA
    p_melanoma, p_benign = predict_with_manual_tta(cls_model, img_path, melanoma_idx)
    
    # Classification decision
    if p_melanoma >= THRESHOLD:
        predicted_label = 'MELANOMA'
        predicted_class = 1
    else:
        predicted_label = 'BENIGN'
        predicted_class = 0
    
    # Determine outcome
    is_correct = (predicted_class == true_class)
    
    if true_class == 1 and predicted_class == 1:
        outcome = 'True Positive (TP)'
    elif true_class == 0 and predicted_class == 0:
        outcome = 'True Negative (TN)'
    elif true_class == 1 and predicted_class == 0:
        outcome = 'False Negative (FN)'
    else:  # true_class == 0 and predicted_class == 1
        outcome = 'False Positive (FP)'
    
    results.append({
        'image': item['image'],
        'true_label': true_label,
        'predicted_label': predicted_label,
        'p_melanoma': p_melanoma,
        'p_benign': p_benign,
        'outcome': outcome,
        'correct': is_correct,
        'true_class': true_class,
        'predicted_class': predicted_class
    })
    
    print(f"Processing: {i+1}/{total_images} | {outcome[:2]}: {p_melanoma:.3f}", end='\r')

print(f"\n✅ Completed inference on {total_images} images.\n")

# ==============================================================================
# 6. METRICS CALCULATION
# ==============================================================================
df = pd.DataFrame(results)

# Extract labels
y_true = df['true_class'].values
y_pred = df['predicted_class'].values

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Specificity
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Per-class metrics
print("="*80)
print("📊 TEST PERFORMANCE METRICS")
print("="*80)
print(f"\n📈 Overall Performance:")
print(f"   Accuracy:        {accuracy*100:.2f}%")
print(f"   Precision:       {precision*100:.2f}%")
print(f"   Recall:          {recall*100:.2f}% (Sensitivity)")
print(f"   Specificity:     {specificity*100:.2f}%")
print(f"   F1-Score:        {f1*100:.2f}%")

print(f"\n🎯 Confusion Matrix:")
print(f"   True Negatives (TN):   {tn:3d}  (Correctly identified benign)")
print(f"   False Positives (FP):  {fp:3d}  (Benign misclassified as melanoma)")
print(f"   False Negatives (FN):  {fn:3d}  (Melanoma misclassified as benign) ⚠️")
print(f"   True Positives (TP):   {tp:3d}  (Correctly identified melanoma)")

print(f"\n📋 Detailed Classification Report:")
print("-" * 80)
print(classification_report(y_true, y_pred, target_names=['Benign', 'Melanoma'], digits=4))
print("="*80)

# Save CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n💾 Results saved to: {OUTPUT_CSV}")

# ==============================================================================
# 7. VISUALIZATION
# ==============================================================================
print("🎨 Generating visualization dashboard...")

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

fig.suptitle(f'MELANOMA TEST DIAGNOSTIC DASHBOARD\n'
             f'Accuracy: {accuracy*100:.1f}% | Precision: {precision*100:.1f}% | '
             f'Recall: {recall*100:.1f}% | F1: {f1*100:.1f}%', 
             fontsize=18, fontweight='bold', y=0.98)

# 1. CONFUSION MATRIX
ax1 = fig.add_subplot(gs[0, 0])
cm_display = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Benign', 'Melanoma'], 
            yticklabels=['Benign', 'Melanoma'],
            cbar_kws={'label': 'Count'})
ax1.set_ylabel('True Label', fontweight='bold')
ax1.set_xlabel('Predicted Label', fontweight='bold')
ax1.set_title('Confusion Matrix', fontweight='bold', fontsize=14)

# 2. OUTCOME PIE CHART
ax2 = fig.add_subplot(gs[0, 1])
outcomes = df['outcome'].value_counts()
colors = {
    'True Positive (TP)': '#27ae60',
    'True Negative (TN)': '#3498db',
    'False Negative (FN)': '#e74c3c',
    'False Positive (FP)': '#f39c12'
}
pie_colors = [colors.get(x, '#95a5a6') for x in outcomes.index]
wedges, texts, autotexts = ax2.pie(outcomes, labels=outcomes.index, autopct='%1.1f%%', 
                                     colors=pie_colors, startangle=90,
                                     textprops={'fontsize': 10, 'fontweight':'bold'})
ax2.set_title('Classification Outcomes', fontweight='bold', fontsize=14)

# 3. METRICS BAR CHART
ax3 = fig.add_subplot(gs[0, 2])
metrics_names = ['Accuracy', 'Precision', 'Recall\n(Sensitivity)', 'Specificity', 'F1-Score']
metrics_values = [accuracy, precision, recall, specificity, f1]
bars = ax3.barh(metrics_names, metrics_values, color=['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c'])
ax3.set_xlim(0, 1.0)
ax3.set_xlabel('Score', fontweight='bold')
ax3.set_title('Performance Metrics', fontweight='bold', fontsize=14)
for i, (bar, val) in enumerate(zip(bars, metrics_values)):
    ax3.text(val + 0.02, i, f'{val*100:.1f}%', va='center', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 4. MELANOMA PROBABILITY DISTRIBUTION (BY TRUE CLASS)
ax4 = fig.add_subplot(gs[1, :2])
benign_probs = df[df['true_label'] == 'BENIGN']['p_melanoma']
melanoma_probs = df[df['true_label'] == 'MELANOMA']['p_melanoma']
ax4.hist(benign_probs, bins=30, alpha=0.6, label='True Benign', color='blue', edgecolor='black')
ax4.hist(melanoma_probs, bins=30, alpha=0.6, label='True Melanoma', color='red', edgecolor='black')
ax4.axvline(THRESHOLD, color='green', linewidth=3, linestyle='--', label=f'Threshold ({THRESHOLD})')
ax4.set_xlabel('Melanoma Probability (TTA Averaged)', fontweight='bold')
ax4.set_ylabel('Count', fontweight='bold')
ax4.set_title('Probability Distribution by True Class', fontweight='bold', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(alpha=0.3)

# 5. ROC-LIKE THRESHOLD SENSITIVITY
ax5 = fig.add_subplot(gs[1, 2])
thresholds = np.linspace(0, 1, 100)
sensitivities = []
specificities = []

for t in thresholds:
    preds = (df['p_melanoma'] >= t).astype(int)
    cm_temp = confusion_matrix(df['true_class'], preds)
    if cm_temp.shape == (2, 2):
        tn_t, fp_t, fn_t, tp_t = cm_temp.ravel()
        sens = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        spec = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
    else:
        sens, spec = 0, 0
    sensitivities.append(sens)
    specificities.append(spec)

ax5.plot(thresholds, sensitivities, linewidth=2, label='Sensitivity (Recall)', color='red')
ax5.plot(thresholds, specificities, linewidth=2, label='Specificity', color='blue')
ax5.axvline(THRESHOLD, color='green', linestyle='--', linewidth=2, label=f'Current ({THRESHOLD})')
ax5.set_xlabel('Threshold', fontweight='bold')
ax5.set_ylabel('Score', fontweight='bold')
ax5.set_title('Sensitivity vs Specificity', fontweight='bold', fontsize=14)
ax5.legend()
ax5.grid(alpha=0.3)

# 6. CLASS DISTRIBUTION
ax6 = fig.add_subplot(gs[2, 0])
class_counts = df['true_label'].value_counts()
ax6.bar(class_counts.index, class_counts.values, color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=2)
ax6.set_ylabel('Count', fontweight='bold')
ax6.set_title('Test Set Distribution', fontweight='bold', fontsize=14)
ax6.grid(axis='y', alpha=0.3)
for i, (label, count) in enumerate(zip(class_counts.index, class_counts.values)):
    ax6.text(i, count + 1, str(count), ha='center', fontweight='bold', fontsize=12)

# 7. PREDICTION DISTRIBUTION
ax7 = fig.add_subplot(gs[2, 1])
pred_counts = df['predicted_label'].value_counts()
ax7.bar(pred_counts.index, pred_counts.values, color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=2)
ax7.set_ylabel('Count', fontweight='bold')
ax7.set_title('Prediction Distribution', fontweight='bold', fontsize=14)
ax7.grid(axis='y', alpha=0.3)
for i, (label, count) in enumerate(zip(pred_counts.index, pred_counts.values)):
    ax7.text(i, count + 1, str(count), ha='center', fontweight='bold', fontsize=12)

# 8. ERROR ANALYSIS
ax8 = fig.add_subplot(gs[2, 2])
error_types = ['Correct\nPredictions', 'False\nPositives', 'False\nNegatives']
error_counts = [tn + tp, fp, fn]
colors_error = ['#2ecc71', '#f39c12', '#e74c3c']
bars = ax8.bar(error_types, error_counts, color=colors_error, edgecolor='black', linewidth=2)
ax8.set_ylabel('Count', fontweight='bold')
ax8.set_title('Error Analysis', fontweight='bold', fontsize=14)
ax8.grid(axis='y', alpha=0.3)
for bar, count in zip(bars, error_counts):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
print(f"🖼️  Dashboard saved to: {OUTPUT_IMG}")

print("\n" + "="*80)
print("✅ DIAGNOSTIC COMPLETE")
print("="*80)

# Show misclassified samples
print("\n⚠️  FALSE NEGATIVES (Melanoma missed):")
fn_samples = df[df['outcome'] == 'False Negative (FN)'].head(5)
if len(fn_samples) > 0:
    for idx, row in fn_samples.iterrows():
        print(f"   {row['image']} - Prob: {row['p_melanoma']:.3f}")
else:
    print("   None! Perfect melanoma detection! 🎉")

print("\n⚠️  FALSE POSITIVES (Benign misclassified):")
fp_samples = df[df['outcome'] == 'False Positive (FP)'].head(5)
if len(fp_samples) > 0:
    for idx, row in fp_samples.iterrows():
        print(f"   {row['image']} - Prob: {row['p_melanoma']:.3f}")
else:
    print("   None! Perfect benign detection! 🎉")

plt.show()
