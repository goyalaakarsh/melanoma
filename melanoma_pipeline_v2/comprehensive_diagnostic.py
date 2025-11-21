import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ultralytics import YOLO
import os
import pandas as pd
import seaborn as sns

# Load the classifier
MODEL_CLS_PATH = r'melanoma_pipeline_v2\models\melanoma_classifier_opt.pt'
cls_model = YOLO(MODEL_CLS_PATH)

IMAGES_FOLDER = r"melanoma_dip_engine\data\train\images"

print("="*100)
print("COMPREHENSIVE CLASSIFIER DIAGNOSTIC - FULL TRAINING SET")
print("="*100)
print(f"\nGround Truth: ALL images in {IMAGES_FOLDER} are MELANOMA")
print("\nProcessing images...\n")

results = []
image_files = sorted([f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith('.jpg')])

total_images = len(image_files)
print(f"Found {total_images} images to analyze\n")

# Process all images
for idx, img_file in enumerate(image_files, 1):
    if idx % 20 == 0:
        print(f"Progress: {idx}/{total_images} ({idx/total_images*100:.1f}%)")
    
    image_path = os.path.join(IMAGES_FOLDER, img_file)
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
    
    is_correct = top1_name.upper() == 'MELANOMA'
    
    results.append({
        'image': img_file,
        'predicted': top1_name.upper(),
        'confidence': top1_conf,
        'melanoma_prob': melanoma_conf,
        'not_melanoma_prob': not_melanoma_conf,
        'correct': is_correct,
        'decision_margin': abs(melanoma_conf - not_melanoma_conf)
    })

print(f"\n✅ Processed all {total_images} images\n")

# Create DataFrame
df = pd.DataFrame(results)

# Calculate comprehensive metrics
true_positives = len(df[df['correct'] == True])
false_negatives = len(df[df['correct'] == False])
total = len(df)

accuracy = true_positives / total
sensitivity = true_positives / total  # Since all are melanoma, sensitivity = accuracy
false_negative_rate = false_negatives / total

print("="*100)
print("AGGREGATE PERFORMANCE METRICS")
print("="*100)
print(f"\nTotal Images Analyzed:        {total}")
print(f"Ground Truth (All Melanoma):  {total}")
print(f"\nClassification Results:")
print(f"  ✅ Correctly Identified:     {true_positives} ({accuracy*100:.1f}%)")
print(f"  ❌ Missed (False Negatives): {false_negatives} ({false_negative_rate*100:.1f}%)")
print(f"\nPerformance Metrics:")
print(f"  Accuracy:                    {accuracy*100:.1f}%")
print(f"  Sensitivity/Recall:          {sensitivity*100:.1f}%")
print(f"  False Negative Rate:         {false_negative_rate*100:.1f}%")
print(f"  ⚠️  CRITICAL MISS RATE:       {false_negative_rate*100:.1f}%")

# Confidence statistics
correct_df = df[df['correct'] == True]
incorrect_df = df[df['correct'] == False]

print(f"\nConfidence Analysis:")
print(f"  Correct Predictions (Melanoma):")
print(f"    Mean Confidence:           {correct_df['confidence'].mean()*100:.1f}%")
print(f"    Median Confidence:         {correct_df['confidence'].median()*100:.1f}%")
print(f"    Min Confidence:            {correct_df['confidence'].min()*100:.1f}%")
print(f"    Max Confidence:            {correct_df['confidence'].max()*100:.1f}%")

if len(incorrect_df) > 0:
    print(f"\n  Incorrect Predictions (Not Melanoma):")
    print(f"    Mean Confidence:           {incorrect_df['confidence'].mean()*100:.1f}%")
    print(f"    Median Confidence:         {incorrect_df['confidence'].median()*100:.1f}%")
    print(f"    Min Confidence:            {incorrect_df['confidence'].min()*100:.1f}%")
    print(f"    Max Confidence:            {incorrect_df['confidence'].max()*100:.1f}%")
    print(f"\n  Melanoma Probability for Misclassified:")
    print(f"    Mean:                      {incorrect_df['melanoma_prob'].mean()*100:.1f}%")
    print(f"    Median:                    {incorrect_df['melanoma_prob'].median()*100:.1f}%")
    print(f"    Range:                     {incorrect_df['melanoma_prob'].min()*100:.1f}% - {incorrect_df['melanoma_prob'].max()*100:.1f}%")

# Decision margin analysis
print(f"\nDecision Margin Analysis:")
print(f"  Correct Predictions:")
print(f"    Mean Margin:               {correct_df['decision_margin'].mean()*100:.1f}%")
print(f"  Incorrect Predictions:")
if len(incorrect_df) > 0:
    print(f"    Mean Margin:               {incorrect_df['decision_margin'].mean()*100:.1f}%")

# Threshold analysis
print(f"\nThreshold Analysis:")
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for thresh in thresholds:
    would_classify_as_melanoma = len(df[df['melanoma_prob'] >= thresh])
    new_sensitivity = would_classify_as_melanoma / total
    print(f"  If threshold = {thresh:.1f}: {would_classify_as_melanoma}/{total} would be melanoma ({new_sensitivity*100:.1f}% sensitivity)")

# Save detailed results
df.to_csv('melanoma_pipeline_v2/comprehensive_diagnostic_results.csv', index=False)
print(f"\n✅ Detailed results saved to: melanoma_pipeline_v2/comprehensive_diagnostic_results.csv")

# Show worst misclassifications
print("\n" + "="*100)
print("TOP 10 WORST MISCLASSIFICATIONS (Highest Confidence in Wrong Prediction)")
print("="*100)
if len(incorrect_df) > 0:
    worst_cases = incorrect_df.nlargest(min(10, len(incorrect_df)), 'confidence')
    for idx, (_, row) in enumerate(worst_cases.iterrows(), 1):
        print(f"{idx:2}. {row['image']:15} → Predicted: NOT_MELANOMA ({row['confidence']*100:.1f}%) | "
              f"True Melanoma Prob: {row['melanoma_prob']*100:.1f}% | Margin: {row['decision_margin']*100:.1f}%")
else:
    print("No misclassifications found!")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

fig.suptitle(f'COMPREHENSIVE CLASSIFIER DIAGNOSTIC\n{total} Images - All Ground Truth: MELANOMA\nAccuracy: {accuracy*100:.1f}% | False Negative Rate: {false_negative_rate*100:.1f}%', 
             fontsize=18, fontweight='bold', y=0.98)

# 1. Prediction Distribution (Pie Chart)
ax1 = fig.add_subplot(gs[0, 0])
prediction_counts = df['predicted'].value_counts()
colors_pie = ['#27ae60' if 'MELANOMA' in pred else '#e74c3c' for pred in prediction_counts.index]
wedges, texts, autotexts = ax1.pie(prediction_counts.values, labels=prediction_counts.index, 
                                     autopct='%1.1f%%', colors=colors_pie, startangle=90,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title('Prediction Distribution\n(Ground Truth: All Melanoma)', fontweight='bold', fontsize=12)

# 2. Confidence Distribution (Histogram)
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(correct_df['confidence'], bins=20, alpha=0.7, color='green', label='Correct (Melanoma)', edgecolor='black')
if len(incorrect_df) > 0:
    ax2.hist(incorrect_df['confidence'], bins=20, alpha=0.7, color='red', label='Wrong (Not Melanoma)', edgecolor='black')
ax2.set_xlabel('Prediction Confidence', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('Confidence Distribution by Outcome', fontweight='bold', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Melanoma Probability Distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(df['melanoma_prob'], bins=30, alpha=0.7, color='purple', edgecolor='black')
ax3.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
ax3.set_xlabel('Melanoma Probability', fontweight='bold')
ax3.set_ylabel('Frequency', fontweight='bold')
ax3.set_title('Melanoma Probability Distribution', fontweight='bold', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Confusion Matrix Style View
ax4 = fig.add_subplot(gs[1, 0])
confusion_data = [[true_positives, 0], [false_negatives, 0]]  # Since no true negatives
sns.heatmap(confusion_data, annot=True, fmt='d', cmap='RdYlGn', 
            xticklabels=['Predicted\nMelanoma', 'Predicted\nNot Melanoma'],
            yticklabels=['Actual\nMelanoma', 'Actual\nNot Melanoma'],
            ax=ax4, cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 16, 'fontweight': 'bold'})
ax4.set_title('Confusion Matrix', fontweight='bold', fontsize=12)

# 5. Decision Margin Distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(correct_df.index, correct_df['decision_margin'], c='green', s=30, alpha=0.6, label='Correct')
if len(incorrect_df) > 0:
    ax5.scatter(incorrect_df.index, incorrect_df['decision_margin'], c='red', s=30, alpha=0.6, label='Incorrect')
ax5.set_xlabel('Image Index', fontweight='bold')
ax5.set_ylabel('Decision Margin (|P(M) - P(NM)|)', fontweight='bold')
ax5.set_title('Decision Margin per Image', fontweight='bold', fontsize=12)
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Threshold Sensitivity Analysis
ax6 = fig.add_subplot(gs[1, 2])
threshold_range = np.linspace(0, 1, 50)
sensitivities = []
for t in threshold_range:
    sens = len(df[df['melanoma_prob'] >= t]) / total
    sensitivities.append(sens)
ax6.plot(threshold_range, sensitivities, linewidth=3, color='navy')
ax6.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='100% Sensitivity')
ax6.axhline(accuracy, color='red', linestyle='--', alpha=0.5, label=f'Current: {accuracy*100:.1f}%')
ax6.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='Current Threshold (0.5)')
ax6.set_xlabel('Melanoma Probability Threshold', fontweight='bold')
ax6.set_ylabel('Sensitivity (Recall)', fontweight='bold')
ax6.set_title('Sensitivity vs Threshold', fontweight='bold', fontsize=12)
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_ylim([0, 1.05])

# 7. Melanoma vs Not Melanoma Probabilities (Scatter)
ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(correct_df['melanoma_prob'], correct_df['not_melanoma_prob'], 
           c='green', s=50, alpha=0.6, label='Correct', edgecolors='black')
if len(incorrect_df) > 0:
    ax7.scatter(incorrect_df['melanoma_prob'], incorrect_df['not_melanoma_prob'], 
               c='red', s=50, alpha=0.6, label='Incorrect', edgecolors='black')
ax7.plot([0, 1], [1, 0], 'k--', alpha=0.5, label='Decision Boundary')
ax7.set_xlabel('P(Melanoma)', fontweight='bold')
ax7.set_ylabel('P(Not Melanoma)', fontweight='bold')
ax7.set_title('Probability Space Analysis', fontweight='bold', fontsize=12)
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Performance Summary Table
ax8 = fig.add_subplot(gs[2, 1:])
ax8.axis('off')

summary_data = [
    ['Total Images', str(total)],
    ['True Positives', f'{true_positives} ({accuracy*100:.1f}%)'],
    ['False Negatives', f'{false_negatives} ({false_negative_rate*100:.1f}%)'],
    ['', ''],
    ['Accuracy', f'{accuracy*100:.1f}%'],
    ['Sensitivity', f'{sensitivity*100:.1f}%'],
    ['False Negative Rate', f'{false_negative_rate*100:.1f}%'],
    ['', ''],
    ['Avg Confidence (Correct)', f'{correct_df["confidence"].mean()*100:.1f}%'],
    ['Avg Confidence (Incorrect)', f'{incorrect_df["confidence"].mean()*100:.1f}%' if len(incorrect_df) > 0 else 'N/A'],
    ['', ''],
    ['Clinical Impact', '⚠️ CRITICAL: Missing melanoma cases!'],
]

table = ax8.table(cellText=summary_data, cellLoc='left', loc='center',
                 colWidths=[0.4, 0.4], bbox=[0.1, 0, 0.8, 1])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style the table
for i in range(len(summary_data)):
    cell = table[(i, 0)]
    cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    cell.set_text_props(weight='bold')
    
    cell = table[(i, 1)]
    cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    
    # Highlight critical metrics
    if i in [2, 6, 11]:  # False negatives and critical rows
        table[(i, 0)].set_facecolor('#ffcccc')
        table[(i, 1)].set_facecolor('#ffcccc')

ax8.set_title('Performance Summary', fontweight='bold', fontsize=14, pad=20)

plt.savefig('melanoma_pipeline_v2/comprehensive_diagnostic_dashboard.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Comprehensive diagnostic dashboard saved to: melanoma_pipeline_v2/comprehensive_diagnostic_dashboard.png")

plt.show()

# Final recommendations
print("\n" + "="*100)
print("CLINICAL SIGNIFICANCE & RECOMMENDATIONS")
print("="*100)
print(f"""
⚠️  CRITICAL FINDINGS:
    - The classifier is missing {false_negative_rate*100:.1f}% of melanoma cases
    - {false_negatives} patients would be incorrectly told they don't have melanoma
    - This is UNACCEPTABLE for a medical diagnostic system
    
📊 MODEL PERFORMANCE:
    - Sensitivity of only {sensitivity*100:.1f}% is far below clinical standards
    - Medical melanoma screening typically requires >95% sensitivity
    - Current model would cause significant patient harm
    
🔧 IMMEDIATE ACTIONS REQUIRED:
    1. DO NOT deploy this model in clinical settings
    2. Retrain with more diverse melanoma examples
    3. Consider lowering classification threshold to 0.3-0.4
    4. Implement ensemble methods for higher sensitivity
    5. Add human review for all borderline cases (0.3-0.7 probability)
    6. Validate on external test sets before any clinical use
    
💡 SUGGESTED THRESHOLD ADJUSTMENT:
    - At threshold 0.3: Would achieve {len(df[df['melanoma_prob'] >= 0.3])/total*100:.1f}% sensitivity
    - At threshold 0.4: Would achieve {len(df[df['melanoma_prob'] >= 0.4])/total*100:.1f}% sensitivity
    - Recommendation: Use threshold of 0.3 to minimize false negatives
""")

print("="*100)
