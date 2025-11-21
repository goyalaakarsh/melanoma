import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ultralytics import YOLO
import os
import pandas as pd

# Load the classifier
MODEL_CLS_PATH = r'melanoma_pipeline_v2\models\melanoma_classifier.pt'
cls_model = YOLO(MODEL_CLS_PATH)

IMAGES_FOLDER = r"melanoma_dip_engine\data\images"

print("="*80)
print("CLASSIFIER PERFORMANCE ANALYSIS")
print("="*80)
print("\nGround Truth: ALL 8 images are MELANOMA")
print("\nDetailed Classification Results:\n")

results = []
image_files = sorted([f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith('.jpg')])

for img_file in image_files:
    image_path = os.path.join(IMAGES_FOLDER, img_file)
    cls_results = cls_model(image_path, verbose=False)[0]
    
    # Get top 2 predictions
    probs = cls_results.probs
    top1_idx = probs.top1
    top1_conf = probs.top1conf.item()
    top1_name = cls_results.names[top1_idx]
    
    # Get all class probabilities
    all_probs = probs.data.cpu().numpy()
    
    # Assuming binary classification: melanoma vs not_melanoma
    melanoma_conf = all_probs[0] if cls_results.names[0].upper() == 'MELANOMA' else all_probs[1]
    not_melanoma_conf = all_probs[1] if cls_results.names[0].upper() == 'MELANOMA' else all_probs[0]
    
    is_correct = top1_name.upper() == 'MELANOMA'
    
    results.append({
        'image': img_file,
        'predicted': top1_name.upper(),
        'confidence': top1_conf,
        'melanoma_prob': melanoma_conf,
        'not_melanoma_prob': not_melanoma_conf,
        'correct': is_correct
    })
    
    status = "✅ CORRECT" if is_correct else "❌ WRONG (FALSE NEGATIVE!)"
    print(f"{img_file:15} → {top1_name.upper():15} ({top1_conf:.1%})  {status}")
    print(f"                  Melanoma: {melanoma_conf:.1%} | Not Melanoma: {not_melanoma_conf:.1%}")
    print()

# Calculate metrics
df = pd.DataFrame(results)
accuracy = df['correct'].sum() / len(df)
false_negatives = len(df[df['correct'] == False])

print("="*80)
print("PERFORMANCE METRICS")
print("="*80)
print(f"Accuracy: {accuracy:.1%} ({df['correct'].sum()}/{len(df)})")
print(f"False Negatives: {false_negatives} (Missed melanoma cases!)")
print(f"Sensitivity/Recall: {df['correct'].sum()}/{len(df)} = {accuracy:.1%}")
print("\n⚠️  CRITICAL: {:.1%} of melanoma cases are being missed!".format(1-accuracy))

# Analyze misclassified cases
print("\n" + "="*80)
print("MISCLASSIFIED IMAGES ANALYSIS")
print("="*80)
misclassified = df[df['correct'] == False]
for _, row in misclassified.iterrows():
    print(f"\n{row['image']}:")
    print(f"  - Predicted as: {row['predicted']} with {row['confidence']:.1%} confidence")
    print(f"  - Melanoma probability was only: {row['melanoma_prob']:.1%}")
    print(f"  - Decision margin: {abs(row['melanoma_prob'] - row['not_melanoma_prob']):.1%}")

# Visualization
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('CLASSIFIER PREDICTIONS - ALL IMAGES ARE MELANOMA (Ground Truth)', 
             fontsize=16, fontweight='bold')

for idx, (ax, row) in enumerate(zip(axes.flat, results)):
    img_path = os.path.join(IMAGES_FOLDER, row['image'])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    ax.imshow(img)
    
    # Color code based on correctness
    if row['correct']:
        color = 'green'
        border_color = 'green'
    else:
        color = 'red'
        border_color = 'red'
    
    title = f"{row['image']}\n{row['predicted']} ({row['confidence']:.1%})"
    ax.set_title(title, fontweight='bold', color=color, fontsize=10)
    ax.axis('off')
    
    # Add colored border
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(4)
    
    # Add probability bars
    prob_text = f"M: {row['melanoma_prob']:.0%} | NM: {row['not_melanoma_prob']:.0%}"
    ax.text(0.5, -0.05, prob_text, transform=ax.transAxes, 
            ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('melanoma_pipeline_v2/classifier_diagnostic.png', dpi=150, bbox_inches='tight')
print("\n✅ Diagnostic visualization saved to: melanoma_pipeline_v2/classifier_diagnostic.png")
plt.show()

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS TO FIX THE CLASSIFIER")
print("="*80)
print("""
1. **Retrain the model** with a more balanced dataset
   - Current model appears undertrained or biased toward NOT_MELANOMA class
   
2. **Check training data quality**
   - Ensure training set has sufficient melanoma examples
   - Verify labels are correct
   
3. **Adjust classification threshold**
   - Consider lowering threshold for melanoma classification
   - Better to have false positives than false negatives in medical diagnosis
   
4. **Use ensemble methods**
   - Combine multiple models to reduce false negatives
   
5. **Add data augmentation**
   - Increase diversity of melanoma training samples
   
6. **Consider using a pretrained medical imaging model**
   - Transfer learning from models trained on dermatology datasets
   
7. **Implement uncertainty quantification**
   - Flag low-confidence predictions for human review
""")
