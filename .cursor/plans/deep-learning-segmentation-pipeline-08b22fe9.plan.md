<!-- 08b22fe9-8d81-46ce-8a72-15dc32d763b7 0bd5f559-5649-4fd1-9482-497acec1ae2f -->
# Deep Learning Melanoma Segmentation with SSL + Mask R-CNN

## Overview

Implement a production-ready deep learning pipeline using Masked Autoencoder (MAE) pre-training followed by Mask R-CNN fine-tuning for melanoma lesion segmentation.

## Phase 1: Environment Setup

### Update Dependencies

- **File**: `requirements.txt`
- Already contains deep learning dependencies (torch, detectron2, transformers, timm, albumentations)
- Verify versions are compatible with CUDA setup

### Notebook Structure

- **File**: `melanoma_dip_engine/train_segmentation_model.ipynb`
- Currently empty - will implement full training pipeline here
- Structure: 7 major sections with ~25-30 code cells total

## Phase 2: Self-Supervised Pre-training (MAE)

### Section 1: Imports and Configuration

```python
import torch, torchvision, timm, detectron2
from transformers import ViTMAEForPreTraining, ViTMAEConfig
```

### Section 2: Dataset for Unlabeled Images

- **Class**: `UnlabeledSkinDataset`
  - Initialize with data directory path (e.g., `data/images/`)
  - Load all images recursively from subdirectories
  - Transform: Resize to 224x224, normalize with ImageNet stats
  - Return: Single image tensor

### Section 3: MAE Pre-training Implementation

- **Approach**: Use Hugging Face's `ViTMAEForPreTraining`
  - Built-in masking (75% mask ratio)
  - Built-in decoder for patch reconstruction
  - MSE loss between reconstructed and original patches
- **Training Loop**:
  - DataLoader with batch_size=32 (adjust for GPU memory)
  - AdamW optimizer, lr=1.5e-4 with cosine scheduler
  - Train for 100-200 epochs on unlabeled data
  - Log reconstruction loss every 10 batches
  - Visualize masked/reconstructed images periodically

### Section 4: Save SSL Backbone

- Extract encoder weights from trained MAE model
- Save as `models/ssl_vit_backbone.pth`
- Save config for later loading

## Phase 3: Mask R-CNN Fine-tuning

### Section 5: Labeled Dataset Registration

- **Class**: `LesionSegmentationDataset`
  - Load images and corresponding masks from directory
  - Convert masks to COCO format (bboxes + segmentation polygons)
  - Return dict with: `file_name`, `image_id`, `height`, `width`, `annotations`
  - Annotations include: `bbox`, `segmentation`, `category_id`, `area`

- **Detectron2 Registration**:
  - Register dataset using `DatasetCatalog.register()`
  - Register metadata using `MetadataCatalog.get()`
  - Split into train/val sets (80/20)

### Section 6: Albumentations Augmentation Pipeline

```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3)
])
```

### Section 7: Custom ViT Backbone for Detectron2

- **Critical Implementation**: Build custom backbone wrapper
  - **Function**: `build_vit_backbone(cfg)`
  - Load ViT architecture from timm: `timm.create_model('vit_base_patch16_224')`
  - Load SSL pre-trained weights from `ssl_vit_backbone.pth`
  - Wrap in Detectron2 `Backbone` class
  - Implement required methods: `forward()`, `output_shape()`
  - Return multi-scale features compatible with FPN

- **Challenge**: ViT outputs single-scale features, but Mask R-CNN expects multi-scale
  - **Solution**: Extract intermediate layers (blocks 3, 6, 9, 12) as different scales
  - Reshape patch tokens back to 2D feature maps
  - Add small conv layers to adjust channel dimensions

### Section 8: Detectron2 Configuration

```python
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Replace backbone
cfg.MODEL.BACKBONE.NAME = "build_vit_backbone"
cfg.MODEL.PIXEL_MEAN = [123.675, 116.28, 103.53]  # ImageNet normalization
cfg.MODEL.PIXEL_STD = [58.395, 57.12, 57.375]

# Training hyperparameters
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.MAX_ITER = 5000
cfg.SOLVER.STEPS = (3000, 4500)
cfg.SOLVER.CHECKPOINT_PERIOD = 500

# Model configuration
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only lesion class
cfg.INPUT.MIN_SIZE_TRAIN = (512, 640, 704, 768)
```

### Section 9: Custom Trainer with Augmentation

- **Class**: `AugmentationTrainer(DefaultTrainer)`
  - Override `build_train_loader()` to use albumentations
  - Integrate augmentation pipeline into Detectron2 dataloader
  - Add validation loss hooks for monitoring

### Section 10: Training Execution

```python
trainer = AugmentationTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

- Monitor training via TensorBoard
- Track: total_loss, loss_cls, loss_box_reg, loss_mask
- Validation every 500 iterations

### Section 11: Evaluation

- Load best checkpoint
- Run inference on validation set
- Calculate Mask AP, AP50, AP75
- Visualize predictions on sample images
- Compare with classical segmentation baseline

### Section 12: Save Final Model

- Save final weights: `models/final_lesion_segmenter.pth`
- Save config: `models/config.yaml`
- Export inference-ready model

## Phase 4: Standalone Inference Script

### Create predict.py

- **File**: `melanoma_dip_engine/predict.py`
- **Function**: `predict_mask(image_path, output_path=None)`
  - Load trained Mask R-CNN model
  - Preprocess input image (resize, normalize)
  - Run inference
  - Extract highest-confidence mask
  - Post-process: smooth edges, fill holes
  - Return binary mask as numpy array
  - Optionally visualize and save overlay

- **Main block** for command-line usage:
  ```python
  python predict.py --image path/to/image.jpg --output path/to/result.png
  ```


## Key Technical Challenges & Solutions

### 1. ViT → Detectron2 Integration

- **Challenge**: ViT produces 1D patch embeddings, Detectron2 needs 2D feature maps
- **Solution**: Reshape patches to 2D, extract multi-scale features from intermediate blocks

### 2. MAE Decoder Implementation

- **Challenge**: Complex reconstruction decoder required
- **Solution**: Use HuggingFace `ViTMAEForPreTraining` with built-in decoder

### 3. COCO Format Conversion

- **Challenge**: Binary masks → polygon annotations
- **Solution**: Use `cv2.findContours()` + `RLE.encode()` for efficient encoding

### 4. Data Augmentation in Detectron2

- **Challenge**: Detectron2 uses custom augmentation format
- **Solution**: Create custom mapper function that applies albumentations then converts to Detectron2 format

## File Structure After Implementation

```
melanoma_dip_engine/
├── train_segmentation_model.ipynb (main training notebook)
├── predict.py (inference script)
├── models/
│   ├── ssl_vit_backbone.pth (MAE pre-trained encoder)
│   ├── final_lesion_segmenter.pth (trained Mask R-CNN)
│   └── config.yaml (model configuration)
└── data/
    ├── images/ (unlabeled images for SSL)
    ├── train/ (labeled images + masks)
    └── val/ (validation images + masks)
```

## Dependencies on Existing Code

- Will use `config.IMAGE_SIZE` for consistency
- `predict.py` will be independent, not modifying `image_processing.py`
- Can be integrated later by calling predict.py from main pipeline

### To-dos

- [ ] Verify requirements.txt and create notebook structure with all sections
- [ ] Implement UnlabeledSkinDataset class for SSL training
- [ ] Implement MAE pre-training loop with HuggingFace ViTMAEForPreTraining
- [ ] Extract and save SSL backbone weights
- [ ] Implement LesionSegmentationDataset with Detectron2 registration
- [ ] Create albumentations augmentation pipeline
- [ ] Build custom ViT backbone wrapper for Detectron2 with multi-scale features
- [ ] Configure Detectron2 with custom backbone and hyperparameters
- [ ] Implement custom trainer with augmentation integration
- [ ] Execute Mask R-CNN training and validation
- [ ] Evaluate model performance and visualize results
- [ ] Create standalone predict.py script for inference