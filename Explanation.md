ssl_rcnn_segmentation explanation 

████████████████████████████
█        CELL 1            █
████████████████████████████
This code cell prepares the full working environment for a deep-learning pipeline using PyTorch, Detectron2, Vision Transformers, and Albumentations. Its purpose is to load required libraries, enable logging, set reproducibility controls, and verify that the environment is ready for training.

The first part imports essential Python, PyTorch, and image-processing libraries. PyTorch handles model building, optimization, and tensor operations, while Dataset and DataLoader manage data organization and batching. Image transformations rely on TorchVision, OpenCV, and PIL, and Matplotlib supports visualization. General utilities like os, json, and Path facilitate file handling and project structure. Together, these imports enable data preparation, image manipulation, visualization, and model construction.

Next, the script loads key Detectron2 modules, including configuration utilities, built-in training pipelines, dataset registration tools, and visualization components. Structures such as Boxes, Instances, and BitMasks support object detection and segmentation outputs. Model building and checkpoint loading utilities prepare the project for training custom architectures. Importing the backbone registry is especially important for integrating custom backbones, such as Vision Transformers, into standard Detectron2 models like Mask R-CNN.

The code also brings in Vision Transformer (ViT) components from HuggingFace, including pretrained and self-supervised ViT models and their configuration classes. Additionally, Albumentations and its PyTorch adapter enable high-quality data augmentations and seamless conversion to tensors. These tools together support advanced preprocessing and transformer-based feature extraction.

Logging is set up by suppressing unnecessary warnings and activating Detectron2’s built-in logger, ensuring clean and structured output during execution.

A reproducibility function is defined to fix random seeds across PyTorch, NumPy, Python’s random module, and CUDA (when available). This ensures consistent results across runs, which is essential for debugging and comparing experiments.

Finally, the script initializes the seed and prints the versions of PyTorch and Detectron2 along with information about whether CUDA (GPU) is available. This confirms that the environment is correctly configured and ready for model training.



████████████████████████████
█        Cell 2            █
████████████████████████████

Purpose:

Initialize the runtime environment for training/evaluation: import required libraries, configure Detectron2, set up reproducible seeds, and print a short environment check. This cell prepares everything the later cells depend on.
Top-level structure

Grouped imports (PyTorch, stdlib, image libs), Detectron2 imports, Hugging Face + Albumentations, setup (warnings/logger), and a reproducibility helper function.
Imports — what and why

PyTorch & utilities:

import torch, import torch.nn as nn, import torch.optim as optim
Core framework for model definition, optimization and device management.
from torch.utils.data import Dataset, DataLoader
Dataset and batching primitives used by MAE pretraining and Detectron2 training helpers.
import torchvision.transforms as transforms, import torchvision.utils as vutils
Common image transforms and utilities (used in the SSL dataset pipeline / debugging/visuals).
Standard library & helpers:

import os, json, random, shutil, warnings
File I/O, config persistence, deterministic seeding, file copying, and warning suppression.
import numpy as np
Numeric operations, mask handling, and conversions between PIL/OpenCV/torch.
from pathlib import Path and from PIL import Image
Filesystem helpers and a robust image loader.
import matplotlib.pyplot as plt
Plotting and visualization (used to show predictions/visualizations inline).
Image I/O / processing:

import cv2 (OpenCV)
Fast image I/O, resizing, contour finding, mask operations used throughout dataset conversion, preprocessing, and visualization.
Detectron2 (core object-detection/instance-segmentation framework):

import detectron2 and from detectron2.utils.logger import setup_logger
Base package and its logger initializer.
from detectron2.config import get_cfg and from detectron2 import model_zoo
Build and modify model configs; model_zoo provides template YAMLs.
from detectron2.engine import DefaultTrainer, default_setup
Training scaffolding; DefaultTrainer handles typical training loops and checkpointing.
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
Dataset registration and train/test dataloader builders (we register COCO-like dataset dicts later).
from detectron2.data import detection_utils as utils
Utility functions for reading images / building inputs in a Detectron2-compatible way.
from detectron2.structures import BoxMode, Instances, Boxes, BitMasks
Low-level data structures representing bounding boxes, instance masks, and batched instances.
from detectron2.utils.visualizer import Visualizer, ColorMode
Visualizer used to draw boxes/masks on images for debug/inspection.
from detectron2.modeling import build_model and from detectron2.checkpoint import DetectionCheckpointer
Programmatic model construction and checkpoint load/save helpers.
from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY and from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool and from detectron2.layers import ShapeSpec
Backbone base classes and FPN building blocks — used to register the custom ViT backbone and wrap it with FPN.
import detectron2.utils.comm as comm
Utilities for distributed training communication (even if not used, often imported for multi-GPU setups).
import fvcore.nn.weight_init as weight_init
Weight initialization helpers (used to initialize convs in the custom backbone).
Hugging Face (ViT / MAE) and Albumentations:

from transformers import ViTMAEForPreTraining, ViTModel, ViTConfig
HuggingFace model classes. ViTMAEForPreTraining is a ready-made MAE pretraining model that returns losses for masked reconstruction. ViTModel/ViTConfig are used to instantiate ViT encoders for backbone conversion or SSL-loading.
import albumentations as A and from albumentations.pytorch import ToTensorV2
Fast augmentations for images, masks and bboxes. ToTensorV2 converts aug results to torch tensors in channel-first format and is commonly used in Detectron2 mappers.
Setup and warnings

warnings.filterwarnings('ignore')
Hides noisy warnings in the notebook — convenient but be careful: important deprecation warnings may be suppressed.
setup_logger()
Initializes Detectron2’s logging system. This sets up the formatting and ensures Detectron2 logs (info/warn) appear in the notebook/console.
Reproducibility helper — seed_everything(seed=42)

Function content:
torch.manual_seed(seed) — sets PyTorch CPU RNG seed.
np.random.seed(seed) — sets NumPy RNG seed.
random.seed(seed) — sets Python stdlib RNG seed.
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed) — seeds CUDA RNG on all devices if GPUs exist.
Why it’s useful:
Reduces run-to-run variance (data shuffling, weight initialization, and some augmentation randomness).
Important caveats:
Full determinism requires extra settings (e.g., torch.backends.cudnn.deterministic = True, torch.use_deterministic_algorithms(True)), which can slow training and sometimes aren’t possible for all ops.
On multi-process dataloaders, some non-determinism can persist unless extra care taken.
On macOS with MPS or CPU-only training, reproducibility depends on underlying BLAS and threading libraries.
Final line — environment check print

print(f"Environment Ready. Torch: {torch.__version__} ({'CUDA' if torch.cuda.is_available() else 'CPU'}), Detectron2: {detectron2.__version__}")
Quick runtime sanity: prints the torch version and whether CUDA is available, plus the detectron2 version. Useful first check to ensure the kernel is using the venv where you installed packages.
Common pitfalls that show up in this cell

Missing packages or version mismatches:
If detectron2 fails to import, it usually means it wasn’t installed for the current Python interpreter (kernel mismatch) or it didn’t build for your Python version.
If transformers or albumentations import fails, install them in the same venv.
Kernel mismatch:
Notebook kernel must be the same Python where you installed packages. Check sys.executable in a quick cell to confirm.
macOS / Apple Silicon specifics:
Detectron2 often needs a source build on macOS arm64 and a specific Python minor version (you already used .venv311 for this reason). Building requires cmake, ninja, and pkg-config via Homebrew.
Suppressing warnings:
warnings.filterwarnings('ignore') hides diagnostic messages that might help diagnosing version incompatibilities — consider temporarily removing it if debugging.
Quick checks you can run (in your zsh)



████████████████████████████
█       Cell 3.            █
████████████████████████████

Config (concise, important details)

Purpose: Centralize all project paths and core hyperparameters so other cells use the same settings. Changing values here adjusts dataset locations, where models are saved, and training/runtime behavior across the notebook.

Key fields (what they mean):

DATA_ROOT / data paths: DATA_ROOT = Path("data") and derived paths:
UNLABELED_IMAGES_PATH — where unlabeled images for MAE pretraining live.
TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, VAL_IMAGES_PATH, VAL_MASKS_PATH — expected PH2-like layout used by get_melanoma_dicts.
MODELS_PATH / model artifacts:
SSL_BACKBONE_PATH — where the MAE ViT encoder weights are saved (models/ssl_vit_backbone.pth).
SSL_CONFIG_JSON — small JSON describing ViT config (patch size, hidden size) used to rebuild the ViT for Detectron2.
FINAL_MODEL_PATH — final Mask R-CNN checkpoint path used by predict.py and evaluate().
IMAGE_SIZE: expected image size for ViT/MAE (default (224,224)). Important because ViT patching and positional embeddings assume patch grid sizes derived from this.
Batching / epochs:
BATCH_SIZE_SSL, BATCH_SIZE_DETECTRON, NUM_EPOCHS_SSL — small defaults tuned for notebook/small-data runs.
DEVICE:
torch.device("cuda" if torch.cuda.is_available() else "cpu") — selects CUDA if available, else CPU.
Important: on Apple Silicon you may want torch.device("mps") or detect MPS explicitly; the current expression will fall back to CPU if CUDA isn’t available (even if MPS is available).
__post_init__ behavior:

Ensures the models and data directories exist by creating them (mkdir(exist_ok=True, parents=True)).
This avoids file-not-found errors later when saving weights or writing metadata.
Why these fields matter elsewhere:

SSL_BACKBONE_PATH and SSL_CONFIG_JSON are read by build_vit_backbone() to load a pre-trained encoder.
FINAL_MODEL_PATH is used by evaluate() and predict.py to load the trained Mask R-CNN.
IMAGE_SIZE & patch-related settings must be consistent with how the ViT embeddings are computed and how positional embeddings are interpolated in the backbone.
DEVICE is propagated into cfg.MODEL.DEVICE (Detectron2 config) and determines where tensors/models are placed.



████████████████████████████
█        CELL 4            █
████████████████████████████

UnlabeledSkinDataset & MAETrainer (concise, important details)

What this cell does (one-liner)
- Provides an unlabeled image Dataset used for MAE (masked autoencoder) pretraining and a small trainer class that runs ViT MAE pretraining and saves the ViT encoder for later use as a backbone.

UnlabeledSkinDataset (key points)
- Purpose: load all unlabeled images under `config.UNLABELED_IMAGES_PATH` and produce tensors suitable for ViT MAE pretraining.
- File discovery: recursively finds files matching `*.jpg, *.png, *.jpeg`.
- Transform pipeline:
  - Resize to 256, RandomCrop to 224, RandomHorizontalFlip, ToTensor, ImageNet normalization (mean/std).
  - Output shape: tensor (3, 224, 224), dtype float32 — batch becomes (B, 3, 224, 224).
- Robustness: __getitem__ returns a zero tensor on load failure so the DataLoader keeps working.
- Why 224/normalize: ViT MAE expects 224×224 patches and typically ImageNet-style normalization if using pretrained components or similarly scaled inputs.

MAETrainer (key points)
- Model instantiation:
  - `ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')` — loads a ready MAE model that computes reconstruction loss for masked patches out of the box.
  - Model moved to `config.DEVICE` (so training runs on GPU/MPS/CPU as available).
- Optimizer:
  - `AdamW` with lr=1.5e-4 and weight decay 0.05 — standard for transformer pretraining.
- Training loop:
  - For each epoch, iterates dataloader batches and computes `loss = self.model(pixel_values=batch.to(self.config.DEVICE)).loss`.
  - Typical HF MAE returns a Namespace with `.loss` when called with pixel inputs.
  - Standard optimizer steps: zero_grad, backward, step.
  - Prints epoch average loss for monitoring.
- Saving:
  - `save_backbone()` saves `self.model.vit.state_dict()` (the encoder only) to `config.SSL_BACKBONE_PATH`.
  - Also writes small `ssl_config.json` describing image/patch/hidden sizes needed later to re-create a compatible `ViTModel`.
  - Important: saving encoder state_dict (not entire HF wrapper) is intentional — `build_vit_backbone()` later constructs a ViTModel and loads these weights.

Operational notes & caveats
- Input expectation: the MAE model is fed `pixel_values` as a (B, C, H, W) tensor already normalized by the dataset transforms. Confirm the HF MAE you use expects normalized values; if not, adjust transforms accordingly.
- Resource usage: MAE pretraining is compute- and memory-intensive (esp. for high batch sizes). On CPU or MPS this will be very slow—consider using smaller batch sizes, fewer epochs, or starting from an existing ViT checkpoint instead of training from scratch.
- Batch size: `config.BATCH_SIZE_SSL` controls DataLoader batch size; lower on limited hardware.
- Determinism: training still non-deterministic due to GPU operations, but seeding helps reproducibility for debugging.
- If no unlabeled data: `run_ssl_training()` checks for images and will print "Skipping SSL" — safe to run even when you don't plan to pretrain.
- Compatibility: The saved `ssl_config.json` must match any code that reconstructs the ViT for backbone conversion (patch size, hidden size, layers). Changing ViT config later requires careful weight mapping.


████████████████████████████
█        CELL 5            █
████████████████████████████

Cell 5 — `get_melanoma_dicts(...)` and dataset registration (concise, important details)

What this cell does (one-sentence)
- Converts your paired image + mask files into Detectron2’s expected dataset dicts and registers them with `DatasetCatalog`/`MetadataCatalog` so Detectron2 can load training and validation data.

Step-by-step (important bits)
- File discovery:
  - Collects image files from `img_dir` using globs for common extensions (case variants included).
- For each image:
  - Builds a `record` with `file_name` (str path) and `image_id` (index).
  - Reads the image via `cv2.imread(...)` and stores `height`, `width` from `image.shape[:2]`.
- Mask resolution:
  - Looks for a mask file named `"<image_stem>_mask<orig_ext>"` and falls back to `"<image_stem>_mask.png"` or `.jpg` if the exact extension isn’t found.
  - If no mask, skips this image (so dataset only contains annotated images).
- Mask reading & contour extraction:
  - Reads mask as grayscale (`cv2.IMREAD_GRAYSCALE`).
  - Uses `cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` to get external contours (each contour is a candidate object).
- Build annotations for each contour:
  - Filters tiny contours: `if cv2.contourArea(c) < 50: continue` (helps remove noise).
  - Flattens polygon points to a list of coordinates: `poly = c.flatten().tolist()` → produces [x0,y0,x1,y1,...].
  - Ensures valid polygon: `if len(poly) < 6: continue` (less than 3 points is invalid).
  - Computes bounding box via `x,y,w,h = cv2.boundingRect(c)` but stores as `[x, y, x+w, y+h]` — this is (x_min,y_min,x_max,y_max).
  - Each annotation dict contains:
    - `"bbox"`: [x_min, y_min, x_max, y_max]
    - `"bbox_mode": BoxMode.XYXY_ABS` (tells Detectron2 these are absolute XYXY coords)
    - `"segmentation"`: [poly] (a list containing one polygon list of floats)
    - `"category_id": 0` (single-class dataset; class index 0 = "lesion")
- Append:
  - If any `objs` (annotations) exist for the image, attach `record["annotations"] = objs` and append `record` to `dataset_dicts`.
- Registration:
  - Registers two datasets, `melanoma_train` and `melanoma_val`, with `DatasetCatalog.register(...)` using lambdas that call `get_melanoma_dicts(...)` with appropriate train/val paths.
  - Sets `MetadataCatalog.get(...).set(thing_classes=["lesion"])` so visualizers know the class names.
  - Note: the lambda uses `lambda d=d: ...` pattern to capture loop variable correctly.

Why this matters (Detectron2 expectations)
- Detectron2 expects a function that returns a list of dicts where each dict has at minimum:
  - `file_name`, `height`, `width`, `image_id` and optionally `annotations` (list of per-instance dicts).
- Annotation format must match Detectron2 conventions:
  - `segmentation`: list(s) of floats representing polygon(s).
  - `bbox` with `BoxMode` specifying the coordinate format.
  - `category_id` as integer class index.



████████████████████████████
█        CELL 6            █
████████████████████████████

**AlbumentationsMapper — concise explanation**

- **Purpose:**  
  - Converts a Detectron2 dataset dict (filename, height, width, annotations) into a model-ready sample with augmentations. It applies strong image/mask/bbox augmentations during training and returns a dict containing `image` (tensor) and `instances` (Detectron2 `Instances`) for the dataloader.

- **Augmentations used:**  
  - **Train aug (`self.aug`)**: `A.Resize(512,512)`, horizontal/vertical flips, `ShiftScaleRotate`, `RandomBrightnessContrast`, `A.Normalize(...)`, `ToTensorV2()`.  
  - **Val aug (`self.val_aug`)**: `A.Resize(512,512)`, `Normalize`, `ToTensorV2()`.  
  - Note: `A.Resize` ensures every sample is the model’s expected spatial size (512×512 here), so downstream feature maps and bboxes remain consistent.

- **Input → internal conversion (important data-shape details):**  
  - Reads the image with `utils.read_image(..., format="BGR")` then converts to RGB for Albumentations.  
  - Builds `masks` as numpy 2D arrays (shape `(height, width)`), `bboxes` as lists `[x_min, y_min, x_max, y_max]`, and `labels` as ints.  
  - Albumentations returns transformed `image` (numpy or torch tensor depending on `ToTensorV2`), `masks` (list of H×W arrays), and `bboxes` (list of pascal_voc boxes). After `ToTensorV2()`, `image` is a torch tensor `(C, H, W)` float32 normalized to ImageNet mean/std.

- **BBox/label plumbing:**  
  - Mapper config uses `bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])`. This tells Albumentations to transform bboxes in Pascal VOC format and keep labels aligned with boxes.
  - Detectron2 expects bboxes as `Boxes(torch.tensor(bboxes, dtype=torch.float32))`.

- **Instances / mask packaging (what the model receives):**  
  - If there are annotations after augmentations, it creates:
    - `target = Instances((img_h, img_w))`
    - `target.gt_boxes` = `Boxes(torch.tensor(bboxes, dtype=torch.float32))`
    - `target.gt_classes` = `torch.tensor(labels, dtype=torch.int64)`
    - `target.gt_masks` = `BitMasks(torch.stack([torch.as_tensor(m, dtype=torch.bool) for m in masks]))` — BitMasks expects a tensor shape `(N, H, W)` of booleans.
  - If no annotations remain, it sets empty placeholders (`Boxes(torch.zeros((0,4)))`, zero-length `gt_classes`, empty `BitMasks`) to avoid downstream crashes.

- **Robustness / edge cases handled:**  
  - Wraps augmentation in `try/except` and falls back to `val_aug` if augmentation fails (helps avoid crashing when an aug removes all bboxes or a particular mask is malformed).  
  - Trims `bboxes`, `masks`, and `labels` to `min_len` to ensure consistent lengths after augmentation (albumentations can drop boxes).  
  - Converts mask polygons into raster masks before augmentation so Albumentations can transform them correctly.

- **Common failure modes & quick fixes:**  
  - **Wrong bbox format** → verify boxes are `[x_min,y_min,x_max,y_max]` (Pascal VOC).  
  - **Mask dtype/values** → `cv2.findContours` and `cv2.fillPoly` expect 0/255 or 0/1; ensure masks are binary before building polygons. If Albumentations returns empty masks or unexpected types, inspect `masks[0].dtype` and `masks[0].max()`.  
  - **Coordinates outside image** → check and clamp bbox coordinates to image dims; Albumentations can produce invalid bboxes if input masks are unexpected.  
  - **NumWorkers / pickling issues** → keep `cfg.DATALOADER.NUM_WORKERS = 0` during debugging (the notebook does this).

- **Performance notes:**  
  - `ToTensorV2()` moves data to CPU memory as float32; pushing many large masks may increase memory. Keep `BATCH_SIZE` small on CPU/MPS.  
  - Resizing to fixed 512×512 simplifies feature-stride math but increases memory compared to smaller sizes (tradeoff: accuracy vs cost).



████████████████████████████
█        CELL 7            █
████████████████████████████

**Training Runner (Cell 7)**

- **Purpose:**  
  - Configure Detectron2 training, register the custom ViT backbone, and run Mask R-CNN training end-to-end via a `DefaultTrainer` subclass.

- **Key pieces (what the code defines):**
  - **`AugTrainer`**: subclass of `DefaultTrainer` that overrides `build_train_loader` to return Detectron2’s train loader but with the `AlbumentationsMapper` (so your custom augmentations + masks/bboxes are used).
  - **`run_mask_rcnn_training()`**: main function that:
    1. **Data check** — exits early with "No data found." if `config.TRAIN_IMAGES_PATH` is empty.
    2. **Backbone registry refresh** — removes any stale `"build_vit_backbone"` entry from `BACKBONE_REGISTRY` and re-registers `build_vit_backbone` to ensure Detectron2 uses your current ViT→FPN builder.
    3. **Load base config** — `cfg = get_cfg()` then `cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))` to use a Mask R-CNN template.
    4. **Prevent default R-50 weights** — the code forces `cfg.MODEL.WEIGHTS` to empty strings so Detectron2 does not automatically load the R-50 backbone weights (important because you’re using a custom ViT backbone).
    5. **Set backbone & FPN** — `cfg.MODEL.BACKBONE.NAME = "build_vit_backbone"` and `cfg.MODEL.FPN.IN_FEATURES = ["res2","res3","res4","res5"]`.
    6. **Dataset / classes** — `cfg.DATASETS.TRAIN = ("melanoma_train",)` and `cfg.DATASETS.TEST = ("melanoma_val",)`; `cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1`.
    7. **Runtime / perf settings** — `cfg.DATALOADER.NUM_WORKERS = 0` (safer for notebooks/mac), `cfg.SOLVER.IMS_PER_BATCH = config.BATCH_SIZE_DETECTRON`, `cfg.SOLVER.MAX_ITER = 500`.
    8. **Device and output** — `cfg.MODEL.DEVICE = str(config.DEVICE)` and `cfg.OUTPUT_DIR` created under `models/detectron2_output`.
    9. **Train** — create `trainer = AugTrainer(cfg)`, call `trainer.resume_or_load(resume=False)` then `trainer.train()`.
   10. **Save final weights** — `DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR).save("final_model")` then copy the produced `final_model.pth` to `config.FINAL_MODEL_PATH`.

- **Why the important lines exist:**
  - **Registry refresh:** ensures your most recent `build_vit_backbone` function is used (prevents stale references if you edited the function in the notebook).
  - **Clearing `MODEL.WEIGHTS`:** stops Detectron2 from loading unrelated ResNet weights which would mismatch your ViT backbone.
  - **`NUM_WORKERS=0`:** avoids multiprocessing/pickling errors inside notebooks and on macOS; increase later on stable Linux/GPU.
  - **`IMS_PER_BATCH`:** total images per step across devices — lowering it reduces memory use.
  - **`resume_or_load(resume=False)`:** starts fresh; set `resume=True` to continue from a checkpoint.

- **Where outputs appear:**  
  - Training logs and checkpoints are written to the directory in `cfg.OUTPUT_DIR` (default `models/detectron2_output`). After training the notebook copies `final_model.pth` to `models/final_lesion_segmenter.pth` (via `config.FINAL_MODEL_PATH`).


████████████████████████████
█        CELL 8            █
████████████████████████████

`evaluate()` (concise, important details)

Purpose
- Load the trained Mask R-CNN model, run a few validation images through it, and display visualized predictions for quick qualitative checks.

What it does (step-by-step)
1. Early exit:
   - `if not config.FINAL_MODEL_PATH.exists(): return` — safe guard if training hasn’t produced `final_model.pth`.
2. Build Detectron2 config:
   - `cfg = get_cfg()` then `cfg.merge_from_file(model_zoo.get_config_file(...mask_rcnn_R_50_FPN_3x.yaml))` to reuse the Mask R-CNN template.
   - Re-apply important overrides: set `cfg.MODEL.DEVICE = str(config.DEVICE)`, `cfg.MODEL.BACKBONE.NAME = "build_vit_backbone"`, `cfg.MODEL.FPN.IN_FEATURES = ["res2","res3","res4","res5"]`, `cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1`.
   - `cfg.MODEL.PIXEL_MEAN = [0,0,0]; cfg.MODEL.PIXEL_STD = [1,1,1]` — because the code applies Albumentations normalization before feeding the tensor.
3. Construct and load model:
   - `model = build_model(cfg)` creates the Detectron2 model object programmatically (it will use the registered `build_vit_backbone`).
   - `DetectionCheckpointer(model).load(str(config.FINAL_MODEL_PATH))` loads saved weights into the model.
   - `model.eval()` puts model in inference mode (disables dropout, uses running stats).
4. Prepare validation examples:
   - `dicts = get_melanoma_dicts(config.VAL_IMAGES_PATH, config.VAL_MASKS_PATH)` builds raw dataset dicts.
   - For the first few items: read image with `cv2.imread` and convert BGR→RGB.
   - Apply the same albumentations normalization used in training (`A.Normalize(...)`, `ToTensorV2()`), producing a tensor `inp` shaped (C,H,W).
   - Move tensor to device with `.to(config.DEVICE)`.
5. Run inference:
   - Call `with torch.no_grad(): out = model([{"image": inp.squeeze(0)}])[0]`. Detectron2 models expect a list of dicts and return a list of outputs.
   - `out` contains an `Instances` object with fields like `pred_boxes`, `pred_masks`, `scores`, `pred_classes`.
6. Visualize:
   - `v = Visualizer(img, metadata=MetadataCatalog.get("melanoma_val"))` — create visualizer with dataset metadata (class names).
   - `res = v.draw_instance_predictions(out["instances"].to("cpu"))` — draw predictions (must move tensors to CPU).
   - Use matplotlib to show `res.get_image()`.

Important notes & gotchas
- Input normalization must match training:
  - The code normalizes images before passing to model (albumentations). If you instead feed raw images, model outputs will be incorrect — keep pixel mean/std consistent.
- Device consistency:
  - Ensure `config.DEVICE` matches where the model weights were trained. Loading GPU-trained weights on CPU may require map_location or re-saving; Detectron2 checkpointer handles common cases but check errors.
- Model/state mismatch:
  - If loading weights fails, confirm `cfg` (backbone name, ROI classes) matches training config and that the saved checkpoint contains compatible keys.
- No predictions shown:
  - If `out["instances"]` is empty, verify training produced any detections, thresholding, or that class indices align (ROI_HEADS.NUM_CLASSES should be 1).
- Visualization expects CPU tensors:
  - Call `.to("cpu")` on `out["instances"]` before passing to `Visualizer`.
- Pixel mean/std override:
  - The notebook sets PIXEL_MEAN/STD to identity because input already normalized. If you change normalization pipeline, update these values accordingly.

