"""
Script to organize PH2 Dataset into training/validation directory structure.
Copies images from PH2 Dataset to the appropriate folders for deep learning training.
"""

import os
import shutil
from pathlib import Path
import random

# Configuration
PH2_DATASET_PATH = r"C:\Users\Aakarsh Goyal\Downloads\archive\PH2Dataset\PH2 Dataset images"
OUTPUT_BASE_PATH = Path(__file__).parent / "data"

# Split ratio (80% train, 20% validation)
TRAIN_RATIO = 0.8

def organize_ph2_dataset():
    """
    Organize PH2 Dataset images into train/val structure.
    
    Structure:
    - Original images: IMDxxx/IMDxxx_Dermoscopic_Image/IMDxxx.bmp
    - Mask images: IMDxxx/IMDxxx_lesion/IMDxxx.bmp
    """
    
    print("=" * 80)
    print("PH2 Dataset Organizer")
    print("=" * 80)
    
    # Create output directories
    train_images_dir = OUTPUT_BASE_PATH / "train" / "images"
    train_masks_dir = OUTPUT_BASE_PATH / "train" / "masks"
    val_images_dir = OUTPUT_BASE_PATH / "val" / "images"
    val_masks_dir = OUTPUT_BASE_PATH / "val" / "masks"
    unlabeled_images_dir = OUTPUT_BASE_PATH / "images"
    
    for dir_path in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, unlabeled_images_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreated output directories")
    print(f"  - Train images: {train_images_dir}")
    print(f"  - Train masks: {train_masks_dir}")
    print(f"  - Val images: {val_images_dir}")
    print(f"  - Val masks: {val_masks_dir}")
    print(f"  - Unlabeled images: {unlabeled_images_dir}")
    
    # Find all image folders
    ph2_path = Path(PH2_DATASET_PATH)
    if not ph2_path.exists():
        print(f"\nERROR: PH2 Dataset path not found: {PH2_DATASET_PATH}")
        print("   Please update PH2_DATASET_PATH in the script")
        return
    
    image_folders = [f for f in ph2_path.iterdir() if f.is_dir() and f.name.startswith("IMD")]
    image_folders.sort()
    
    print(f"\nFound {len(image_folders)} image folders")
    
    if len(image_folders) == 0:
        print(f"\nERROR: No IMD folders found in {PH2_DATASET_PATH}")
        return
    
    # Shuffle for random train/val split
    random.seed(42)  # For reproducibility
    random.shuffle(image_folders)
    
    # Calculate split point
    split_idx = int(len(image_folders) * TRAIN_RATIO)
    train_folders = image_folders[:split_idx]
    val_folders = image_folders[split_idx:]
    
    print(f"\nSplit: {len(train_folders)} training, {len(val_folders)} validation")
    
    # Process training images
    print("\nProcessing training images...")
    copied_train = 0
    for folder in train_folders:
        result = copy_image_and_mask(folder, train_images_dir, train_masks_dir, unlabeled_images_dir)
        if result:
            copied_train += 1
            if copied_train % 10 == 0:
                print(f"   Processed {copied_train}/{len(train_folders)} training images...")
    
    print(f"Copied {copied_train} training images")
    
    # Process validation images
    print("\nProcessing validation images...")
    copied_val = 0
    for folder in val_folders:
        result = copy_image_and_mask(folder, val_images_dir, val_masks_dir, unlabeled_images_dir)
        if result:
            copied_val += 1
            if copied_val % 10 == 0:
                print(f"   Processed {copied_val}/{len(val_folders)} validation images...")
    
    print(f"Copied {copied_val} validation images")
    
    # Summary
    print("\n" + "=" * 80)
    print("DATASET ORGANIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Total images processed: {copied_train + copied_val}")
    print(f"  - Training images: {copied_train}")
    print(f"  - Validation images: {copied_val}")
    print(f"  - Unlabeled images (for SSL): {copied_train + copied_val}")
    print(f"\nOutput directories:")
    print(f"  - Training: {train_images_dir}")
    print(f"  - Validation: {val_images_dir}")
    print(f"  - Unlabeled: {unlabeled_images_dir}")
    
    print(f"\nReady for training!")
    print(f"   Run: train_segmentation_model.ipynb")
    print("=" * 80)


def copy_image_and_mask(folder, images_dir, masks_dir, unlabeled_dir):
    """
    Copy original image and mask from PH2 folder structure.
    
    Args:
        folder: Path to IMDxxx folder
        images_dir: Destination for original images
        masks_dir: Destination for mask images
        unlabeled_dir: Destination for unlabeled images (SSL)
    
    Returns:
        bool: True if successful, False otherwise
    """
    folder_name = folder.name  # e.g., "IMD427"
    
    # Find original image
    dermoscopic_folder = folder / f"{folder_name}_Dermoscopic_Image"
    original_image = dermoscopic_folder / f"{folder_name}.bmp"
    
    # Find mask image
    lesion_folder = folder / f"{folder_name}_lesion"
    mask_image = lesion_folder / f"{folder_name}_lesion.bmp"
    
    # Check if both exist
    if not original_image.exists():
        print(f"Warning: Original image not found for {folder_name}")
        return False
    
    if not mask_image.exists():
        print(f"Warning: Mask image not found for {folder_name}")
        return False
    
    try:
        # Copy original image (convert to jpg for consistency)
        dest_image = images_dir / f"{folder_name}.jpg"
        shutil.copy2(original_image, dest_image)
        
        # Copy mask image (keep as bmp or convert to png)
        dest_mask = masks_dir / f"{folder_name}_mask.jpg"
        shutil.copy2(mask_image, dest_mask)
        
        # Copy original to unlabeled directory (for SSL pre-training)
        dest_unlabeled = unlabeled_dir / f"{folder_name}.jpg"
        shutil.copy2(original_image, dest_unlabeled)
        
        return True
        
    except Exception as e:
        print(f"Error copying {folder_name}: {e}")
        return False


def verify_dataset():
    """Verify the organized dataset structure."""
    
    print("\n" + "=" * 80)
    print("DATASET VERIFICATION")
    print("=" * 80)
    
    base_path = OUTPUT_BASE_PATH
    
    # Count files
    train_images = list((base_path / "train" / "images").glob("*.jpg"))
    train_masks = list((base_path / "train" / "masks").glob("*.jpg"))
    val_images = list((base_path / "val" / "images").glob("*.jpg"))
    val_masks = list((base_path / "val" / "masks").glob("*.jpg"))
    unlabeled = list((base_path / "images").glob("*.jpg"))
    
    print(f"\nFile counts:")
    print(f"  - Training images: {len(train_images)}")
    print(f"  - Training masks: {len(train_masks)}")
    print(f"  - Validation images: {len(val_images)}")
    print(f"  - Validation masks: {len(val_masks)}")
    print(f"  - Unlabeled images: {len(unlabeled)}")
    
    # Check for matching pairs
    train_match = len(train_images) == len(train_masks)
    val_match = len(val_images) == len(val_masks)
    
    print(f"\nMatching pairs:")
    print(f"  - Training: {'OK' if train_match else 'MISMATCH'}")
    print(f"  - Validation: {'OK' if val_match else 'MISMATCH'}")
    
    # Sample verification
    if train_images:
        sample_image = train_images[0]
        sample_name = sample_image.stem  # e.g., "IMD427"
        sample_mask = base_path / "train" / "masks" / f"{sample_name}_mask.jpg"
        
        print(f"\nSample files:")
        print(f"  - Image: {sample_image.name}")
        print(f"  - Mask: {sample_mask.name}")
        print(f"  - Mask exists: {'YES' if sample_mask.exists() else 'NO'}")
    
    print("\n" + "=" * 80)
    
    return train_match and val_match


if __name__ == "__main__":
    print("\nStarting PH2 Dataset organization...\n")
    
    # Run organization
    organize_ph2_dataset()
    
    # Verify results
    verify_dataset()
    
    print("\nAll done! You can now run train_segmentation_model.ipynb\n")

