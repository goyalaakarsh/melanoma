#!/usr/bin/env python3
"""
Standalone inference script for melanoma lesion segmentation.
Uses trained Mask R-CNN model with ViT backbone for prediction.

Usage:
    python predict.py --image path/to/image.jpg --output path/to/result.png
    python predict.py --image path/to/image.jpg  # Display result without saving
"""

import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the current directory to Python path to import from notebook
sys.path.append(str(Path(__file__).parent))

# Import required modules (these would be imported from the notebook)
try:
    import detectron2
    from detectron2.utils.logger import setup_logger
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # Import our custom deep learning models
    from deep_learning_models import build_vit_backbone
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure all dependencies are installed:")
    print("pip install torch torchvision detectron2 transformers timm albumentations")
    sys.exit(1)

# Setup logging
setup_logger()

class MelanomaSegmenter:
    """
    Melanoma lesion segmentation using trained Mask R-CNN model.
    """
    
    def __init__(self, model_path, config_path, device='cuda'):
        """
        Initialize the segmenter.
        
        Args:
            model_path (str): Path to trained model weights
            config_path (str): Path to model configuration
            device (str): Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.cfg = None
        self.metadata = None
        
        # Load model and configuration
        self.load_model(model_path, config_path)
        
        # Setup metadata
        self.setup_metadata()
        
        # Preprocessing transforms
        self.setup_transforms()
    
    def load_model(self, model_path, config_path):
        """Load trained model and configuration"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load configuration
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        
        # Set device
        self.cfg.MODEL.DEVICE = str(self.device)
        
        # Build model
        self.model = build_model(self.cfg)
        self.model.to(self.device)
        self.model.eval()
        
        # Load weights
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(model_path)
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Running on device: {self.device}")
    
    def setup_metadata(self):
        """Setup metadata for visualization"""
        self.metadata = MetadataCatalog.get("melanoma_val")
        self.metadata.thing_classes = ["lesion"]
    
    def setup_transforms(self):
        """Setup preprocessing transforms"""
        self.transform = A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        return image_tensor
    
    def predict(self, image_path, confidence_threshold=0.5):
        """
        Run inference on a single image.
        
        Args:
            image_path (str): Path to input image
            confidence_threshold (float): Minimum confidence for predictions
            
        Returns:
            dict: Prediction results containing masks, boxes, and scores
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Run inference
        with torch.no_grad():
            # Move to device and add batch dimension
            inputs = [{"image": image_tensor.to(self.device)}]
            outputs = self.model(inputs)[0]
        
        # Filter by confidence
        instances = outputs["instances"]
        instances = instances[instances.scores > confidence_threshold]
        
        # Convert to CPU for processing
        instances = instances.to("cpu")
        
        # Extract results
        results = {
            'masks': instances.pred_masks.numpy() if len(instances) > 0 else np.array([]),
            'boxes': instances.pred_boxes.tensor.numpy() if len(instances) > 0 else np.array([]),
            'scores': instances.scores.numpy() if len(instances) > 0 else np.array([]),
            'classes': instances.pred_classes.numpy() if len(instances) > 0 else np.array([]),
            'num_instances': len(instances)
        }
        
        return results
    
    def get_best_mask(self, image_path, confidence_threshold=0.5):
        """
        Get the best segmentation mask from predictions.
        
        Args:
            image_path (str): Path to input image
            confidence_threshold (float): Minimum confidence for predictions
            
        Returns:
            np.ndarray: Binary mask of the best prediction, or None if no predictions
        """
        results = self.predict(image_path, confidence_threshold)
        
        if results['num_instances'] == 0:
            print(f"No predictions above confidence threshold {confidence_threshold}")
            return None
        
        # Get the mask with highest confidence
        best_idx = np.argmax(results['scores'])
        best_mask = results['masks'][best_idx]
        
        # Convert to binary mask
        binary_mask = (best_mask > 0.5).astype(np.uint8) * 255
        
        return binary_mask
    
    def visualize_prediction(self, image_path, confidence_threshold=0.5, save_path=None):
        """
        Visualize prediction results.
        
        Args:
            image_path (str): Path to input image
            confidence_threshold (float): Minimum confidence for predictions
            save_path (str, optional): Path to save visualization
        """
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get predictions
        results = self.predict(image_path, confidence_threshold)
        
        if results['num_instances'] == 0:
            print(f"No predictions above confidence threshold {confidence_threshold}")
            # Show original image
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.title("No predictions found")
            plt.axis('off')
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            return
        
        # Create visualization
        # Resize image to match model output size
        image_resized = cv2.resize(image, (512, 512))
        
        # Convert to tensor for Detectron2 visualizer
        image_tensor = torch.as_tensor(image_resized.transpose(2, 0, 1).astype("float32"))
        
        # Create instances for visualization
        from detectron2.structures import Instances, Boxes
        
        instances = Instances((512, 512))
        instances.pred_masks = torch.as_tensor(results['masks'])
        instances.pred_boxes = Boxes(torch.as_tensor(results['boxes']))
        instances.scores = torch.as_tensor(results['scores'])
        instances.pred_classes = torch.as_tensor(results['classes'])
        
        # Visualize
        v = Visualizer(image_resized, metadata=self.metadata, scale=1.2)
        out = v.draw_instance_predictions(instances)
        
        # Display
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image_resized)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(out.get_image())
        plt.title(f"Predictions (confidence > {confidence_threshold})")
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()

def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(description="Melanoma lesion segmentation inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to save output visualization")
    parser.add_argument("--model", default="models/final_lesion_segmenter.pth", 
                       help="Path to trained model")
    parser.add_argument("--config", default="models/config.yaml",
                       help="Path to model configuration")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold for predictions")
    parser.add_argument("--mask-only", action="store_true",
                       help="Return only the binary mask")
    
    args = parser.parse_args()
    
    # Check if input image exists
    if not os.path.exists(args.image):
        print(f"Error: Input image not found: {args.image}")
        return
    
    # Check if model files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first using the notebook")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        print("Please train the model first using the notebook")
        return
    
    try:
        # Initialize segmenter
        segmenter = MelanomaSegmenter(args.model, args.config)
        
        if args.mask_only:
            # Return only the mask
            mask = segmenter.get_best_mask(args.image, args.confidence)
            if mask is not None:
                if args.output:
                    cv2.imwrite(args.output, mask)
                    print(f"Mask saved to {args.output}")
                else:
                    print("Binary mask generated (use --output to save)")
            else:
                print("No mask generated")
        else:
            # Full visualization
            segmenter.visualize_prediction(args.image, args.confidence, args.output)
    
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
