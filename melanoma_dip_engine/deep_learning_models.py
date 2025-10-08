"""
Deep Learning Models for Melanoma Segmentation.
This module contains the custom ViT backbone and related components for Detectron2.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, List

# Detectron2 imports
import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

# Hugging Face and timm imports
try:
    import timm
    from transformers import ViTModel, ViTConfig
except ImportError:
    print("Warning: timm or transformers not available. Deep learning features disabled.")
    timm = None
    ViTModel = None
    ViTConfig = None

class ViTFeatureExtractor(nn.Module):
    """
    Extract multi-scale features from ViT encoder.
    Converts 1D patch tokens back to 2D feature maps at different scales.
    """
    
    def __init__(self, vit_model, feature_layers=[3, 6, 9, 12]):
        super().__init__()
        self.vit_model = vit_model
        self.feature_layers = feature_layers
        self.patch_size = vit_model.config.patch_size
        self.hidden_size = vit_model.config.hidden_size
        
        # Freeze ViT parameters
        for param in self.vit_model.parameters():
            param.requires_grad = False
        
        # Feature projection layers to match Detectron2's expected channels
        self.feature_projections = nn.ModuleList([
            nn.Conv2d(self.hidden_size, 256, 1) for _ in range(len(feature_layers))
        ])
        
        # Initialize projection layers
        for proj in self.feature_projections:
            weight_init.c2_msra_fill(proj)
    
    def forward(self, x):
        """
        Extract multi-scale features from ViT.
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            List of feature maps at different scales
        """
        B, C, H, W = x.shape
        
        # Reshape to patches and add positional embedding
        patches = self.vit_model.embeddings.patch_embeddings(x)  # (B, num_patches, hidden_size)
        
        # Add CLS token and positional embeddings
        cls_tokens = self.vit_model.embeddings.cls_token.expand(B, -1, -1)
        embeddings = torch.cat((cls_tokens, patches), dim=1)
        embeddings = embeddings + self.vit_model.embeddings.position_embeddings
        
        # Apply transformer blocks and extract features at specified layers
        hidden_states = embeddings
        features = []
        
        for i, block in enumerate(self.vit_model.encoder.layer):
            hidden_states = block(hidden_states)[0]
            
            # Extract features at specified layers (excluding CLS token)
            if i + 1 in self.feature_layers:
                # Remove CLS token and reshape to 2D
                patch_features = hidden_states[:, 1:]  # Remove CLS token
                
                # Calculate patch grid size
                num_patches_per_dim = int(np.sqrt(patch_features.shape[1]))
                patch_features = patch_features.reshape(
                    B, num_patches_per_dim, num_patches_per_dim, self.hidden_size
                )
                patch_features = patch_features.permute(0, 3, 1, 2)  # (B, C, H, W)
                
                # Project to expected channel size
                projected_features = self.feature_projections[self.feature_layers.index(i + 1)](patch_features)
                features.append(projected_features)
        
        return features

class ViTBackbone(Backbone):
    """
    ViT backbone for Detectron2.
    Provides multi-scale features compatible with FPN.
    """
    
    def __init__(self, vit_model, feature_layers=[3, 6, 9, 12]):
        super().__init__()
        self.feature_extractor = ViTFeatureExtractor(vit_model, feature_layers)
        
        # Define feature output shapes
        # Assuming input size of 224x224 and patch size of 16
        self._out_feature_channels = {
            "res3": 256,  # 14x14 features
            "res4": 256,  # 14x14 features  
            "res5": 256,  # 14x14 features
            "res6": 256,  # 14x14 features
        }
        
        self._out_feature_strides = {
            "res3": 16,  # 224/14 = 16
            "res4": 16,
            "res5": 16, 
            "res6": 16,
        }
    
    def forward(self, x):
        """
        Forward pass through ViT backbone.
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Dict of feature maps at different scales
        """
        features = self.feature_extractor(x)
        
        return {
            "res3": features[0],
            "res4": features[1], 
            "res5": features[2],
            "res6": features[3],
        }
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], 
                stride=self._out_feature_strides[name]
            )
            for name in self._out_feature_channels.keys()
        }

@BACKBONE_REGISTRY.register()
def build_vit_backbone(cfg, input_shape: ShapeSpec):
    """
    Build ViT backbone for Detectron2.
    This function is called by Detectron2's model builder.
    """
    
    # Check if SSL backbone exists
    ssl_path = Path("models/ssl_vit_backbone.pth")
    if not ssl_path.exists():
        print(f"Warning: SSL backbone not found at {ssl_path}")
        print("Using ImageNet pre-trained ViT instead")
        
        if timm is None or ViTModel is None or ViTConfig is None:
            raise ImportError("timm and transformers are required for ViT backbone")
        
        # Use ImageNet pre-trained ViT
        vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        
        # Convert to HuggingFace format for consistency
        # Create ViT config
        vit_config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )
        
        # Create model and load weights
        vit_model_hf = ViTModel(vit_config)
        
        # Map timm weights to HuggingFace format
        timm_state = vit_model.state_dict()
        hf_state = {}
        
        # Map embedding layer
        hf_state['embeddings.patch_embeddings.projection.weight'] = timm_state['patch_embed.proj.weight']
        hf_state['embeddings.patch_embeddings.projection.bias'] = timm_state['patch_embed.proj.bias']
        hf_state['embeddings.cls_token'] = timm_state['cls_token']
        hf_state['embeddings.position_embeddings'] = timm_state['pos_embed'][:, 1:]  # Remove class token
        
        # Map transformer layers
        for i in range(12):
            prefix = f'encoder.layer.{i}'
            timm_prefix = f'blocks.{i}'
            
            # Attention
            hf_state[f'{prefix}.attention.attention.query.weight'] = timm_state[f'{timm_prefix}.attn.qkv.weight'][:768]
            hf_state[f'{prefix}.attention.attention.query.bias'] = timm_state[f'{timm_prefix}.attn.qkv.bias'][:768]
            hf_state[f'{prefix}.attention.attention.key.weight'] = timm_state[f'{timm_prefix}.attn.qkv.weight'][768:1536]
            hf_state[f'{prefix}.attention.attention.key.bias'] = timm_state[f'{timm_prefix}.attn.qkv.bias'][768:1536]
            hf_state[f'{prefix}.attention.attention.value.weight'] = timm_state[f'{timm_prefix}.attn.qkv.weight'][1536:]
            hf_state[f'{prefix}.attention.attention.value.bias'] = timm_state[f'{timm_prefix}.attn.qkv.bias'][1536:]
            hf_state[f'{prefix}.attention.output.dense.weight'] = timm_state[f'{timm_prefix}.attn.proj.weight']
            hf_state[f'{prefix}.attention.output.dense.bias'] = timm_state[f'{timm_prefix}.attn.proj.bias']
            
            # Layer norm
            hf_state[f'{prefix}.layernorm_before.weight'] = timm_state[f'{timm_prefix}.norm1.weight']
            hf_state[f'{prefix}.layernorm_before.bias'] = timm_state[f'{timm_prefix}.norm1.bias']
            hf_state[f'{prefix}.layernorm_after.weight'] = timm_state[f'{timm_prefix}.norm2.weight']
            hf_state[f'{prefix}.layernorm_after.bias'] = timm_state[f'{timm_prefix}.norm2.bias']
            
            # MLP
            hf_state[f'{prefix}.intermediate.dense.weight'] = timm_state[f'{timm_prefix}.mlp.fc1.weight']
            hf_state[f'{prefix}.intermediate.dense.bias'] = timm_state[f'{timm_prefix}.mlp.fc1.bias']
            hf_state[f'{prefix}.output.dense.weight'] = timm_state[f'{timm_prefix}.mlp.fc2.weight']
            hf_state[f'{prefix}.output.dense.bias'] = timm_state[f'{timm_prefix}.mlp.fc2.bias']
        
        # Load mapped weights
        vit_model_hf.load_state_dict(hf_state)
        vit_model = vit_model_hf
        
    else:
        print(f"Loading SSL backbone from {ssl_path}")
        
        if ViTModel is None or ViTConfig is None:
            raise ImportError("transformers is required for SSL backbone loading")
        
        # Load SSL backbone
        # Load config
        with open(Path("models/ssl_config.json"), 'r') as f:
            config_dict = json.load(f)
        
        # Create ViT model
        vit_config = ViTConfig(**config_dict)
        vit_model = ViTModel(vit_config)
        
        # Load SSL weights
        ssl_weights = torch.load(ssl_path, map_location='cpu')
        vit_model.load_state_dict(ssl_weights)
        
        print("SSL backbone loaded successfully!")
    
    # Create backbone
    backbone = ViTBackbone(vit_model)
    return backbone
