"""Test script to verify SSL backbone loading fix"""
import torch
import json
from transformers import ViTModel, ViTConfig

# Load config
with open('models/ssl_config.json', 'r') as f:
    config_dict = json.load(f)

# Create ViT config and model WITHOUT pooler layer
vit_config = ViTConfig(**config_dict)
model = ViTModel(vit_config, add_pooling_layer=False)

# Load SSL weights
ssl_weights = torch.load('models/ssl_vit_backbone.pth', map_location='cpu')

# Try loading
missing, unexpected = model.load_state_dict(ssl_weights, strict=False)

print(f'Missing keys: {len(missing)}')
print(f'Unexpected keys: {len(unexpected)}')

if len(missing) == 0 and len(unexpected) == 0:
    print('\n✓ PERFECT MATCH! All keys loaded successfully!')
else:
    print('\n⚠ KEYS MISMATCH')
    if missing:
        print(f'  Missing ({len(missing)}): {missing[:5]}')
    if unexpected:
        print(f'  Unexpected ({len(unexpected)}): {unexpected[:5]}')
