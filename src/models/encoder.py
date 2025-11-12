"""
Encoder models for Stage 1 pre-training
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def create_encoder(
    model_name: str,
    pretrained: bool = True,
    num_classes: Optional[int] = None
) -> Tuple[nn.Module, int]:
    """
    Create an encoder backbone from torchvision models.
    
    Args:
        model_name: Name of the model ('resnet18', 'resnet50', 'efficientnet_b0', etc.)
        pretrained: If True, load ImageNet pretrained weights
        num_classes: Number of output classes (if None, returns feature extractor only)
        
    Returns:
        Tuple of (encoder, feature_dim) where feature_dim is the output dimension
    """
    if 'resnet18' in model_name:
        encoder = models.resnet18(pretrained=pretrained)
        feature_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()  # Remove classification head
        
    elif 'resnet50' in model_name:
        encoder = models.resnet50(pretrained=pretrained)
        feature_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()
        
    elif 'resnet34' in model_name:
        encoder = models.resnet34(pretrained=pretrained)
        feature_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()
        
    elif 'efficientnet_b0' in model_name:
        encoder = models.efficientnet_b0(pretrained=pretrained)
        feature_dim = encoder.classifier[1].in_features
        encoder.classifier = nn.Identity()
        
    elif 'efficientnet_b1' in model_name:
        encoder = models.efficientnet_b1(pretrained=pretrained)
        feature_dim = encoder.classifier[1].in_features
        encoder.classifier = nn.Identity()
        
    elif 'mobilenet_v2' in model_name:
        encoder = models.mobilenet_v2(pretrained=pretrained)
        feature_dim = encoder.classifier[1].in_features
        encoder.classifier = nn.Identity()
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    logger.info(f"Created encoder: {model_name}, pretrained={pretrained}, feature_dim={feature_dim}")
    
    return encoder, feature_dim


class EncoderClassifier(nn.Module):
    """
    Classification model with encoder backbone.
    Used for Stage 1 pre-training with EuroSAT.
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Initialize encoder classifier.
        
        Args:
            encoder_name: Name of the encoder backbone
            num_classes: Number of output classes
            pretrained: If True, use ImageNet pretrained weights
            dropout: Dropout rate before final classification layer
        """
        super(EncoderClassifier, self).__init__()
        
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        
        # Create encoder
        self.encoder, feature_dim = create_encoder(encoder_name, pretrained)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        logger.info(f"Initialized EncoderClassifier: {encoder_name}, classes={num_classes}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Extract features
        features = self.encoder(x)
        
        # Handle encoders that already include global pooling internally
        if features.dim() == 4:
            features = self.global_pool(features)
            features = features.flatten(1)
        elif features.dim() == 2:
            # Already pooled/flattened by the backbone
            pass
        else:
            raise ValueError(
                f"Unexpected encoder output shape {features.shape} from {self.encoder_name}"
            )
        
        # Classification
        logits = self.classifier(features)  # (batch_size, num_classes)
        
        return logits
    
    def get_encoder(self):
        """Get the encoder backbone (without classification head)."""
        return self.encoder
    
    def load_encoder_weights(self, weights_path: str):
        """
        Load encoder weights from a checkpoint.
        
        Args:
            weights_path: Path to the checkpoint file
        """
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Extract encoder weights
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                encoder_state_dict[new_key] = value
        
        # Load weights
        self.encoder.load_state_dict(encoder_state_dict, strict=False)
        logger.info(f"Loaded encoder weights from {weights_path}")
    
    def save_encoder_weights(self, save_path: str):
        """
        Save encoder weights to a file.
        
        Args:
            save_path: Path to save the encoder weights
        """
        encoder_state_dict = self.encoder.state_dict()
        torch.save({'encoder_state_dict': encoder_state_dict}, save_path)
        logger.info(f"Saved encoder weights to {save_path}")


class FeatureExtractor(nn.Module):
    """
    Feature extractor that returns intermediate features from encoder.
    Used for U-Net skip connections.
    """
    
    def __init__(self, encoder_name: str = 'resnet50', pretrained: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            encoder_name: Name of the encoder backbone
            pretrained: If True, use ImageNet pretrained weights
        """
        super(FeatureExtractor, self).__init__()
        
        self.encoder_name = encoder_name
        
        if 'resnet' in encoder_name:
            # Create ResNet and extract layers
            if 'resnet18' in encoder_name:
                resnet = models.resnet18(pretrained=pretrained)
            elif 'resnet34' in encoder_name:
                resnet = models.resnet34(pretrained=pretrained)
            elif 'resnet50' in encoder_name:
                resnet = models.resnet50(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported ResNet variant: {encoder_name}")
            
            # Extract layers for skip connections
            self.layer0 = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu
            )
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            
            # Feature dimensions for ResNet
            if 'resnet18' in encoder_name or 'resnet34' in encoder_name:
                self.feature_dims = [64, 64, 128, 256, 512]
            else:  # ResNet50+
                self.feature_dims = [64, 256, 512, 1024, 2048]
        
        else:
            raise NotImplementedError(f"Feature extraction not implemented for {encoder_name}")
        
        logger.info(f"Initialized FeatureExtractor: {encoder_name}")
        logger.info(f"Feature dimensions: {self.feature_dims}")
    
    def forward(self, x):
        """
        Forward pass with skip connections.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            List of feature maps at different scales
        """
        features = []
        
        # Initial convolution
        x = self.layer0(x)
        features.append(x)  # 1/2 resolution
        
        # MaxPool and encoder blocks
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)  # 1/4 resolution
        
        x = self.layer2(x)
        features.append(x)  # 1/8 resolution
        
        x = self.layer3(x)
        features.append(x)  # 1/16 resolution
        
        x = self.layer4(x)
        features.append(x)  # 1/32 resolution
        
        return features
    
    def get_feature_dims(self):
        """Get feature dimensions for each scale."""
        return self.feature_dims

