"""
SegNet architecture for semantic segmentation
SegNet uses pooling indices for upsampling instead of skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

from .encoder import FeatureExtractor

logger = logging.getLogger(__name__)


class SegNetEncoder(nn.Module):
    """
    SegNet Encoder: VGG-like architecture with max pooling indices saved.
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        """
        Initialize SegNet Encoder.
        
        Args:
            in_channels: Number of input channels
            base_channels: Number of channels in the first layer
        """
        super(SegNetEncoder, self).__init__()
        
        # Encoder Block 1
        self.enc1_1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.enc1_2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Block 2
        self.enc2_1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.enc2_2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Block 3
        self.enc3_1 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.enc3_2 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)
        self.enc3_3 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Block 4
        self.enc4_1 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1)
        self.enc4_2 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1)
        self.enc4_3 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Block 5 (Bottleneck)
        self.enc5_1 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1)
        self.enc5_2 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1)
        self.enc5_3 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Batch normalization layers
        self.bn1_1 = nn.BatchNorm2d(base_channels)
        self.bn1_2 = nn.BatchNorm2d(base_channels)
        self.bn2_1 = nn.BatchNorm2d(base_channels * 2)
        self.bn2_2 = nn.BatchNorm2d(base_channels * 2)
        self.bn3_1 = nn.BatchNorm2d(base_channels * 4)
        self.bn3_2 = nn.BatchNorm2d(base_channels * 4)
        self.bn3_3 = nn.BatchNorm2d(base_channels * 4)
        self.bn4_1 = nn.BatchNorm2d(base_channels * 8)
        self.bn4_2 = nn.BatchNorm2d(base_channels * 8)
        self.bn4_3 = nn.BatchNorm2d(base_channels * 8)
        self.bn5_1 = nn.BatchNorm2d(base_channels * 8)
        self.bn5_2 = nn.BatchNorm2d(base_channels * 8)
        self.bn5_3 = nn.BatchNorm2d(base_channels * 8)
        
        logger.info(f"Initialized SegNetEncoder: in_channels={in_channels}, base_channels={base_channels}")
    
    def forward(self, x):
        """
        Forward pass through encoder.
        Returns features and pooling indices for each level.
        """
        # Encoder Block 1
        x = F.relu(self.bn1_1(self.enc1_1(x)))
        x = F.relu(self.bn1_2(self.enc1_2(x)))
        x, indices1 = self.pool1(x)
        size1 = x.size()
        
        # Encoder Block 2
        x = F.relu(self.bn2_1(self.enc2_1(x)))
        x = F.relu(self.bn2_2(self.enc2_2(x)))
        x, indices2 = self.pool2(x)
        size2 = x.size()
        
        # Encoder Block 3
        x = F.relu(self.bn3_1(self.enc3_1(x)))
        x = F.relu(self.bn3_2(self.enc3_2(x)))
        x = F.relu(self.bn3_3(self.enc3_3(x)))
        x, indices3 = self.pool3(x)
        size3 = x.size()
        
        # Encoder Block 4
        x = F.relu(self.bn4_1(self.enc4_1(x)))
        x = F.relu(self.bn4_2(self.enc4_2(x)))
        x = F.relu(self.bn4_3(self.enc4_3(x)))
        x, indices4 = self.pool4(x)
        size4 = x.size()
        
        # Encoder Block 5
        x = F.relu(self.bn5_1(self.enc5_1(x)))
        x = F.relu(self.bn5_2(self.enc5_2(x)))
        x = F.relu(self.bn5_3(self.enc5_3(x)))
        x, indices5 = self.pool5(x)
        size5 = x.size()
        
        return x, [indices1, indices2, indices3, indices4, indices5], [size1, size2, size3, size4, size5]


class SegNetDecoder(nn.Module):
    """
    SegNet Decoder: Uses unpooling with saved indices.
    """
    
    def __init__(self, num_classes: int = 10, base_channels: int = 64):
        """
        Initialize SegNet Decoder.
        
        Args:
            num_classes: Number of output classes
            base_channels: Number of channels in the first layer (must match encoder)
        """
        super(SegNetDecoder, self).__init__()
        
        # Decoder Block 5 (from bottleneck)
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec5_3 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1)
        self.dec5_2 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1)
        self.dec5_1 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1)
        
        # Decoder Block 4
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec4_3 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1)
        self.dec4_2 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1)
        self.dec4_1 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1)
        
        # Decoder Block 3
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec3_3 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)
        self.dec3_2 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)
        self.dec3_1 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        
        # Decoder Block 2
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec2_2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)
        self.dec2_1 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        
        # Decoder Block 1
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec1_2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.dec1_1 = nn.Conv2d(base_channels, num_classes, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn5_3 = nn.BatchNorm2d(base_channels * 8)
        self.bn5_2 = nn.BatchNorm2d(base_channels * 8)
        self.bn5_1 = nn.BatchNorm2d(base_channels * 8)
        self.bn4_3 = nn.BatchNorm2d(base_channels * 8)
        self.bn4_2 = nn.BatchNorm2d(base_channels * 8)
        self.bn4_1 = nn.BatchNorm2d(base_channels * 4)
        self.bn3_3 = nn.BatchNorm2d(base_channels * 4)
        self.bn3_2 = nn.BatchNorm2d(base_channels * 4)
        self.bn3_1 = nn.BatchNorm2d(base_channels * 2)
        self.bn2_2 = nn.BatchNorm2d(base_channels * 2)
        self.bn2_1 = nn.BatchNorm2d(base_channels)
        self.bn1_2 = nn.BatchNorm2d(base_channels)
        
        logger.info(f"Initialized SegNetDecoder: num_classes={num_classes}, base_channels={base_channels}")
    
    def forward(self, x, indices_list, sizes_list):
        """
        Forward pass through decoder.
        
        Args:
            x: Input features from encoder bottleneck
            indices_list: List of pooling indices from encoder [indices1, ..., indices5]
            sizes_list: List of feature sizes before pooling [size1, ..., size5]
        """
        # Decoder Block 5
        x = self.unpool5(x, indices_list[4])
        x = F.relu(self.bn5_3(self.dec5_3(x)))
        x = F.relu(self.bn5_2(self.dec5_2(x)))
        x = F.relu(self.bn5_1(self.dec5_1(x)))
        
        # Decoder Block 4
        x = self.unpool4(x, indices_list[3])
        x = F.relu(self.bn4_3(self.dec4_3(x)))
        x = F.relu(self.bn4_2(self.dec4_2(x)))
        x = F.relu(self.bn4_1(self.dec4_1(x)))
        
        # Decoder Block 3
        x = self.unpool3(x, indices_list[2])
        x = F.relu(self.bn3_3(self.dec3_3(x)))
        x = F.relu(self.bn3_2(self.dec3_2(x)))
        x = F.relu(self.bn3_1(self.dec3_1(x)))
        
        # Decoder Block 2
        x = self.unpool2(x, indices_list[1])
        x = F.relu(self.bn2_2(self.dec2_2(x)))
        x = F.relu(self.bn2_1(self.dec2_1(x)))
        
        # Decoder Block 1
        x = self.unpool1(x, indices_list[0])
        x = F.relu(self.bn1_2(self.dec1_2(x)))
        x = self.dec1_1(x)  # No activation, will apply softmax in loss
        
        return x


class SegNet(nn.Module):
    """
    Standard SegNet architecture for semantic segmentation.
    Uses pooling indices for upsampling instead of skip connections.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64
    ):
        """
        Initialize SegNet.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            base_channels: Number of channels in the first layer
        """
        super(SegNet, self).__init__()
        
        self.encoder = SegNetEncoder(in_channels, base_channels)
        self.decoder = SegNetDecoder(num_classes, base_channels)
        
        logger.info(f"Initialized SegNet: in_channels={in_channels}, num_classes={num_classes}")
    
    def forward(self, x):
        """Forward pass through SegNet."""
        # Encode
        encoded, indices_list, sizes_list = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded, indices_list, sizes_list)
        
        return decoded


class SegNetWithPretrainedEncoder(nn.Module):
    """
    SegNet with pre-trained encoder backbone.
    Uses a pre-trained encoder (ResNet, EfficientNet, etc.) instead of SegNet encoder.
    The decoder uses transposed convolutions since we don't have pooling indices.
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        num_classes: int = 10,
        encoder_pretrained: bool = True,
        encoder_weights_path: Optional[str] = None,
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        """
        Initialize SegNet with pre-trained encoder.
        
        Args:
            encoder_name: Name of the encoder backbone
            num_classes: Number of output classes
            encoder_pretrained: If True, use ImageNet pretrained weights
            encoder_weights_path: Path to custom encoder weights (from Stage 1)
            dropout: Dropout rate
            freeze_encoder: If True, freeze encoder weights
        """
        super(SegNetWithPretrainedEncoder, self).__init__()
        
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        
        # Create encoder with feature extraction
        self.encoder = FeatureExtractor(encoder_name, encoder_pretrained)
        feature_dims = self.encoder.get_feature_dims()
        
        # Load custom encoder weights if provided
        if encoder_weights_path:
            self.load_encoder_weights(encoder_weights_path)
        
        # Freeze encoder if requested
        if freeze_encoder:
            self.freeze_encoder()
        
        # Decoder using transposed convolutions
        # feature_dims = [64, 256, 512, 1024, 2048] for ResNet50
        # We'll use a simpler decoder since we don't have pooling indices
        self.decoder = SegNetTransposedDecoder(
            bottleneck_channels=feature_dims[4],
            num_classes=num_classes,
            feature_dims=feature_dims,
            dropout=dropout
        )
        
        logger.info(f"Initialized SegNetWithPretrainedEncoder: {encoder_name}, classes={num_classes}")
        logger.info(f"Feature dimensions: {feature_dims}")
        logger.info(f"Encoder frozen: {freeze_encoder}")
    
    def forward(self, x):
        """Forward pass through SegNet with pre-trained encoder."""
        # Get features from encoder at different scales
        features = self.encoder(x)
        # features = [f0, f1, f2, f3, f4] at resolutions [1/2, 1/4, 1/8, 1/16, 1/32]
        
        # Decode using bottleneck feature
        decoded = self.decoder(features[4], features)
        
        return decoded
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen")
    
    def load_encoder_weights(self, weights_path: str):
        """
        Load encoder weights from a checkpoint.
        
        Args:
            weights_path: Path to the checkpoint file
        """
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
        elif 'model_state_dict' in checkpoint:
            # Extract encoder weights from full model
            state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key.startswith('encoder.'):
                    new_key = key.replace('encoder.', '')
                    state_dict[new_key] = value
        else:
            state_dict = checkpoint
        
        # Load weights
        missing_keys, unexpected_keys = self.encoder.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys when loading encoder: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading encoder: {unexpected_keys}")
        
        logger.info(f"Loaded encoder weights from {weights_path}")
    
    def get_encoder_parameters(self):
        """Get encoder parameters for separate optimization."""
        return self.encoder.parameters()
    
    def get_decoder_parameters(self):
        """Get decoder parameters for separate optimization."""
        decoder_params = []
        for name, param in self.named_parameters():
            if not name.startswith('encoder.'):
                decoder_params.append(param)
        return decoder_params


class SegNetTransposedDecoder(nn.Module):
    """
    SegNet-style decoder using transposed convolutions.
    Used when encoder doesn't provide pooling indices.
    """
    
    def __init__(
        self,
        bottleneck_channels: int,
        num_classes: int,
        feature_dims: list,
        dropout: float = 0.1
    ):
        """
        Initialize transposed decoder.
        
        Args:
            bottleneck_channels: Number of channels in bottleneck (e.g., 2048 for ResNet50)
            num_classes: Number of output classes
            feature_dims: List of feature dimensions from encoder [f0, f1, f2, f3, f4]
            dropout: Dropout rate
        """
        super(SegNetTransposedDecoder, self).__init__()
        
        # Decoder blocks with transposed convolutions
        # Start from bottleneck (f4) and upsample
        self.dec5 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, feature_dims[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[3]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(feature_dims[3], feature_dims[3], kernel_size=2, stride=2)
        )
        
        self.dec4 = nn.Sequential(
            nn.Conv2d(feature_dims[3], feature_dims[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(feature_dims[2], feature_dims[2], kernel_size=2, stride=2)
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(feature_dims[2], feature_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(feature_dims[1], feature_dims[1], kernel_size=2, stride=2)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(feature_dims[1], feature_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(feature_dims[0], feature_dims[0], kernel_size=2, stride=2)
        )
        
        # Final classification layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(feature_dims[0], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        logger.info(f"Initialized SegNetTransposedDecoder: bottleneck={bottleneck_channels}, classes={num_classes}")
    
    def forward(self, bottleneck, features_list):
        """
        Forward pass through decoder.
        
        Args:
            bottleneck: Bottleneck features (f4)
            features_list: List of all encoder features [f0, f1, f2, f3, f4]
        """
        # Start from bottleneck
        x = self.dec5(bottleneck)
        
        # Continue upsampling
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        
        # Final classification
        x = self.final_conv(x)
        
        return x

