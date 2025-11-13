"""
U-Net architecture for semantic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

from .encoder import FeatureExtractor

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Double convolution block used in U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.1):
        super(UpBlock, self).__init__()
        
        # Upsampling
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Convolution after concatenation
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels, dropout)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size mismatch
        if x.shape != skip.shape:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2])
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture for semantic segmentation.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize U-Net.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            base_channels: Number of channels in the first layer
            dropout: Dropout rate
        """
        super(UNet, self).__init__()
        
        # Encoder (contracting path)
        self.enc1 = ConvBlock(in_channels, base_channels, dropout)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(base_channels, base_channels * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, dropout)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16, dropout)
        
        # Decoder (expansive path)
        self.up1 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8, dropout)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4, dropout)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, dropout)
        self.up4 = UpBlock(base_channels * 2, base_channels, base_channels, dropout)
        
        # Final classification layer
        self.out = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        
        logger.info(f"Initialized U-Net: in_channels={in_channels}, num_classes={num_classes}")
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder
        dec1 = self.up1(bottleneck, enc4)
        dec2 = self.up2(dec1, enc3)
        dec3 = self.up3(dec2, enc2)
        dec4 = self.up4(dec3, enc1)
        
        # Output
        out = self.out(dec4)
        
        return out


class UNetWithPretrainedEncoder(nn.Module):
    """
    U-Net with pre-trained encoder backbone.
    Used for Stage 2 training with BigEarthNet.
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
        Initialize U-Net with pre-trained encoder.
        
        Args:
            encoder_name: Name of the encoder backbone
            num_classes: Number of output classes
            encoder_pretrained: If True, use ImageNet pretrained weights
            encoder_weights_path: Path to custom encoder weights (from Stage 1)
            dropout: Dropout rate
            freeze_encoder: If True, freeze encoder weights
        """
        super(UNetWithPretrainedEncoder, self).__init__()
        
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        
        # Create encoder with skip connections
        self.encoder = FeatureExtractor(encoder_name, encoder_pretrained)
        feature_dims = self.encoder.get_feature_dims()
        
        # Load custom encoder weights if provided
        if encoder_weights_path:
            self.load_encoder_weights(encoder_weights_path)
        
        # Freeze encoder if requested
        if freeze_encoder:
            self.freeze_encoder()
        
        # Decoder
        # feature_dims = [64, 256, 512, 1024, 2048] for ResNet50
        self.up1 = UpBlock(feature_dims[4], feature_dims[3], 512, dropout)
        self.up2 = UpBlock(512, feature_dims[2], 256, dropout)
        self.up3 = UpBlock(256, feature_dims[1], 128, dropout)
        self.up4 = UpBlock(128, feature_dims[0], 64, dropout)
        
        # Final upsampling to original resolution
        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
        logger.info(f"Initialized UNetWithPretrainedEncoder: {encoder_name}, classes={num_classes}")
        logger.info(f"Feature dimensions: {feature_dims}")
        logger.info(f"Encoder frozen: {freeze_encoder}")
    
    def forward(self, x):
        # Get features from encoder at different scales
        features = self.encoder(x)
        # features = [f0, f1, f2, f3, f4] at resolutions [1/2, 1/4, 1/8, 1/16, 1/32]
        
        # Decoder with skip connections
        dec1 = self.up1(features[4], features[3])  # 1/16
        dec2 = self.up2(dec1, features[2])         # 1/8
        dec3 = self.up3(dec2, features[1])         # 1/4
        dec4 = self.up4(dec3, features[0])         # 1/2
        
        # Final upsampling to original resolution
        out = self.final_up(dec4)                  # 1/1
        out = self.final_conv(out)
        
        return out
    
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