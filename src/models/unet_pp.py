"""
U-Net++ (Nested U-Net) architecture for semantic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

from .encoder import FeatureExtractor

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Double convolution block used in U-Net++."""
   
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


class UNetPlusPlus(nn.Module):
    """
    U-Net++ (Nested U-Net) architecture for semantic segmentation.
    Features dense skip connections and optional deep supervision.
    """
   
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        dropout: float = 0.1,
        deep_supervision: bool = False
    ):
        """
        Initialize U-Net++.
       
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            base_channels: Number of channels in the first layer
            dropout: Dropout rate
            deep_supervision: If True, output from multiple decoder levels
        """
        super(UNetPlusPlus, self).__init__()
        
        self.deep_supervision = deep_supervision
        
        # Channel dimensions for each level
        nb_filter = [base_channels, base_channels*2, base_channels*4, 
                     base_channels*8, base_channels*16]
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder path (X^0_i)
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0], dropout)
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1], dropout)
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2], dropout)
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3], dropout)
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4], dropout)
        
        # Nested dense skip pathways (X^i_j where i+j <= 4)
        # Level 1
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0], dropout)
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1], dropout)
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2], dropout)
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3], dropout)
        
        # Level 2
        self.conv0_2 = ConvBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0], dropout)
        self.conv1_2 = ConvBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1], dropout)
        self.conv2_2 = ConvBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2], dropout)
        
        # Level 3
        self.conv0_3 = ConvBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0], dropout)
        self.conv1_3 = ConvBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1], dropout)
        
        # Level 4
        self.conv0_4 = ConvBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0], dropout)
        
        # Upsampling layers
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(nb_filter[i+1], nb_filter[i+1], kernel_size=2, stride=2)
            for i in range(4)
        ])
        
        # Output layers
        if self.deep_supervision:
            # Multiple outputs for deep supervision
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            # Single output
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        
        logger.info(f"Initialized U-Net++: in_channels={in_channels}, num_classes={num_classes}")
        logger.info(f"Deep supervision: {deep_supervision}")
   
    def forward(self, x):
        # Encoder path with nested dense skip connections
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._upsample(self.up[0](x1_0), x0_0)], 1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._upsample(self.up[1](x2_0), x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._upsample(self.up[0](x1_1), x0_0)], 1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._upsample(self.up[2](x3_0), x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._upsample(self.up[1](x2_1), x1_0)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._upsample(self.up[0](x1_2), x0_0)], 1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._upsample(self.up[3](x4_0), x3_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._upsample(self.up[2](x3_1), x2_0)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._upsample(self.up[1](x2_2), x1_0)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._upsample(self.up[0](x1_3), x0_0)], 1))
        
        # Output
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output
    
    def _upsample(self, x, target):
        """Upsample x to match target size."""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)
        return x


class UNetPlusPlusWithPretrainedEncoder(nn.Module):
    """
    U-Net++ with pre-trained encoder backbone.
    Uses nested dense skip connections with pretrained features.
    """
   
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        num_classes: int = 10,
        encoder_pretrained: bool = True,
        encoder_weights_path: Optional[str] = None,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        deep_supervision: bool = False
    ):
        """
        Initialize U-Net++ with pre-trained encoder.
       
        Args:
            encoder_name: Name of the encoder backbone
            num_classes: Number of output classes
            encoder_pretrained: If True, use ImageNet pretrained weights
            encoder_weights_path: Path to custom encoder weights
            dropout: Dropout rate
            freeze_encoder: If True, freeze encoder weights
            deep_supervision: If True, output from multiple levels
        """
        super(UNetPlusPlusWithPretrainedEncoder, self).__init__()
        
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        # Create encoder
        self.encoder = FeatureExtractor(encoder_name, encoder_pretrained)
        feature_dims = self.encoder.get_feature_dims()
        # feature_dims = [64, 256, 512, 1024, 2048] for ResNet50
        
        # Load custom weights if provided
        if encoder_weights_path:
            self.load_encoder_weights(encoder_weights_path)
        
        if freeze_encoder:
            self.freeze_encoder()
        
        # Decoder channels
        decoder_channels = [256, 128, 64, 32, 16]
        
        # Upsampling layers
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(feature_dims[i+1], decoder_channels[i], kernel_size=2, stride=2)
            for i in range(4)
        ])
        
        # Nested dense convolution blocks
        # X^0_1: combines encoder level 0 with upsampled level 1
        self.conv0_1 = ConvBlock(feature_dims[0] + decoder_channels[0], decoder_channels[0], dropout)
        
        # X^1_1: combines encoder level 1 with upsampled level 2
        self.conv1_1 = ConvBlock(feature_dims[1] + decoder_channels[1], decoder_channels[1], dropout)
        
        # X^0_2: combines [encoder_0, conv0_1, upsampled conv1_1]
        self.conv0_2 = ConvBlock(feature_dims[0] + decoder_channels[0]*2, decoder_channels[0], dropout)
        
        # X^2_1: combines encoder level 2 with upsampled level 3
        self.conv2_1 = ConvBlock(feature_dims[2] + decoder_channels[2], decoder_channels[2], dropout)
        
        # X^1_2: combines [encoder_1, conv1_1, upsampled conv2_1]
        self.conv1_2 = ConvBlock(feature_dims[1] + decoder_channels[1]*2, decoder_channels[1], dropout)
        
        # X^0_3: combines [encoder_0, conv0_1, conv0_2, upsampled conv1_2]
        self.conv0_3 = ConvBlock(feature_dims[0] + decoder_channels[0]*3, decoder_channels[0], dropout)
        
        # X^3_1: combines encoder level 3 with upsampled level 4
        self.conv3_1 = ConvBlock(feature_dims[3] + decoder_channels[3], decoder_channels[3], dropout)
        
        # X^2_2: combines [encoder_2, conv2_1, upsampled conv3_1]
        self.conv2_2 = ConvBlock(feature_dims[2] + decoder_channels[2]*2, decoder_channels[2], dropout)
        
        # X^1_3: combines [encoder_1, conv1_1, conv1_2, upsampled conv2_2]
        self.conv1_3 = ConvBlock(feature_dims[1] + decoder_channels[1]*3, decoder_channels[1], dropout)
        
        # X^0_4: final output combining all level 0 features
        self.conv0_4 = ConvBlock(feature_dims[0] + decoder_channels[0]*4, decoder_channels[0], dropout)
        
        # Final classification layers
        if self.deep_supervision:
            self.final1 = nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1)
        
        logger.info(f"Initialized UNet++WithPretrainedEncoder: {encoder_name}, classes={num_classes}")
        logger.info(f"Feature dimensions: {feature_dims}")
        logger.info(f"Deep supervision: {deep_supervision}")
   
    def forward(self, x):
        # Get encoder features [f0, f1, f2, f3, f4]
        features = self.encoder(x)
        
        # Nested dense skip connections
        # First column (j=1)
        x0_1 = self.conv0_1(torch.cat([features[0], self._upsample(self.up[0](features[1]), features[0])], 1))
        
        # Second column (j=2)
        x1_1 = self.conv1_1(torch.cat([features[1], self._upsample(self.up[1](features[2]), features[1])], 1))
        x0_2 = self.conv0_2(torch.cat([features[0], x0_1, self._upsample(self.up[0](x1_1), features[0])], 1))
        
        # Third column (j=3)
        x2_1 = self.conv2_1(torch.cat([features[2], self._upsample(self.up[2](features[3]), features[2])], 1))
        x1_2 = self.conv1_2(torch.cat([features[1], x1_1, self._upsample(self.up[1](x2_1), features[1])], 1))
        x0_3 = self.conv0_3(torch.cat([features[0], x0_1, x0_2, self._upsample(self.up[0](x1_2), features[0])], 1))
        
        # Fourth column (j=4)
        x3_1 = self.conv3_1(torch.cat([features[3], self._upsample(self.up[3](features[4]), features[3])], 1))
        x2_2 = self.conv2_2(torch.cat([features[2], x2_1, self._upsample(self.up[2](x3_1), features[2])], 1))
        x1_3 = self.conv1_3(torch.cat([features[1], x1_1, x1_2, self._upsample(self.up[1](x2_2), features[1])], 1))
        x0_4 = self.conv0_4(torch.cat([features[0], x0_1, x0_2, x0_3, self._upsample(self.up[0](x1_3), features[0])], 1))
        
        # Output
        if self.deep_supervision:
            # Upsample all outputs to original resolution
            output1 = self._upsample(self.final1(x0_1), x)
            output2 = self._upsample(self.final2(x0_2), x)
            output3 = self._upsample(self.final3(x0_3), x)
            output4 = self._upsample(self.final4(x0_4), x)
            return [output1, output2, output3, output4]
        else:
            output = self._upsample(self.final(x0_4), x)
            return output
    
    def _upsample(self, x, target):
        """Upsample x to match target size."""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)
        return x
    
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
        """Load encoder weights from checkpoint."""
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        if 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key.startswith('encoder.'):
                    new_key = key.replace('encoder.', '')
                    state_dict[new_key] = value
        else:
            state_dict = checkpoint
        
        missing_keys, unexpected_keys = self.encoder.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        logger.info(f"Loaded encoder weights from {weights_path}")

    def get_encoder_parameters(self):
        """Return encoder parameters for optimizer."""
        return self.encoder.parameters()

    def get_decoder_parameters(self):
        """Return all decoder parameters (everything except encoder)."""
        decoder_params = []
        for name, param in self.named_parameters():
            if not name.startswith("encoder."):
                decoder_params.append(param)
        return decoder_params
