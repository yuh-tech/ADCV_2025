"""
DeepLabV3+ architecture for semantic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

from .encoder import FeatureExtractor  # Pre-trained backbone encoder

logger = logging.getLogger(__name__)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module used in DeepLabV3+."""

    def __init__(self, in_channels: int, out_channels: int, atrous_rates: list = [6, 12, 18]):
        super(ASPP, self).__init__()
        self.branches = nn.ModuleList()

        # 1x1 convolution
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # 3x3 convolutions with different dilation rates
        for rate in atrous_rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # Image pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        size = x.shape[2:]
        res = [branch(x) for branch in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode='bilinear', align_corners=False)
        res.append(gp)
        res = torch.cat(res, dim=1)
        return self.project(res)


class DecoderBlock(nn.Module):
    """Decoder block for DeepLabV3+."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        # Upsample
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ for semantic segmentation with optional pre-trained encoder.
    """

    def __init__(
        self,
        encoder_name: str = 'resnet50',
        num_classes: int = 21,
        encoder_pretrained: bool = True,
        encoder_weights_path: Optional[str] = None,
        atrous_rates: list = [6, 12, 18],
        low_level_channels: int = 256
    ):
        super(DeepLabV3Plus, self).__init__()

        # Encoder
        self.encoder = FeatureExtractor(encoder_name, encoder_pretrained)
        feature_dims = self.encoder.get_feature_dims()  # e.g., [64, 256, 512, 1024, 2048]

        if encoder_weights_path:
            self.load_encoder_weights(encoder_weights_path)

        # ASPP module
        self.aspp = ASPP(feature_dims[-1], 256, atrous_rates)

        # Decoder
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(feature_dims[1], low_level_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True)
        )
        self.decoder = DecoderBlock(256, low_level_channels, 256)

        # Final classification
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

        logger.info(f"Initialized DeepLabV3Plus: {encoder_name}, classes={num_classes}")
        logger.info(f"Feature dimensions: {feature_dims}")

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        # features = [f0, f1, f2, f3, f4] at decreasing resolutions

        low_level_feat = self.low_level_conv(features[1])
        x = self.aspp(features[-1])
        x = self.decoder(x, low_level_feat)

        # Upsample to input size
        x = F.interpolate(x, size=x.shape[2]*4, mode='bilinear', align_corners=False)  # Adjust factor based on skip resolution
        x = self.classifier(x)
        return x

    def load_encoder_weights(self, weights_path: str):
        """Load encoder weights from checkpoint."""
        checkpoint = torch.load(weights_path, map_location='cpu')
        if 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = {k.replace('encoder.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
        else:
            state_dict = checkpoint
        self.encoder.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded encoder weights from {weights_path}")

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen")

    def get_encoder_parameters(self):
        return self.encoder.parameters()

    def get_decoder_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith('encoder.')]
