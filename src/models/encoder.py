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
    # ----------------- ResNet -----------------
    if 'resnet18' in model_name:
        encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()

    elif 'resnet50' in model_name:
        encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()

    elif 'resnet34' in model_name:
        encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()

    # ----------------- EfficientNet -----------------
    elif 'efficientnet_b0' in model_name:
        encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_dim = encoder.classifier[1].in_features
        encoder.classifier = nn.Identity()

    elif 'efficientnet_b1' in model_name:
        encoder = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_dim = encoder.classifier[1].in_features
        encoder.classifier = nn.Identity()

    # ----------------- MobileNetV2 -----------------
    elif 'mobilenet_v2' in model_name:
        encoder = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_dim = encoder.classifier[1].in_features
        encoder.classifier = nn.Identity()

    # ----------------- MobileNetV3 -----------------
    elif 'mobilenet_v3' in model_name:
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = mobilenet_v3_large(weights=weights)

        # tìm lớp Linear cuối cùng trong classifier
        linear_layer = next(l for l in reversed(encoder.classifier) if isinstance(l, nn.Linear))
        feature_dim = linear_layer.in_features

        # bỏ head classification
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
        super().__init__()

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
        features = self.encoder(x)

        if features.dim() == 4:
            features = self.global_pool(features)
            features = features.flatten(1)
        elif features.dim() == 2:
            pass
        else:
            raise ValueError(f"Unexpected encoder output shape {features.shape} from {self.encoder_name}")

        logits = self.classifier(features)
        return logits

    def get_encoder(self):
        return self.encoder

    def load_encoder_weights(self, weights_path: str):
        checkpoint = torch.load(weights_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                encoder_state_dict[new_key] = value

        self.encoder.load_state_dict(encoder_state_dict, strict=False)
        logger.info(f"Loaded encoder weights from {weights_path}")

    def save_encoder_weights(self, save_path: str):
        encoder_state_dict = self.encoder.state_dict()
        torch.save({'encoder_state_dict': encoder_state_dict}, save_path)
        logger.info(f"Saved encoder weights to {save_path}")


class FeatureExtractor(nn.Module):
    """
    Feature extractor that returns intermediate features from encoder.
    Used for U-Net skip connections.
    """

    def __init__(self, encoder_name: str = 'resnet50', pretrained: bool = True):
        super().__init__()

        self.encoder_name = encoder_name

        if 'resnet' in encoder_name:
            if 'resnet18' in encoder_name:
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            elif 'resnet34' in encoder_name:
                resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            elif 'resnet50' in encoder_name:
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            else:
                raise ValueError(f"Unsupported ResNet variant: {encoder_name}")

            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4

            if 'resnet18' in encoder_name or 'resnet34' in encoder_name:
                self.feature_dims = [64, 64, 128, 256, 512]
            else:
                self.feature_dims = [64, 256, 512, 1024, 2048]

        else:
            raise NotImplementedError(f"Feature extraction not implemented for {encoder_name}")

        logger.info(f"Initialized FeatureExtractor: {encoder_name}")
        logger.info(f"Feature dimensions: {self.feature_dims}")

    def forward(self, x):
        features = []
        x = self.layer0(x)
        features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features

    def get_feature_dims(self):
        return self.feature_dims
