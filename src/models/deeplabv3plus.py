import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large


# ---------------------------
# ASPP (giữ nguyên tên)
# ---------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super().__init__()

        self.blocks = nn.ModuleList()

        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )

        for r in rates:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3,
                              padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        h, w = x.shape[2:]

        res = [blk(x) for blk in self.blocks]

        img = self.image_pool(x)
        img = F.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)
        res.append(img)

        out = torch.cat(res, dim=1)
        return self.project(out)


# ---------------------------
# DecoderBlock (giữ nguyên tên)
# ---------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


# ---------------------------
# DeepLabV3Plus (giữ nguyên tên)
# ---------------------------
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()

        backbone = mobilenet_v3_large(pretrained=pretrained)

        self.low_level = backbone.features[0:3]     # stride /4
        self.high_level = backbone.features[3:]     # stride /16

        self.low_level_channels = 24
        self.high_level_channels = 960

        self.low_reduce = nn.Sequential(
            nn.Conv2d(self.low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(self.high_level_channels, 256)
        self.decoder = DecoderBlock(256, 48, 256)

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[2:]

        x_low = self.low_level(x)
        x_high = self.high_level(x_low)

        x_aspp = self.aspp(x_high)
        x_low_r = self.low_reduce(x_low)

        x_dec = self.decoder(x_aspp, x_low_r)

        out = F.interpolate(x_dec, size=(h, w), mode="bilinear", align_corners=False)

        return self.classifier(out)
