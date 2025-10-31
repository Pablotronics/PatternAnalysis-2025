# modules.py
# Project 3: U-Net + CAN (dilated conv) modules
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
    def forward(self, x): return self.net(x)

class CANBlock(nn.Module):
    """
    Context Aggregation Network block:
    a stack of same-channel 3x3 convs with increasing dilation rates to aggregate global context
    without changing spatial resolution.
    """
    def __init__(self, channels, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        layers = []
        for d in dilations:
            layers += [
                nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.01, inplace=True),
            ]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class UNet2D_CAN(nn.Module):
    """
    Standard 2D U-Net with a CANBlock at the bottleneck.
    """
    def __init__(self, n_classes=2, base=64, can_dilations=(1,2,4,8,16,32)):
        super().__init__()
        self.inc   = DoubleConv(1, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base,   base*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))

        self.can   = CANBlock(base*8, dilations=can_dilations)

        self.up1   = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.conv1 = DoubleConv(base*8, base*4)
        self.up2   = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv2 = DoubleConv(base*4, base*2)
        self.up3   = nn.ConvTranspose2d(base*2, base,   2, stride=2)
        self.conv3 = DoubleConv(base*2, base)
        self.outc  = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)       # (B, base, H, W)
        x2 = self.down1(x1)    # (B, 2b, H/2, W/2)
        x3 = self.down2(x2)    # (B, 4b, H/4, W/4)
        x4 = self.down3(x3)    # (B, 8b, H/8, W/8)

        x4 = self.can(x4)      # CAN context

        x  = self.up1(x4)      # (B, 4b, H/4, W/4)
        x  = torch.cat([x, x3], dim=1)
        x  = self.conv1(x)

        x  = self.up2(x)       # (B, 2b, H/2, W/2)
        x  = torch.cat([x, x2], dim=1)
        x  = self.conv2(x)

        x  = self.up3(x)       # (B, b, H, W)
        x  = torch.cat([x, x1], dim=1)
        x  = self.conv3(x)

        return self.outc(x)
