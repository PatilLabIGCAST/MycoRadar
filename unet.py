import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_c, out_c, dropout=0.2):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout2d(p=dropout),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout2d(p=dropout)
    )

class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, out_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.W_x = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.psi = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        g1 = self.W_g(x)
        x1 = self.W_x(g)
        psi = self.sigmoid(self.psi(F.relu(g1 + x1)))
        return x * psi

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dropout=0.2):
        super(UNet, self).__init__()

        self.enc1 = conv_block(in_channels, 32, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(64, 128, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(128, 256, dropout)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(256, 512, dropout)

        self.attn4 = AttentionGate(256, 512, 256)
        self.attn3 = AttentionGate(128, 256, 128)
        self.attn2 = AttentionGate(64, 128, 64)
        self.attn1 = AttentionGate(32, 64, 32)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = conv_block(512, 256, dropout)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(256, 128, dropout)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64, dropout)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32, dropout)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.bottleneck(self.pool4(x4))

        d4 = self.up4(x5)
        d4 = F.interpolate(d4, size=x4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.attn4(d4, x4)
        d4 = self.dec4(torch.cat([d4, x4], dim=1))

        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.attn3(d3, x3)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.attn2(d2, x2)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.attn1(d1, x1)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        return self.final_conv(d1)
