import torch
import torch.nn as nn
import torch.nn.functional as functional


class ConvINormConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x


class DownArm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, stride=2)
        self.conv = ConvINormConv(in_channels, out_channels)

    def forward(self, x):
        x = self.down_sample(x)
        x = self.conv(x)
        return x


class UpArm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvINormConv(in_channels, out_channels)

    def forward(self, x):