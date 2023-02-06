import torch
import torch.nn as nn


class ConvINormConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        assert x.ndim == 5
        assert x.shape[1] == self.in_channels

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
        self.in_channels = in_channels
        self.down_sample = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, stride=2)
        self.conv = ConvINormConv(in_channels, out_channels)

    def forward(self, x):
        assert x.ndim == 5
        assert x.shape[1] == self.in_channels

        x = self.down_sample(x)
        x = self.conv(x)
        return x


class UpArm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.up_sample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv = ConvINormConv(in_channels, out_channels)

    def forward(self, x1, x2):
        assert x1.ndim == 5 and x2.ndim == 5
        C1 = x1.shape[1]
        C2 = x2.shape[1]
        assert C1 + C2 == self.in_channels

        x1 = self.up_sample(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class MultilayerCNN(nn.Module):
    def __init__(self, input_shape, input_channels, first_layer_channels, downarm_channels):
        super().__init__()

        self.input_shape = torch.Size(input_shape)
        self.input_channels = input_channels
        self.first_layer_channels = first_layer_channels
        self.downarm_channels = downarm_channels

        self.input_layer = ConvINormConv(input_channels, first_layer_channels)

        in_channels = first_layer_channels
        self.downarm = nn.ModuleList()
        for nc in downarm_channels:
            self.downarm.append(DownArm(in_channels, nc))
            in_channels = nc

    def forward(self, x):
        assert x.shape[-3:] == self.input_shape

        x = self.input_layer(x)
        for layer in self.downarm:
            x = layer(x)

        return x


class ConvMaxpool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU()
        self.pool = nn.MaxPool3d((2, 2, 2))

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, input_shape, input_channels, downarm_channels):
        super().__init__()

        self.input_shape = torch.Size(input_shape)
        self.input_channels = input_channels
        self.downarm_channels = downarm_channels

        in_channels = input_channels
        convs = []
        for nc in downarm_channels:
            convs.append(ConvMaxpool(in_channels, nc))
            in_channels = nc

        self.downarm = nn.Sequential(*convs)

    def forward(self, x):
        assert x.shape[-3:] == self.input_shape

        x = self.downarm(x)
        return x
