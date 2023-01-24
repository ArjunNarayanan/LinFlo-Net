from src.unet import *


class UnetSegment(nn.Module):
    def __init__(self, input_channels, unet_first_channels, downarm_channels, uparm_channels, output_channels):
        super().__init__()
        self.encoder = Unet(input_channels, unet_first_channels, downarm_channels, uparm_channels)
        decoder_input_channels = uparm_channels[-1]
        self.decoder = nn.Conv3d(decoder_input_channels, output_channels, kernel_size=1)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.activation(x)
        return x
