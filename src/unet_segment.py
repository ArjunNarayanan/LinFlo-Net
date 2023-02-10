from src.unet import *


class UnetSegment(nn.Module):
    def __init__(self, input_size, input_channels, unet_first_channels, downarm_channels, uparm_channels,
                 output_channels):
        super().__init__()
        self.encoder = Unet(input_size, input_channels, unet_first_channels, downarm_channels, uparm_channels)
        decoder_input_channels = uparm_channels[-1]
        self.decoder = nn.Conv3d(decoder_input_channels, output_channels, kernel_size=1)

    @classmethod
    def from_dict(cls, definition):
        input_size = definition["input_size"]
        input_channels = definition["input_channels"]
        unet_first_channels = definition["unet_first_channels"]
        downarm_channels = definition["downarm_channels"]
        uparm_channels = definition["uparm_channels"]
        output_channels = definition["output_channels"]
        return cls(input_size, input_channels, unet_first_channels, downarm_channels, uparm_channels, output_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
