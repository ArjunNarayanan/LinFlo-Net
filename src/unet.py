from src.unet_components import *


class Unet(nn.Module):
    def __init__(self, input_size, input_channels, first_layer_channels, downarm_channels, uparm_channels):
        super().__init__()
        assert len(downarm_channels) == len(uparm_channels)
        assert len(downarm_channels) > 1
        assert len(input_size) == 3

        self.input_size = torch.Size(input_size)
        self.input_channels = input_channels
        self.first_layer_channels = first_layer_channels
        self.downarm_channels = downarm_channels
        self.uparm_channels = uparm_channels

        self.input_layer = ConvINormConv(input_channels, first_layer_channels)

        in_channels = first_layer_channels
        self.downarm = nn.ModuleList()
        for nc in downarm_channels:
            self.downarm.append(DownArm(in_channels, nc))
            in_channels = nc

        self.uparm = nn.ModuleList()
        reverse_downarm_channels = downarm_channels[-1::-1]
        prev_nc = reverse_downarm_channels[0]
        for (idx, down_nc) in enumerate(reverse_downarm_channels[1:]):
            inc = down_nc + prev_nc
            outc = uparm_channels[idx]
            self.uparm.append(UpArm(inc, outc))
            prev_nc = outc

        inc = first_layer_channels + prev_nc
        outc = uparm_channels[-1]
        self.uparm.append(UpArm(inc, outc))

    def forward_downarm(self, x):
        # assert x.shape[-3:] == self.input_size

        x = self.input_layer(x)
        downarm_encodings = [x]
        for layer in self.downarm:
            x = layer(x)
            downarm_encodings.append(x)
        return downarm_encodings

    def forward_uparm(self, downarm_encodings):
        prev_x = downarm_encodings.pop()

        assert len(downarm_encodings) == len(self.uparm)
        for layer in self.uparm:
            x = downarm_encodings.pop()
            x = layer(prev_x, x)
            prev_x = x

        return x

    def forward(self, x):
        downarm_encodings = self.forward_downarm(x)
        x = self.forward_uparm(downarm_encodings)
        return x
