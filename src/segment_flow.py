from src.unet import *


class SegmentFlow(nn.Module):
    def __init__(self,
                 input_size,
                 input_channels,
                 unet_first_layer_channels,
                 downarm_channels,
                 uparm_channels,
                 num_classes,
                 clip_flow):
        assert clip_flow > 0
        super().__init__()

        self.encoder = Unet(input_size, input_channels, unet_first_layer_channels, downarm_channels, uparm_channels)
        decoder_input_channels = uparm_channels[-1]
        self.segmentation = nn.Conv3d(decoder_input_channels, num_classes, kernel_size=3, padding=1)
        self.flow = nn.Conv3d(decoder_input_channels, 3, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.clip_flow = clip_flow

    @classmethod
    def from_dict(cls, definition):
        input_size = definition["input_size"]
        input_channels = definition["input_channels"]
        unet_first_channels = definition["unet_first_channels"]
        downarm_channels = definition["downarm_channels"]
        uparm_channels = definition["uparm_channels"]
        num_classes = definition["num_classes"]
        clip_flow = definition["clip_flow"]
        return cls(input_size, input_channels, unet_first_channels, 
            downarm_channels, uparm_channels, num_classes, clip_flow)

    def _clip_flow_field(self, flow):
        clip_flow = self.clip_flow
        norm = torch.norm(flow, dim=1, keepdim=True)
        norm = torch.clamp(norm, min=clip_flow)
        flow = clip_flow * (flow / norm)
        return flow

    def forward(self, image):
        encoding = self.encoder(image)
        segmentation = self.segmentation(encoding)
        flow = self.flow(encoding)
        flow = self._clip_flow_field(flow)
        return segmentation, flow
