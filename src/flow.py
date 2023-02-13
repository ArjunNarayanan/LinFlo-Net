from src.unet import *
import src.finite_difference as fd


class Flow(nn.Module):
    def __init__(self,
                 input_size,
                 input_channels,
                 unet_first_layer_channels,
                 downarm_channels,
                 uparm_channels,
                 decoder_hidden_channels,
                 clip_flow):
        assert clip_flow > 0
        super().__init__()

        self.input_size = input_size
        input_size_list = 3 * [input_size]
        self.encoder = Unet(input_size_list, input_channels, unet_first_layer_channels, downarm_channels,
                            uparm_channels)
        decoder_input_channels = uparm_channels[-1]

        self.flow_decoder = ConvINormConv(decoder_input_channels, decoder_hidden_channels)
        self.flow = nn.Conv3d(decoder_hidden_channels, 3, kernel_size=1)
        self.clip_flow = clip_flow

        # Initialize flow weights to small value
        self.flow.weight = nn.Parameter(torch.distributions.normal.Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    @classmethod
    def from_dict(cls, definition):
        input_size = definition["input_size"]
        input_channels = definition["input_channels"]
        unet_first_channels = definition["unet_first_channels"]
        downarm_channels = definition["downarm_channels"]
        uparm_channels = definition["uparm_channels"]
        decoder_hidden_channels = definition["decoder_hidden_channels"]
        clip_flow = definition["clip_flow"]
        return cls(input_size,
                   input_channels,
                   unet_first_channels,
                   downarm_channels,
                   uparm_channels,
                   decoder_hidden_channels,
                   clip_flow)

    def _clip_flow_field(self, flow):
        clip_flow = self.clip_flow
        norm = torch.norm(flow, dim=1, keepdim=True)
        norm = torch.clamp(norm, min=clip_flow)
        flow = clip_flow * (flow / norm)
        return flow

    def get_flow_field(self, image):
        encoding = self.encoder(image)
        encoding = self.flow_decoder(encoding)
        flow = self.flow(encoding)
        flow = self._clip_flow_field(flow)
        return flow


class SegmentFlow(Flow):
    def __init__(self,
                 input_size,
                 input_channels,
                 unet_first_layer_channels,
                 downarm_channels,
                 uparm_channels,
                 decoder_hidden_channels,
                 num_classes,
                 clip_flow):
        super().__init__(input_size,
                         input_channels,
                         unet_first_layer_channels,
                         downarm_channels,
                         uparm_channels,
                         decoder_hidden_channels,
                         clip_flow)

        decoder_input_channels = uparm_channels[-1]
        self.num_classes = num_classes
        self.segmentation_decoder = ConvINormConv(decoder_input_channels, decoder_hidden_channels)
        self.segmentation = nn.Conv3d(decoder_hidden_channels, num_classes, kernel_size=1)

    @classmethod
    def from_dict(cls, definition):
        input_size = definition["input_size"]
        input_channels = definition["input_channels"]
        unet_first_channels = definition["unet_first_channels"]
        downarm_channels = definition["downarm_channels"]
        uparm_channels = definition["uparm_channels"]
        decoder_hidden_channels = definition["decoder_hidden_channels"]
        num_classes = definition["num_classes"]
        clip_flow = definition["clip_flow"]
        return cls(input_size,
                   input_channels,
                   unet_first_channels,
                   downarm_channels,
                   uparm_channels,
                   decoder_hidden_channels,
                   num_classes,
                   clip_flow)

    def get_segmentation_and_flow(self, image):
        encoding = self.encoder(image)

        seg_encoding = self.segmentation_decoder(encoding)
        segmentation = self.segmentation(seg_encoding)

        flow_encoding = self.flow_encoder(encoding)
        flow = self.flow(flow_encoding)
        flow = self._clip_flow_field(flow)

        return segmentation, flow


class FlowDiv:
    def __init__(self, input_size):
        assert input_size > 1
        self.spacing = 1.0 / (input_size - 1)

    def get_flow_div(self, flow_field):
        flow_div = fd.batch_divergence3d(flow_field, self.spacing)
        flow_and_div = torch.cat([flow_field, flow_div], dim=1)
        return flow_and_div


class FlowPredictor(nn.Module):
    def __init__(self, flow, integrator):
        super().__init__()

        self.flow = flow
        self.integrator = integrator

    def forward(self, image, vertices):
        flow = self.flow.get_flow_field(image)
        deformed_vertices = self.integrator.integrate(flow, vertices)
        return deformed_vertices

