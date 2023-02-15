from src.unet import *
import src.finite_difference as fd
from src.utilities import batch_occupancy_map_from_vertices, occupancy_map


class Flow(nn.Module):
    def __init__(self,
                 input_shape,
                 input_channels,
                 unet_first_layer_channels,
                 downarm_channels,
                 uparm_channels,
                 decoder_hidden_channels,
                 clip_flow):
        assert clip_flow > 0
        super().__init__()

        self.input_shape = input_shape
        input_size_list = 3 * [input_shape]
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
        input_shape = definition["input_shape"]
        input_channels = definition["input_channels"]
        unet_first_channels = definition["first_layer_channels"]
        downarm_channels = definition["downarm_channels"]
        uparm_channels = definition["uparm_channels"]
        decoder_hidden_channels = definition["decoder_hidden_channels"]
        clip_flow = definition["clip_flow"]
        return cls(input_shape,
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
        assert image.ndim == 5
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


class EncodeLinearTransformFlow(nn.Module):
    def __init__(self, encoder, linear_transform, flow, integrator):
        super().__init__()

        self.encoder = encoder
        self.linear_transform = linear_transform
        self.flow = flow
        self.integrator = integrator

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.linear_transform.parameters():
            param.requires_grad = False

    def get_occupancy(self, vertices, batch_size, input_shape):
        if isinstance(vertices, list):
            occupancy = batch_occupancy_map_from_vertices(vertices, batch_size, input_shape)
        elif isinstance(vertices, torch.Tensor):
            occupancy = occupancy_map(vertices, input_shape)
            occupancy = occupancy.unsqueeze(0)
        else:
            raise ValueError("Expected vertices to be list of tensors or tensor, got ", type(vertices))

        return occupancy

    def forward(self, image, vertices):
        assert image.ndim == 5
        batch_size = image.shape[0]

        encoding = self.encoder(image)
        encoding = torch.cat([image, encoding], dim=1)

        lt_deformed_vertices = self.linear_transform(encoding, vertices)
        occupancy = self.get_occupancy(vertices, batch_size, self.flow.input_shape)

        encoding = torch.cat([encoding, occupancy], dim=1)

        flow = self.flow.get_flow_field(encoding)
        deformed_vertices = self.integrator.integrate(flow, lt_deformed_vertices)

        return deformed_vertices
