from src.unet import *
import src.finite_difference as fd
from src.utilities import batch_occupancy_map_from_vertices, occupancy_map
from src.template import BatchTemplate


class FlowDiv:
    def __init__(self, input_size):
        assert input_size > 1
        self.spacing = 1.0 / (input_size - 1)

    def get_flow_div(self, flow_field):
        flow_div = fd.batch_divergence3d(flow_field, self.spacing)
        flow_and_div = torch.cat([flow_field, flow_div], dim=1)
        return flow_and_div


class ClipFlow:
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def clip_flow_field(self, flow):
        clip_flow = self.clip_value
        if clip_flow > 0.0:
            assert flow.ndim == 5
            assert flow.shape[1] == 3

            norm = torch.norm(flow, dim=1, keepdim=True)
            norm = torch.clamp(norm, min=clip_flow)
            flow = clip_flow * (flow / norm)

        return flow


class SoftClipFlow:
    def __init__(self, clip_value):
        assert clip_value > 0
        self.clip_value = clip_value
        self.activation = nn.Tanh()

    def clip_flow_field(self, flow):
        assert flow.ndim == 5
        assert flow.shape[1] == 3

        flow = self.clip_value * self.activation(flow)
        return flow


class Decoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.conv = ConvINormConv(input_channels, hidden_channels)
        self.decoder = nn.Conv3d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.decoder(x)
        return x


class FlowDecoder(Decoder):
    def __init__(self, input_channels, hidden_channels, clip_value):
        super().__init__(input_channels, hidden_channels, 3)
        self.clip_flow = ClipFlow(clip_value)

        self.decoder.weight = nn.Parameter(torch.distributions.normal.Normal(0, 1e-5).sample(self.decoder.weight.shape))
        self.decoder.bias = nn.Parameter(torch.zeros(self.decoder.bias.shape))

    def forward(self, x):
        flow = super().forward(x)
        flow = self.clip_flow.clip_flow_field(flow)
        return flow


class SoftClipFlowDecoder(Decoder):
    def __init__(self, input_channels, hidden_channels, clip_value):
        super().__init__(input_channels, hidden_channels, 3)
        self.clip_flow = SoftClipFlow(clip_value)

        self.decoder.weight = nn.Parameter(torch.distributions.normal.Normal(0, 1e-5).sample(self.decoder.weight.shape))
        self.decoder.bias = nn.Parameter(torch.zeros(self.decoder.bias.shape))

    def forward(self, x):
        flow = super().forward(x)
        flow = self.clip_flow.clip_flow_field(flow)
        return flow


def get_occupancy(vertices, batch_size, input_shape):
    if isinstance(vertices, list):
        occupancy = batch_occupancy_map_from_vertices(vertices, batch_size, input_shape)
    elif isinstance(vertices, torch.Tensor):
        occupancy = occupancy_map(vertices, input_shape)
        occupancy = occupancy.unsqueeze(0)
    else:
        raise ValueError("Expected vertices to be list of tensors or tensor, got ", type(vertices))

    return occupancy


class EncodeLinearTransformSegmentFlow(nn.Module):
    def __init__(self,
                 input_size,
                 pretrained_encoder,
                 pretrained_linear_transform,
                 encoder,
                 segment_decoder,
                 flow_decoder,
                 integrator):
        super().__init__()

        self.input_size = input_size
        self.pretrained_encoder = pretrained_encoder
        self.pretrained_linear_transform = pretrained_linear_transform
        self.encoder = encoder
        self.segment_decoder = segment_decoder
        self.flow_decoder = flow_decoder
        self.integrator = integrator

        for param in self.pretrained_encoder.parameters():
            param.requires_grad = False

        for param in self.pretrained_linear_transform.parameters():
            param.requires_grad = False

    def get_encoder_input(self, image):
        pre_encoding = self.pretrained_encoder(image)
        pre_encoding = torch.cat([image, pre_encoding], dim=1)
        return pre_encoding

    @staticmethod
    def _get_flow_decoder_input(encoding, occupancy):
        return torch.cat([encoding, occupancy], dim=1)

    @staticmethod
    def get_flow_decoder_input(encoding, vertices, batch_size, input_shape):
        occupancy = get_occupancy(vertices, batch_size, input_shape)
        encoding = torch.cat([encoding, occupancy], dim=1)
        return encoding

    def get_segmentation(self, image):
        assert image.ndim == 5

        pre_encoding = self.get_encoder_input(image)
        encoding = self.encoder(pre_encoding)
        segmentation = self.segment_decoder(encoding)

        return segmentation

    def forward(self, image, vertices):
        assert image.ndim == 5
        batch_size = image.shape[0]

        pre_encoding = self.get_encoder_input(image)
        lt_deformed_vertices = self.pretrained_linear_transform(pre_encoding, vertices)

        encoding = self.encoder(pre_encoding)

        input_shape = image.shape[-1]
        encoding = self.get_flow_decoder_input(encoding, lt_deformed_vertices, batch_size, input_shape)
        flow = self.flow_decoder(encoding)
        deformed_vertices = self.integrator.integrate(flow, lt_deformed_vertices)

        return deformed_vertices

    def predict(self, image, template):
        flow_div = FlowDiv(self.input_size)

        batch_size = image.shape[0]

        batched_template = BatchTemplate.from_single_template(template, batch_size)
        batched_verts = batched_template.batch_vertex_coordinates()

        pre_encoding = self.get_encoder_input(image)
        lt_deformed_vertices = self.pretrained_linear_transform(pre_encoding, batched_verts)

        encoding = self.encoder(pre_encoding)
        predicted_segmentation = self.segment_decoder(encoding)

        encoding = self.get_flow_decoder_input(encoding, lt_deformed_vertices, batch_size, INPUT_SHAPE)
        flow_field = self.flow_decoder(encoding)
        flow_and_div = flow_div.get_flow_div(flow_field)

        deformed_verts, div_integral = self.integrator.integrate_flow_and_div(flow_and_div, lt_deformed_vertices)
        batched_template.update_batched_vertices(deformed_verts, detach=False)

        predictions = {"meshes": batched_template.meshes_list,
                       "segmentation": predicted_segmentation,
                       "divergence_integral": div_integral}

        return predictions


class LinearTransformSegmentFlow(nn.Module):
    def __init__(self,
                 input_size,
                 pretrained_linear_transform,
                 encoder,
                 segment_decoder,
                 flow_decoder,
                 integrator):
        super().__init__()

        self.input_size = input_size
        self.pretrained_linear_transform = pretrained_linear_transform
        self.encoder = encoder
        self.segment_decoder = segment_decoder
        self.flow_decoder = flow_decoder
        self.integrator = integrator

        for param in self.pretrained_linear_transform.parameters():
            param.requires_grad = False

    @staticmethod
    def get_encoder_input(image, occupancy):
        return torch.cat([image, occupancy], dim=1)

    def forward(self, image, vertices):
        assert image.ndim == 5
        batch_size = image.shape[0]

        lt_deformed_vertices = self.pretrained_linear_transform(image, vertices)
        input_shape = image.shape[-1]
        occupancy = get_occupancy(lt_deformed_vertices, batch_size, input_shape)
        encoder_input = self.get_encoder_input(image, occupancy)

        encoding = self.encoder(encoder_input)
        # segmentation = self.segment_decoder(encoding)

        flow = self.flow_decoder(encoding)
        deformed_vertices = self.integrator.integrate(flow, lt_deformed_vertices)

        return deformed_vertices

    def predict(self, image, template):
        flow_div = FlowDiv(self.input_size)
        batch_size = image.shape[0]

        batched_template = BatchTemplate.from_single_template(template, batch_size)
        batched_verts = batched_template.batch_vertex_coordinates()

        lt_deformed_vertices = self.pretrained_linear_transform(image, batched_verts)
        occupancy = batch_occupancy_map_from_vertices(lt_deformed_vertices, batch_size, self.input_size)
        encoder_input = self.get_encoder_input(image, occupancy)

        encoding = self.encoder(encoder_input)
        predicted_segmentation = self.segment_decoder(encoding)
        flow_field = self.flow_decoder(encoding)

        flow_and_div = flow_div.get_flow_div(flow_field)
        deformed_verts, div_integral = self.integrator.integrate_flow_and_div(flow_and_div, lt_deformed_vertices)
        batched_template.update_batched_vertices(deformed_verts, detach=False)

        predictions = {"meshes": batched_template.meshes_list,
                       "segmentation": predicted_segmentation,
                       "divergence_integral": div_integral}

        return predictions
