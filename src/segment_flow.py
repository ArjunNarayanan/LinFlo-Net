from src.unet import *
import src.finite_difference as fd
from src.utilities import batch_occupancy_map_from_vertices, occupancy_map
from src.linear_transform import linear_transform_image


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
        if clip_value <= 0.0:
            print("WARNING: CLIP VALUE <= 0, FLOW WILL NOT BE CLIPPED")
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
        if clip_value <= 0.0:
            print("WARNING: CLIP VALUE <= 0.0, FLOW WILL NOT BE CLIPPED")
        self.clip_value = clip_value
        self.activation = nn.Tanh()

    def clip_flow_field(self, flow):
        if self.clip_flow > 0.0:
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


class LinearFlowDecoder(nn.Module):
    def __init__(self, input_channels, clip_value):
        super().__init__()
        self.clip_flow = ClipFlow(clip_value)
        self.decoder = nn.Conv3d(input_channels, 3, kernel_size=1)

        self.decoder.weight = nn.Parameter(torch.distributions.normal.Normal(0, 1e-5).sample(self.decoder.weight.shape))
        self.decoder.bias = nn.Parameter(torch.zeros(self.decoder.bias.shape))

    def forward(self, x):
        flow = self.decoder(x)
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
    def __init__(
            self,
            input_size,
            pretrained_encoder,
            pretrained_linear_transform,
            encoder,
            segment_decoder,
            flow_decoder,
            integrator
    ):

        super().__init__()

        self.input_size = input_size
        self.pretrained_encoder = pretrained_encoder
        self.pretrained_linear_transform = pretrained_linear_transform
        self.encoder = encoder
        self.segment_decoder = segment_decoder
        self.flow_decoder = flow_decoder
        self.integrator = integrator
        self.flow_div = FlowDiv(self.input_size)

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

    @staticmethod
    def _fix_input_shape(image):
        if image.ndim == 4:
            image = image.unsqueeze(0)
        assert image.ndim == 5
        return image

    def forward(self, image, vertices, multiplication_factor=1.0):
        image = self._fix_input_shape(image)
        batch_size = image.shape[0]

        with torch.no_grad():
            pre_encoding = self.get_encoder_input(image)
            lt_deformed_vertices = self.pretrained_linear_transform(image, vertices)

        encoding = self.encoder(pre_encoding)
        input_shape = self.input_size
        encoding = self.get_flow_decoder_input(encoding, lt_deformed_vertices, batch_size, input_shape)
        flow = self.flow_decoder(encoding) * multiplication_factor
        deformed_vertices = self.integrator.integrate(flow, lt_deformed_vertices)

        return deformed_vertices

    def predict(self, image, batched_verts, multiplication_factor=1.0):
        image = self._fix_input_shape(image)
        batch_size = image.shape[0]
        with torch.no_grad():
            pre_encoding = self.get_encoder_input(image)
            lt_deformed_vertices = self.pretrained_linear_transform(image, batched_verts)

        encoding = self.encoder(pre_encoding)
        predicted_segmentation = self.segment_decoder(encoding)

        encoding = self.get_flow_decoder_input(encoding, lt_deformed_vertices, batch_size, self.input_size)
        flow_field = self.flow_decoder(encoding) * multiplication_factor
        flow_and_div = self.flow_div.get_flow_div(flow_field)

        deformed_verts, div_integral = self.integrator.integrate_flow_and_div(flow_and_div, lt_deformed_vertices)

        predictions = {"deformed_vertices": deformed_verts,
                       "segmentation": predicted_segmentation,
                       "divergence_integral": div_integral}

        return predictions


class UDFEncodeLinearTransformSegmentFlow(nn.Module):
    def __init__(
            self,
            input_size,
            pretrained_encoder,
            pretrained_linear_transform,
            encoder,
            segment_decoder,
            flow_decoder,
            integrator
    ):
        super().__init__()

        self.input_size = input_size
        self.pretrained_encoder = pretrained_encoder
        self.pretrained_linear_transform = pretrained_linear_transform
        self.encoder = encoder
        self.segment_decoder = segment_decoder
        self.flow_decoder = flow_decoder
        self.integrator = integrator
        self.flow_div = FlowDiv(self.input_size)

        for param in self.pretrained_encoder.parameters():
            param.requires_grad = False

        for param in self.pretrained_linear_transform.parameters():
            param.requires_grad = False

    def get_encoder_input(self, image):
        pre_encoding = self.pretrained_encoder(image)
        pre_encoding = torch.cat([image, pre_encoding], dim=1)
        return pre_encoding

    @staticmethod
    def _fix_input_shape(image):
        if image.ndim == 4:
            image = image.unsqueeze(0)
        assert image.ndim == 5
        return image

    def predict(self, image, vertices, distance_map):
        image = self._fix_input_shape(image)

        with torch.no_grad():
            pre_encoding = self.pretrained_encoder(image)
            lt_parameters = self.pretrained_linear_transform.linear_transform_parameters_from_image(image)

        scale, translate, rotate = lt_parameters
        transformed_distance_map = batch_transform_distance_map(
            distance_map,
            scale,
            translate,
            rotate
        )
        lt_deformed_vertices = self.pretrained_linear_transform.transform_from_parameters(
            vertices,
            scale,
            translate,
            rotate
        )

        pre_encoding = torch.cat([pre_encoding, transformed_distance_map], dim=1)
        encoding = self.encoder(pre_encoding)
        predicted_segmentation = self.segment_decoder(encoding)

        # flow_encoding = torch.cat([encoding, transformed_distance_map], dim=1)
        flow_field = self.flow_decoder(encoding)
        flow_and_div = self.flow_div.get_flow_div(flow_field)
        deformed_verts, div_integral = self.integrator.integrate_flow_and_div(flow_and_div, lt_deformed_vertices)

        predictions = {"deformed_vertices": deformed_verts,
                       "segmentation": predicted_segmentation,
                       "divergence_integral": div_integral}

        return predictions

    def forward(self, image, vertices, distance_map):
        image = self._fix_input_shape(image)

        with torch.no_grad():
            pre_encoding = self.pretrained_encoder(image)
            lt_parameters = self.pretrained_linear_transform.linear_transform_parameters_from_image(image)

        scale, translate, rotate = lt_parameters
        transformed_distance_map = batch_transform_distance_map(
            distance_map,
            scale,
            translate,
            rotate
        )
        lt_deformed_vertices = self.pretrained_linear_transform.transform_from_parameters(
            vertices,
            scale,
            translate,
            rotate
        )

        pre_encoding = torch.cat([pre_encoding, transformed_distance_map], dim=1)
        encoding = self.encoder(pre_encoding)

        # flow_encoding = torch.cat([encoding, transformed_distance_map], dim=1)
        flow_field = self.flow_decoder(encoding)
        deformed_verts = self.integrator.integrate(flow_field, lt_deformed_vertices)

        return deformed_verts


class LinearTransformSegmentFlow(nn.Module):
    def __init__(
            self,
            input_size,
            pretrained_linear_transform,
            encoder,
            segment_decoder,
            flow_decoder,
            integrator
    ):
        super().__init__()

        self.input_size = input_size
        self.pretrained_linear_transform = pretrained_linear_transform
        self.encoder = encoder
        self.segment_decoder = segment_decoder
        self.flow_decoder = flow_decoder
        self.integrator = integrator
        self.flow_div = FlowDiv(self.input_size)

        for param in self.pretrained_linear_transform.parameters():
            param.requires_grad = False

    @staticmethod
    def get_encoder_input(image, occupancy):
        return torch.cat([image, occupancy], dim=1)

    def forward(self, image, vertices):
        assert image.ndim == 5
        batch_size = image.shape[0]

        with torch.no_grad():
            lt_deformed_vertices = self.pretrained_linear_transform(image, vertices)

        occupancy = get_occupancy(lt_deformed_vertices, batch_size, self.input_size)
        encoder_input = self.get_encoder_input(image, occupancy)

        encoding = self.encoder(encoder_input)

        flow = self.flow_decoder(encoding)
        deformed_vertices = self.integrator.integrate(flow, lt_deformed_vertices)

        return deformed_vertices

    def predict(self, image, batched_verts):
        batch_size = image.shape[0]

        with torch.no_grad():
            lt_deformed_vertices = self.pretrained_linear_transform(image, batched_verts)

        occupancy = batch_occupancy_map_from_vertices(lt_deformed_vertices, batch_size, self.input_size)
        encoder_input = self.get_encoder_input(image, occupancy)

        encoding = self.encoder(encoder_input)
        predicted_segmentation = self.segment_decoder(encoding)
        flow_field = self.flow_decoder(encoding)

        flow_and_div = self.flow_div.get_flow_div(flow_field)
        deformed_verts, div_integral = self.integrator.integrate_flow_and_div(flow_and_div, lt_deformed_vertices)

        predictions = {"deformed_vertices": deformed_verts,
                       "segmentation": predicted_segmentation,
                       "divergence_integral": div_integral}

        return predictions


def linear_transform_distance_map(distance_map, scale, translate, rotate):
    assert scale.ndim == translate.ndim == rotate.ndim == 1
    assert scale.shape[0] == translate.shape[0] == rotate.shape[0] == 3
    assert distance_map.ndim == 4

    transformed_map = linear_transform_image(
        distance_map,
        scale,
        translate,
        rotate
    )
    return transformed_map


def batch_transform_distance_map(distance_map, scale, translate, rotate):
    assert scale.ndim == translate.ndim == rotate.ndim == 2
    assert scale.shape[0] == translate.shape[0] == rotate.shape[0]
    transformed_maps = []
    batch_size = scale.shape[0]

    for batch in range(batch_size):
        transformed_maps.append(
            linear_transform_distance_map(
                distance_map,
                scale[batch],
                translate[batch],
                rotate[batch]
            )
        )

    return torch.stack(transformed_maps)


class UDFLinearTransformSegmentFlow(nn.Module):
    def __init__(
            self,
            input_size,
            pretrained_linear_transform,
            encoder,
            segment_decoder,
            flow_decoder,
            integrator
    ):
        super().__init__()

        self.input_size = input_size
        self.pretrained_linear_transform = pretrained_linear_transform
        self.encoder = encoder
        self.segment_decoder = segment_decoder
        self.flow_decoder = flow_decoder
        self.integrator = integrator
        self.flow_div = FlowDiv(self.input_size)

        for param in self.pretrained_linear_transform.parameters():
            param.requires_grad = False

    def forward(self, image, vertices, distance_map):
        assert image.ndim == 5

        with torch.no_grad():
            lt_parameters = self.pretrained_linear_transform.linear_transform_parameters_from_image(image)

        scale, translate, rotate = lt_parameters
        transformed_distance_map = batch_transform_distance_map(
            distance_map,
            scale,
            translate,
            rotate
        )
        
        lt_deformed_vertices = self.pretrained_linear_transform.transform_from_parameters(
            vertices,
            scale,
            translate,
            rotate
        )

        encoder_input = torch.cat([image, transformed_distance_map], dim=1)
        encoding = self.encoder(encoder_input)

        flow = self.flow_decoder(encoding)
        deformed_vertices = self.integrator.integrate(flow, lt_deformed_vertices)

        return deformed_vertices

    def predict(self, image, vertices, distance_map):
        assert image.ndim == 5

        with torch.no_grad():
            lt_parameters = self.pretrained_linear_transform.linear_transform_parameters_from_image(image)

        scale, translate, rotate = lt_parameters
        transformed_distance_map = batch_transform_distance_map(
            distance_map,
            scale,
            translate,
            rotate
        )
        lt_deformed_vertices = self.pretrained_linear_transform.transform_vertices_list_from_parameters(
            vertices,
            scale,
            translate,
            rotate
        )

        encoder_input = torch.cat([image, transformed_distance_map], dim=1)
        encoding = self.encoder(encoder_input)

        predicted_segmentation = self.segment_decoder(encoding)
        flow_field = self.flow_decoder(encoding)

        flow_and_div = self.flow_div.get_flow_div(flow_field)
        deformed_verts, div_integral = self.integrator.integrate_flow_and_div(flow_and_div, lt_deformed_vertices)

        predictions = {"deformed_vertices": deformed_verts,
                       "segmentation": predicted_segmentation,
                       "divergence_integral": div_integral}

        return predictions
