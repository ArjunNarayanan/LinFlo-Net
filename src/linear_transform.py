import torch
import torch.nn as nn
from src.unet_components import MultilayerCNN
from pytorch3d.transforms import Scale, Translate, Rotate, euler_angles_to_matrix


class LinearTransformer:
    def __init__(self, scale_factors, translations, euler_angles):
        self.scale = Scale(scale_factors)
        self.translate = Translate(translations)
        rot_mat = euler_angles_to_matrix(euler_angles, "XYZ")
        self.rotate = Rotate(rot_mat)

    def transform(self, vertices):
        # transform vertices to [-1,1]
        vertices = 2 * vertices - 1
        vertices = self.scale.transform_points(vertices)
        vertices = self.rotate.transform_points(vertices)
        vertices = self.translate.transform_points(vertices)

        # transform vertices to [0,1]
        vertices = (vertices + 1) / 2
        return vertices


class LinearTransformNet(nn.Module):
    def __init__(self, input_shape, input_channels, first_layer_channels, downarm_channels):
        super().__init__()

        self.encoder = MultilayerCNN(input_shape, input_channels, first_layer_channels, downarm_channels)

        num_conv = len(downarm_channels)
        self.encoder_output_channels = downarm_channels[-1]
        self.encoder_output_size = input_shape // (2 ** num_conv)
        assert self.encoder_output_size > 2

        self.num_encoder_features = (self.encoder_output_size ** 3) * self.encoder_output_channels
        self.linear_transform_parameters = nn.Sequential(nn.Linear(self.num_encoder_features, 128),
                                                         nn.LeakyReLU(),
                                                         nn.Linear(128, 9))

        # Initialize weights to small values
        final_layer = self.linear_transform_parameters[-1]
        final_layer.weight = nn.Parameter(
            torch.distributions.normal.Normal(0, 1e-5).sample(final_layer.weight.shape))
        final_layer.bias = nn.Parameter(torch.zeros(final_layer.bias.shape))

        self.scale_rotation = 2 * torch.pi

    @classmethod
    def from_dict(cls, definition):
        input_shape = definition["input_shape"]
        input_channels = definition["input_channels"]
        first_layer_channels = definition["first_layer_channels"]
        downarm_channels = definition["downarm_channels"]
        return cls(input_shape, input_channels, first_layer_channels, downarm_channels)

    def get_linear_transformer(self, encoding, multiplication_factor):
        x = encoding.reshape([encoding.shape[0], -1])
        transform_parameters = self.linear_transform_parameters(x)

        scale_by = (1.0 - transform_parameters[:, 0:3] * multiplication_factor)
        translate_by = transform_parameters[:, 3:6] * multiplication_factor
        rotate_by = self.scale_rotation * transform_parameters[:, 6:9] * multiplication_factor

        return LinearTransformer(scale_by, translate_by, rotate_by)

    def _fix_input_shape(self, image):
        if image.ndim == 4:
            image = image.unsqueeze(0)
        assert image.ndim == 5
        return image

    def _linear_transform_vertices_list(self, image, verts_list, multiplication_factor):
        image = self._fix_input_shape(image)
        batch_size = image.shape[0]
        assert isinstance(verts_list, list)
        assert all([v.ndim == 3 for v in verts_list])
        assert all([v.shape[0] == batch_size for v in verts_list])
        assert all([v.shape[-1] == 3 for v in verts_list])

        encoding = self.encoder(image)
        transformer = self.get_linear_transformer(encoding, multiplication_factor)
        deformed_verts_list = [transformer.transform(verts) for verts in verts_list]

        return deformed_verts_list

    def _linear_transform_vertices(self, image, vertices, multiplication_factor):
        assert isinstance(vertices, torch.Tensor)

        verts = self._linear_transform_vertices_list(image, [vertices], multiplication_factor)
        return verts[0]

    def forward(self, image, vertices, multiplication_factor=1.0):
        if isinstance(vertices, list):
            return self._linear_transform_vertices_list(image, vertices, multiplication_factor)
        elif isinstance(vertices, torch.Tensor):
            return self._linear_transform_vertices(image, vertices, multiplication_factor)
        else:
            raise TypeError("Expected vertices to be list of tensors or tensor, got ", type(vertices))

    def deform_vertices(self, image, vertices, multiplication_factor=1.0):
        return self.forward(image, vertices, multiplication_factor)

    def predict(self, image, vertices, multiplication_factor=1.0):
        deformed_vertices = self(image, vertices, multiplication_factor)
        predictions = {"deformed_vertices": deformed_vertices}
        return predictions

class IdentityLinearTransform(nn.Identity):
    def __init__(self):
        super().__init__()
    
    def forward(self, image, vertices):
        return vertices


class LinearTransformWithEncoder(nn.Module):
    def __init__(self, pretrained_encoder, linear_transform):
        super().__init__()

        self.pretrained_encoder = pretrained_encoder
        self.linear_transform = linear_transform

        for param in self.pretrained_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, vertices):
        encoding = self.pretrained_encoder(image)
        encoding = torch.cat([image, encoding], dim=1)
        deformed_vertices = self.linear_transform(encoding, vertices)
        return deformed_vertices
