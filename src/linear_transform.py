import torch
import torch.nn as nn
from src.unet_components import MultilayerCNN, SimpleCNN
from pytorch3d.transforms import Scale, Translate, Rotate, euler_angles_to_matrix


class ApplyLinearTransform():
    def __init__(self, scale_factors, translations, euler_angles):
        self.scale = Scale(scale_factors)
        self.translate = Translate(translations)
        rot_mat = euler_angles_to_matrix(euler_angles, "XYZ")
        self.rotate = Rotate(rot_mat)

    def transform(self, vertices):
        vertices = 2 * vertices - 1
        vertices = self.scale.transform_points(vertices)
        vertices = self.rotate.transform_points(vertices)
        vertices = self.translate.transform_points(vertices)
        vertices = (vertices + 1) / 2
        return vertices


class LinearTransform(nn.Module):
    def __init__(self, encoder_name, definition):
        super().__init__()
        self.encoder_name = encoder_name

        if encoder_name == "MultilayerCNN":
            self.encoder = MultilayerCNN(**definition)
        elif encoder_name == "SimpleCNN":
            self.encoder = SimpleCNN(**definition)
        else:
            raise ValueError("Unexpected value for encoder_name : ", encoder_name)

        downarm_channels = definition["downarm_channels"]
        input_shape = definition["input_shape"]
        num_conv = len(downarm_channels)
        self.encoder_output_channels = downarm_channels[-1]
        self.encoder_output_size = input_shape[0] // (2 ** num_conv)
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

        self.scale_rotation = torch.pi

    @classmethod
    def from_dict(cls, definition):
        encoder_name = definition["encoder_name"]
        encoder_definition = definition["encoder_definition"]
        return cls(encoder_name, encoder_definition)

    def encode(self, image):
        encoding = self.encoder(image)
        return encoding

    def get_linear_transformer(self, encoding):
        x = encoding.reshape([encoding.shape[0], -1])
        transform_parameters = self.linear_transform_parameters(x)

        scale_by = (1 - transform_parameters[:, 0:3])
        translate_by = transform_parameters[:, 3:6]
        rotate_by = self.scale_rotation * transform_parameters[:, 6:9]

        return ApplyLinearTransform(scale_by, translate_by, rotate_by)

    def _fix_input_shape(self, image):
        if image.ndim == 4:
            image = image.unsqueeze(0)
        assert image.ndim == 5
        return image

    def _linear_transform_vertices_list(self, image, verts_list):
        image = self._fix_input_shape(image)
        batch_size = image.shape[0]
        assert all([v.ndim == 3 for v in verts_list])
        assert all([v.shape[0] == batch_size for v in verts_list])
        assert all([v.shape[-1] == 3 for v in verts_list])

        encoding = self.encode(image)
        transformer = self.get_linear_transformer(encoding)
        deformed_verts_list = [transformer.transform(verts) for verts in verts_list]

        return deformed_verts_list

    def _linear_transform_vertices(self, image, vertices):
        assert isinstance(vertices, torch.Tensor)

        verts = self._linear_transform_vertices_list(image, [vertices])
        return verts[0]

    def forward(self, image, vertices):
        if isinstance(vertices, list):
            return self._linear_transform_vertices_list(image, vertices)
        elif isinstance(vertices, torch.Tensor):
            return self._linear_transform_vertices(image, vertices)
        else:
            raise TypeError("Expected vertices to be list of tensors or tensor, got ", type(vertices))

    def deform_vertices(self, image, vertices):
        return self.forward(image, vertices)


class LinearTransformWithEncoder(LinearTransform):
    def __init__(self, pretrained_encoder, encoder_name, definition):
        super().__init__(encoder_name, definition)

        self.pretrained_encoder = pretrained_encoder
        for param in self.pretrained_encoder.parameters():
            param.requires_grad = False

    @classmethod
    def from_dict(cls, definition):
        pretrained_encoder_fn = definition["pretrained_encoder"]
        pretrained_encoder = torch.load(pretrained_encoder_fn, map_location=torch.device("cpu"))
        encoder_name = definition["encoder_name"]
        encoder_definition = definition["encoder_definition"]
        return cls(pretrained_encoder, encoder_name, encoder_definition)

    def encode(self, image):
        encoding = self.pretrained_encoder(image)
        encoding = torch.cat([image, encoding], dim=1)
        encoding = self.encoder(encoding)
        return encoding
