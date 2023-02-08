import torch.nn.functional as func
import torch


class GridSample:
    """
    Local pooling operation.
    """

    def __init__(self, interpolation):
        self.interp_mode = interpolation

    def interpolate(self, flow_field, vertices):
        grid = 2 * vertices - 1
        grid = grid.unsqueeze(1).unsqueeze(1).flip(dims=(-1,))

        interp = func.grid_sample(flow_field, grid, mode=self.interp_mode, padding_mode="border", align_corners=True)
        interp = interp.squeeze(2).squeeze(2)
        out = torch.transpose(interp, 1, 2)

        return out


class IntegrateRK4:
    def __init__(self, num_steps, interpolation="bilinear"):
        assert num_steps > 0
        self.interpolator = GridSample(interpolation)
        self.num_steps = num_steps

        self.weights = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]
        self.coefs = [0.0, 0.5, 0.5, 1.0]

    def step(self, flow, x):
        prev_k, step = 0.0, 0.0
        for j in range(len(self.weights)):
            prev_k = self.interpolator.interpolate(flow, x + self.coefs[j] * prev_k)
            step += self.weights[j] * prev_k
        x = x + step

        return x

    def _integrate_tensor(self, flow, x):
        assert isinstance(x, torch.Tensor)

        for step in range(self.num_steps):
            x = self.step(flow, x)
        return x

    def _integrate_tensor_list(self, flow, x_list):
        assert isinstance(x_list, list)
        y_list = [self._integrate_tensor(flow, x) for x in x_list]
        return y_list

    def integrate(self, flow, vertices):
        if isinstance(vertices, torch.Tensor):
            return self._integrate_tensor(flow, vertices)
        elif isinstance(vertices, list):
            return self._integrate_tensor_list(flow, vertices)
        else:
            raise TypeError("Expected vertices to be torch tensor or list but got", type(vertices))