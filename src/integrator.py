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
            raise TypeError("Expected vertices to be torch tensor or list but got " + type(vertices))


class IntegrateFlowDivRK4(IntegrateRK4):
    def __init__(self, num_steps, interpolation="bilinear"):
        super().__init__(num_steps, interpolation)

    def step_flow_and_div(self, flow_and_div, vertex_coordinates, div_integral):
        assert flow_and_div.ndim == 5
        assert vertex_coordinates.ndim == 3
        assert flow_and_div.shape[1] == 4
        assert flow_and_div.shape[0] == vertex_coordinates.shape[0]

        stage, integral = 0.0, 0.0
        interp_at = vertex_coordinates
        for j in range(len(self.weights)):
            stage = self.interpolator.interpolate(flow_and_div, interp_at)
            integral = integral + self.weights[j] * stage
            interp_at = vertex_coordinates + self.coefs[j] * stage.narrow(-1, 0, 3)

        vertex_coordinates = vertex_coordinates + integral.narrow(-1, 0, 3)
        div_integral = div_integral + integral.select(-1, -1)

        return vertex_coordinates, div_integral

    def _integrate_flow_and_div_tensor(self, flow_and_div, vertex_coordinates):
        assert isinstance(vertex_coordinates, torch.Tensor)

        div_integral = 0
        for step in range(self.num_steps):
            vertex_coordinates, div_integral = self.step(flow_and_div, vertex_coordinates, div_integral)

        return vertex_coordinates, div_integral

    def _integrate_flow_and_div_tensor_list(self, flow_and_div, vertex_coordinates_list):
        assert isinstance(vertex_coordinates_list, list)

        deformed_coordinates_list = []
        div_integral_list = []

        for coords in vertex_coordinates_list:
            dc, di = self._integrate_tensor(flow_and_div, coords)
            deformed_coordinates_list.append(dc)
            div_integral_list.append(di)

        return deformed_coordinates_list, div_integral_list

    def integrate_flow_and_div(self, flow_and_div, vertices):
        if isinstance(vertices, torch.Tensor):
            return self._integrate_tensor(flow_and_div, vertices)
        elif isinstance(vertices, list):
            return self._integrate_tensor_list(flow_and_div, vertices)
        else:
            raise TypeError("Expected tensor or list of tensors for vertices but got " + type(vertices))

