import torch


def centered_finite_difference(x, dim, h):
    N = x.shape[dim]
    diff = (x.narrow(dim, 2, N - 2) - x.narrow(dim, 0, N - 2)) / (2 * h)
    N = diff.shape[dim]
    derivative = torch.cat((diff.narrow(dim, 0, 1), diff, diff.narrow(dim, N - 1, 1)), dim=dim)
    return derivative


def gradient3d(x, component, dim, h):
    assert x.ndim == 4
    assert 0 <= component <= x.shape[0]
    assert 0 <= dim <= 2

    x = x[component]
    dx = centered_finite_difference(x, dim, h).unsqueeze(0)
    return dx


def divergence(u, h):
    du_dx = gradient3d(u, 0, 0, h)
    du_dy = gradient3d(u, 1, 1, h)
    du_dz = gradient3d(u, 2, 2, h)
    div = du_dx + du_dy + du_dz
    return div


def batch_divergence3d(flow, grid_spacing):
    assert flow.ndim == 5
    b, c, h, w, d = flow.shape
    assert c == 3

    dfdx = centered_finite_difference(flow.narrow(1, 0, 1), -3, grid_spacing)
    dfdy = centered_finite_difference(flow.narrow(1, 1, 1), -2, grid_spacing)
    dfdz = centered_finite_difference(flow.narrow(1, 2, 1), -1, grid_spacing)
    div = dfdx + dfdy + dfdz
    return div