import torch

from fadin.kernels import DiscreteKernelFiniteSupport


def identity(x, **param):
    return x


def linear_zero_one(x, **params):
    temp = 2 * x
    mask = x > 1
    temp[mask] = 0.
    return temp


def reverse_linear_zero_one(x, **params):
    temp = 2 - 2 * x
    mask = x > 1
    temp[mask] = 0.
    return temp


def truncated_gaussian(x, **params):
    rc = DiscreteKernelFiniteSupport(delta=0.01, n_dim=1, kernel='truncated_gaussian')
    mu = params['mu']
    sigma = params['sigma']
    kernel_values = rc.kernel_eval(
        [torch.Tensor(mu), torch.Tensor(sigma)], torch.tensor(x))

    return kernel_values.double().numpy()
