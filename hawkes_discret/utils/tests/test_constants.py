import numpy as np
import torch

from hawkes_discret.utils.compute_constants_np import get_zG, \
    get_zN, get_ztzG, get_zLG, get_zLN, get_zLtzG
from hawkes_discret.utils.validation import check_random_state


def test_zG():
    """Check that the inner product of zG with a kernel
    equal the sum of the convolution product over the grid
    """
    n_discrete = 100
    n_grid = 100 * n_discrete
    random_state = None

    rng = check_random_state(random_state)

    kernel = rng.randn(n_discrete)
    events = rng.randn(n_grid)

    zG, _ = get_zG(events.reshape(1, n_grid), n_discrete)
    kzG = zG @ kernel
    kzG_conv = torch.conv_transpose1d(
        torch.tensor(events).view(
            1, n_grid), torch.tensor(kernel).view(
            1, 1, n_discrete))[
                :, :-n_discrete + 1]

    assert np.isclose(kzG, kzG_conv.sum().numpy())


def test_zN():
    """Check that the inner product of zN with a kernel
    equal the sum of the convolution product over the timestamps
    """
    n_discrete = 100
    n_grid = 100 * n_discrete
    random_state = None

    rng = check_random_state(random_state)

    kernel = rng.randn(n_discrete)
    events = rng.randn(n_grid)

    zN, _ = get_zN(events.reshape(1, n_grid), n_discrete)
    kzN = zN @ kernel

    kzN_conv = torch.conv_transpose1d(
        torch.tensor(events).view(
            1, n_grid), torch.tensor(kernel).view(
            1, 1, n_discrete))[
                :, :-n_discrete + 1]
    kzN_conv *= events

    assert np.isclose(kzN, kzN_conv.squeeze().sum().numpy())


def test_ztzG():
    """Check that the cross inner product of ztzG with kernels
    equal the correlation of the two convolution products over the grid
    """
    n_discrete = 100
    n_grid = 100 * n_discrete
    random_state = None

    rng = check_random_state(random_state)

    kernel1 = rng.randn(n_discrete)
    kernel2 = rng.randn(n_discrete)
    events = rng.randn(n_grid)

    ztzG, _ = get_ztzG(events.reshape(1, n_grid), n_discrete)

    K = np.outer(kernel1, kernel2)
    K_ztzG = K * ztzG

    conv1 = torch.conv_transpose1d(
        torch.tensor(events).view(
            1, n_grid), torch.tensor(kernel1).view(
            1, 1, n_discrete))[
                :, :-n_discrete + 1]
    conv2 = torch.conv_transpose1d(
        torch.tensor(events).view(
            1, n_grid), torch.tensor(kernel2).view(
            1, 1, n_discrete))[
                :, :-n_discrete + 1]

    assert np.isclose(K_ztzG.sum(), (conv1 * conv2).sum())
