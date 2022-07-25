import numpy as np
import torch
from hawkes_discret.kernels import KernelExpDiscret
from hawkes_discret.loss_and_gradient import l2loss_precomputation, l2loss_conv,\
    term1, term2, term3, term4, const_loss
from hawkes_discret.utils.validation import check_random_state
from hawkes_discret.utils.compute_constants_np import get_zG, get_zN, get_ztzG


def set_data(end_time, n_discrete, random_state):
    """Set data for the following tests.
    """
    n_dim = 2

    rng = check_random_state(random_state)

    d1 = torch.zeros(n_discrete * end_time, dtype=torch.float64)
    d2 = torch.zeros(n_discrete * end_time, dtype=torch.float64)
    idx1 = rng.choice(np.arange(n_discrete * end_time), int(end_time / 10))
    idx2 = rng.choice(np.arange(n_discrete * end_time), int(end_time / 7))
    d1[idx1] = 1.
    d2[idx2] = 2.
    d1[idx2] = 3.
    events = torch.stack([d1, d2])
    baseline = torch.tensor(rng.randn(n_dim))
    adjacency = torch.tensor(rng.randn(n_dim, n_dim))
    decay = torch.tensor(rng.randn(n_dim, n_dim))
    discrete = torch.linspace(0, 1, n_discrete)

    return events, baseline, adjacency, decay, discrete


def test_squared_term_l2loss():
    """Check that the quadratic part of the l2loss are the same  
    using precomputation and convolution tools
    """
    end_time = 10000
    delta = 0.01
    n_discrete = 100
    random_state = None

    events, baseline, adjacency, decay, discrete = set_data(end_time,
                                                            n_discrete,
                                                            random_state)

    model = KernelExpDiscret(decay, 1, delta)
    kernel = model.eval(discrete)
    intens = model.intensity_eval(baseline, adjacency, events)
    squared_conv = ((intens**2).sum(1) * 0.5 * delta).sum() / end_time

    zG = get_zG(events.numpy(), n_discrete)
    ztzG = get_ztzG(events.numpy(), n_discrete)
    const = const_loss(delta, end_time)
    term_1 = term1(baseline)
    term_2 = 2 * const * term2(torch.tensor(zG), baseline, adjacency, kernel)
    term_3 = const * term3(torch.tensor(ztzG), adjacency, kernel)

    squared_precomp = torch.tensor(
        term_1 + term_2 + term_3,
        dtype=torch.float64)

    assert torch.isclose(squared_conv, squared_precomp)


def test_right_term_l2loss():
    """Check that the right part of the l2loss are the same  
    using precomputation and convolution tools
    """
    end_time = 10000
    delta = 0.01
    n_discrete = 100
    random_state = None

    events, baseline, adjacency, decay, discrete = set_data(end_time,
                                                            n_discrete,
                                                            random_state)
    n_events = events.sum(1)

    model = KernelExpDiscret(decay, 1, delta)
    kernel = model.eval(discrete)
    intens = model.intensity_eval(baseline, adjacency, events)
    right_term_conv = (intens * events).sum() / end_time

    zN = get_zN(events.numpy(), n_discrete)
    right_term_precomp = term4(
        torch.tensor(zN),      
        baseline,
        adjacency,
        kernel,
        n_events)
    const = const_loss(delta, end_time)
    c = (2 / delta) * const
    right_term_precomp = torch.tensor(
        c * right_term_precomp, dtype=torch.float64)

    assert torch.isclose(right_term_conv, right_term_precomp)


def test_l2loss():
    """Check that the l2loss are the same  
    using precomputation and convolution tools
    """
    end_time = 10000
    delta = 0.01
    n_discrete = 100
    random_state = None

    events, baseline, adjacency, decay, discrete = set_data(end_time,
                                                            n_discrete,
                                                            random_state)
    n_events = events.sum(1)

    model = KernelExpDiscret(decay, 1, delta)
    kernel = model.eval(discrete)
    intens = model.intensity_eval(baseline, adjacency, events)
    loss_conv = l2loss_conv(intens, events, delta, end_time)

    zG = get_zG(events.numpy(), n_discrete)
    zN = get_zN(events.numpy(), n_discrete)
    ztzG = get_ztzG(events.numpy(), n_discrete)

    loss_precomp = l2loss_precomputation(torch.tensor(zG),
                                         torch.tensor(zN),
                                         torch.tensor(ztzG),
                                         baseline, adjacency,
                                         kernel, n_events, 
                                         delta, end_time)

    assert torch.isclose(loss_conv, loss_precomp)
