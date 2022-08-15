import numpy as np
import torch
from hawkes_discret.kernels import KernelExpDiscret
from hawkes_discret.loss_and_gradient import l2loss_precomputation, l2loss_conv,\
    term1, term2, term3, term4, get_grad_mu, get_grad_alpha, get_grad_theta
from hawkes_discret.utils.validation import check_random_state
from hawkes_discret.utils.compute_constants_np import get_zG, get_zN, get_ztzG


def set_data(end_time, n_discrete, random_state):
    """Set data for the following tests.
    """
    n_dim = 2

    rng = check_random_state(random_state)

    d1 = torch.zeros(n_discrete * end_time, dtype=torch.float64)
    d2 = torch.zeros(n_discrete * end_time, dtype=torch.float64)
    idx1 = rng.choice(np.arange(n_discrete * end_time), int(10*end_time))
    idx2 = rng.choice(np.arange(n_discrete * end_time), int(15*end_time))
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

    model = KernelExpDiscret(1, delta)
    kernel = model.eval(decay, discrete)
    intens = model.intensity_eval(baseline, adjacency, 
                                  decay, events, discrete)
    squared_conv = 2 * ((intens**2).sum(1) * 0.5 * delta).sum() / events.sum()

    zG, _ = get_zG(events.numpy(), n_discrete)
    ztzG, _ = get_ztzG(events.numpy(), n_discrete)

    term_1 = end_time * term1(baseline)
    term_2 = 2 * delta * term2(torch.tensor(zG), baseline, adjacency, kernel)
    term_3 = delta * term3(torch.tensor(ztzG), adjacency, kernel)

    squared_precomp = (term_1 + term_2 + term_3).double() / events.sum()

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

    model = KernelExpDiscret(1, delta)
    kernel = model.eval(decay, discrete)
    intens = model.intensity_eval(baseline, adjacency, 
                                  decay, events, discrete)
    right_term_conv = 2 * (intens * events).sum() / events.sum()

    zN, _ = get_zN(events.numpy(), n_discrete)
    right_term_precomp = term4(
        torch.tensor(zN),      
        baseline,
        adjacency,
        kernel,
        n_events)
    right_term_precomp = 2 * right_term_precomp.double() / events.sum()

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

    model = KernelExpDiscret(1, delta)
    kernel = model.eval(decay, discrete)
    intens = model.intensity_eval(baseline, adjacency, 
                                  decay, events, discrete)
    loss_conv = l2loss_conv(intens, events, delta)

    zG, _ = get_zG(events.numpy(), n_discrete)
    zN, _ = get_zN(events.numpy(), n_discrete)
    ztzG, _ = get_ztzG(events.numpy(), n_discrete)

    loss_precomp = l2loss_precomputation(torch.tensor(zG),
                                         torch.tensor(zN),
                                         torch.tensor(ztzG),
                                         baseline, adjacency,
                                         kernel, n_events, 
                                         delta, end_time)

    assert torch.isclose(loss_conv, loss_precomp)


def test_gradients():
    """Check that the implemented gradients
    are equal to those of pytorch autodiff
    """
    end_time = 10000
    delta = 0.01
    n_discrete = 100
    random_state = None

    events, baseline, adjacency, decay, discrete = set_data(end_time,
                                                            n_discrete,
                                                            random_state)


    zG, _ = get_zG(events.numpy(), n_discrete)
    zN, _ = get_zN(events.numpy(), n_discrete)
    ztzG, _ = get_ztzG(events.numpy(), n_discrete)
    n_events = events.sum(1)
   
    baseline_ = baseline.clone().requires_grad_(True)
    adjacency_ = adjacency.clone().requires_grad_(True)
    decay_ = decay.clone().requires_grad_(True)

    model = KernelExpDiscret(1, delta)
    kernel = model.eval(decay_, discrete)  

    loss_precomp = l2loss_precomputation(torch.tensor(zG),
                                         torch.tensor(zN),
                                         torch.tensor(ztzG),
                                         baseline_, adjacency_,
                                         kernel, n_events, 
                                         delta, end_time)
    loss_precomp.backward()

    grad_mu = get_grad_mu(torch.tensor(zG), 
                          baseline, adjacency, 
                          kernel, delta,  
                          n_events, end_time)
                          
    assert torch.allclose(baseline_.grad.float(), grad_mu)

    grad_alpha = get_grad_alpha(torch.tensor(zG), 
                                torch.tensor(zN), 
                                ztzG, 
                                baseline, adjacency,
                                kernel.detach(), delta, n_events)

    assert torch.allclose(adjacency_.grad.float(), grad_alpha)

    grad_kernel = model.compute_grad(decay, discrete)
    grad_theta = get_grad_theta(torch.tensor(zG), 
                                torch.tensor(zN),
                                ztzG,  baseline, 
                                adjacency, kernel, 
                                grad_kernel, delta, n_events)

    assert torch.allclose(decay_.grad.float(), grad_theta)