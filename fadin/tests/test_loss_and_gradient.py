import numpy as np
import torch
from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.loss_and_gradient import l2loss_precomputation, l2loss_conv,\
    term1, term2, term3, term4, get_grad_mu, get_grad_alpha, get_grad_theta
from fadin.utils.validation import check_random_state
from fadin.utils.compute_constants import get_zG, get_zN, get_ztzG  # , get_ztzG_


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
    alpha = torch.tensor(rng.randn(n_dim, n_dim))
    decay = torch.tensor(rng.randn(n_dim, n_dim))
    sigma = torch.tensor(rng.randn(n_dim, n_dim))
    u = sigma.clone() + 1
    discrete = torch.linspace(0, 1, n_discrete)

    return events, baseline, alpha, decay, u, sigma, discrete


def test_squared_term_l2loss():
    """Check that the quadratic part of the l2loss are the same
    using precomputation and convolution tools
    """
    end_time = 10000
    delta = 0.01
    n_discrete = 100
    random_state = None

    events, baseline, alpha, decay, u, sigma, discrete = set_data(end_time, n_discrete,
                                                                  random_state)

    model_EXP = DiscreteKernelFiniteSupport(0, 1, delta, kernel='Exponential', n_dim=2)
    kernel_EXP = model_EXP.eval([decay], discrete)
    intens_EXP = model_EXP.intensity_eval(baseline, alpha, [decay], events, discrete)
    squared_conv_EXP = 2 * ((intens_EXP**2).sum(1) * 0.5 * delta).sum()

    zG = get_zG(events.numpy(), n_discrete)
    ztzG = get_ztzG(events.numpy(), n_discrete)

    term_1_EXP = end_time * term1(baseline)
    term_2_EXP = 2 * delta * term2(torch.tensor(zG), baseline, alpha, kernel_EXP)
    term_3_EXP = delta * term3(torch.tensor(ztzG), alpha, kernel_EXP)

    squared_precomp_EXP = term_1_EXP + term_2_EXP + term_3_EXP

    assert torch.isclose(squared_conv_EXP, squared_precomp_EXP)

    model_RC = DiscreteKernelFiniteSupport(0, 1, delta, kernel='RaisedCosine', n_dim=2)

    kernel_RC = model_RC.eval([u, sigma], discrete).double()
    intens_RC = model_RC.intensity_eval(baseline, alpha.double(), [u, sigma],
                                        events, discrete)
    squared_conv_RC = 2 * ((intens_RC**2).sum(1) * 0.5 * delta).sum()

    term_1_RC = end_time * term1(baseline)
    term_2_RC = 2 * delta * term2(torch.tensor(zG), baseline, alpha, kernel_RC)
    term_3_RC = delta * term3(torch.tensor(ztzG), alpha, kernel_RC)

    squared_precomp_RC = term_1_RC + term_2_RC + term_3_RC

    assert torch.isclose(squared_conv_RC, squared_precomp_RC)


def test_right_term_l2loss():
    """Check that the right part of the l2loss are the same
    using precomputation and convolution tools
    """
    end_time = 10000
    delta = 0.01
    n_discrete = 100
    random_state = None

    events, baseline, alpha, decay, u, sigma, discrete = set_data(end_time, n_discrete,
                                                                  random_state)
    n_events = events.sum(1)

    model_EXP = DiscreteKernelFiniteSupport(0, 1, delta, kernel='Exponential', n_dim=2)
    kernel_EXP = model_EXP.eval([decay], discrete).double()
    intens_EXP = model_EXP.intensity_eval(baseline, alpha, [decay], events, discrete)
    right_term_conv_EXP = 2 * (intens_EXP * events).sum()

    zN = get_zN(events.numpy(), n_discrete)
    right_term_precomp_EXP = term4(torch.tensor(zN), baseline, alpha,
                                   kernel_EXP, n_events)

    assert torch.isclose(right_term_conv_EXP, 2 * right_term_precomp_EXP)

    model_RC = DiscreteKernelFiniteSupport(0, 1, delta, kernel='RaisedCosine', n_dim=2)
    kernel_RC = model_RC.eval([u, sigma], discrete).double()
    intens_RC = model_RC.intensity_eval(baseline, alpha, [u, sigma], events, discrete)
    right_term_conv_RC = 2 * (intens_RC * events).sum()

    right_term_precomp_RC = term4(torch.tensor(zN), baseline, alpha,
                                  kernel_RC, n_events)

    assert torch.isclose(right_term_conv_RC, 2 * right_term_precomp_RC)


def test_l2loss():
    """Check that the l2loss are the same
    using precomputation and convolution tools
    """
    end_time = 10000
    delta = 0.01
    n_discrete = 100
    random_state = None

    events, baseline, alpha, decay, u, sigma, discrete = set_data(end_time,
                                                                  n_discrete,
                                                                  random_state)
    n_events = events.sum(1)

    model_EXP = DiscreteKernelFiniteSupport(0, 1, delta, kernel='Exponential', n_dim=2)
    kernel_EXP = model_EXP.eval([decay], discrete)
    intens_EXP = model_EXP.intensity_eval(baseline, alpha, [decay], events, discrete)
    loss_conv_EXP = l2loss_conv(intens_EXP, events, delta)

    zG = get_zG(events.numpy(), n_discrete)
    zN = get_zN(events.numpy(), n_discrete)
    ztzG = get_ztzG(events.numpy(), n_discrete)

    loss_precomp_EXP = l2loss_precomputation(torch.tensor(zG),
                                             torch.tensor(zN),
                                             torch.tensor(ztzG),
                                             baseline, alpha,
                                             kernel_EXP, n_events,
                                             delta, end_time)

    assert torch.isclose(loss_conv_EXP, loss_precomp_EXP)

    model_RC = DiscreteKernelFiniteSupport(0, 1, delta, kernel='RaisedCosine', n_dim=2)
    kernel_RC = model_RC.eval([u, sigma], discrete).double()
    intens_RC = model_RC.intensity_eval(baseline, alpha, [u, sigma], events, discrete)
    loss_conv_RC = l2loss_conv(intens_RC, events, delta)

    loss_precomp_RC = l2loss_precomputation(torch.tensor(zG),
                                            torch.tensor(zN),
                                            torch.tensor(ztzG),
                                            baseline, alpha,
                                            kernel_RC, n_events,
                                            delta, end_time)

    assert torch.isclose(loss_conv_RC, loss_precomp_RC)


def test_gradients():
    """Check that the implemented gradients
     are equal to those of pytorch autodiff
    """
    end_time = 10000
    delta = 0.01
    n_discrete = 100
    random_state = None

    events, baseline, adjacency, decay, u, sigma, discrete = set_data(end_time,
                                                                      n_discrete,
                                                                      random_state)

    zG = get_zG(events.numpy(), n_discrete)
    zN = get_zN(events.numpy(), n_discrete)
    ztzG = get_ztzG(events.numpy(), n_discrete)
    n_events = events.sum(1)

    baseline_ = baseline.clone().requires_grad_(True)
    baseline__ = baseline.clone().requires_grad_(True)
    adjacency_ = adjacency.clone().requires_grad_(True)
    adjacency__ = adjacency.clone().requires_grad_(True)
    decay_ = decay.clone().requires_grad_(True)
    u_ = u.clone().requires_grad_(True)
    sigma_ = sigma.clone().requires_grad_(True)

    model_EXP = DiscreteKernelFiniteSupport(0, 1, delta, kernel='Exponential', n_dim=2)
    kernel_EXP = model_EXP.eval([decay_], discrete).double()

    loss_precomp_EXP = l2loss_precomputation(torch.tensor(zG),
                                             torch.tensor(zN),
                                             torch.tensor(ztzG),
                                             baseline_, adjacency_,
                                             kernel_EXP, n_events,
                                             delta, end_time)
    loss_precomp_EXP.backward()

    model_RC = DiscreteKernelFiniteSupport(0, 1, delta, kernel='RaisedCosine', n_dim=2)
    kernel_RC = model_RC.eval([u_, sigma_], discrete).double()
    loss_precomp_RC = l2loss_precomputation(torch.tensor(zG),
                                            torch.tensor(zN),
                                            torch.tensor(ztzG),
                                            baseline__, adjacency__,
                                            kernel_RC, n_events,
                                            delta, end_time)
    loss_precomp_RC.backward()

    grad_mu_EXP = get_grad_mu(torch.tensor(zG),
                              baseline, adjacency,
                              kernel_EXP, delta,
                              n_events, end_time)

    assert torch.allclose(baseline_.grad.float(), grad_mu_EXP)

    grad_mu_RC = get_grad_mu(torch.tensor(zG),
                             baseline, adjacency,
                             kernel_RC, delta,
                             n_events, end_time)

    assert torch.allclose(baseline__.grad.float(), grad_mu_RC)

    grad_alpha_EXP = get_grad_alpha(torch.tensor(zG),
                                    torch.tensor(zN),
                                    ztzG,
                                    baseline, adjacency,
                                    kernel_EXP.detach(), delta, n_events)

    assert torch.allclose(adjacency_.grad.float(), grad_alpha_EXP)

    grad_alpha_RC = get_grad_alpha(torch.tensor(zG),
                                   torch.tensor(zN),
                                   ztzG,
                                   baseline, adjacency,
                                   kernel_RC.detach(), delta, n_events)

    assert torch.allclose(adjacency__.grad.float(), grad_alpha_RC)

    grad_kernel_EXP = model_EXP.get_grad([decay], discrete)
    grad_theta_EXP = get_grad_theta(torch.tensor(zG),
                                    torch.tensor(zN),
                                    torch.tensor(ztzG),
                                    baseline,
                                    adjacency, kernel_EXP,
                                    grad_kernel_EXP[0], delta, n_events)

    assert torch.allclose(decay_.grad.float(), grad_theta_EXP)

    grad_kernel_RC = model_RC.get_grad([u, sigma], discrete)
    grad_theta_RC = get_grad_theta(torch.tensor(zG),
                                   torch.tensor(zN),
                                   torch.tensor(ztzG),
                                   baseline,
                                   adjacency, kernel_RC,
                                   grad_kernel_RC[0].double(), delta, n_events)

    assert torch.allclose(u_.grad.float(), grad_theta_RC)

    grad_theta_RC = get_grad_theta(torch.tensor(zG),
                                   torch.tensor(zN),
                                   torch.tensor(ztzG),
                                   baseline,
                                   adjacency, kernel_RC,
                                   grad_kernel_RC[1].double(), delta, n_events)

    assert torch.allclose(sigma_.grad.float(), grad_theta_RC)
