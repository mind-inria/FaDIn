# %%
import numpy as np
import torch
from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.loss_and_gradient import discrete_l2_loss_precomputation, \
    discrete_l2_loss_conv, squared_compensator_1, squared_compensator_2, \
    squared_compensator_3, intens_events, get_grad_baseline, get_grad_alpha, \
    get_grad_eta
from fadin.utils.utils import check_random_state
from fadin.utils.compute_constants import get_zG, get_zN, get_ztzG


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
    events = torch.stack([d1, d2]).float()
    baseline = torch.tensor(rng.randn(n_dim))
    alpha = torch.tensor(rng.randn(n_dim, n_dim))
    decay = 1 + torch.abs(torch.tensor(rng.randn(n_dim, n_dim)))
    sigma = torch.tensor(rng.randn(n_dim, n_dim))
    u = sigma.clone() + 1
    discrete = torch.linspace(0, 1, n_discrete)

    return events, baseline, alpha, decay, u, sigma, discrete

# %%


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

    model_EXP = DiscreteKernelFiniteSupport(delta, n_dim=2,
                                            kernel='truncated_exponential')
    kernel_EXP = model_EXP.kernel_eval([decay], discrete)
    intens_EXP = model_EXP.intensity_eval(baseline, alpha, [decay], events, discrete)
    squared_conv_EXP = 2 * ((intens_EXP**2).sum(1) * 0.5 * delta).sum()

    zG = get_zG(events.numpy(), n_discrete)
    ztzG = get_ztzG(events.numpy(), n_discrete)

    term_1_EXP = end_time * squared_compensator_1(baseline)
    term_2_EXP = 2 * delta * squared_compensator_2(torch.tensor(zG),
                                                   baseline, alpha, kernel_EXP)
    term_3_EXP = delta * squared_compensator_3(torch.tensor(ztzG), alpha, kernel_EXP)

    squared_precomp_EXP = term_1_EXP + term_2_EXP + term_3_EXP

    assert torch.isclose(squared_conv_EXP, squared_precomp_EXP)

    model_RC = DiscreteKernelFiniteSupport(delta, n_dim=2, kernel='raised_cosine')

    kernel_RC = model_RC.kernel_eval([u, sigma], discrete).double()
    intens_RC = model_RC.intensity_eval(baseline, alpha.double(), [u, sigma],
                                        events, discrete)
    squared_conv_RC = 2 * ((intens_RC**2).sum(1) * 0.5 * delta).sum()

    term_1_RC = end_time * squared_compensator_1(baseline)
    term_2_RC = 2 * delta * squared_compensator_2(torch.tensor(zG),
                                                  baseline, alpha, kernel_RC)
    term_3_RC = delta * squared_compensator_3(torch.tensor(ztzG), alpha, kernel_RC)

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

    model_EXP = DiscreteKernelFiniteSupport(delta, n_dim=2,
                                            kernel='truncated_exponential')
    kernel_EXP = model_EXP.kernel_eval([decay], discrete).double()
    intens_EXP = model_EXP.intensity_eval(baseline, alpha, [decay], events, discrete)
    right_term_conv_EXP = 2 * (intens_EXP * events).sum()

    zN = get_zN(events.numpy(), n_discrete)
    right_term_precomp_EXP = intens_events(torch.tensor(zN), baseline.float(), alpha,
                                           kernel_EXP, n_events)

    assert torch.isclose(right_term_conv_EXP, 2 * right_term_precomp_EXP)

    model_RC = DiscreteKernelFiniteSupport(delta, n_dim=2, kernel='raised_cosine')
    kernel_RC = model_RC.kernel_eval([u, sigma], discrete).double()
    intens_RC = model_RC.intensity_eval(baseline, alpha, [u, sigma], events, discrete)
    right_term_conv_RC = 2 * (intens_RC * events).sum()

    right_term_precomp_RC = intens_events(torch.tensor(zN), baseline.float(), alpha,
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

    model_EXP = DiscreteKernelFiniteSupport(delta, n_dim=2,
                                            kernel='truncated_exponential')
    kernel_EXP = model_EXP.kernel_eval([decay], discrete)
    intens_EXP = model_EXP.intensity_eval(baseline, alpha, [decay], events, discrete)
    loss_conv_EXP = discrete_l2_loss_conv(intens_EXP, events, delta)

    zG = get_zG(events.numpy(), n_discrete)
    zN = get_zN(events.numpy(), n_discrete)
    ztzG = get_ztzG(events.numpy(), n_discrete)

    loss_precomp_EXP = discrete_l2_loss_precomputation(torch.tensor(zG),
                                                       torch.tensor(zN),
                                                       torch.tensor(ztzG),
                                                       baseline.float(), alpha,
                                                       kernel_EXP, n_events,
                                                       delta, end_time)

    assert torch.isclose(loss_conv_EXP, loss_precomp_EXP)

    model_RC = DiscreteKernelFiniteSupport(delta, n_dim=2, kernel='raised_cosine')
    kernel_RC = model_RC.kernel_eval([u, sigma], discrete).double()
    intens_RC = model_RC.intensity_eval(baseline, alpha, [u, sigma], events, discrete)
    loss_conv_RC = discrete_l2_loss_conv(intens_RC, events, delta)

    loss_precomp_RC = discrete_l2_loss_precomputation(torch.tensor(zG),
                                                      torch.tensor(zN),
                                                      torch.tensor(ztzG),
                                                      baseline.float(), alpha,
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

    events, baseline, alpha, decay, u, sigma, discrete = set_data(end_time,
                                                                  n_discrete,
                                                                  random_state)

    zG = get_zG(events.numpy(), n_discrete)
    zN = get_zN(events.numpy(), n_discrete)
    ztzG = get_ztzG(events.numpy(), n_discrete)
    n_events = events.sum(1)

    baseline_ = baseline.clone().requires_grad_(True)
    baseline__ = baseline.clone().requires_grad_(True)
    alpha_ = alpha.clone().requires_grad_(True)
    alpha__ = alpha.clone().requires_grad_(True)
    decay_ = decay.clone().requires_grad_(True)
    u_ = u.clone().requires_grad_(True)
    sigma_ = sigma.clone().requires_grad_(True)

    model_EXP = DiscreteKernelFiniteSupport(delta, n_dim=2,
                                            kernel='truncated_exponential')
    kernel_EXP = model_EXP.kernel_eval([decay_], discrete).double()

    loss_precomp_EXP = discrete_l2_loss_precomputation(torch.tensor(zG),
                                                       torch.tensor(zN),
                                                       torch.tensor(ztzG),
                                                       baseline_.float(), alpha_,
                                                       kernel_EXP, n_events,
                                                       delta, end_time)
    loss_precomp_EXP.backward()

    model_RC = DiscreteKernelFiniteSupport(delta, n_dim=2, kernel='raised_cosine')
    kernel_RC = model_RC.kernel_eval([u_, sigma_], discrete).double()
    loss_precomp_RC = discrete_l2_loss_precomputation(torch.tensor(zG),
                                                      torch.tensor(zN),
                                                      torch.tensor(ztzG),
                                                      baseline__.float(), alpha__,
                                                      kernel_RC, n_events,
                                                      delta, end_time)
    loss_precomp_RC.backward()

    grad_mu_EXP = get_grad_baseline(torch.tensor(zG),
                                    baseline, alpha,
                                    kernel_EXP, delta,
                                    n_events, end_time)

    assert torch.allclose(baseline_.grad, grad_mu_EXP)

    grad_mu_RC = get_grad_baseline(torch.tensor(zG),
                                   baseline, alpha,
                                   kernel_RC, delta,
                                   n_events, end_time)

    assert torch.allclose(baseline__.grad, grad_mu_RC)

    grad_alpha_EXP = get_grad_alpha(torch.tensor(zG),
                                    torch.tensor(zN),
                                    torch.tensor(ztzG),
                                    baseline, alpha,
                                    kernel_EXP.detach(), delta, n_events)

    assert torch.allclose(alpha_.grad, grad_alpha_EXP)

    grad_alpha_RC = get_grad_alpha(torch.tensor(zG),
                                   torch.tensor(zN),
                                   torch.tensor(ztzG),
                                   baseline, alpha,
                                   kernel_RC.detach(), delta, n_events)

    assert torch.allclose(alpha__.grad, grad_alpha_RC)

    grad_kernel_EXP = model_EXP.grad_eval([decay], discrete)
    grad_eta_EXP = get_grad_eta(torch.tensor(zG),
                                torch.tensor(zN),
                                torch.tensor(ztzG),
                                baseline,
                                alpha, kernel_EXP,
                                grad_kernel_EXP[0], delta, n_events)

    assert torch.allclose(decay_.grad, grad_eta_EXP)

    grad_kernel_RC = model_RC.grad_eval([u, sigma], discrete)
    grad_eta_RC = get_grad_eta(torch.tensor(zG),
                               torch.tensor(zN),
                               torch.tensor(ztzG),
                               baseline,
                               alpha, kernel_RC,
                               grad_kernel_RC[0].double(), delta, n_events)

    assert torch.allclose(u_.grad, grad_eta_RC)

    grad_eta_RC = get_grad_eta(torch.tensor(zG),
                               torch.tensor(zN),
                               torch.tensor(ztzG),
                               baseline,
                               alpha, kernel_RC,
                               grad_kernel_RC[1].double(), delta, n_events)

    assert torch.allclose(sigma_.grad, grad_eta_RC)


"""
def test_optim_grad():
    delta = 0.01
    n_dim = 10
    L = int(1 / delta)
    end_time = 1000
    baseline = torch.randn(n_dim)
    alpha = torch.randn(n_dim, n_dim)
    zG = torch.randn(n_dim, L)
    zN = torch.randn(n_dim, n_dim, L)
    ztzG = torch.randn(n_dim, n_dim, L, L)
    kernel = torch.randn(n_dim, n_dim, L)
    grad_kernel = torch.randn(n_dim, n_dim, L)
    n_events = torch.ones(n_dim) + 10

    g_baseline = get_grad_baseline(zG, baseline, alpha, kernel,
                                   delta, n_events, end_time)

    g_baseline_ = get_grad_baseline_(zG, baseline, alpha, kernel,
                                     delta, n_events, end_time)

    assert torch.allclose(g_baseline, g_baseline_)

    g_alpha = get_grad_alpha(zG, zN, ztzG, baseline, alpha, kernel, delta, n_events)

    g_alpha_ = get_grad_alpha_(zG, zN, ztzG, baseline, alpha, kernel, delta, n_events)

    assert torch.allclose(g_alpha, g_alpha_, atol=10e-4)

    g_eta = get_grad_eta(zG, zN, ztzG, baseline, alpha, kernel,
                             grad_kernel, delta, n_events)

    g_eta_ = get_grad_eta_(zG, zN, ztzG, baseline, alpha, kernel,
                               grad_kernel, delta, n_events)

    assert torch.allclose(g_eta, g_eta_, atol=10e-4)
"""
