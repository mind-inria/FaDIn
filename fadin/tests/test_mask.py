"""Test implementation of baseline_mask and alpha_mask in FaDIn solver"""
# %%
import torch
from fadin.solver import FaDIn
import numpy as np
from fadin.utils.utils_simu import simu_hawkes_cluster
from fadin.kernels import DiscreteKernelFiniteSupport


# %% Function
def maskedsolver(events, T, kernel, max_iter=1000, ztzG_approx=False,
                 optim_mask=None, init='random', random_state=0):
    solver = FaDIn(
        n_dim=n_dim,
        kernel=kernel,
        optim_mask=optim_mask,
        init=init,
        kernel_length=kernel_length,
        delta=dt, optim="RMSprop",
        params_optim=params_optim,
        max_iter=max_iter,
        ztzG_approx=ztzG_approx,
        random_state=random_state
    )
    solver.fit(events, T)
    estimated_baseline = solver.baseline_
    estimated_alpha = solver.alpha_
    param_kernel = solver.kernel_
    if kernel == 'raised_cosine':
        # multiply alpha by 2* sigma
        estimated_alpha = 2 * estimated_alpha * param_kernel[1]

    return estimated_baseline, estimated_alpha


# %% Here, we set the parameters
# Simulation parameters
baseline = np.array([.1, .1])
alpha = np.array([[0.01, 0.05], [0.9, 0.05]])
mu = np.array([[0.4, 0.6], [0.55, 0.6]])
sigma = np.array([[0.3, 0.3], [0.25, 0.3]])
lambda_exp = 2.
u = mu - sigma
params_rc = dict(u=u, sigma=sigma)
params_tg = {'a': 0, 'b': 3, 'scale': 0.5, 'loc': 1}
simu_random_state = 30
# Solver parameters
n_dim = 2
dt = 0.01
T = 20_000
kernel_length = 3
L = int(1 / dt)
size_grid = int(T / dt) + 1
discretization = torch.linspace(0, kernel_length, L)
optim_mask = {
    'baseline': torch.Tensor([0, 0]),
    'alpha': torch.Tensor([[0, 0], [1, 0]])
}
init1_rc = {
    'alpha': torch.Tensor([[0.2, 0.4], [0.7, 0.9]]),
    'baseline': torch.Tensor([0.7, 0.4]),
    'kernel': [torch.Tensor([[0.5, 0.5], [0.5, 0.5]]),
               torch.Tensor([[0.25, 0.25], [0.25, 0.25]])]
}
init1_exp = {
    'alpha': torch.Tensor([[0.2, 0.4], [0.7, 0.9]]),
    'baseline': torch.Tensor([0.7, 0.4]),
    'kernel': [torch.Tensor([[0.5, 0.5], [0.5, 0.5]])]
}
init2 = 'random'
init3 = 'moment_matching_mean'
init4 = 'moment_matching_max'
max_iter = 5000
ztzG_approx = True
params_optim = {'lr': 1e-3}


def test_exp_mask():
    # Simulate Hawkes Process with exponentiel kernels
    kernel = 'expon'
    events = simu_hawkes_cluster(T,
                                 baseline,
                                 alpha,
                                 kernel,
                                 params_kernel={'scale': 1/lambda_exp},
                                 random_state=simu_random_state)

    # Fit Hawkes process to exponential simulation
    for init in [init1_exp, init2]:
        exp_bl, exp_alpha = maskedsolver(
            kernel='truncated_exponential',
            events=events, T=T,
            optim_mask=optim_mask,
            init=init,
            ztzG_approx=ztzG_approx,
            random_state=simu_random_state
        )
    assert torch.allclose(exp_bl, torch.Tensor([0., 0.]))
    assert torch.allclose(exp_alpha * torch.Tensor([[1., 1.], [0., 1.]]),
                          torch.zeros(2, 2))


def raised_cosine(x, **params):
    rc = DiscreteKernelFiniteSupport(delta=0.01, n_dim=2,
                                     kernel='raised_cosine')
    u = params['u']
    sigma = params['sigma']
    kernel_values = rc.kernel_eval([torch.Tensor(u), torch.Tensor(sigma)],
                                   torch.tensor(x))
    return kernel_values.double().numpy()


def test_rc_mask():
    # Simulate raised cosine kernels
    events_rc = simu_hawkes_cluster(T, baseline, alpha, raised_cosine,
                                    params_kernel=params_rc,
                                    random_state=simu_random_state)

    # %% Fit Hawkes process to raised_cosine simulation
    for init in [init1_rc, init2, init3, init4]:
        rc_bl, rc_alpha = maskedsolver(
            kernel='raised_cosine',
            events=events_rc,
            T=T,
            optim_mask=optim_mask,
            init=init,
            ztzG_approx=ztzG_approx,
            random_state=simu_random_state
        )
        assert torch.allclose(rc_bl, torch.Tensor([0., 0.]))
        assert torch.allclose(rc_alpha * torch.Tensor([[1., 1.], [0., 1.]]),
                              torch.zeros(2, 2))
