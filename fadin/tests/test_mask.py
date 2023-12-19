"""Test implementation of baseline_mask and alpha_mask in FaDIn solver"""
# %%
import torch
from fadin.solver import FaDIn
import numpy as np
from fadin.utils.utils_simu import simu_hawkes_cluster
from fadin.kernels import DiscreteKernelFiniteSupport


# %% Function
def testmaskedsolver(events, T, kernel, max_iter=1000, ztzG_approx=False,
                     baseline_mask=None, baseline_init=None,
                     alpha_mask=None, alpha_init=None,
                     random_state=0):
    solver = FaDIn(n_dim=n_dim,
                   kernel=kernel,
                   baseline_mask=baseline_mask,
                   baseline_init=baseline_init,
                   alpha_init=alpha_init,
                   alpha_mask=alpha_mask,
                   kernel_length=kernel_length,
                   delta=dt, optim="RMSprop",
                   params_optim=params_optim,
                   max_iter=max_iter, criterion='l2',
                   ztzG_approx=ztzG_approx,
                   random_state=random_state
                   )
    solver.fit(events, T)
    estimated_baseline = solver.params_intens[0]
    estimated_alpha = solver.params_intens[1]
    param_kernel = solver.params_intens[2:]
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
baseline_mask = torch.Tensor([0, 0])
alpha_mask = torch.Tensor([[0, 0], [1, 0]])
alpha_init = torch.Tensor([[0.2, 0.4], [0.7, 0.9]])
baseline_init = torch.Tensor([0.7, 0.4])
max_iter = 5000
ztzG_approx = True
params_optim = {'lr': 1e-3}

# %% Simulate Hawkes Process with exponentiel kernels
kernel = 'expon'
events = simu_hawkes_cluster(T,
                             baseline,
                             alpha,
                             kernel,
                             params_kernel={'scale': 1/lambda_exp},
                             random_state=simu_random_state)

# %% Fit Hawkes process to exponential simulation
exp_bl, exp_alpha = testmaskedsolver(kernel='truncated_exponential',
                                     events=events, T=T,
                                     baseline_mask=baseline_mask,
                                     alpha_mask=alpha_mask,
                                     baseline_init=baseline_init,
                                     alpha_init=alpha_init,
                                     ztzG_approx=ztzG_approx,
                                     random_state=simu_random_state)
assert torch.allclose(exp_bl, torch.Tensor([0., 0.]))
assert torch.allclose(exp_alpha * torch.Tensor([[1, 1], [0, 1]]),
                      torch.zeros(2, 2))


# %% Simulate raised cosine kernels
def raised_cosine(x, **params):
    rc = DiscreteKernelFiniteSupport(delta=0.01, n_dim=2,
                                     kernel='raised_cosine')
    u = params['u']
    sigma = params['sigma']
    kernel_values = rc.kernel_eval([torch.Tensor(u), torch.Tensor(sigma)],
                                   torch.tensor(x))
    return kernel_values.double().numpy()


events_rc = simu_hawkes_cluster(T, baseline, alpha, raised_cosine,
                                params_kernel=params_rc,
                                random_state=simu_random_state)

# %% Fit Hawkes process to raised_cosine simulation
rc_bl, rc_alpha = testmaskedsolver(kernel='raised_cosine', events=events_rc,
                                   T=T,
                                   baseline_mask=baseline_mask,
                                   alpha_mask=alpha_mask,
                                   baseline_init=baseline_init,
                                   alpha_init=alpha_init,
                                   ztzG_approx=ztzG_approx,
                                   random_state=simu_random_state)
assert torch.allclose(rc_bl, torch.Tensor([0., 0.]))
assert torch.allclose(rc_alpha * torch.Tensor([[1, 1], [0, 1]]),
                      torch.zeros(2, 2))
