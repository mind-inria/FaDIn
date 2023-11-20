"""Test implementation of baseline_mask and alpha_mask in FaDIn solver"""
# %%
import torch
from fadin.solver import FaDIn
import numpy as np
from fadin.utils.utils_simu import simu_hawkes_cluster
import matplotlib.pyplot as plt
from fadin.kernels import DiscreteKernelFiniteSupport


# %% Function
def testmaskedsolver(events, T, kernel, max_iter=1000, ztzG_approx=False,
                     baseline_mask=None, alpha_mask=None, alpha_init=None,
                     random_state=0):
    solver = FaDIn(n_dim=n_dim,
                   kernel=kernel,
                   baseline_mask=baseline_mask,
                   alpha_init=alpha_init,
                   alpha_mask=alpha_mask,
                   kernel_length=kernel_length,
                   delta=dt, optim="RMSprop",
                   params_optim=params_optim,
                   max_iter=max_iter, criterion='l2',
                   log=True,
                   ztzG_approx=ztzG_approx,
                   radom_state=random_state
                   )
    # print('Init baseline: ', solver.params_intens[0])
    print('Init alpha: ', solver.params_intens[1])
    # print('Init kernel parameters: ', solver.kernel_params_fixed)
    solver.fit(events, T)
    estimated_baseline = solver.params_intens[0]
    estimated_alpha = solver.params_intens[1]
    param_kernel = solver.params_intens[2:]
    if kernel == 'raised_cosine':
        # multiply alpha by 2* sigma
        estimated_alpha = 2 * estimated_alpha * param_kernel[1]

    print('Estimated baseline:', estimated_baseline)
    print('Estimated alpha:', estimated_alpha)
    print('Estimated parameters of the', kernel, 'kernel:', param_kernel)
    plt.plot(np.arange(len(solver.v_loss)), solver.v_loss)
    del solver


# %% Here, we set the parameters
# Simulation parameters
baseline = np.array([.1, .1])
alpha = np.array([[0.2, 0.5], [0.9, 0.3]])
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
T = 1_000_000
kernel_length = 3
L = int(1 / dt)
size_grid = int(T / dt) + 1
discretization = torch.linspace(0, kernel_length, L)
baseline_mask = None  # torch.Tensor([0, 0])
alpha_mask = None  # torch.Tensor([[0, 0], [1, 0]])
alpha_init = None  # torch.Tensor([[0.2, 0.2], [0.8, 0.2]])
max_iter = 5000
ztzG_approx = False
params_optim = {'lr': 1e-3}

# %% Simulate Hawkes Process with exponentiel kernels
kernel = 'expon'  # 'norm'
events = simu_hawkes_cluster(T,
                             baseline,
                             alpha,
                             kernel,
                             params_kernel={'scale': 1/lambda_exp},
                             random_state=simu_random_state)

# %% Fit Hawkes process to exponential simulation
testmaskedsolver(kernel='truncated_exponential', events=events, T=T,
                 baseline_mask=baseline_mask, alpha_mask=alpha_mask,
                 ztzG_approx=False)
print('Simulation baseline:', baseline)
print('Simulation alpha:', alpha)


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
testmaskedsolver(kernel='raised_cosine', events=events_rc, T=T,
                 max_iter=max_iter, baseline_mask=baseline_mask,
                 alpha_init=alpha_init, alpha_mask=alpha_mask,
                 ztzG_approx=ztzG_approx)
print('Simulation baseline:', baseline)
print('Simulation alpha:', alpha)

# %% Simulate truncated gaussian kernels
kernel_tg = 'truncnorm'
events_tg = simu_hawkes_cluster(T,
                                baseline,
                                alpha,
                                kernel_tg,
                                params_kernel=params_tg, 
                                random_state=simu_random_state)

# %% Fit Hawkes process to truncated_gaussian simulation
testmaskedsolver(kernel='truncated_gaussian', events=events_tg, T=T,
                 max_iter=max_iter,
                 baseline_mask=baseline_mask,
                 alpha_init=alpha_init, alpha_mask=alpha_mask,
                 ztzG_approx=ztzG_approx)
print('Simulation baseline:', baseline)
print('Simulation alpha:', alpha)
