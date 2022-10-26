# %% import stuff
# import libraries
import time
import numpy as np
import torch
from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc

from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.solver import FaDIn

################################
# Meta parameters
################################
dt = 0.01
T = 100000
size_grid = int(T / dt) + 1

# mem = Memory(location=".", verbose=2)

# %% Experiment
################################


# @mem.cache
def simulate_data(baseline, alpha, mu, sigma, T, dt, seed=0):
    L = int(1 / dt)
    discretization = torch.linspace(0, 1, L)
    u = mu - sigma
    n_dim = u.shape[0]
    RC = DiscreteKernelFiniteSupport(0, 1, dt, kernel='RaisedCosine', n_dim=n_dim)
    kernel_values = RC.eval(
        [torch.Tensor(u), torch.Tensor(sigma)], discretization
    )
    kernel_values = kernel_values * alpha[:, :, None]

    t_values = discretization.double().numpy()
    k = kernel_values[0, 0].double().numpy()

    tf = HawkesKernelTimeFunc(t_values=t_values, y_values=k)
    kernels = [[tf]]
    hawkes = SimuHawkes(
        baseline=baseline, kernels=kernels, end_time=T, verbose=False, seed=int(seed)
    )

    hawkes.simulate()
    events = hawkes.timestamps
    return events

# %% solver
##


# @mem.cache
def run_solver(events, u_init, sigma_init, baseline_init, alpha_init, dt, T, seed=0):
    start = time.time()
    max_iter = 2000
    solver = FaDIn("RaisedCosine",
                             [torch.tensor(u_init),
                              torch.tensor(sigma_init)],
                             torch.tensor(baseline_init),
                             torch.tensor(alpha_init),
                             dt, solver="RMSprop",
                             step_size=1e-3,
                             max_iter=max_iter,
                             log=False,
                             random_state=0,
                             device="cpu",
                             optimize_kernel=True)
    print(time.time() - start)
    results = solver.fit(events, T)
    results_ = dict(param_baseline=results['param_baseline'][-10:].mean().item(),
                    param_alpha=results['param_alpha'][-10:].mean().item(),
                    param_kernel=[results['param_kernel'][0][-10:].mean().item(),
                                  results['param_kernel'][1][-10:].mean().item()])
    results_["time"] = time.time() - start
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = dt
    return results_

# %% Test


baseline = np.array([1.1])
alpha = np.array([[0.8]])
mu = np.array([[0.5]])
sigma = np.array([[0.3]])
u = mu - sigma

events = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=0)

v = 0.2
baseline_init = baseline + v
alpha_init = alpha + v
mu_init = mu + v
sigma_init = sigma - v
u_init = mu_init - sigma_init
results = run_solver(events, u_init, sigma_init,
                     baseline_init, alpha_init,
                     dt, T, seed=0)

print(np.abs(results['param_baseline'] - baseline))
print(np.abs(results['param_alpha'] - alpha))
print(np.abs(results['param_kernel'][0] - u))
print(np.abs(results['param_kernel'][1] - sigma))

# %% try kernel functions
