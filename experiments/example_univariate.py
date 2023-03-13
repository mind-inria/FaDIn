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
T = 1_000_000
kernel_length = 1.5
size_grid = int(T / dt) + 1

# mem = Memory(location=".", verbose=2)

# %% Experiment
################################


# @mem.cache
def simulate_data(baseline, alpha, a, b, kernel_length, T, dt, seed=0):
    L = int(kernel_length / dt)
    discretization = torch.linspace(0, kernel_length, L)
    n_dim = a.shape[0]
    kuma = DiscreteKernelFiniteSupport(dt, n_dim, kernel='kumaraswamy')
    kernel_values = kuma.kernel_eval([torch.Tensor(a), torch.Tensor(b)],
                                     discretization)
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


# @mem.cache
def run_solver(events, u_init, sigma_init, baseline_init,
               alpha_init, kernel_length, dt, T, seed=0):
    start = time.time()
    max_iter = 10000
    solver = FaDIn(1,
                   "kumaraswamy",
                   [torch.tensor(u_init),
                    torch.tensor(sigma_init)],
                   torch.tensor(baseline_init),
                   torch.tensor(alpha_init),
                   kernel_length=kernel_length,
                   delta=dt, optim="RMSprop",
                   max_iter=max_iter, criterion='l2'
                   )

    print(time.time() - start)
    solver.fit(events, T)
    results_ = dict(param_baseline=solver.param_baseline[-10:].mean().item(),
                    param_alpha=solver.param_alpha[-10:].mean().item(),
                    param_kernel=[solver.param_kernel[0][-10:].mean().item(),
                                  solver.param_kernel[1][-10:].mean().item()]
                    )

    results_["time"] = time.time() - start
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = dt
    return results_

# %% Test


baseline = np.array([.1])
alpha = np.array([[0.8]])
middle = kernel_length / 2
a = np.array([[2.]])
b = np.array([[2.]])


events = simulate_data(baseline, alpha, a, b, kernel_length, T, dt, seed=0)

v = 0.2
baseline_init = baseline + v
alpha_init = alpha + v
a_init = np.array([[1.]])
b_init = np.array([[1.]])

results = run_solver(events, a_init, b_init,
                     baseline_init, alpha_init, 1.5,
                     dt, T, seed=0)

print(np.abs(results['param_baseline'] - baseline))
print(np.abs(results['param_alpha'] - alpha))
print(np.abs(results['param_kernel'][0] - a))
print(np.abs(results['param_kernel'][1] - b))

# %%
import matplotlib.pyplot as plt
%matplotlib inline
kernel_length = 1.5
n_dim = 1
L = int(kernel_length / dt)
discretization = torch.linspace(0, kernel_length, L)
kuma = DiscreteKernelFiniteSupport(dt, n_dim, kernel='kumaraswamy')
kernel_values = kuma.kernel_eval([torch.Tensor(a), torch.Tensor(b)],
                                     discretization)

kuma_ = DiscreteKernelFiniteSupport(dt, n_dim, kernel='kumaraswamy')
kernel_values_ = kuma_.kernel_eval([torch.Tensor([[results['param_kernel'][0]]]), torch.tensor([[results['param_kernel'][1]]])], discretization)

plt.plot(kernel_values.squeeze(), label='true kernel')
plt.plot(kernel_values_.squeeze(), label='estimated kernel')
plt.legend()
# %%
