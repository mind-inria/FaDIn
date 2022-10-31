# %% import stuff
# import libraries
import time
import numpy as np
import torch
import pandas as pd
from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc

from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.solver import FaDIn

################################
# Meta parameters
################################
dt = 0.01
L = int(1 / dt)
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
    k11 = kernel_values[0, 0].double().numpy()
    k12 = kernel_values[0, 1].double().numpy()
    k21 = kernel_values[1, 0].double().numpy()
    k22 = kernel_values[1, 1].double().numpy()

    tf11 = HawkesKernelTimeFunc(t_values=t_values, y_values=k11)
    tf12 = HawkesKernelTimeFunc(t_values=t_values, y_values=k12)
    tf21 = HawkesKernelTimeFunc(t_values=t_values, y_values=k21)
    tf22 = HawkesKernelTimeFunc(t_values=t_values, y_values=k22)
    
    kernels = [[tf11, tf12], [tf21, tf22]]
    hawkes = SimuHawkes(
        baseline=baseline, kernels=kernels, end_time=T, verbose=False, seed=int(seed)
    )

    hawkes.simulate()
    events = hawkes.timestamps
    return events


# @mem.cache
def run_solver(events, u_init, sigma_init, baseline_init, alpha_init, dt, T, 
               side_effects, seed=0):
    start = time.time()
    max_iter = 1000
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
                    optimize_kernel=True,
                    precomputations=True,
                    side_effects=side_effects)
    print(time.time() - start)
    results = solver.fit(events, T)
    results_ = dict(param_baseline=results['param_baseline'][-10:].mean(0),
                    param_alpha=results['param_alpha'][-10:].mean(0),
                    param_kernel=[results['param_kernel'][0][-10:].mean(0),
                                  results['param_kernel'][1][-10:].mean(0)])
    results_["time"] = time.time() - start
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = dt
    return results_
# %% Run experiment


baseline = np.array([.1, .2,])
alpha = np.array([[1.5, 0.1], [0.1, 1.5]])
mu = np.array([[0.4, 0.6], [0.55, 0.6]])
sigma = np.array([[0.3, 0.3], [0.25, 0.3]])
u = mu - sigma
events = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=1)

print("events of the first process: ", events[0].shape[0])
print("events of the second process: ", events[1].shape[0])


v = 0.2
baseline_init = baseline + v
alpha_init = alpha + v
mu_init = mu 
sigma_init = sigma + v
u_init = mu_init - sigma_init
side_effects = False
results = run_solver(events, u_init, sigma_init,
                     baseline_init, alpha_init,
                     dt, T, side_effects, seed=0)
print(results)
# df = pd.DataFrame([results])
# df.to_csv('results/test_autodiff_multivariate.csv', index=False)
# %%
n_dim = 2
L = 10
a = torch.randn(n_dim, n_dim, L)
a_ = torch.randn(n_dim, n_dim, L)
b = torch.randn(n_dim, n_dim, L, L)

res1 = torch.zeros(n_dim, n_dim, n_dim)
res1_ = torch.zeros(n_dim, n_dim, n_dim)
res2 = torch.zeros(n_dim, n_dim, n_dim)
res3 = torch.zeros(n_dim, n_dim, n_dim)
for i in range(n_dim):
    for k in range(n_dim):
        for j in range(n_dim):
            res1[i, j, k] = (a[i, k].view(1, L) * (b[j, k] * a_[i, j].view(L, 1)).sum(0)).sum()
            res1_[i, j, k] = (a_[i, j].view(1, L) * (b[k, j] * a[i, k].view(L, 1)).sum(0)).sum()
            temp = 0
            temp2 = 0
            for tau in range(L):
                temp2 += a[i, j, tau] * (a[i, k] @ b[j, k, tau])
                for tau_p in range(L):
                    temp += (a[i, j, tau] * a[i, k, tau_p]) * b[j, k, tau, tau_p]
            res2[i, j, k] = temp
            res3[i, j, k] = temp2
print(res1 == res1_)
#### test grad

n_dim = 2
L = 10
a = torch.randn(n_dim, n_dim, L)
a_ = torch.randn(n_dim, n_dim, L)
b = torch.randn(n_dim, n_dim, L, L)

res1 = torch.zeros(n_dim, n_dim, n_dim)
res2 = torch.zeros(n_dim, n_dim, n_dim)
res1_ = torch.zeros(n_dim, n_dim, n_dim)
res2_ = torch.zeros(n_dim, n_dim, n_dim)
for m in range(n_dim):
    for n in range(n_dim):
        for k in range(n_dim):
            res1[m, n, k] = (a[m, k].view(1, L) * (b[n, k] * a_[m, n].view(L, 1)).sum(0)).sum()
            res2[m, n, k] = (a_[m, n].view(1, L) * (b[k, n] * a[m, k].view(L, 1)).sum(0)).sum()
            temp0 = 0
            temp1 = 0
            for tau in range(L):
                for taup in range(L):
                    temp0 += a_[m, n, tau] * a[m, k, taup] * b[n, k, tau, taup]
                    temp1 += a_[m, n, taup] * a[m, k, tau] * b[k, n, tau, taup]
            res1_[m, n, k] = temp0
            res2_[m, n, k] = temp1

# %%
