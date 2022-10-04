# %% import stuff
## import libraries
from ast import increment_lineno
import itertools
import pickle
import time
import numpy as np
import torch
from joblib import Memory, Parallel, delayed
import matplotlib.pyplot as plt
from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc

from hawkes_discret.kernels import KernelTruncatedGaussianDiscret
from hawkes_discret.hawkes_discret_l2 import HawkesDiscretL2


################################
## Meta parameters
################################

dt = 0.01
T = 10_000_000
size_grid = int(T / dt) + 1

mem = Memory(location=".", verbose=2)

# %% simulate data
# Simulated data
################################

baseline = np.array([.1])
alpha = np.array([[0.8]])
m = np.array([[0.5]])
sigma = np.array([[0.3]])

#@mem.cache
def simulate_data(baseline, alpha, m, sigma, T, dt, seed=0):
    L = int(1 / dt)
    discretization = torch.linspace(0, 1, L)
    TG = KernelTruncatedGaussianDiscret(0, 1, dt)
    kernel_values = TG.eval(
        [torch.Tensor(m), torch.Tensor(sigma)], discretization
    )  # * dt
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

events = simulate_data(baseline, alpha, m, sigma, T, dt, seed=0)


#@mem.cache
def run_solver(events, m_init, sigma_init, baseline_init, alpha_init, T, dt, seed=0):
    start = time.time()
    max_iter = 2000
    solver = HawkesDiscretL2(
        "TruncatedGaussian",
        [torch.tensor(m_init),
        torch.tensor(sigma_init)],
        torch.tensor(baseline_init),
        torch.tensor(alpha_init),
        dt,
        solver="RMSprop",
        step_size=1e-3,
        max_iter=max_iter,
        log=False,
        random_state=0,
        device="cpu",
        optimize_kernel=True
    )
    print(time.time()-start)
    results = solver.fit(events, T)

    results_ = dict(param_baseline=results['param_baseline'][-10:].mean().item(),
                    param_adjacency=results['param_adjacency'][-10:].mean().item(),
                    param_kernel=[results['param_kernel'][0][-10:].mean().item(),
                                  results['param_kernel'][1][-10:].mean().item()])
    results_["time"] = time.time() - start
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = dt
    return results_

np.random.seed(0)
baseline_init = np.array([np.random.rand()*0.5])
alpha_init = np.array([[np.random.rand()]])
m_init = np.array([[np.random.rand()]])
sigma_init = np.array([[np.random.rand()*0.5]])

start = time.time()
results_1 = run_solver(
    events, m_init, sigma_init, baseline_init, alpha_init, T, dt, seed=0
)
baseline_our = results_1['param_baseline']
adjacency_our = results_1['param_adjacency']
print(np.abs(results_1['param_baseline'])) #- baseline))
print(np.abs(results_1['param_adjacency'])) #- alpha))
print(np.abs(results_1['param_kernel'][0] ))#- decay))
print(np.abs(results_1['param_kernel'][1] ))
# %%

# %%
# %% eval on grid
##
def run_experiment(baseline, alpha, m, sigma, T, dt, seed=0):
    v =  0.2
    events = simulate_data(baseline, alpha, m, sigma, T, dt, seed=seed)
    baseline_init = baseline + v 
    alpha_init = alpha + v 
    m_init = m + v 
    sigma_init = sigma - v 
    results = run_solver(events, m_init, sigma_init, baseline_init, alpha_init, T, dt, seed)

    return results

T_list = [1000, 10_000, 100_000, 1_000_000]
dt_list = np.logspace(1, 3, 10) / 10e3
seeds = np.arange(100)
info = dict(T_list=T_list, dt_list=dt_list, seeds=seeds)

n_jobs=40
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, alpha, m, sigma, T, dt, seed=seed)
    for T, dt, seed in itertools.product(
        T_list, dt_list, seeds
    )
)
all_results.append(info)
file_name = "results/error_discrete_TG.pkl"
open_file = open(file_name, "wb")
pickle.dump(all_results, open_file)
open_file.close()