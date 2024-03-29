# %% import stuff
# import libraries
import itertools
import time
import pickle
import numpy as np
import torch
from joblib import Parallel, delayed
from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc
from tick.hawkes import HawkesBasisKernels

from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.solver import FaDIn


# %% simulate data
# Simulated data
################################

def simulate_data(n_dim, T, dt, seed=0):
    L = int(1 / dt)
    discretization = torch.linspace(0, 1, L)

    baseline = 0.1 + np.zeros(n_dim)

    alpha = 0.001 + np.zeros((n_dim, n_dim))
    decay = 5 + np.zeros((n_dim, n_dim))

    Exp = DiscreteKernelFiniteSupport(dt, n_dim, kernel='truncated_exponential')
    kernel_values = Exp.kernel_eval([torch.Tensor(decay)], discretization)
    kernel_values = kernel_values * alpha[:, :, None]

    t_values = discretization.double().numpy()
    ker = []
    for i in range(n_dim):
        for j in range(n_dim):
            k = kernel_values[i, j].double().numpy()
            tf = HawkesKernelTimeFunc(t_values=t_values, y_values=k)
            ker.append(tf)

    kernels = [[ker[j * n_dim + i] for i in range(n_dim)] for j in range(n_dim)]
    hawkes = SimuHawkes(
        baseline=baseline, kernels=kernels, end_time=T, verbose=False, seed=int(seed)
    )

    hawkes.simulate()
    events = hawkes.timestamps
    return events


# %%
# %% solver

def run_solver(events, decay_init, baseline_init, alpha_init, T, dt, seed=0):
    start = time.time()
    max_iter = 800
    n_dim = baseline_init.shape[0]
    solver = FaDIn(n_dim,
                   "truncated_exponential",
                   [torch.tensor(decay_init)],
                   torch.tensor(baseline_init),
                   torch.tensor(alpha_init),
                   delta=dt, optim="RMSprop",
                   step_size=1e-3,
                   max_iter=max_iter,
                   log=False,
                   random_state=0,
                   device="cpu",
                   optimize_kernel=True,
                   precomputations=True,
                   ztzG_approx=True)

    print(time.time() - start)
    results = solver.fit(events, T)
    results_ = dict(param_baseline=results['param_baseline'][-10:].mean(0),
                    param_alpha=results['param_alpha'][-10:].mean(0),
                    param_kernel=[results['param_kernel'][0][-10:].mean(0)])
    results_["time"] = time.time() - start
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = dt
    return results_


def run_experiment(n_dim, T, dt, seed=0):
    events = simulate_data(n_dim, T, dt, seed=seed)
    baseline_init = np.array([np.random.rand()])
    alpha_init = np.array([[np.random.rand()]])
    decay_init = np.array([[np.random.rand()]])

    start_our = time.time()
    run_solver(events, decay_init, baseline_init, alpha_init,
               T, dt, seed=0)
    time_our = time.time() - start_our

    start_tick = time.time()
    non_param = HawkesBasisKernels(1, n_basis=1, kernel_size=int(1 / dt),
                                   max_iter=800, ode_tol=1e-15)
    non_param.fit(events)
    time_tick = time.time() - start_tick

    res_our = dict(comp_time=time_our, n_dim=n_dim, T=T, dt=dt, seed=seed)

    res_tick = dict(comp_time=time_tick, n_dim=n_dim, T=T, dt=dt, seed=seed)

    return res_our, res_tick


n_dim = 10
dt = 0.01
T = 1000

us, tick = run_experiment(n_dim, T, dt, seed=0)

print("us is:", us['comp_time'])
print("tick is:", tick['comp_time'])
# %% run


n_dim_list = [2, 5, 10, 50, 100]
dt_list = [0.1]
T_list = [100_000, 1_000_000]
seeds = np.arange(10)

info = dict(n_dim_list=n_dim_list, T_list=T_list, dt_list=dt_list, seeds=seeds)

n_jobs = 70
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(n_dim, T, dt, seed=seed)
    for n_dim, T, dt, seed in itertools.product(
        n_dim_list, T_list, dt_list, seeds
    )
)


all_results.append(info)
file_name = "results/comp_time_dim.pkl"
open_file = open(file_name, "wb")
pickle.dump(all_results, open_file)
open_file.close()

# %%
