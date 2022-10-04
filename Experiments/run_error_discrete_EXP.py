# %% import stuff
## import libraries
from ast import increment_lineno
import itertools
import pickle
import time
import numpy as np
import torch
from joblib import Memory, Parallel, delayed
from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc

from hawkes_discret.kernels import KernelExpDiscret
from hawkes_discret.hawkes_discret_l2 import HawkesDiscretL2


################################
## Meta parameters
################################

dt = 0.01
T = 10_000_000
size_grid = int(T / dt) + 1

#mem = Memory(location=".", verbose=2)

# %% simulate data
# Simulated data
################################

baseline = np.array([.1])
alpha = np.array([[0.8]])
decay = np.array([[5]])

#@mem.cache
def simulate_data(baseline, alpha, decay, T, dt, seed=0):
    L = int(1 / dt)
    discretization = torch.linspace(0, 1, L)
    Exp = KernelExpDiscret(0, 1, dt)
    kernel_values = Exp.eval(
        [torch.Tensor(decay)], discretization
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

events = simulate_data(baseline, alpha, decay, T, dt, seed=0)


#@mem.cache
def run_solver(events, decay_init, baseline_init, alpha_init, T, dt, seed=0):
    start = time.time()
    max_iter = 2000
    solver = HawkesDiscretL2(
        "KernelExpDiscret",
        [torch.tensor(decay_init)],
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
                    param_kernel=[results['param_kernel'][0][-10:].mean().item()])
    results_["time"] = time.time() - start
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = dt
    return results_

# %%

# %%
# %% eval on grid
##
def run_experiment(baseline, alpha, decay, T, dt, seed=0):
    v =  0.2
    events = simulate_data(baseline, alpha, decay, T, dt, seed=seed)
    baseline_init = baseline + v 
    alpha_init = alpha + v 
    decay_init = decay + v

    results = run_solver(events, decay_init, baseline_init, alpha_init, T, dt, seed)

    return results

T_list = [1000, 10_000, 100_000, 1_000_000]
dt_list = np.logspace(1, 3, 10) / 10e3
seeds = np.arange(100)
info = dict(T_list=T_list, dt_list=dt_list, seeds=seeds)

n_jobs=40
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, alpha, decay, T, dt, seed=seed)
    for T, dt, seed in itertools.product(
        T_list, dt_list, seeds
    )
)
all_results.append(info)
file_name = "results/error_discrete_EXP.pkl"
open_file = open(file_name, "wb")
pickle.dump(all_results, open_file)
open_file.close()