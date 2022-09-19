# %% import stuff
## import libraries
import itertools
import pickle
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from joblib import Memory, Parallel, delayed

from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc, HawkesExpKern, HawkesKernelExp

from hawkes_discret.kernels import KernelRaisedCosineDiscret
from hawkes_discret.hawkes_discret_l2 import HawkesDiscretL2

################################
## Meta parameters
################################
dt = 0.01
T = 100000
size_grid = int(T / dt) + 1

mem = Memory(location=".", verbose=2)

# %% simulate data
# Simulated data
################################

baseline = np.array([1.1])
alpha = np.array([[0.8]])
mu = np.array([[0.5]])
sigma = np.array([[0.3]])
u = mu - sigma

@mem.cache
def simulate_data(baseline, alpha, mu, sigma, T, dt, seed=0):
    L = int(1 / dt)
    discretization = torch.linspace(0, 1, L)
    u = mu - sigma
    RC = KernelRaisedCosineDiscret(dt)
    kernel_values = RC.eval(
        [torch.Tensor(u), torch.Tensor(sigma)], discretization
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

@mem.cache
def run_solver(events, u_init, sigma_init, baseline_init, alpha_init, dt, T, seed=0):
    start = time.time()
    max_iter = 800
    solver = HawkesDiscretL2(
        "RaisedCosine",
        [torch.tensor(u_init),
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
    results["time"] = time.time() - start
    results["seed"] = seed
    results["T"] = T
    results["dt"] = dt
    return results

def run_experiment(baseline, alpha, mu, sigma, T, dt, seed=0):
    v =  0.2
    events = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=seed)
    baseline_init = baseline + v 
    alpha_init = alpha + v 
    mu_init = mu + v 
    sigma_init = sigma - v 
    u_init = mu_init - sigma_init 
    results = run_solver(events, u_init, sigma_init, baseline_init, alpha_init, dt, T, seed)
    return results

T_list = [1000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
dt_list = [0.01]
seeds = np.arange(100)
info = dict(T_list=T_list, dt_list=dt_list, seeds=seeds)

n_jobs=40
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, alpha, mu, sigma, T, dt, seed=seed)
    for T, dt, seed in itertools.product(
        T_list, dt_list, seeds
    )
)
all_results.append(info)
file_name = "results/error_stat.pkl"
open_file = open(file_name, "wb")
pickle.dump(all_results, open_file)
open_file.close()
1/0
# %% get results
##
#file_name = "erreur_stat.pkl"
#open_file = open(file_name, "rb")
#all_results = pickle.load(open_file)
#open_file.close()
