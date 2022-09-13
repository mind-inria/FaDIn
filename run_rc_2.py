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

from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc

from hawkes_discret.kernels import KernelRaisedCosineDiscret
from hawkes_discret.hawkes_discret_l2 import HawkesDiscretL2

from hawkes_discret.utils.compute_constants_np import get_ztzG, _get_ztzG
from hawkes_discret.utils.utils import projected_grid
################################
## Meta parameters
################################
dt = 0.01
T = 100_000
n_jobs = 30
size_grid = int(T / dt) + 1

mem = Memory(location=".", verbose=2)

# %% simulate data
# Simulated data
################################

baseline = np.array([1.1])
alpha = np.array([[0.8]])
mu = np.array([[0.5]])
sigma = np.array([[0.3]])


@mem.cache
def simulate_data(baseline, alpha, mu, sigma, T, dt, seed=0):
    L = int(1 / dt)
    discretization = torch.linspace(0, 1, L)
    u = mu - sigma
    RC = KernelRaisedCosineDiscret(dt)
    kernel_values = RC.eval(
        torch.Tensor(u), torch.Tensor(sigma), discretization
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


events = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=0)

# %% solver
##


@mem.cache
def run_solver(events, u_init, sigma_init, baseline_init, alpha_init, dt, T, seed=0):
    start = time.time()
    max_iter = 1000
    solver = HawkesDiscretL2(
        "RaisedCosine",
        torch.tensor(u_init),
        torch.tensor(sigma_init),
        torch.tensor(baseline_init),
        torch.tensor(alpha_init),
        dt,
        solver="RMSprop",
        step_size=1e-3,
        max_iter=max_iter,
        log=False,
        random_state=0,
        device="cpu",
    )
    results = solver.fit(events, T)
    results["time"] = time.time() - start
    results["seed"] = seed
    results["T"] = T
    results["dt"] = dt
    return results



baseline_init = baseline + np.random.rand()*0.5
alpha_init = alpha + np.random.rand()*0.5
mu_init = mu + np.random.rand()*0.5
sigma_init = sigma + np.random.rand()*0.2
u_init = mu - sigma 

results_1 = run_solver(
    events, u_init, sigma_init, baseline_init, alpha_init, dt, T, seed=0
)

file_name = "test2.pkl"
open_file = open(file_name, "wb")
pickle.dump(results_1, open_file)
open_file.close()

# %% eval on grid
##
def run_experiment(baseline, alpha, mu, sigma, T, dt, seed=0):
    v =  0.15
    events = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=seed)
    baseline_init = baseline + v #np.random.rand()*0.5
    alpha_init = alpha + v #np.random.rand()*0.5
    mu_init = mu + v #np.random.rand()*0.5
    sigma_init = sigma - v #np.random.rand()*0.2
    u_init = mu_init - sigma_init 
    results = run_solver(events, u_init, sigma_init, baseline_init, alpha_init, dt, T, seed)
    return results

T_list = [1000]
dt_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02,  
0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
seeds = np.arange(100)
info = dict(T_list=T_list, dt_list=dt_list, seeds=seeds)
n_jobs=30
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, alpha, mu, sigma, T, dt, seed=seed)
    for T, dt, seed in itertools.product(
        T_list, dt_list, seeds
    )
)
all_results.append(info)
file_name = "benchmark3.pkl"
open_file = open(file_name, "wb")
pickle.dump(all_results, open_file)
open_file.close()

##############################################################################
"""
L = int(1/dt)
events_ = projected_grid(events, dt, size_grid)
start = time.time()
#q = _get_ztzG(events_.numpy(), L)
qq, _ = get_ztzG(events_.numpy(), L)
print('premiere:', time.time() - start)
def strided_method(ar):
    a = np.concatenate(( ar, ar ))
    L = len(ar)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a[L:], (L,L), (-n,n))


def test(events, n_discrete):
    n_dim, _ = events.shape

    ztzG = np.zeros(shape=(n_dim, n_dim,
                           n_discrete,
                           n_discrete))
    for i in range(n_dim):
        ei = events[i]
        for j in range(n_dim):
            ej = events[j]
            #eij = ei * ej
            #eij_sum = eij[n_discrete:-n_discrete].sum()
            temp = np.zeros(n_discrete)
            temp[0] =  ei @ ej        
            for tau in range(1, n_discrete):
                temp[tau] = ei[:-tau] @ ej[tau:]
            temp_ = strided_method(temp)
            ztzG[i, j] = np.triu(temp_) + np.triu(temp_, 1).T
            #for tau in range(n_discrete):
            #    for tau_p in range(n_discrete):
            #        idx = np.absolute(tau-tau_p)
            #        ztzG[i, j, tau, tau_p] = temp[idx]

    return ztzG
start = time.time()
ztzG_test = test(events_.numpy(), L)
print("deuxième", time.time() - start)
start = time.time()
ztzG_test2 = _test2(events_.numpy(), L)
print("troisième:", time.time() - start)
"""