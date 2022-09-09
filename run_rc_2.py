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
        baseline=baseline, kernels=kernels, end_time=T, verbose=False, seed=seed
    )

    hawkes.simulate()
    events = hawkes.timestamps
    return events


events = simulate_data(baseline, alpha, mu, sigma, T, dt)

# %% solver
##


@mem.cache
def run_solver(events, u_init, sigma_init, baseline_init, alpha_init, dt, T, seed=0):
    start = time.time()
    max_iter = 2000
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

def run_experiment(baseline, alpha, mu, sigma, T, dt, seed=0):
    events = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=seed)
    baseline_init = baseline + np.random.rand()*0.5
    alpha_init = alpha + np.random.rand()*0.5
    mu_init = mu + np.random.rand()*0.5
    sigma_init = sigma + np.random.rand()*0.2
    u_init = mu_init - sigma_init 
    results = run_solver(events, u_init, sigma_init, baseline_init, alpha_init, dt, T)
    return results

T_list = [100_000, 200_000, 300_000, 400_000, 500_000]
dt_list = [0.1, 0.01]
seeds = np.arange(10)
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    #delayed(run_experiment)(baseline, alpha, mu, sigma, T, dt, seed=seed)
    delayed(run_solver)(
        events, u_init, sigma_init, baseline_init, alpha_init, dt, T, seed=seed
    )
    for T, dt, seed in itertools.product(
        T_list, dt_list, seeds
    )
)
file_name = "benchmark.pkl"
open_file = open(file_name, "wb")
pickle.dump(all_results, open_file)
open_file.close()
# %% name
n_T = len(T_list)
n_dt = len(dt_list) 
n_seeds = len(seeds)
n_xp = n_T * n_dt * n_seeds

# %% plot loss
%matplotlib inline
matplotlib.rc("xtick", labelsize=13)
matplotlib.rc("ytick", labelsize=13)
lw = 5
fontsize = 18
fig, axs = plt.subplots(n_T, n_dt, figsize=(15, 10))


for i in range(n_T):
    for j in range(n_dt):
        for l in range(n_seeds):
            idx = i*n_seeds + j*n_seeds + l
            axs[i, j].plot(all_results[idx]["v_loss"], lw=lw, label='seed={}'.format(seeds[l]))
            axs[i, j].set_title("model: T={}, dt={} ".format(T_list[i], dt_list[j]), size=fontsize)

fig.tight_layout()

# %% plot param

matplotlib.rc("xtick", labelsize=13)
matplotlib.rc("ytick", labelsize=13)
lw = 5
fontsize = 18
fig, axs = plt.subplots(n_T, n_dt, figsize=(15, 10))

for i in range(n_T):
    for j in range(n_dt):
        for l in range(n_seeds):
            idx = j*n_T + i* + l*n_seeds
            axs[i, j].plot(all_results__[idx]["param_adjacency"], lw=lw, label='seed={}'.format(seeds[l]))
            axs[i, j].set_title("model: T={}, dt={} ".format(T_list[i]).format(dt_list[j]), size=fontsize)

fig.tight_layout()


# %% plot
#% matplotlib inline
import matplotlib


def plot_results(results):
    matplotlib.rc("xtick", labelsize=13)
    matplotlib.rc("ytick", labelsize=13)
    lw = 5
    fontsize = 18
    n_dim = 1
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))

    for i in range(n_dim):
        axs[0, 0].plot(results["grad_baseline"], lw=lw)
        axs[0, 0].set_title("grad_baseline", size=fontsize)

        axs[1, 0].plot(results["param_baseline_e"], lw=lw)
        axs[1, 0].set_title("mu", size=fontsize)

        for j in range(n_dim):
            axs[0, 1].plot(results["grad_adjacency"][:, i, j], lw=lw, label=(i, j)
            )
            axs[0, 1].set_title("grad_alpha", size=fontsize)
            axs[0, 1].legend(fontsize=fontsize - 5)

            axs[0, 2].plot(grad_u[:, i, j], lw=lw)
            axs[0, 2].set_title("grad_u", size=fontsize)

            axs[0, 3].plot(grad_sigma[:, i, j], lw=lw)
            axs[0, 3].set_title("grad_sigma", size=fontsize)

            axs[1, 1].plot(param_adjacency_e[:, i, j], lw=lw)
            axs[1, 1].set_title("alpha", size=fontsize)

            axs[1, 2].plot(param_u_e[:, i, j], lw=lw)
            axs[1, 2].set_title("u", size=fontsize)

            axs[1, 3].plot(param_sigma_e[:, i, j], lw=lw)
            axs[1, 3].set_title("sigma", size=fontsize)

    plt.figure(figsize=(12, 4))

    plt.tight_layout()
    plt.plot(epochs, loss, lw=10)
    plt.title("Loss", size=fontsize)


# %%
