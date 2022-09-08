# %% import stuff
## import libraries
import itertools
import pickle
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
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
    return results


v = 0.15
baseline_init = baseline + v
alpha_init = alpha + v
sigma_init = sigma + v
u_init = mu - sigma + v

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
    v = 0.15
    baseline_init = baseline + v
    alpha_init = alpha + v
    sigma_init = sigma + v
    u_init = mu - sigma + v
    results = run_solver(events, u_init, sigma_init, baseline_init, alpha_init, dt, T)
    return results

T_list = [100_000, 200_000, 300_000, 400_000, 500_000]
seeds = np.arange(10)
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_solver)(
        events, u_init, sigma_init, baseline_init, alpha_init, dt, T, seed=seed
    )
    for T, seed in itertools.product(
        T_list, seeds
    )
)

# %% name
# loss = results_1[0]
# grad_baseline = results_1[1]
# grad_adjacency = results_1[2]
# grad_u = results_1[3]
# grad_sigma = results_1[4]
# param_baseline_e = torch.abs(results_1[5])  # - baseline)
# param_adjacency_e = torch.abs(results_1[6])  # - adjacency)
# param_u_e = torch.abs(results_1[7])  # - u)
# param_sigma_e = torch.abs(results_1[8])  # - sigma)
# epochs = torch.arange(max_iter)
# epochss = torch.arange(max_iter + 1)

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
        axs[0, 0].plot(epochs, results["grad_baseline"], lw=lw)
        axs[0, 0].set_title("grad_baseline", size=fontsize)

        axs[1, 0].plot(epochss, results["param_baseline_e"], lw=lw)
        axs[1, 0].set_title("mu", size=fontsize)

        for j in range(n_dim):
            axs[0, 1].plot(
                epochs, results["grad_adjacency"][:, i, j], lw=lw, label=(i, j)
            )
            axs[0, 1].set_title("grad_alpha", size=fontsize)
            axs[0, 1].legend(fontsize=fontsize - 5)

            axs[0, 2].plot(epochs, grad_u[:, i, j], lw=lw)
            axs[0, 2].set_title("grad_u", size=fontsize)

            axs[0, 3].plot(epochs, grad_sigma[:, i, j], lw=lw)
            axs[0, 3].set_title("grad_sigma", size=fontsize)

            axs[1, 1].plot(epochss, param_adjacency_e[:, i, j], lw=lw)
            axs[1, 1].set_title("alpha", size=fontsize)

            axs[1, 2].plot(epochss, param_u_e[:, i, j], lw=lw)
            axs[1, 2].set_title("u", size=fontsize)

            axs[1, 3].plot(epochss, param_sigma_e[:, i, j], lw=lw)
            axs[1, 3].set_title("sigma", size=fontsize)

    plt.figure(figsize=(12, 4))

    plt.tight_layout()
    plt.plot(epochs, loss, lw=10)
    plt.title("Loss", size=fontsize)


# %%
