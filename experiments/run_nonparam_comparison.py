# %% import stuff
# import libraries
import itertools
import pickle
import time
import numpy as np
import torch
from joblib import Memory, Parallel, delayed
from scipy.stats import skewnorm

from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc
from tick.hawkes import HawkesBasisKernels

from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.solver import FaDIn

# %% simulate data
# Simulated data
################################
dt = 0.01
T = 10000
n_jobs = 30
size_grid = int(T / dt) + 1

mem = Memory(location=".", verbose=2)

baseline = np.array([1.1])
alpha = np.array([[0.8]])
mu = np.array([[0.5]])
sigma = np.array([[0.3]])
u = mu - sigma


@mem.cache
def simulate_data(baseline, alpha, mu, sigma, T, dt, seed=0, kernel='RC'):
    L = int(1 / dt)
    discretization = torch.linspace(0, 1, L)
    u = mu - sigma
    if kernel == 'RC':
        RC = DiscreteKernelFiniteSupport(0, 1, dt, kernel='RaisedCosine')
        kernel_values = RC.eval(
            [torch.Tensor(u), torch.Tensor(sigma)], discretization
        )  # * dt
        kernel_values = kernel_values * alpha[:, :, None]
        k = kernel_values[0, 0].double().numpy()
    elif kernel == 'SG':
        kernel_values = torch.tensor(skewnorm.pdf(np.linspace(-3, 3, L), 3))
        kernel_values = kernel_values * alpha[:, :, None]
        k = kernel_values.squeeze().numpy()

    t_values = discretization.double().numpy()

    tf = HawkesKernelTimeFunc(t_values=t_values, y_values=k)
    kernels = [[tf]]
    hawkes = SimuHawkes(
        baseline=baseline, kernels=kernels, end_time=T, verbose=False, seed=int(seed)
    )

    hawkes.simulate()
    events = hawkes.timestamps
    return events, hawkes


events, hawkes = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=1, kernel='SG')

# %% solver
##


@mem.cache
def run_solver(events, u_init, sigma_init, baseline_init, alpha_init, T, dt, seed=0):
    max_iter = 800
    solver = FaDIn("RaisedCosine",
                   [torch.tensor(u_init),
                    torch.tensor(sigma_init)],
                   torch.tensor(baseline_init),
                   torch.tensor(alpha_init),
                   dt,
                   solver="RMSprop",
                   step_size=1e-3,
                   max_iter=max_iter,
                   log=False,
                   random_state=seed,
                   device="cpu",
                   optimize_kernel=True)
    results = solver.fit(events, T)
    return results


# %%

def run_experiment(baseline, alpha, mu, sigma, T, dt, seed=0, kernel='RC'):
    events, _ = simulate_data(baseline, alpha, mu,
                              sigma, T, dt, seed=seed,
                              kernel=kernel)
    baseline_init = np.array([np.random.rand()])
    alpha_init = np.array([[np.random.rand()]])
    mu_init = np.array([[np.random.rand()]])
    sigma_init = 10
    while (sigma_init > mu_init):
        sigma_init = np.array([[np.random.rand()]])
    u_init = mu_init - sigma_init

    start_our = time.time()
    results = run_solver(events, u_init, sigma_init,
                         baseline_init, alpha_init,
                         T, dt, seed=0)
    time_our = time.time() - start_our

    start_tick = time.time()
    non_param = HawkesBasisKernels(1, n_basis=1, kernel_size=int(1 / dt), max_iter=800)
    non_param.fit(events)
    time_tick = time.time() - start_tick

    discretization = torch.linspace(0, 1, int(1 / dt))
    u_hd = results['param_kernel'][0][-1]
    sigma_hd = results['param_kernel'][1][-1]
    alpha_hd = results['param_alpha'][-1]

    RC = DiscreteKernelFiniteSupport(0, 1, dt, kernel='RaisedCosine')
    kernel_values = RC.eval([torch.Tensor(u_hd),
                            torch.Tensor(sigma_hd)],
                            discretization).squeeze().numpy()
    kernel_values *= alpha_hd.item()

    res_our = dict(kernel=kernel_values, comp_time=time_our,
                   kernel_name=kernel, T=T, dt=dt, seed=seed)

    tick_values = non_param.get_kernel_values(0, 0, discretization[:-1])
    tick_values *= alpha.item()

    res_tick = dict(kernel=tick_values, comp_time=time_tick,
                    kernel_name=kernel, T=T, dt=dt, seed=seed)

    return res_our, res_tick


# %% run

dt_list = [0.1, 0.01, 0.001]
T_list = [1000, 10_000, 100_000, 1_000_000]
seeds = np.arange(100)
kernel_ = ['RC', 'SG']
info = dict(kernel=kernel_, T_list=T_list, dt_list=dt_list, seeds=seeds)

n_jobs = 60
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, alpha, mu, sigma, T, dt, seed=seed, kernel=kernel)
    for kernel, T, dt, seed in itertools.product(
        kernel_, T_list, dt_list, seeds
    )
)
all_results.append(info)
file_name = "results/non_param.pkl"
open_file = open(file_name, "wb")
pickle.dump(all_results, open_file)
open_file.close()
