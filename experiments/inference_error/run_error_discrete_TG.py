# %% import stuff
# import libraries

import itertools
import pandas as pd
import time
import numpy as np
import torch
from joblib import Memory, Parallel, delayed

from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.solver import FaDIn
from fadin.utils.utils_simu import simu_hawkes_cluster

################################
# Meta parameters
################################

dt = 0.01
T = 10_000
size_grid = int(T / dt) + 1

mem = Memory(location=".", verbose=2)

# %% simulate data
# Simulated data
################################

baseline = np.array([.1])
alpha = np.array([[0.8]])
m = np.array([[0.5]])
sigma = np.array([[0.3]])


def simulate_data(baseline, alpha, m, sigma, T, dt, seed=0):
    params = {'mu': m, 'sigma': sigma}

    def truncated_gaussian(x, **params):
        tg = DiscreteKernelFiniteSupport(delta=dt, n_dim=1,
                                         kernel='truncated_gaussian')
        mu = params['mu']
        sigma = params['sigma']
        kernel_values = tg.kernel_eval(
            [torch.Tensor(mu), torch.Tensor(sigma)], torch.tensor(x))

        return kernel_values.double().numpy()

    events = simu_hawkes_cluster(T, baseline, alpha,
                                 truncated_gaussian,
                                 params_kernel=params,
                                 random_state=seed)
    return events


events = simulate_data(baseline, alpha, m, sigma, T, dt, seed=0)

# %%


def run_solver(events, T, dt, seed=0):
    start = time.time()
    max_iter = 2000
    solver = FaDIn(1,
                   "truncated_gaussian",
                   delta=dt,
                   optim="RMSprop",
                   max_iter=max_iter,
                   log=False,
                   random_state=seed)

    print(time.time() - start)
    solver.fit(events, T)

    results_ = dict(param_baseline=solver.param_baseline[-10:].mean().item(),
                    param_alpha=solver.param_alpha[-10:].mean().item(),
                    param_kernel=[solver.param_kernel[0][-10:].mean().item(),
                                  solver.param_kernel[1][-10:].mean().item()])
    results_["time"] = time.time() - start
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = dt
    return results_


np.random.seed(0)
baseline_init = np.array([np.random.rand() * 0.5])
alpha_init = np.array([[np.random.rand()]])
m_init = np.array([[np.random.rand()]])
sigma_init = np.array([[np.random.rand() * 0.5]])

start = time.time()
results_1 = run_solver(events, T, dt, seed=0)
baseline_our = results_1['param_baseline']
alpha_our = results_1['param_alpha']
print(np.abs(results_1['param_baseline']))  # baseline))
print(np.abs(results_1['param_alpha']))  # alpha))
print(np.abs(results_1['param_kernel'][0]))  # decay))
print(np.abs(results_1['param_kernel'][1]))

# %% eval on grid


def run_experiment(baseline, alpha, m, sigma, T, dt, seed=0):
    events = simulate_data(baseline, alpha, m, sigma, T, dt, seed=seed)
    results = run_solver(events, T, dt, seed)
    return results


T_list = [1000, 10_000, 100_000, 1_000_000]
dt_list = np.logspace(1, 3, 10) / 10e3
seeds = np.arange(100)

n_jobs = 40
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, alpha, m, sigma, T, dt, seed=seed)
    for T, dt, seed in itertools.product(
        T_list, dt_list, seeds
    )
)

# save results
df = pd.DataFrame(all_results)

df['param_m'] = df['param_kernel'].apply(lambda x: x[0])
df['param_sigma'] = df['param_kernel'].apply(lambda x: x[1])
true_param = {'baseline': .1, 'alpha': 0.8, 'm': 0.5, 'sigma': 0.3}
for param, value in true_param.items():
    df[param] = value


def compute_norm2_error(s):
    return np.sqrt(np.array([(s[param] - s[f'param_{param}'])**2
                            for param in ['baseline', 'alpha', 'm', 'sigma']]).sum())


df['err_norm2'] = df.apply(
    lambda x: compute_norm2_error(x), axis=1)

df.to_csv('results/error_discrete_TG.csv', index=False)

# %%
