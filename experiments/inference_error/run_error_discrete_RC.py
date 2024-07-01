# %% import stuff
# import libraries
import itertools
import pandas as pd
import time
import numpy as np
import torch
from joblib import Parallel, delayed, Memory

from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.solver import FaDIn
from fadin.utils.utils_simu import simu_hawkes_cluster

################################
# Meta parameters
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


def simulate_data(baseline, alpha, mu, sigma, T, dt, seed=0):
    u = mu - sigma
    params = {'u': u, 'sigma': sigma}

    def raised_cosine(x, **params):
        rc = DiscreteKernelFiniteSupport(delta=dt, n_dim=1,
                                         kernel='raised_cosine')
        u = params['u']
        sigma = params['sigma']
        kernel_values = rc.kernel_eval([torch.Tensor(u), torch.Tensor(sigma)],
                                       torch.tensor(x))

        return kernel_values.double().numpy()

    events = simu_hawkes_cluster(T, baseline, alpha,
                                 raised_cosine,
                                 params_kernel=params,
                                 random_state=seed)
    return events


events = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=0)
# %% solver
##


def run_solver(events, dt, T, seed=0):
    start = time.time()
    max_iter = 2000

    solver = FaDIn(1,
                   "raised_cosine",
                   delta=dt,
                   optim="RMSprop",
                   max_iter=max_iter,
                   log=False,
                   random_state=0)

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


# %% eval on grid
##
def run_experiment(baseline, alpha, mu, sigma, T, dt, seed=0):
    events = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=seed)
    results = run_solver(events, dt, T, seed)

    return results


T_list = [1000, 10_000, 100_000, 1_000_000]
dt_list = np.logspace(1, 3, 10) / 10e3
seeds = np.arange(100)

n_jobs = 60
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, alpha, mu, sigma, T, dt, seed=seed)
    for T, dt, seed in itertools.product(
        T_list, dt_list, seeds
    )
)

# save results
df = pd.DataFrame(all_results)

df['param_u'] = df['param_kernel'].apply(lambda x: x[0])
df['param_sigma'] = df['param_kernel'].apply(lambda x: x[1])
true_param = {'baseline': 1.1, 'alpha': 0.8, 'u': 0.2, 'sigma': 0.3}
for param, value in true_param.items():
    df[param] = value


def compute_norm2_error(s):
    return np.sqrt(np.array([(s[param] - s[f'param_{param}'])**2
                            for param in ['baseline', 'alpha', 'u', 'sigma']]).sum())


df['err_norm2'] = df.apply(
    lambda x: compute_norm2_error(x), axis=1)

df.to_csv('results/error_discrete_RC.csv', index=False)

# %%
