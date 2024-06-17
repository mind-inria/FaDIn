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
from fadin.utils.utils import l2_error
from fadin.utils.utils_simu import simu_hawkes_cluster

mem = Memory(location=".", verbose=2)

# %% simulate data
# Simulated data
################################

baseline = np.array([.1, .2])
alpha = np.array([[0.8, 0.1], [0.1, 0.8]])
m = np.array([[0.4, 0.6], [0.55, 0.6]])
sigma = np.array([[0.3, 0.3], [0.25, 0.3]])


def simulate_data(baseline, alpha, m, sigma, T, dt, seed=0):
    params = {'mu': m, 'sigma': sigma}

    def truncated_gaussian(x, **params):
        tg = DiscreteKernelFiniteSupport(delta=dt, n_dim=2,
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

# %% solver


@mem.cache
def run_solver(events, dt, T, seed=0):
    start = time.time()
    max_iter = 2000

    solver = FaDIn(2,
                   "truncated_gaussian",
                   delta=dt, optim="RMSprop",
                   max_iter=max_iter,
                   log=False,
                   random_state=seed,
                   ztzG_approx=True)

    print(time.time() - start)
    solver.fit(events, T)
    results_ = dict(param_baseline=solver.param_baseline[-10:].mean(0),
                    param_alpha=solver.param_alpha[-10:].mean(0),
                    param_kernel=[solver.param_kernel[0][-10:].mean(0),
                                  solver.param_kernel[1][-10:].mean(0)])
    results_["time"] = time.time() - start
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = dt
    return results_

# %% eval on grid


def run_experiment(baseline, alpha, m, sigma, T, dt, seed=0):
    events = simulate_data(baseline, alpha, m, sigma, T, dt, seed=seed)
    results = run_solver(events, dt, T, seed)
    return results


T_list = [1000, 10_000, 100_000, 1_000_000]
dt_list = np.logspace(1, 3, 10) / 10e3
seeds = np.arange(10)

n_jobs = 60
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

df['err_baseline'] = df['param_baseline'].apply(lambda x: l2_error(x, baseline))
df['err_alpha'] = df['param_alpha'].apply(lambda x: l2_error(x, alpha))
df['err_m'] = df['param_m'].apply(lambda x: l2_error(x, m))
df['err_sigma'] = df['param_sigma'].apply(lambda x: l2_error(x, sigma))
df['err_sum'] = np.sqrt(df['err_baseline']**2 + df['err_alpha']**2 +
                        df['err_m']**2 + df['err_sigma']**2)

df.to_csv('results/error_discrete_TG_m.csv', index=False)
