# %% import stuff
# import libraries
import itertools
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from fadin.solver import FaDIn
from fadin.utils.utils_simu import simu_hawkes_cluster

################################
# Meta parameters
################################

dt = 0.01
T = 10_000
size_grid = int(T / dt) + 1

# mem = Memory(location="__cache__", verbose=2)

# %% simulate data
# Simulated data
################################

baseline = np.array([.1])
alpha = np.array([[0.3]])
decay = np.array([[5]])


# @mem.cache
def simulate_data(baseline, alpha, decay, T, seed=0):
    kernel = 'expon'
    events = simu_hawkes_cluster(T, baseline, alpha, kernel,
                                 params_kernel={'scale': 1 / decay},
                                 random_state=seed)
    return events


events = simulate_data(baseline, alpha, decay, T, seed=0)


# @mem.cache
def run_solver(events, T, dt, seed=0):
    start = time.time()
    max_iter = 2000

    solver = FaDIn(1,
                   "truncated_exponential",
                   delta=dt,
                   optim="RMSprop",
                   max_iter=max_iter,
                   kernel_length=10,
                   log=False,
                   random_state=0
                   )

    print(time.time() - start)
    solver.fit(events, T)
    results_ = dict(param_baseline=solver.param_baseline[-10:].mean().item(),
                    param_alpha=solver.param_alpha[-10:].mean().item(),
                    param_kernel=[solver.param_kernel[0][-10:].mean().item()])
    results_["time"] = time.time() - start
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = dt
    return results_


# %% eval on grid


def run_experiment(baseline, alpha, decay, T, dt, seed=0):
    events = simulate_data(baseline, alpha, decay, T, seed=seed)
    results = run_solver(events, T, dt, seed)

    return results


T_list = [1000, 10_000, 100_000, 1_000_000]
dt_list = np.logspace(1, 3, 10) / 10e3
seeds = np.arange(10)

n_jobs = 60
all_results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)(
    delayed(run_experiment)(baseline, alpha, decay, T, dt, seed=seed)
    for T, dt, seed in itertools.product(
        T_list, dt_list, seeds
    )
)

# save results
df = pd.DataFrame(all_results)
df['param_decay'] = df['param_kernel'].apply(lambda x: x[0])
true_param = {'baseline': .1, 'alpha': 0.3, 'decay': 5}
for param, value in true_param.items():
    df[param] = value


def compute_norm2_error(s):
    return np.sqrt(np.array([(s[param] - s[f'param_{param}'])**2
                            for param in ['baseline', 'alpha', 'decay']]).sum())


df['err_norm2'] = df.apply(
    lambda x: compute_norm2_error(x), axis=1)

df.to_csv('results/error_discrete_EXP.csv', index=False)

# df['param_sigma'] = df['param_kernel'].apply(lambda x: x[1])
# , 'sigma': 0.3}

# %%
