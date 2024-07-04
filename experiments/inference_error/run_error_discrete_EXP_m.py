# %% import stuff
# import libraries
import itertools
import pandas as pd
import time
import numpy as np
from joblib import Parallel, delayed

from fadin.solver import FaDIn
from fadin.utils.utils import l2_error
from fadin.utils.utils_simu import simu_hawkes_cluster

# mem = Memory(location=".", verbose=2)


# %% simulate data
# Simulated data
################################

baseline = np.array([.1, .2])
alpha = np.array([[0.3, 0.1], [0.1, 0.3]])
decay = np.array([[5, 5], [5, 5]])


# @mem.cache
def simulate_data(baseline, alpha, decay, T, seed=0):
    kernel = 'expon'
    events = simu_hawkes_cluster(T, baseline, alpha, kernel,
                                 params_kernel={'scale': 1 / decay},
                                 random_state=seed)
    return events

# %% solver


def run_solver(events, dt, T, seed=0):
    start = time.time()
    max_iter = 2000
    solver = FaDIn(2,
                   "truncated_exponential",
                   delta=dt,
                   optim="RMSprop",
                   max_iter=max_iter,
                   log=False,
                   random_state=0,
                   ztzG_approx=True)

    print(time.time() - start)
    solver.fit(events, T)
    results_ = dict(param_baseline=solver.param_baseline[-10:].mean(0),
                    param_alpha=solver.param_alpha[-10:].mean(0),
                    param_kernel=[solver.param_kernel[0][-10:].mean(0)])
    results_["time"] = time.time() - start
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = dt
    return results_

# %% eval on grid


def run_experiment(baseline, alpha, decay, T, dt, seed=0):
    events = simulate_data(baseline, alpha, decay, T, seed=seed)
    results = run_solver(events, dt, T, seed)

    return results


T_list = [1000, 10_000, 100_000, 1_000_000]
dt_list = np.logspace(1, 3, 10) / 10e3
seeds = np.arange(10)

n_jobs = 30
all_results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)(
    delayed(run_experiment)(baseline, alpha, decay, T, dt, seed=seed)
    for T, dt, seed in itertools.product(
        T_list, dt_list, seeds
    )
)

# save results
df = pd.DataFrame(all_results)

df['param_decay'] = df['param_kernel'].apply(lambda x: x[0])

df['err_baseline'] = df['param_baseline'].apply(lambda x: l2_error(x, baseline))
df['err_alpha'] = df['param_alpha'].apply(lambda x: l2_error(x, alpha))
df['err_decay'] = df['param_decay'].apply(lambda x: l2_error(x, decay))

df['err_sum'] = np.sqrt(df['err_baseline']**2 + df['err_alpha']**2 +
                        df['err_decay']**2)

df.to_csv('results/error_discrete_EXP_m.csv', index=False)

# %%
