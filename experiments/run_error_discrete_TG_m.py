# %% import stuff
# import libraries
import itertools
import pandas as pd
import time
import numpy as np
import torch
from joblib import Parallel, delayed, Memory
from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc

from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.solver import FaDIn
from fadin.utils.utils import l2_error

mem = Memory(location=".", verbose=2)

# %% simulate data
# Simulated data
################################

baseline = np.array([.1, .2])
alpha = np.array([[0.8, 0.1], [0.1, 0.8]])
m = np.array([[0.4, 0.6], [0.55, 0.6]])
sigma = np.array([[0.3, 0.3], [0.25, 0.3]])


@mem.cache
def simulate_data(baseline, alpha, m, sigma, T, dt, seed=0):
    L = int(1 / dt)
    discretization = torch.linspace(0, 1, L)
    n_dim = m.shape[0]
    TG = DiscreteKernelFiniteSupport(dt, n_dim, kernel='truncated_gaussian')

    kernel_values = TG.kernel_eval([torch.Tensor(m), torch.Tensor(sigma)],
                                   discretization)
    kernel_values = kernel_values * alpha[:, :, None]

    t_values = discretization.double().numpy()
    k11 = kernel_values[0, 0].double().numpy()
    k12 = kernel_values[0, 1].double().numpy()
    k21 = kernel_values[1, 0].double().numpy()
    k22 = kernel_values[1, 1].double().numpy()

    tf11 = HawkesKernelTimeFunc(t_values=t_values, y_values=k11)
    tf12 = HawkesKernelTimeFunc(t_values=t_values, y_values=k12)
    tf21 = HawkesKernelTimeFunc(t_values=t_values, y_values=k21)
    tf22 = HawkesKernelTimeFunc(t_values=t_values, y_values=k22)

    kernels = [[tf11, tf12], [tf21, tf22]]
    hawkes = SimuHawkes(
        baseline=baseline, kernels=kernels, end_time=T, verbose=False, seed=int(seed)
    )

    hawkes.simulate()
    events = hawkes.timestamps
    return events

# %% solver


@mem.cache
def run_solver(events, m_init, sigma_init, baseline_init, alpha_init, dt, T, seed=0):
    start = time.time()
    max_iter = 2000
    solver = FaDIn("truncated_gaussian",
                   [torch.tensor(m_init),
                    torch.tensor(sigma_init)],
                   torch.tensor(baseline_init),
                   torch.tensor(alpha_init),
                   delta=dt, optim="RMSprop",
                   step_size=1e-3,
                   max_iter=max_iter,
                   log=False,
                   random_state=0,
                   device="cpu",
                   optimize_kernel=True,
                   precomputations=True,
                   ztzG_approx=True)

    print(time.time() - start)
    results = solver.fit(events, T)
    results_ = dict(param_baseline=results['param_baseline'][-10:].mean(0),
                    param_alpha=results['param_alpha'][-10:].mean(0),
                    param_kernel=[results['param_kernel'][0][-10:].mean(0),
                                  results['param_kernel'][1][-10:].mean(0)])
    results_["time"] = time.time() - start
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = dt
    return results_

# %% eval on grid


def run_experiment(baseline, alpha, m, sigma, T, dt, seed=0):
    v = 0.2
    events = simulate_data(baseline, alpha, m, sigma, T, dt, seed=seed)
    baseline_init = baseline + v
    alpha_init = alpha + v
    m_init = m + v
    sigma_init = sigma + v

    results = run_solver(events, m_init, sigma_init,
                         baseline_init, alpha_init,
                         dt, T, seed)
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
