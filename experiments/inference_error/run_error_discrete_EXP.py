# %% import stuff
# import libraries
import itertools
import time
import numpy as np
import torch
import pandas as pd
from joblib import Parallel, delayed
from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc

from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.solver import FaDIn


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
def simulate_data(baseline, alpha, decay, T, dt, seed=0):
    L = int(1 / dt)
    discretization = torch.linspace(0, 1, L)
    Exp = DiscreteKernelFiniteSupport(dt, 1, kernel='truncated_exponential')
    kernel_values = Exp.kernel_eval([torch.Tensor(decay)], discretization)
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


events = simulate_data(baseline, alpha, decay, T, dt, seed=0)


# @mem.cache
def run_solver(events, decay_init, baseline_init, alpha_init, T, dt, seed=0):
    start = time.time()
    max_iter = 2000
    solver = FaDIn(2,
                   "truncated_exponential",
                   [torch.tensor(decay_init)],
                   torch.tensor(baseline_init),
                   torch.tensor(alpha_init),
                   delta=dt,
                   optim="RMSprop",
                   step_size=1e-3,
                   max_iter=max_iter,
                   kernel_length=10,
                   log=False,
                   random_state=0,
                   device="cpu",
                   optimize_kernel=True
                   )
    print(time.time() - start)
    results = solver.fit(events, T)
    results_ = dict(param_baseline=results['param_baseline'][-10:].mean().item(),
                    param_alpha=results['param_alpha'][-10:].mean().item(),
                    param_kernel=[results['param_kernel'][0][-10:].mean().item()])
    results_["time"] = time.time() - start
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = dt
    return results_


# %% eval on grid


def run_experiment(baseline, alpha, decay, T, dt, seed=0):
    v = 0.2
    events = simulate_data(baseline, alpha, decay, T, dt, seed=seed)
    baseline_init = baseline + v
    alpha_init = alpha + v
    decay_init = decay + v

    results = run_solver(events, decay_init, baseline_init, alpha_init, T, dt, seed)

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
