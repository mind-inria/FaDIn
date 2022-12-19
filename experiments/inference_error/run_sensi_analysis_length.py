# %% import stuff
# import libraries
import time
import itertools
import numpy as np
import torch
from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc, HawkesKernelExp
from joblib import Parallel, delayed
import pandas as pd

from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.solver import FaDIn

################################
# Meta parameters
################################
dt = 0.01
T = 10_000
kernel_length = 20

# %% Experiment
################################


def simulate_data(baseline, alpha, decay, kernel_length, T, dt, seed=0):
    L = int(kernel_length / dt)
    discretization = torch.linspace(0, kernel_length, L)
    n_dim = decay.shape[0]
    EXP = DiscreteKernelFiniteSupport(dt, n_dim, kernel='truncated_exponential',
                                      kernel_length=kernel_length,
                                      upper=kernel_length)
    kernel_values = EXP.kernel_eval([torch.Tensor(decay)],
                                    discretization)
    kernel_values = kernel_values * alpha[:, :, None]

    t_values = discretization.double().numpy()
    k = kernel_values[0, 0].double().numpy()

    tf = HawkesKernelTimeFunc(t_values=t_values, y_values=k)
    kernels = [[tf]]
    hawkes = SimuHawkes(
        baseline=baseline, kernels=kernels, end_time=T, verbose=False, seed=int(seed)
    )

    hawkes.simulate()
    events_discrete = hawkes.timestamps

    tf = HawkesKernelExp(alpha.item(), decay.item())
    kernels = [[tf]]
    hawkes = SimuHawkes(
        baseline=baseline, kernels=kernels, end_time=T, verbose=False, seed=int(seed)
    )

    hawkes.simulate()
    events_continuous = hawkes.timestamps

    return events_discrete, events_continuous


baseline = np.array([.1])
alpha = np.array([[0.8]])
decay = np.array([[1.]])

events_d, events_c = simulate_data(baseline, alpha, decay, kernel_length, T, dt, seed=0)
# %% solver


def run_solver(events, decay_init, baseline_init,
               alpha_init, kernel_length, T, dt, seed=0):
    start = time.time()
    max_iter = 2000
    solver = FaDIn(1,
                   "truncated_exponential",
                   [torch.tensor(decay_init)],
                   torch.tensor(baseline_init),
                   torch.tensor(alpha_init),
                   kernel_length=kernel_length,
                   delta=dt, optim="RMSprop",
                   step_size=1e-3, max_iter=max_iter
                   )

    print(time.time() - start)
    results = solver.fit(events, T)
    results_ = dict(param_baseline=results['param_baseline'][-10:].mean().item(),
                    param_alpha=results['param_alpha'][-10:].mean().item(),
                    param_kernel=[results['param_kernel'][0][-10:].mean().item()]
                    )

    results_["time"] = time.time() - start
    results_['W'] = kernel_length
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = dt
    return results_


# %% Test


baseline = np.array([1.1])
alpha = np.array([[0.8]])
decay = np.array([[0.5]])

events_d, events_c = simulate_data(baseline, alpha, decay, kernel_length, T, dt, seed=0)

v = 0.2
baseline_init = baseline + v
alpha_init = alpha + v
decay_init = decay + v

results = run_solver(events_c, decay_init,
                     baseline_init, alpha_init,
                     kernel_length, T, dt, seed=0)

print(np.abs(results['param_baseline'] - baseline))
print(np.abs(results['param_alpha'] - alpha))
print(np.abs(results['param_kernel'][0] - decay))

# %%


def run_experiment(baseline, alpha, decay, kernel_length, T, dt, seed=0):
    v = 0.2
    events_d, events_c = simulate_data(baseline, alpha, decay, kernel_length,
                                       T, dt, seed=seed)
    baseline_init = baseline + v
    alpha_init = alpha + v
    decay_init = decay + v

    # results_d = run_solver(events_d, decay_init, baseline_init, alpha_init,
    #                        kernel_length, T, dt, seed)
    results_c = run_solver(events_c, decay_init, baseline_init, alpha_init,
                           kernel_length, T, dt, seed)

    return results_c  # results_d


W_list = [1, 5, 10, 20, 50, 100]
T_list = [1000, 10000, 100_000, 1_000_000]
dt_list = [0.01]
seeds = np.arange(10)


n_jobs = 60
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, alpha, decay, W, T, dt, seed=seed)
    for W, T, dt, seed in itertools.product(
        W_list, T_list, dt_list, seeds
    )
)

# save results
df = pd.DataFrame(all_results)
df['param_decay'] = df['param_kernel'].apply(lambda x: x[0])
true_param = {'baseline': 1.1, 'alpha': 0.8, 'decay': 0.5}
for param, value in true_param.items():
    df[param] = value


def compute_norm2_error(s):
    return np.sqrt(np.array([(s[param] - s[f'param_{param}'])**2
                            for param in ['baseline', 'alpha', 'decay']]).sum())


df['err_norm2'] = df.apply(
    lambda x: compute_norm2_error(x), axis=1)

df.to_csv('results/sensitivity_length.csv', index=False)
