# %% import stuff
# import libraries
import itertools
import pandas as pd
import time
import numpy as np
import torch
from joblib import Parallel, delayed
from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc

from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.solver import FaDIn, FaDIn_loglikelihood

# %% simulate data
# Simulated data
################################


def simulate_data(baseline, alpha, kernel_params,
                  T, dt, seed=0, kernel='raised_cosine'):

    L = int(1 / dt)
    discretization = torch.linspace(0, 1, L)
    if kernel == 'raised_cosine':
        u, sigma = kernel_params
        RC = DiscreteKernelFiniteSupport(dt, 1, kernel=kernel)
        kernel_values = RC.kernel_eval([torch.Tensor(u), torch.Tensor(sigma)],
                                       discretization)
    elif kernel == 'truncated_gaussian':
        m, sigma = kernel_params
        TG = DiscreteKernelFiniteSupport(dt, 1, kernel=kernel)
        kernel_values = TG.kernel_eval([torch.Tensor(m), torch.Tensor(sigma)],
                                       discretization)
    elif kernel == 'truncated_exponential':
        decay = kernel_params[0]
        EXP = DiscreteKernelFiniteSupport(dt, 1, kernel=kernel)
        kernel_values = EXP.kernel_eval([torch.Tensor(decay)],
                                        discretization)
    else:
        raise NameError('this kernel is not implemented')

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


def run_solver(criterion, events, kernel_params_init,
               baseline_init, alpha_init, T, dt, seed=0,
               kernel='raised_cosine'):
    max_iter = 2000
    init = {
        'alpha': torch.tensor(alpha_init),
        'baseline': torch.tensor(baseline_init),
        'kernel': [torch.tensor(a) for a in kernel_params_init]
    }
    if criterion == 'l2':
        solver = FaDIn(
            1,
            kernel,
            init=init,
            delta=dt,
            optim="RMSprop",
            step_size=1e-3,
            max_iter=max_iter,
            log=False,
            random_state=seed,
            device="cpu"
        )
    elif criterion == 'll':
        solver = FaDIn_loglikelihood(
            1,
            kernel,
            init=init,
            delta=dt,
            optim="RMSprop",
            step_size=1e-3,
            max_iter=max_iter,
            log=False,
            random_state=seed,
            device="cpu"
        )
    results = solver.fit(events, T)
    if kernel == 'truncated_exponential':
        results_ = dict(param_baseline=results['param_baseline'][-10:].mean().item(),
                        param_alpha=results['param_alpha'][-10:].mean().item(),
                        param_kernel=[results['param_kernel'][0][-10:].mean().item()])
    else:
        results_ = dict(param_baseline=results['param_baseline'][-10:].mean().item(),
                        param_alpha=results['param_alpha'][-10:].mean().item(),
                        param_kernel=[results['param_kernel'][0][-10:].mean().item(),
                                      results['param_kernel'][1][-10:].mean().item()])
    return results_


def run_experiment(baseline, alpha, kernel_params, T, dt, seed=0,
                   kernel='raised_cosine'):
    res = dict(T=T, dt=dt, seed=seed)
    # simulate data
    events = simulate_data(baseline, alpha, kernel_params, T, dt,
                           seed=seed, kernel=kernel)
    v = 0.2
    baseline_init = np.random.uniform(size=(1))
    alpha_init = np.random.uniform(size=(1, 1))

    if kernel == 'raised_cosine':
        u, sigma = kernel_params
        sigma_init = sigma + v
        u_init = u + v
        kernel_params_init = [u_init, sigma_init]

    elif kernel == 'truncated_gaussian':
        m, sigma = kernel_params
        m_init = m + v
        sigma_init = sigma + v
        kernel_params_init = [m_init, sigma_init]

    elif kernel == 'truncated_exponential':
        decay = kernel_params[0]
        decay_init = decay + v
        kernel_params_init = [decay_init]

    start_fadin = time.time()
    res_fadin = run_solver('l2', events, kernel_params_init,
                           baseline_init, alpha_init, T, dt,
                           seed=seed, kernel=kernel)

    res['err_fadin_baseline'] = ((res_fadin['param_baseline'] - baseline)**2).item()
    res['err_fadin_alpha'] = ((res_fadin['param_alpha'] - alpha)**2).item()
    res['err_fadin_k0'] = ((res_fadin['param_kernel'][0] - kernel_params[0])**2).item()
    if kernel == 'truncated_exponential':
        res['err_fadin'] = np.sqrt(res['err_fadin_baseline'] + res['err_fadin_alpha'] +
                                   res['err_fadin_k0']).item()
    else:
        res['err_fadin_k1'] = ((res_fadin['param_kernel'][1] -
                                kernel_params[1])**2).item()
        res['err_fadin'] = np.sqrt(res['err_fadin_baseline'] + res['err_fadin_alpha'] +
                                   res['err_fadin_k0'] + res['err_fadin_k1']).item()
    res['time_fadin'] = time.time() - start_fadin

    start_ll = time.time()
    res_ll = run_solver('ll', events, kernel_params_init,
                        baseline_init, alpha_init, T, dt, seed=seed, kernel=kernel)
    res['err_ll_baseline'] = ((res_ll['param_baseline'] - baseline)**2).item()
    res['err_ll_alpha'] = ((res_ll['param_alpha'] - alpha)**2).item()
    res['err_ll_k0'] = ((res_ll['param_kernel'][0] - kernel_params[0])**2).item()
    if kernel == 'truncated_exponential':
        res['err_ll'] = np.sqrt(res['err_ll_baseline'] + res['err_ll_alpha'] +
                                res['err_ll_k0']).item()
    else:
        res['err_ll_k1'] = ((res_ll['param_kernel'][1] - kernel_params[1])**2).item()
        res['err_ll'] = np.sqrt(res['err_ll_baseline'] + res['err_ll_alpha'] +
                                res['err_ll_k0'] + res['err_ll_k1']).item()

    res['time_ll'] = time.time() - start_ll

    return res


# kernel = 'raised_cosine'
# kernel = 'truncated_gaussian'
kernel = 'truncated_exponential'

baseline = np.array([.1])
alpha = np.array([[0.8]])
mu = np.array([[0.5]])
sigma = np.array([[0.3]])
decay = np.array([[5]])
u = mu - sigma

if kernel == 'raised_cosine':
    kernel_params = [u, sigma]
elif kernel == 'truncated_gaussian':
    kernel_params = [mu, sigma]
elif kernel == 'truncated_exponential':
    alpha = np.array([[0.3]])
    kernel_params = [decay]


T_list = [100, 1000, 10_000, 100_000]  # , 1_000_000]
dt_list = [0.1, 0.01]
seeds = np.arange(10)

n_jobs = 80
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, alpha, kernel_params, T,
                            dt, seed=seed, kernel=kernel)
    for T, dt, seed in itertools.product(
        T_list, dt_list, seeds
    )
)

# save results
df = pd.DataFrame(all_results)
df.to_csv(f'results/comparison_ll_{kernel}.csv', index=False)

# %%
