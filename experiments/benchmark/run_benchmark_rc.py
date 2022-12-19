# %% import stuff
# import libraries
import itertools
import time
import numpy as np
import torch
from joblib import Parallel, delayed
import pandas as pd
from pybasicbayes.util.text import progprint_xrange

from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc
from tick.hawkes import HawkesBasisKernels

from pyhawkes.models import \
    DiscreteTimeNetworkHawkesModelGammaMixture, \
    DiscreteTimeStandardHawkesModel
from pyhawkes.internals.network import StochasticBlockModel

from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.solver import FaDIn
from fadin.utils.utils import projected_grid


def simulate_data(baseline, alpha, mu, sigma, T, dt, seed=0):
    L = int(1 / dt)
    discretization = torch.linspace(0, 1, L)
    u = mu - sigma
    n_dim = u.shape[0]
    RC = DiscreteKernelFiniteSupport(dt, n_dim=n_dim, kernel='raised_cosine',
                                     lower=0, upper=1)

    kernel_values = RC.kernel_eval([torch.Tensor(u), torch.Tensor(sigma)],
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

    events_grid = projected_grid(events, dt, L * T + 1)
    intens = RC.intensity_eval(torch.tensor(baseline),
                               torch.tensor(alpha),
                               [torch.Tensor(u), torch.Tensor(sigma)],
                               events_grid, discretization)
    return events, intens


def run_fadin(events, u_init, sigma_init, baseline_init, alpha_init, T, dt, seed=0):
    start = time.time()
    max_iter = 2000
    solver = FaDIn(2,
                   "raised_cosine",
                   [torch.tensor(u_init),
                    torch.tensor(sigma_init)],
                   torch.tensor(baseline_init),
                   torch.tensor(alpha_init),
                   delta=dt, optim="RMSprop",
                   step_size=1e-3, max_iter=max_iter,
                   optimize_kernel=True, precomputations=True,
                   ztzG_approx=True, device='cpu', log=False
                   )

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


def run_gibbs(S, size_grid, dt, seed=0):
    np.random.seed(seed)
    init_len = size_grid
    start = time.time()
    init_model = DiscreteTimeStandardHawkesModel(K=2, dt=dt, dt_max=1, B=5,
                                                 alpha=1.0, beta=1.0)
    init_model.add_data(S[:init_len, :])

    init_model.initialize_to_background_rate()
    init_model.fit_with_bfgs()

    ###########################################################
    # Create a test weak spike-and-slab model
    ###########################################################
    # Copy the network hypers.
    # Give the test model p, but not c, v, or m
    test_model = DiscreteTimeNetworkHawkesModelGammaMixture(K=2, dt=dt, dt_max=1, B=5)
    test_model.add_data(S)

    # Initialize with the standard model parameters
    if init_model is not None:
        test_model.initialize_with_standard_model(init_model)

    ###########################################################
    # Fit the test model with Gibbs sampling
    ###########################################################
    N_samples = 250
    samples = []
    lps = []
    for itr in progprint_xrange(N_samples):
        lps.append(test_model.log_probability())
        samples.append(test_model.copy_sample())
        test_model.resample_model()

    results = dict(intens_gibbs=test_model.compute_rate(), time=time.time() - start)

    return results


def run_vb(S, size_grid, dt, seed=0):
    np.random.seed(seed)
    init_len = size_grid
    start = time.time()
    init_model = DiscreteTimeStandardHawkesModel(K=2, dt=dt, dt_max=1, B=5,
                                                 alpha=1.0, beta=1.0)
    init_model.add_data(S[:init_len, :])

    init_model.initialize_to_background_rate()
    init_model.fit_with_bfgs()

    ###########################################################
    # Create a test weak spike-and-slab model
    ###########################################################

    test_network = StochasticBlockModel(K=2, C=1)
    test_model = DiscreteTimeNetworkHawkesModelGammaMixture(K=2, dt=dt, dt_max=1, B=5,
                                                            network=test_network
                                                            )
    test_model.add_data(S)

    # Initialize with the standard model parameters
    if init_model is not None:
        test_model.initialize_with_standard_model(init_model)

    ###########################################################
    # Fit the test model with variational Bayesian inference
    ###########################################################
    # VB coordinate descent
    N_iters = 2000
    vlbs = []
    samples = []
    for itr in range(N_iters):
        vlbs.append(test_model.meanfield_coordinate_descent_step())
        print("VB Iter: ", itr, "\tVLB: ", vlbs[-1])
        if itr > 0:
            if (vlbs[-2] - vlbs[-1]) > 1e-1:
                print("WARNING: VLB is not increasing!")

        # Resample from variational distribution and plot
        test_model.resample_from_mf()
        samples.append(test_model.copy_sample())

    results = dict(intens_vb=test_model.compute_rate(), time=time.time() - start)

    return results


def run_non_param(S, size_grid, dt, seed=0):
    np.random.seed(seed)
    # Make a model to initialize the parameters
    init_len = size_grid
    start = time.time()
    init_model = DiscreteTimeStandardHawkesModel(K=2, dt=dt, B=5, beta=1.0)
    init_model.add_data(S[:init_len, :])

    print("Initializing with BFGS on first ", init_len, " time bins.")
    init_model.fit_with_bfgs()

    # Make another model for inference
    test_model = DiscreteTimeStandardHawkesModel(K=2, dt=dt, B=5, beta=1.0)
    # Initialize with the BFGS parameters
    test_model.weights = init_model.weights
    # Add the data in minibatches
    test_model.add_data(S, minibatchsize=size_grid)

    # Gradient descent
    N_steps = 2000
    learning_rate = 0.01 * np.ones(N_steps)
    momentum = 0.8 * np.ones(N_steps)
    prev_velocity = None
    for itr in range(N_steps):
        _, _, prev_velocity = test_model.sgd_step(prev_velocity,
                                                  learning_rate[itr],
                                                  momentum[itr])

    results = dict(intens_nonparam_sgd=test_model.compute_rate(),
                   time=time.time() - start)

    return results


def run_non_param_tick(events, size_grid, dt, seed=0):
    np.random.seed(seed)
    L = int(1 / dt)
    start = time.time()
    non_param = HawkesBasisKernels(1, n_basis=1, kernel_size=int(1 / dt), max_iter=2000)
    non_param.fit(events)
    baseline = non_param.baseline
    discretization = np.linspace(0, 1, L)
    tick_kernel_values = np.zeros((2, 2, L))
    for i in range(2):
        for j in range(2):
            tick_kernel_values[i, j] = non_param.get_kernel_values(i, j, discretization)
    tick_kernel_values *= alpha.reshape(2, 2, 1)

    tick_kernel_values = torch.tensor(tick_kernel_values)

    events_grid = projected_grid(events, dt, size_grid)
    intensity_temp = torch.zeros(2, 2, size_grid)
    for i in range(2):
        intensity_temp[i, :, :] = torch.conv_transpose1d(
            events_grid[i].view(1, size_grid),
            tick_kernel_values[:, i].reshape(1, 2, L).float())[
                :, :-L + 1]
    intensity_tick = intensity_temp.sum(0) + torch.tensor(baseline).unsqueeze(1)
    results = dict(intens_tick=intensity_tick,
                   time=time.time() - start)

    return results


def run_experiment(baseline, alpha, mu, sigma, T, dt, seed=0):
    res = dict(T=T, dt=dt, seed=seed)
    L = int(1 / dt)
    size_grid = L * T + 1
    # simulate data
    events, intens = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=seed)
    events_grid = projected_grid(events, dt, size_grid)
    S = events_grid.T.numpy()
    S = np.array(S, dtype=int)

    # run fadin
    v = 0.2
    baseline_init = np.random.uniform(size=(2))
    alpha_init = np.random.uniform(size=(2, 2))
    mu_init = mu
    sigma_init = sigma + v
    u_init = mu_init - sigma_init
    results = run_fadin(events, u_init, sigma_init, baseline_init,
                        alpha_init, T, dt, seed=0)
    baseline_hat = results['param_baseline']
    alpha_hat = results['param_alpha']
    u_hat = results['param_kernel'][0]
    sigma_hat = results['param_kernel'][1]

    RC = DiscreteKernelFiniteSupport(dt, n_dim=2, kernel='raised_cosine',
                                     lower=0, upper=1)
    intens_fadin = RC.intensity_eval(torch.tensor(baseline_hat),
                                     torch.tensor(alpha_hat),
                                     [torch.Tensor(u_hat),
                                      torch.Tensor(sigma_hat)],
                                     events_grid, torch.linspace(0, 1, L))

    res['err_fadin'] = np.absolute(intens.numpy() - intens_fadin.numpy()).mean()
    res['time_fadin'] = results['time']

    results = run_gibbs(S, size_grid, dt, seed=seed)
    intens_gibbs = results['intens_gibbs']
    res['err_gibbs'] = np.absolute(intens.numpy() - intens_gibbs.T).mean()
    res['time_gibbs'] = results['time']

    results = run_vb(S, size_grid, dt, seed=seed)
    intens_vb = results['intens_vb']
    res['err_vb'] = np.absolute(intens.numpy() - intens_vb.T).mean()
    res['time_vb'] = results['time']

    results = run_non_param(S, size_grid, dt, seed=seed)
    intens_np = results['intens_nonparam_sgd']
    res['err_nonparam'] = np.absolute(intens.numpy() - intens_np.T).mean()
    res['time_nonparam'] = results['time']

    results = run_non_param_tick(events, size_grid, dt, seed=seed)
    intens_tick = results['intens_tick']
    res['err_nonparam_tick'] = np.absolute(intens.numpy() - intens_tick.numpy()).mean()
    res['time_nonparam_tick'] = results['time']

    return res


baseline = np.array([.1, .2])
alpha = np.array([[1.5, 0.1], [0.1, 1.5]])
mu = np.array([[0.4, 0.6], [0.55, 0.6]])
sigma = np.array([[0.3, 0.3], [0.25, 0.3]])


T = 1000
dt = 0.01
# res = run_experiment(baseline, alpha, mu, sigma, T, dt, seed=0)
# res
# %%
dt_list = [0.01]
T_list = [1000, 10000, 100000]  # , 1_000_000]
seeds = np.arange(10)

n_jobs = 40
all_results = Parallel(n_jobs=n_jobs, prefer='threads', verbose=1)(
    delayed(run_experiment)(baseline, alpha, mu, sigma, T, dt, seed=seed)
    for T, dt, seed in itertools.product(
        T_list, dt_list, seeds
    )
)

df = pd.DataFrame(all_results)
df.to_csv('results/benchmark.csv', index=False)

# %%

df = pd.read_csv('results/benchmark.csv')
