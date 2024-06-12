# %% import stuff
# import libraries
import itertools
import pickle
import time
import numpy as np
import torch
from joblib import Memory, Parallel, delayed

from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc

from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.solver import FaDIn
from fadin.loss_and_gradient import discrete_l2_loss_conv

################################
# Define solver without precomputations
################################


def optim_iteration_l2_noprecomput(solver, events_grid, discretization,
                                   i, n_events, end_time):
    """One optimizer iteration of FaDIn_no_precomputations solver,
    with l2 loss and no precomputations."""
    intens = solver.kernel_model.intensity_eval(
        solver.params_intens[0],
        solver.params_intens[1],
        solver.params_intens[2:],
        events_grid,
        discretization
    )
    loss = discrete_l2_loss_conv(intens, events_grid, solver.delta)
    loss.backward()


class FaDInNoPrecomputations(FaDIn):
    """Define the FaDIn framework for estimated Hawkes processes *without
    precomputations*."""
    optim_iteration = staticmethod(optim_iteration_l2_noprecomput)
    precomputations = False


################################
# Meta parameters
################################
dt = 0.1
T = 1000
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


@mem.cache
def simulate_data(baseline, alpha, mu, sigma, T, dt, seed=0):
    L = int(1 / dt)
    discretization = torch.linspace(0, 1, L)
    u = mu - sigma
    RC = DiscreteKernelFiniteSupport(dt, 1, kernel='raised_cosine')
    kernel_values = RC.kernel_eval([torch.Tensor(u), torch.Tensor(sigma)],
                                   discretization)  # * dt
    kernel_values = kernel_values * alpha[:, :, None]

    t_values = discretization.double().numpy()
    k = kernel_values[0, 0].double().numpy()

    tf = HawkesKernelTimeFunc(t_values=t_values, y_values=k)
    kernels = [[tf]]
    hawkes = SimuHawkes(
        baseline=baseline,
        kernels=kernels,
        end_time=T,
        verbose=False,
        seed=int(seed)
    )

    hawkes.simulate()
    events = hawkes.timestamps
    return events


@mem.cache
def run_solver(events, u_init, sigma_init, baseline_init, alpha_init, T, dt,
               seed=0):
    max_iter = 800
    init = {
        'alpha': torch.tensor(alpha_init),
        'baseline': torch.tensor(baseline_init),
        'kernel': [torch.tensor(u_init), torch.tensor(sigma_init)]
    }
    solver_autodiff = FaDInNoPrecomputations(
        1,
        "raised_cosine",
        init=init,
        delta=dt, optim="RMSprop",
        step_size=1e-3, max_iter=max_iter,
        random_state=0
    )
    start_autodiff = time.time()
    solver_autodiff.fit(events, T)
    time_autodiff = time.time() - start_autodiff
    init = {
        'alpha': torch.tensor(alpha_init),
        'baseline': torch.tensor(baseline_init),
        'kernel': [torch.tensor(u_init), torch.tensor(sigma_init)]
    }
    solver_FaDIn = FaDIn(
        1,
        "raised_cosine",
        init=init,
        delta=dt, optim="RMSprop",
        step_size=1e-3, max_iter=max_iter,
        random_state=0
    )
    start_FaDIn = time.time()
    solver_FaDIn.fit(events, T)
    time_FaDIn = time.time() - start_FaDIn

    results = dict(time_autodiff=time_autodiff, time_FaDIn=time_FaDIn)

    results["seed"] = seed
    results["T"] = T
    results["dt"] = dt

    return results


# %% example
v = 0.2
events = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=0)
baseline_init = baseline + v
alpha_init = alpha + v
mu_init = mu + v
sigma_init = sigma - v
u_init = mu_init - sigma_init
# run_solver(events, u_init, sigma_init, baseline_init, alpha_init, dt, T, seed=0)
# %% eval on grid


def run_experiment(baseline, alpha, mu, sigma, T, dt, seed=0):
    v = 0.2
    events = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=seed)
    baseline_init = baseline + v
    alpha_init = alpha + v
    mu_init = mu + v
    sigma_init = sigma - v
    u_init = mu_init - sigma_init
    results = run_solver(events, u_init, sigma_init, baseline_init, alpha_init,
                         T, dt, seed)
    return results


T_list = [100000, 1000000]
dt_list = [0.1, 0.01]
seeds = np.arange(10)
info = dict(T_list=T_list, dt_list=dt_list, seeds=seeds)

n_jobs = 1
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, alpha, mu, sigma, T, dt, seed=seed)
    for T, dt, seed in itertools.product(
        T_list, dt_list, seeds
    )
)
all_results.append(info)
file_name = "results/comp_autodiff.pkl"
open_file = open(file_name, "wb")
pickle.dump(all_results, open_file)
open_file.close()
