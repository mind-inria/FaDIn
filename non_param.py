# %% import stuff
## import libraries
import itertools
import pickle
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from joblib import Memory, Parallel, delayed
from scipy.stats import skewnorm #, beta

from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc
from tick.hawkes import HawkesBasisKernels

from hawkes_discret.kernels import KernelRaisedCosineDiscret
from hawkes_discret.hawkes_discret_l2 import HawkesDiscretL2

# %% simulate data
# Simulated data
################################
dt = 0.01
T = 10000
n_jobs = 30
size_grid = int(T/dt) + 1

mem = Memory(location=".", verbose=2)

baseline = np.array([1.1])
alpha = np.array([[0.8]])
mu = np.array([[0.5]])
sigma = np.array([[0.3]])
u = mu - sigma


@mem.cache
def simulate_data(baseline, alpha, mu, sigma, T, dt, seed=0, kernel='RC'):
    L = int(1 / dt)
    discretization = torch.linspace(0, 1, L)
    u = mu - sigma
    if kernel == 'RC':
        RC = KernelRaisedCosineDiscret(dt)
        kernel_values = RC.eval(
            [torch.Tensor(u), torch.Tensor(sigma)], discretization
        )  # * dt
        kernel_values = kernel_values * alpha[:, :, None]
        k = kernel_values[0, 0].double().numpy()
    elif kernel == 'SG':
        kernel_values = torch.tensor(skewnorm.pdf(np.linspace(-3, 3, L), 3))
        kernel_values = kernel_values * alpha[:, :, None]
        k = kernel_values.squeeze().numpy()
        
    #kernel_values = torch.tensor(beta.pdf(discretization.numpy(), 2, 3))

    t_values = discretization.double().numpy()
    

    tf = HawkesKernelTimeFunc(t_values=t_values, y_values=k)
    kernels = [[tf]]
    hawkes = SimuHawkes(
        baseline=baseline, kernels=kernels, end_time=T, verbose=False, seed=int(seed)
    )

    hawkes.simulate()
    events = hawkes.timestamps
    return events, hawkes


events, hawkes = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=1, kernel='SG')

# %% solver
##


@mem.cache
def run_solver(events, u_init, sigma_init, baseline_init, alpha_init, T, dt, seed=0):
    start = time.time()
    max_iter = 800
    solver = HawkesDiscretL2(
        "RaisedCosine",
        [torch.tensor(u_init),
        torch.tensor(sigma_init)],
        torch.tensor(baseline_init),
        torch.tensor(alpha_init),
        dt,
        solver="RMSprop",
        step_size=1e-3,
        max_iter=max_iter,
        log=False,
        random_state=seed,
        device="cpu",
        optimize_kernel=True
    )
    results = solver.fit(events, T) 
    return results


# %%

def run_experiment(baseline, alpha, mu, sigma, T, dt, seed=0, kernel='RC'):
    
    events, _ = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=seed, kernel=kernel)
    baseline_init = np.array([np.random.rand()])
    alpha_init = np.array([[np.random.rand()]])
    mu_init = np.array([[np.random.rand()]])
    sigma_init = 10
    while (sigma_init > mu_init):
        sigma_init = np.array([[np.random.rand()]])
    u_init = mu_init - sigma_init 

    start_our = time.time()
    results = run_solver(events, u_init, sigma_init, baseline_init, alpha_init, T, dt, seed=0)
    time_our = time.time() - start_our
    
    start_tick = time.time()
    non_param = HawkesBasisKernels(1, n_basis=1, kernel_size=int(1/dt), max_iter=800)
    non_param.fit(events)
    time_tick = time.time() - start_tick

    discretization = torch.linspace(0, 1, int(1/dt))
    u_hd = results['param_kernel'][0][-1]
    sigma_hd = results['param_kernel'][1][-1]
    alpha_hd = results['param_adjacency'][-1]

    RC = KernelRaisedCosineDiscret(dt)
    kernel_values = RC.eval([torch.Tensor(u_hd), 
                            torch.Tensor(sigma_hd)], 
                            discretization).squeeze().numpy() 
    kernel_values *= alpha_hd.item()



    res_our = dict(kernel=kernel_values, comp_time=time_our, kernel_name=kernel, T=T, dt=dt, seed=seed)

    tick_values = non_param.get_kernel_values(0, 0, discretization[:-1])
    tick_values *= alpha.item() 
    
    res_tick = dict(kernel=tick_values, comp_time=time_tick, kernel_name=kernel, T=T, dt=dt, seed=seed)

    return res_our, res_tick
# %% run 

dt_list = np.logspace(1, 3, 10) / 10e3
T_list = [1000, 10_000, 100_000]#, 10_000, 100_000]
seeds = np.arange(100)
kernel_ = ['RC', 'SG']
info = dict(kernel=kernel_, T_list=T_list, dt_list=dt_list, seeds=seeds)

n_jobs=30
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, alpha, mu, sigma, T, dt, seed=seed, kernel=kernel)
    for kernel, T, dt, seed in itertools.product(
        kernel_, T_list, dt_list, seeds
    )
)
all_results.append(info)
file_name = "non_param.pkl"
open_file = open(file_name, "wb")
pickle.dump(all_results, open_file)
open_file.close()
1/0


# %% get results
##
file_name = "non_param.pkl"
open_file = open(file_name, "rb")
all_results = pickle.load(open_file)
open_file.close()

def get_results(results):
    dt_list = results[-1]['dt_list']; T_list = results[-1]['T_list'];
    seeds = results[-1]['seeds']; kernel = results[-1]['kernel']
    n_dt = len(dt_list); n_seeds = len(seeds); n_T = len(T_list)

    our_results = np.zeros((2, n_T, n_seeds, n_dt))
    tick_results = np.zeros((2, n_T, n_seeds, n_dt))
    n_xp = len(results) - 1
    for i in range(2):
         for j in range(n_T):
            for k in range(n_dt):
                for l in range(n_seeds):
                    idx = i*(n_T*n_dt*n_seeds) + j*(n_dt*n_seeds) + k*(n_seeds) + l 
                    our_results[i, j, k, l] = all_results[idx][0]['comp_time']
                    tick_results[i, j, k, l] = all_results[idx][1]['comp_time']
    return our_results, tick_results

comp_time_our, comp_time_tick = get_results(all_results)
# %% plot kernel 

def plot_nonparam(all_results, comp_time_our, comp_time_tick, l, T_idx, kernel='RC'):


    dt = all_results[l][0]['dt']

    L = int(1/dt)
    discretization = torch.linspace(0,1, L)
    discretisation = np.linspace(0, 1, L)

    if kernel == 'RC':
        K_idx = 0
        RC = KernelRaisedCosineDiscret(dt) 
        true_values = RC.eval(
            [torch.Tensor(u), torch.Tensor(sigma)], discretization
        ).squeeze().numpy()  
    else:
        K_idx = 1
        true_values = skewnorm.pdf(np.linspace(-3, 3, L), 3)

    true_values *= alpha.item() 

    our = comp_time_our[K_idx, T_idx, :, :]
    tick = comp_time_tick[K_idx, T_idx, :, :]

    our_kernel = all_results[l][0]['kernel']
    tick_kernel = all_results[l][1]['kernel']
    T = all_results[-1]['T_list'][T_idx]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Comparison to non parametric kernel approach, T={}'.format(T), fontsize=20)
    fontsize=20
    lw=2
    ax[0].step(discretisation[:-1], tick_kernel, label='Non-param', linewidth=lw, c='b')
    ax[0].plot(discretisation, our_kernel, label='Our', lw=lw, c='r')
    ax[0].plot(discretisation, true_values, label='True kernel', lw=lw, c='k')
    ax[0].set_xlim(0, 1)
    ax[0].set_title('Estimated kernel, dt={}'.format(np.round(all_results[l][0]['dt'], 3)), size=fontsize)
    ax[0].legend(fontsize=fontsize-5)

    ax[1].semilogy(dt_list, our.mean(1), label='Our', lw=lw, c='r')
    ax[1].semilogy(dt_list, tick.mean(1), label='Non-param', lw=lw, c='b')
    ax[1].set_title('Computation time', size=fontsize)
    ax[1].fill_between(dt_list, np.percentile(our, 10, axis=1), 
                        np.percentile(our, 90, axis=1), alpha=0.1, color='r')
    ax[1].fill_between(dt_list, np.percentile(tick, 10, axis=1), 
                        np.percentile(tick, 90, axis=1), alpha=0.1, color='b')
    ax[1].set_xlim(dt_list[0], dt_list[-1])
    ax[1].set_xlabel('dt', size=fontsize)
    ax[1].invert_xaxis()
    ax[1].legend(fontsize=20)
    ax[1].set_xticks(dt_list)
    ax[1].set_xscale('log')

    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    return 0;

#%matplotlib inline
matplotlib.rc("xtick", labelsize=13)
matplotlib.rc("ytick", labelsize=13)


##dt = 0.01
plot_nonparam(all_results, comp_time_our, comp_time_tick, 50, 0, kernel='RC')
plot_nonparam(all_results, comp_time_our, comp_time_tick, 150, 1, kernel='RC')
plot_nonparam(all_results, comp_time_our, comp_time_tick, 250, 2, kernel='RC')

## dt=0.001
plot_nonparam(all_results, comp_time_our, comp_time_tick, 60, 0, kernel='RC')
plot_nonparam(all_results, comp_time_our, comp_time_tick, 101, 1, kernel='RC')
plot_nonparam(all_results, comp_time_our, comp_time_tick, 206, 2, kernel='RC')

##dt = 0.01
plot_nonparam(all_results, comp_time_our, comp_time_tick, 350, 0, kernel='SG')
plot_nonparam(all_results, comp_time_our, comp_time_tick, 450, 1, kernel='SG')
plot_nonparam(all_results, comp_time_our, comp_time_tick, 550, 2, kernel='SG')

## dt=0.001
# we cannot take dt=0.001 for T=1000; not enough data to estimate with non param
plot_nonparam(all_results, comp_time_our, comp_time_tick, 360, 0, kernel='SG')
plot_nonparam(all_results, comp_time_our, comp_time_tick, 401, 1, kernel='SG')
plot_nonparam(all_results, comp_time_our, comp_time_tick, 506, 2, kernel='SG')

