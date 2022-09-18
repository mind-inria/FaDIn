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

from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc

from hawkes_discret.kernels import KernelRaisedCosineDiscret
from hawkes_discret.hawkes_discret_l2 import HawkesDiscretL2

################################
## Meta parameters
################################
dt = 0.01
T = 100000
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
    RC = KernelRaisedCosineDiscret(dt)
    kernel_values = RC.eval(
        [torch.Tensor(u), torch.Tensor(sigma)], discretization
    )  # * dt
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


events = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=0)

# %% solver
##


@mem.cache
def run_solver(events, u_init, sigma_init, baseline_init, alpha_init, dt, T, seed=0):
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
        random_state=0,
        device="cpu",
        optimize_kernel=True
    )
    print(time.time()-start)
    results = solver.fit(events, T)
    results["time"] = time.time() - start
    results["seed"] = seed
    results["T"] = T
    results["dt"] = dt
    return results



baseline_init = baseline + np.random.rand()*0.5
alpha_init = alpha + np.random.rand()*0.5
mu_init = mu + np.random.rand()*0.5
sigma_init = sigma + np.random.rand()*0.2
u_init = mu_init - sigma_init 

start = time.time()
results_1 = run_solver(
    events, u_init, sigma_init, baseline_init, alpha_init, dt, T, seed=0
)
print(torch.abs(results_1['param_baseline'][-1] - baseline))
print(torch.abs(results_1['param_adjacency'][-1] - alpha))
print(torch.abs(results_1['param_kernel'][0][-1] - u))
print(torch.abs(results_1['param_kernel'][1][-1] - sigma))

print(time.time() - start)
file_name = "test2.pkl"
open_file = open(file_name, "wb")
pickle.dump(results_1, open_file)
open_file.close()

# %% eval on grid
##
def run_experiment(baseline, alpha, mu, sigma, T, dt, seed=0):
    v =  0.2
    events = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=seed)
    baseline_init = baseline + v #np.random.rand()*0.5
    alpha_init = alpha + v #np.random.rand()*0.5
    mu_init = mu + v #np.random.rand()*0.5
    sigma_init = sigma - v #np.random.rand()*0.2
    u_init = mu_init - sigma_init 
    results = run_solver(events, u_init, sigma_init, baseline_init, alpha_init, dt, T, seed)
    return results

T_list = [1000, 10_000, 100_000]
dt_list = np.logspace(1, 3, 10) / 10e3
#[0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02,  
#0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
seeds = np.arange(100)
info = dict(T_list=T_list, dt_list=dt_list, seeds=seeds)
n_jobs=60
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, alpha, mu, sigma, T, dt, seed=seed)
    for T, dt, seed in itertools.product(
        T_list, dt_list, seeds
    )
)
all_results.append(info)
file_name = "erreur_discret.pkl"
open_file = open(file_name, "wb")
pickle.dump(all_results, open_file)
open_file.close()

# %% get results
##
file_name = "erreur_discret.pkl"
open_file = open(file_name, "rb")
all_results = pickle.load(open_file)
open_file.close()

def get_results(results):
    baseline = np.array([1.1]); alpha = np.array([[0.8]]); 
    mu = np.array([[0.5]]); sigma = np.array([[0.3]]); u = mu - sigma
    dt_list = results[-1]['dt_list']; T_list = results[-1]['T_list'];
    seeds = results[-1]['seeds'];
    n_dt = len(dt_list); n_seeds = len(seeds); n_T = len(T_list)

    mu_vis = np.zeros((n_T, n_seeds, n_dt))
    alpha_vis = np.zeros((n_T, n_seeds, n_dt))
    u_vis = np.zeros((n_T, n_seeds, n_dt))
    sigma_vis = np.zeros((n_T, n_seeds, n_dt))
    loss_vis = np.zeros((n_T, n_seeds, n_dt))
    comptime_vis = np.zeros((n_T, n_seeds, n_dt))
    n_xp = len(results) - 1

    for j in range(n_T):
        for k in range(n_dt):
            for l in range(n_seeds):
                idx = j*(n_dt*n_seeds) + k*(n_seeds)  + l 
                mu_vis[j, k, l] = np.abs(results[idx]['param_baseline'][-1] - baseline)
                alpha_vis[j, k, l] = np.abs(results[idx]['param_adjacency'][-1] - alpha)
                u_vis[j, k, l] = np.abs(results[idx]['param_u'][-1] - u)
                sigma_vis[j, k, l] = np.abs(results[idx]['param_sigma'][-1] - sigma)
                loss_vis[j, k, l] = results[idx]['v_loss'][-1]
                comptime_vis[j, k, l] = results[idx]['time']

    return [mu_vis, alpha_vis, u_vis, sigma_vis, loss_vis, comptime_vis]

data = get_results(all_results)

def plot_one_curve(axs, data, title, dt_list, i, j, col, T):   
    lw = 2
    fontsize = 18
    if (i == 2 and j == 0):
        axs[2, 0].plot(dt_list, data.mean(0), lw=lw, label='T={}'.format(T),  c=col)
    else:
        axs[i, j].semilogy(dt_list, data.mean(0), lw=lw,  label='T={}'.format(T),  c=col)
    #axs[0, 0].axhline(y = baseline, color = 'k', linestyle = ':')
    axs[i, j].set_title(title, size=fontsize)
    axs[i, j].fill_between(dt_list, np.percentile(data, 10, axis=0), 
                        np.percentile(data, 90, axis=0), alpha=0.1, color=col)
    axs[i, j].set_xlim(dt_list[-1], dt_list[0])
    axs[i, j].set_xlabel('dt', size=fontsize)
    axs[i, j].invert_xaxis()
    axs[i, j].legend(fontsize=20)
    axs[i, j].set_xticks(dt_list)
    axs[i, j].set_xscale('log')
    return 0;

titles = ['Baseline', 'Alpha', 'u', 'Sigma', 'Loss', 'Computation time']
# %% plot loss
#%matplotlib inline
matplotlib.rc("xtick", labelsize=13)
matplotlib.rc("ytick", labelsize=13)

fig, axs = plt.subplots(3, 2, figsize=(15, 20))
fig.suptitle('Approximation w.r.t. dt for various T', fontsize=30)

for i in range(3):
    for j in range(2):
        idx = i*2+j
        plot_one_curve(axs, data[idx][0], titles[idx], dt_list, i, j, 'g', T_list[0])
        plot_one_curve(axs, data[idx][1], titles[idx], dt_list, i, j, 'b', T_list[1])
        plot_one_curve(axs, data[idx][2], titles[idx], dt_list, i, j, 'r', T_list[2])

