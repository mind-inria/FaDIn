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

from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc, HawkesExpKern, HawkesKernelExp

from hawkes_discret.kernels import KernelExpDiscret
from hawkes_discret.hawkes_discret_l2 import HawkesDiscretL2

from hawkes_discret.utils.compute_constants_np import get_ztzG, _get_ztzG
from hawkes_discret.utils.utils import projected_grid
################################
## Meta parameters
################################
dt = 0.01
T = 10000
size_grid = int(T / dt) + 1

mem = Memory(location=".", verbose=2)

# %% simulate data
# Simulated data
################################

baseline = np.array([.1])
alpha = np.array([[0.8]])
decay= np.array([[7]])

def exp_f(decay, discretisation):
    return decay*np.exp(-decay*discretisation)

@mem.cache
def simulate_data(baseline, alpha, decay, T, dt, seed=0):
    L = int(1 / dt)
    discretization = torch.linspace(0, 1, 10000)

    #Exp = KernelExpDiscret(1, dt)
    #kernel_values = Exp.eval([torch.tensor(decay)], discretization
    #)  # * dt
    #kernel_values = exp_f(torch.tensor(decay), discretization)
    #kernel_values = kernel_values * alpha[:, :, None]
    
    t_values = discretization.double().numpy()
    #k = kernel_values[0, 0].double().numpy()
    
    #tf = HawkesKernelTimeFunc(t_values=t_values, y_values=k)
    tf = HawkesKernelExp(alpha.item(), decay.item())
    kernels = [[tf]]
    hawkes = SimuHawkes(
        baseline=baseline, kernels=kernels, end_time=T, verbose=False, seed=int(seed)
    )

    hawkes.simulate()
    events = hawkes.timestamps
    return events


events = simulate_data(baseline, alpha, decay, T, dt, seed=0)


#plt.plot(decay.item()*np.exp(-decay.item()*np.linspace(0, 1, 1000)))

# %% one run test
# 
@mem.cache
def run_solver(events, decay, baseline_init, alpha_init,  T, dt, seed=0):
    start = time.time()
    max_iter = 2000
    solver = HawkesDiscretL2(
        "KernelExpDiscret",
        [torch.tensor(decay)],
        torch.tensor(baseline_init),
        torch.tensor(alpha_init),
        dt,
        solver='GD',
        step_size=1e-3,
        max_iter=max_iter,
        log=False,
        random_state=0,
        device="cpu",
        optimize_kernel=False
    )
    print(time.time()-start)
    results = solver.fit(events, T)
    results["time"] = time.time() - start
    results["seed"] = seed
    results["T"] = T
    results["dt"] = dt
    return results


np.random.seed(3)
baseline_init = np.array([np.random.rand()*0.5])
alpha_init = np.array([[np.random.rand()]])
#decay_init = decay + np.random.rand()*0.5


start = time.time()
results_1 = run_solver(
    events, decay, baseline_init, alpha_init, T, dt, seed=0
)
baseline_our = results_1['param_baseline'][-1]
adjacency_our = results_1['param_adjacency'][-1]
print(torch.abs(results_1['param_baseline'][-1])) #- baseline))
print(torch.abs(results_1['param_adjacency'][-1])) #- alpha))
print(torch.abs(results_1['param_kernel'][0][-1] ))#- decay))
# %% Tick solver 

solver_tick = HawkesExpKern(7, gofit='least-squares', penalty='none',
 C=1000.0, solver='gd', step=None, tol=1e-05, max_iter=2000, 
 verbose=True, print_every=10, record_every=10, 
 elastic_net_ratio=0.95, random_state=0)
solver_tick.fit(events, start=np.array([baseline_init.item(),alpha_init.item()]))
baseline_tick = solver_tick.baseline
adjacency_tick = solver_tick.adjacency
print(baseline_tick)
print(adjacency_tick)

# %% Experiment

def run_experiment(baseline, alpha, decay, T, dt, seed=0):
    np.random.seed(seed)
    events = simulate_data(baseline, alpha, decay, T, dt, seed=0)
    
    baseline_init = np.array([np.random.rand()*0.5])
    alpha_init = np.array([[np.random.rand()]]) 

    start = time.time()
    results = run_solver(events, decay, 
                         baseline_init, alpha_init, T, dt, seed=0)
    our_time =    time.time() - start                  
    print('our solver in:', our_time)

    start = time.time()
    solver_tick = HawkesExpKern(decay.item(), gofit='least-squares', penalty='none',
                C=1000.0, solver='agd', step=None, tol=1e-05, max_iter=2000, 
                verbose=False, print_every=10, record_every=10, 
                elastic_net_ratio=0.95, random_state=0)
    solver_tick.fit(events, start=np.array([baseline_init.item(),alpha_init.item()]))
    tick_time = time.time() - start
    print('tick solver in:', tick_time)

    baseline_our = results['param_baseline'][-10:].mean()
    adjacency_our = results['param_adjacency'][-10:].mean()

    baseline_tick = solver_tick.baseline
    adjacency_tick = solver_tick.adjacency

    baseline_diff = np.abs(baseline_our.numpy() - baseline_tick)
    adjacency_diff = np.abs(adjacency_our.numpy() - adjacency_tick)

    res = dict(baseline_diff=baseline_diff, adjacency_diff=adjacency_diff, 
               our_time=our_time, tick_time=tick_time, T=T, dt=dt, seed=seed)
    return res


baseline = np.array([.2])
alpha = np.array([[0.8]])
decay= np.array([[7]])
T_list = [1000, 10_000, 100_000]
dt_list = np.logspace(1, 3, 20) / 10e3
#[0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02,  
#0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
seeds = np.arange(100)
info = dict(T_list=T_list, dt_list=dt_list, seeds=seeds)
n_jobs=60
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, alpha, decay, T, dt, seed=seed)
    for T, dt, seed in itertools.product(
        T_list, dt_list, seeds
    )
)
all_results.append(info)
file_name = "erreur_discret_tick.pkl"
open_file = open(file_name, "wb")
pickle.dump(all_results, open_file)
open_file.close()

1/0
# %% get results:

# %% get results
##
file_name = "erreur_discret_tick.pkl"
open_file = open(file_name, "rb")
all_results = pickle.load(open_file)
open_file.close()

def get_results(results):
    dt_list = results[-1]['dt_list']; T_list = results[-1]['T_list'];
    seeds = results[-1]['seeds'];
    n_dt = len(dt_list); n_seeds = len(seeds); n_T = len(T_list)

    mu_vis = np.zeros((n_T, n_seeds, n_dt))
    alpha_vis = np.zeros((n_T, n_seeds, n_dt))
    comptime_vis = np.zeros((n_T, 2, n_seeds, n_dt))

    for j in range(n_T):
        for k in range(n_dt):
            for l in range(n_seeds):
                idx = j*(n_dt*n_seeds) + k*(n_seeds)  + l 
                mu_vis[j, l, k] = np.abs(results[idx]['baseline_diff'].item())
                alpha_vis[j, l, k] = np.abs(results[idx]['adjacency_diff'].item())
                comptime_vis[j, 0, l, k] = results[idx]['our_time']
                comptime_vis[j, 1, l, k] = results[idx]['tick_time']

    return [mu_vis, alpha_vis, comptime_vis]

data = get_results(all_results)

def plot_one_curve(axs, data, title, dt_list, i, col, T):   
    lw = 2
    fontsize = 18
    if i == 2:
        axs[i].semilogy(dt_list, data[0].mean(0), lw=lw,  
                        linestyle='dashed', label='our model T={}'.format(T),  c='c')
        axs[i].semilogy(dt_list, data[1].mean(0), lw=lw, 
                        linestyle='dashdot',  label='tick T={}'.format(T),  c='m')
        axs[i].fill_between(dt_list, np.percentile(data[0], 10, axis=0), 
                        np.percentile(data[0], 90, axis=0), alpha=0.1, color='c')
        axs[i].fill_between(dt_list, np.percentile(data[1], 10, axis=0), 
                        np.percentile(data[1], 90, axis=0), alpha=0.1, color='m')                                               
    else:    
        axs[i].semilogy(dt_list, data.mean(0), lw=lw,  label='T={}'.format(T),  c=col)
        axs[i].fill_between(dt_list, np.percentile(data, 10, axis=0), 
                        np.percentile(data, 90, axis=0), alpha=0.1, color=col)
    #axs[0, 0].axhline(y = baseline, color = 'k', linestyle = ':')
    axs[i].set_title(title, size=fontsize)

    axs[i].set_xlim(dt_list[0], dt_list[-1])
    axs[i].set_xlabel('dt', size=fontsize)
    axs[i].invert_xaxis()
    axs[i].legend(fontsize=20)
    axs[i].set_xticks(dt_list)
    axs[i].set_xscale('log')
    return 0;

titles = ['Baseline', 'Alpha', 'Computation time']
# %% plot loss
#%matplotlib inline
matplotlib.rc("xtick", labelsize=13)
matplotlib.rc("ytick", labelsize=13)

dt_list = all_results[-1]['dt_list']; T_list = all_results[-1]['T_list'];
seeds = all_results[-1]['seeds'];

fig, axs = plt.subplots(2, 1, figsize=(15, 12))
fig.suptitle('Approximation w.r.t. continuous solver for various T', fontsize=30)
for i in range(2):
    plot_one_curve(axs, data[i][0], titles[i], dt_list, i,  'g', T_list[0])
    plot_one_curve(axs, data[i][1], titles[i], dt_list, i,  'b', T_list[1])
    #plot_one_curve(axs, data[i][2], titles[i], dt_list, i, 'r', T_list[2])
fig.tight_layout()
fig.savefig('results/approx_dt_tick.pdf')
# %%
