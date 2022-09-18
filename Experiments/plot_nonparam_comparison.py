# %% get results
#

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import skewnorm 

from hawkes_discret.kernels import KernelRaisedCosineDiscret

file_name = "results/non_param.pkl"
open_file = open(file_name, "rb")
all_results = pickle.load(open_file)
open_file.close()

def get_results(results):
    dt_list = results[-1]['dt_list'] 
    T_list = results[-1]['T_list']
    seeds = results[-1]['seeds']
    n_dt = len(dt_list); n_seeds = len(seeds); n_T = len(T_list)

    our_results = np.zeros((2, n_T, n_seeds, n_dt))
    tick_results = np.zeros((2, n_T, n_seeds, n_dt))
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
    dt_list = all_results[-1]['dt_list']
    L = int(1/dt)
    discretization = torch.linspace(0,1, L)
    discretisation = np.linspace(0, 1, L)
    alpha = np.array([[0.8]])
    if kernel == 'RC':
        K_idx = 0
        mu = np.array([[0.5]])
        sigma = np.array([[0.3]])
        u = mu - sigma  
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

#fig.savefig('plots/comparison_nonparam.pdf')
# %%
