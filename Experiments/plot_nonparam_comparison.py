# %% get results
#

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from scipy.stats import skewnorm 

from hawkes_discret.kernels import KernelRaisedCosineDiscret

# Load non-param results and FaDIn
file_name = "non_param.pkl"
open_file = open(file_name, "rb")
all_results = pickle.load(open_file)
open_file.close()

def get_results(results):
    dt_list = results[-1]['dt_list'] 
    T_list = results[-1]['T_list']
    seeds = results[-1]['seeds']
    n_dt = len(dt_list); n_seeds = len(seeds); n_T = len(T_list)

    our_results = np.zeros((2, n_T, n_dt, n_seeds))
    tick_results = np.zeros((2, n_T, n_dt, n_seeds))
    for i in range(2):
         for j in range(n_T):
            for k in range(n_dt):
                for l in range(n_seeds):
                    idx = i*(n_T*n_dt*n_seeds) + j*(n_dt*n_seeds) + k*(n_seeds) + l 
                    our_results[i, j, k, l] = all_results[idx][0]['comp_time']
                    tick_results[i, j, k, l] = all_results[idx][1]['comp_time']
    return our_results, tick_results

comp_time_FaDIn, comp_time_tick = get_results(all_results)
## %% plot kernel 


## Load results on autodiff solver
file_name = "comp_autodiff_long_run.pkl"
open_file = open(file_name, "rb")
results_autodiff = pickle.load(open_file)
open_file.close()

dt_list_ = results_autodiff[-1]['dt_list']
T_list_ = results_autodiff[-1]['T_list']
seeds_ = results_autodiff[-1]['seeds']

def get_results_autodiff(results, T_list, dt_list, seeds):
    n_dt = len(dt_list); n_seeds = len(seeds); n_T = len(T_list)

    comptime_autodiff = np.zeros((n_T, n_dt, n_seeds))
    for j in range(n_T):
        for k in range(n_dt):
            for l in range(n_seeds):
                idx = j*(n_dt*n_seeds) + k*(n_seeds)  + l 
                comptime_autodiff[j, k, l] = results[idx]['time_autodiff']

    return  comptime_autodiff

comptime_autodiff = get_results_autodiff(results_autodiff, T_list_, dt_list_, seeds_)




def mean_int(x, y, dt):
    c = y.sum()
    return (x*y).sum() / c  #come from the reparametrization
def plot_nonparam(all_results, comp_time_our, comp_time_tick, comptime_autodiff, l, T_idx, kernel='RC'):


    dt = all_results[l][0]['dt']
    dt_list = all_results[-1]['dt_list']
    T_list = all_results[-1]['T_list']
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
    our_ = comp_time_our[K_idx, 3, :, :]
    tick_ = comp_time_tick[K_idx, 3, :, :]

    autodiff = comptime_autodiff[0]

    our_kernel = all_results[l][0]['kernel']
    tick_kernel = all_results[l][1]['kernel']
    tick_kernel = np.insert(tick_kernel, [-1], tick_kernel[-1])
    T = all_results[-1]['T_list'][T_idx]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fontsize=15
    lw=2
    
    ax[0].step(discretisation, tick_kernel/0.8, label='Non-param', linewidth=lw, c='orange')
    ax[0].plot(discretisation, our_kernel, label='FaDIn', lw=lw, c='b')
    ax[0].plot(discretisation, true_values, label='True kernel', lw=lw, c='k')
    ax[0].set_xlim(0,1)

    if T_idx == 0:
        custom_lines_M = [Line2D([0], [0], color='b', lw=3),
                          Line2D([0], [0], color='orange', lw=3), 
                          Line2D([0], [0], color='k', lw=3)]

        ax[0].legend(custom_lines_M, 
                    ['FaDIn', 'Non-param', 'True kernel'], 
                    fontsize=fontsize, 
                    bbox_to_anchor=(0.92, 1.2), 
                    ncol=2)  

    ax[1].loglog(dt_list, our.mean(1), lw=lw, c='b', linestyle=':')
    ax[1].loglog(dt_list, tick.mean(1), lw=lw, c='orange', linestyle=':')
    ax[1].loglog([0.1, 0.01], autodiff.mean(1), lw=lw, c='r', linestyle=':')
    
    ax[1].fill_between(dt_list, np.percentile(our, 20, axis=1), 
                        np.percentile(our, 80, axis=1), alpha=0.1, color='b')
    ax[1].fill_between(dt_list, np.percentile(tick, 20, axis=1), 
                        np.percentile(tick, 80, axis=1), alpha=0.1, color='orange')
    ax[1].fill_between([0.1, 0.01], np.percentile(autodiff, 20, axis=1), 
                        np.percentile(autodiff, 80, axis=1), alpha=0.1, color='r')


    ax[1].loglog(dt_list, our_.mean(1),  lw=lw, c='b', linestyle='--')
    ax[1].loglog(dt_list, tick_.mean(1),  lw=lw, c='orange', linestyle='--')

    ax[1].fill_between(dt_list, np.percentile(our_, 20, axis=1), 
                        np.percentile(our_, 80, axis=1), alpha=0.1, color='b')
    ax[1].fill_between(dt_list, np.percentile(tick_, 20, axis=1), 
                        np.percentile(tick_, 80, axis=1), alpha=0.1, color='orange')                       
    ax[1].set_xlim(dt_list[0], dt_list[-1])
    ax[1].set_xlabel('$\Delta$', size=fontsize)
    ax[1].set_ylabel('Time (s.)', size=fontsize)


    custom_lines_T = [Line2D([0], [0], color='k', lw=3, ls=':'),
                      Line2D([0], [0], color='k', lw=3, ls='--')]

    custom_lines_m = [Line2D([0], [0], color='b', lw=3),
                      Line2D([0], [0], color='orange', lw=3),
                      Line2D([0], [0], color='r', lw=3)]    

    first_legend = ax[1].legend(custom_lines_T, 
                                ['T={}'.format(T_list[T_idx]), 
                                'T={}'.format(T_list[3])], 
                                fontsize=fontsize, 
                                bbox_to_anchor=(0.43, 1.2))
    ax[1].add_artist(first_legend)          
    ax[1].legend(custom_lines_m, 
                ['FaDIn', 'Non-param', 'L2-Autodiff'], 
                fontsize=fontsize, 
                bbox_to_anchor=(0.9, 1.2))      
                
    ax[1].set_xticks(dt_list)

    fig.tight_layout()
    fig.savefig(
    'plots/nonparam/ker_comparison_nonparam_T={}_dt={}_K={}.pdf'.format(T, \
        np.round(all_results[l][0]['dt'], 3), kernel),
    # we need a bounding box in inches
    bbox_inches=mtransforms.Bbox(
        # This is in "figure fraction" for the bottom half
        # input in [[xmin, ymin], [xmax, ymax]]
        [[0, 0], [0.48, 1]]
    ).transformed(fig.transFigure - fig.dpi_scale_trans
    ),
    )
    fig.savefig(
    'plots/nonparam/time_comparison_nonparam_T={}_K={}.pdf'.format(T, kernel),
    bbox_inches=mtransforms.Bbox([[0.48, 0], [1, 1]]).transformed(
        fig.transFigure - fig.dpi_scale_trans
    ),
)
    return 0;

#%matplotlib inline
matplotlib.rc("xtick", labelsize=20)
matplotlib.rc("ytick", labelsize=20)


##dt = 0.01
plot_nonparam(all_results, comp_time_FaDIn, comp_time_tick, comptime_autodiff, 10, 0, kernel='RC')
plot_nonparam(all_results, comp_time_FaDIn, comp_time_tick, comptime_autodiff, 70, 2, kernel='RC')

#fig.savefig('plots/comparison_nonparam.pdf')
# %%
