# %% get results
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

file_name = "results/erreur_discret_tick.pkl"
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
fig.savefig('plots/approx_dt_tick.pdf')
# %%
