# %% import libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

file_name = "results/error_stat.pkl"
open_file = open(file_name, "rb")
all_results = pickle.load(open_file)
open_file.close()



# %% plot results
##
dt_list = all_results[-1]['dt_list']
T_list = all_results[-1]['T_list']
seeds = all_results[-1]['seeds']

def get_results(results, T_list, dt_list, seeds):
    baseline = np.array([1.1]); alpha = np.array([[0.8]]); 
    mu = np.array([[0.5]]); sigma = np.array([[0.3]]); u = mu - sigma

    n_dt = len(dt_list); n_seeds = len(seeds); n_T = len(T_list)

    mu_vis = np.zeros((n_T, n_dt, n_seeds))
    alpha_vis = np.zeros((n_T, n_dt, n_seeds))
    u_vis = np.zeros((n_T, n_dt, n_seeds))
    sigma_vis = np.zeros((n_T, n_dt, n_seeds))
    loss_vis = np.zeros((n_T, n_dt, n_seeds))
    comptime_vis = np.zeros((n_T, n_dt, n_seeds))
    n_xp = len(results) - 1

    for j in range(n_T):
        for k in range(n_dt):
            for l in range(n_seeds):
                idx = j*(n_dt*n_seeds) + k*(n_seeds)  + l 
                mu_vis[j, k, l] = np.abs(results[idx]['param_baseline'][-10:].mean() - baseline)
                alpha_vis[j, k, l] = np.abs(results[idx]['param_adjacency'][-10:].mean() - alpha)
                u_vis[j, k, l] = np.abs(results[idx]['param_kernel'][0][-10:].mean() - u)
                sigma_vis[j, k, l] = np.abs(results[idx]['param_kernel'][1][-10:].mean() - sigma)
                loss_vis[j, k, l] = results[idx]['v_loss'][-1]
                comptime_vis[j, k, l] = results[idx]['time']

    return [mu_vis, alpha_vis, u_vis, sigma_vis, loss_vis, comptime_vis]

data = get_results(all_results, T_list, dt_list, seeds)


def plot_one_curve(axs, data, title, T_list, i, j, col, dt):   
    lw = 2
    fontsize = 18
    if (i == 2 and j == 0):
        axs[2, 0].plot(T_list, data.mean(1), lw=lw, label='dt={}'.format(dt),  c=col)
    else:
        axs[i, j].semilogy(T_list, data.mean(1), lw=lw,  label='dt={}'.format(dt),  c=col)

    axs[i, j].set_title(title, size=fontsize)
    axs[i, j].fill_between(T_list, np.percentile(data, 10, axis=1), 
                        np.percentile(data, 90, axis=1), alpha=0.1, color=col)
    axs[i, j].set_xlim(T_list[0], T_list[-1])
    axs[i, j].set_xlabel('T', size=fontsize+10)
    #axs[i, j].invert_xaxis()
    axs[i, j].legend(fontsize=20)
    axs[i, j].set_xticks(T_list)
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
        plot_one_curve(axs, data[idx][:, 0, :], titles[idx], T_list, i, j, 'b', dt_list[0])
        plot_one_curve(axs, data[idx][:, 1, :], titles[idx], T_list, i, j, 'r', dt_list[1])
        #plot_one_curve(axs, data[idx][2], titles[idx], dt_list, i, j, 'r', T_list[2])
fig.tight_layout()
fig.subplots_adjust(top=0.9)
fig.savefig('plots/error_stat.pdf')
# %%
