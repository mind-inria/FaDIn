# %% import libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms

file_name = "results/error_discrete_RC.pkl"
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
    comptime_vis = np.zeros((n_T, n_dt, n_seeds))

    for j in range(n_T):
        for k in range(n_dt):
            for l in range(n_seeds):
                idx = j*(n_dt*n_seeds) + k*(n_seeds)  + l 
                mu_vis[j, k, l] = np.abs(results[idx]['param_baseline'] - baseline)**2
                alpha_vis[j, k, l] = np.abs(results[idx]['param_adjacency'] - alpha)**2
                u_vis[j, k, l] = np.abs(results[idx]['param_kernel'][0] - u)**2
                sigma_vis[j, k, l] = np.abs(results[idx]['param_kernel'][1] - sigma)**2
                comptime_vis[j, k, l] = results[idx]['time']

    return [mu_vis, alpha_vis, u_vis, sigma_vis,  comptime_vis]

data = get_results(all_results, T_list, dt_list, seeds)

def plot_one_curve(axs, data, title, dt_list, i, j, col, T):   
    lw = 4
    fontsize = 18

    axs[i, j].loglog(dt_list, data.mean(1), lw=lw,  label='T={}'.format(T),  c=col)
    axs[i, j].fill_between(dt_list, np.percentile(data, 10, axis=1), 
                        np.percentile(data, 90, axis=1), alpha=0.1, color=col)
    axs[i, j].set_xlim(dt_list[-1], dt_list[0])
    axs[i, j].set_ylabel('Error', size=fontsize)
    axs[i, j].set_xlabel('Stepsize', size=fontsize)

    if i == 1 and j==1:
        axs[i, j].legend(fontsize=22, loc='lower left')

    return 0;

titles = ['Baseline', 'Alpha', 'm', 'Sigma',  'Computation time']
# %%
# %% plot loss
# %matplotlib inline
matplotlib.rc("xtick", labelsize=15)
matplotlib.rc("ytick", labelsize=15)

fig, axs = plt.subplots(2, 2, figsize=(15, 12))

for i in range(2):
    for j in range(2):
        idx = i*2+j
        plot_one_curve(axs, data[idx][0],  titles[idx], dt_list, i, j, 'b', T_list[0])
        plot_one_curve(axs, data[idx][1],  titles[idx], dt_list, i, j, 'orange', T_list[1])
        plot_one_curve(axs, data[idx][2],  titles[idx], dt_list, i, j, 'r', T_list[2])
        plot_one_curve(axs, data[idx][3],  titles[idx], dt_list, i, j, 'g', T_list[3])

fig.tight_layout()

fig.savefig(
    'plots/approx/approx_RC_baseline.pdf',
    # we need a bounding box in inches
    bbox_inches=mtransforms.Bbox([[0, 0.5], [0.5, 1]]
    ).transformed(fig.transFigure - fig.dpi_scale_trans
    ),
)
fig.savefig(
    'plots/approx/approx_RC_alpha.pdf',
    bbox_inches=mtransforms.Bbox([[0.5, 0.5], [1, 1]]).transformed(
        fig.transFigure - fig.dpi_scale_trans
    ),
)
fig.savefig(
    'plots/approx/approx_RC_u.pdf',
    bbox_inches=mtransforms.Bbox([[0, 0], [0.5, 0.5]]).transformed(
        fig.transFigure - fig.dpi_scale_trans
    ),
)
fig.savefig(
    'plots/approx/approx_RC_sigma.pdf',
    bbox_inches=mtransforms.Bbox([[0.5, 0], [1, 0.5]]).transformed(
        fig.transFigure - fig.dpi_scale_trans
    ),
)
#fig.savefig('plots/approx_discrete_RC.pdf')
# %%
# %% plot loss

lw = 4
fontsize = 18
norm_l2 = data[0] + data[1] + data[2] + data[3]
comp_time = data[4]
plt.figure(figsize=(8,6))
palette = ['b', 'orange', 'r', 'g']
for i in range(4):
    comptime = comp_time[i]
    norm = norm_l2[i]
    absc = dt_list
    plt.loglog(absc, np.median(norm, axis=1), lw=lw, 
                           label='T={}'.format(T_list[i]),  c=palette[i])

    plt.fill_between(absc, np.percentile(norm, 10, axis=1),
                     np.percentile(norm, 90, axis=1), 
                     alpha=0.1, color=palette[i])
 
plt.xlim(dt_list[-1], dt_list[0])
plt.xlabel('Stepsize', fontsize=23)
plt.ylabel(r'$\ell_2$ error', fontsize=fontsize)

custom_lines_T = [Line2D([0], [0], color=palette[0], lw=3),
                  Line2D([0], [0], color=palette[1], lw=3), 
                  Line2D([0], [0], color=palette[2], lw=3), 
                  Line2D([0], [0], color=palette[3], lw=3)]

plt.legend(custom_lines_T, ['T=1000', 'T=10000', 'T=100000', 'T=1000000'], fontsize=20, 
        bbox_to_anchor=(0.92, 1.3), ncol=2)  
plt.tight_layout()
#plt.savefig('plots/approx_discrete_RC_l2.pdf')