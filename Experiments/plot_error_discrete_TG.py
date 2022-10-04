# %% import libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.transforms as mtransforms

file_name = "results/error_discrete_TG.pkl"
open_file = open(file_name, "rb")
all_results = pickle.load(open_file)
open_file.close()

# %% read scripts
##
dt_list = all_results[-1]['dt_list']
T_list = all_results[-1]['T_list']
seeds = all_results[-1]['seeds']

def get_results(results, T_list, dt_list, seeds):
    baseline = np.array([.1])
    alpha = np.array([[0.8]])
    m = np.array([[0.5]])
    sigma = np.array([[0.3]])

    n_dt = len(dt_list); n_seeds = len(seeds); n_T = len(T_list)

    mu_vis = np.zeros((n_T, n_dt, n_seeds))
    alpha_vis = np.zeros((n_T, n_dt, n_seeds))
    m_vis = np.zeros((n_T, n_dt, n_seeds))
    sigma_vis = np.zeros((n_T, n_dt, n_seeds))
    comptime_vis = np.zeros((n_T, n_dt, n_seeds))

    for j in range(n_T):
        for k in range(n_dt):
            for l in range(n_seeds):
                idx = j*(n_dt*n_seeds) + k*(n_seeds)  + l 
                mu_vis[j, k, l] = np.abs(results[idx]['param_baseline'] - baseline)**2
                alpha_vis[j, k, l] = np.abs(results[idx]['param_adjacency'] - alpha)**2
                m_vis[j, k, l] = np.abs(results[idx]['param_kernel'][0] - m)**2
                sigma_vis[j, k, l] = np.abs(results[idx]['param_kernel'][1] - sigma)**2
                comptime_vis[j, k, l] = results[idx]['time']

    return [mu_vis, alpha_vis, m_vis, sigma_vis,  comptime_vis]

data = get_results(all_results, T_list, dt_list, seeds)

# %% Figure norm l2
#%matplotlib inline
palette = [matplotlib.cm.viridis_r(x) for x in np.linspace(0, 1, 4)]

lw = 4
fontsize = 18
norm_l2 = data[0] + data[1] + data[2] + data[3]
comp_time = data[4]


for i in range(4):
    comptime = comp_time[i]
    norm = norm_l2[i]
    absc = dt_list
    plt.loglog(absc, np.median(norm, axis=1), lw=lw, 
                        label='T={}'.format(T_list[i]),  color=palette[i] )

    plt.fill_between(absc, np.percentile(norm, 10, axis=1),
                    np.percentile(norm, 90, axis=1), 
                    alpha=0.1, color=palette[i])

plt.xlim(dt_list[-1], dt_list[0])
plt.xlabel(r'\Delta$', fontsize=27)
plt.ylabel(r'$\ell_2$ error', fontsize=fontsize+15)
#plt.title('approx')
custom_lines_T = [Line2D([0], [0], color=palette[0], lw=3),
                Line2D([0], [0], color=palette[1], lw=3), 
                Line2D([0], [0], color=palette[2], lw=3), 
                Line2D([0], [0], color=palette[3], lw=3)]

plt.legend(custom_lines_T, ['T={:.0e}'.format(1000), 'T={:.0e}'.format(10000),
                            'T={:.0e}'.format(100000), 'T={:.0e}'.format(1000000)], fontsize=20, 
        bbox_to_anchor=(0.92, 1.3), ncol=2)  
plt.tight_layout()
#plt.savefig('plots/approx_discrete_TG_l2.pdf')
#                    

# %% Plot with detailed parameters
def plot_one_curve(axs, data,  dt_list, i, j, col, T):   
    lw = 4
    fontsize = 18

    axs[i, j].loglog(dt_list, np.median(data, axis=1), lw=lw,  label='T={}'.format(T),  c=col)
    axs[i, j].fill_between(dt_list, np.percentile(data, 10, axis=1), 
                            np.percentile(data, 90, axis=1), alpha=0.1, color=col)
    axs[i, j].set_xlim(dt_list[-1], dt_list[0])
    axs[i, j].set_ylabel('Error', size=fontsize)
    axs[i, j].set_xlabel('Stepsize', size=fontsize)
    if i==0 and j==1:
        axs[i, j].legend(fontsize=20, loc="lower left")

    return 0;

titles = ['Baseline', 'Alpha', 'm', 'Sigma',  'Computation time']

%matplotlib inline
matplotlib.rc("xtick", labelsize=13)
matplotlib.rc("ytick", labelsize=13)

fig, axs = plt.subplots(2, 2, figsize=(15, 12))

for i in range(2):
    for j in range(2):
        idx = i*2+j
        plot_one_curve(axs, data[idx][0], dt_list, i, j, 'b', T_list[0])
        plot_one_curve(axs, data[idx][1], dt_list, i, j, 'orange', T_list[1])
        plot_one_curve(axs, data[idx][2], dt_list, i, j, 'r', T_list[2])
        plot_one_curve(axs, data[idx][3], dt_list, i, j, 'g', T_list[3])

fig.savefig(
    'plots/approx/approx_TG_baseline.pdf',
    # we need a bounding box in inches
    bbox_inches=mtransforms.Bbox([[0.05, 0.48], [0.493, 0.91]]
    ).transformed(fig.transFigure - fig.dpi_scale_trans
    ),
)
fig.savefig(
    'plots/approx/approx_TG_alpha.pdf',
    bbox_inches=mtransforms.Bbox([[0.48, 0.48], [0.93, 0.91]]).transformed(
        fig.transFigure - fig.dpi_scale_trans
    ),
)
fig.savefig(
    'plots/approx/approx_TG_m.pdf',
    bbox_inches=mtransforms.Bbox([[0.05, 0.08], [0.493, 0.493]]).transformed(
        fig.transFigure - fig.dpi_scale_trans
    ),
)
fig.savefig(
    'plots/approx/approx_TG_sigma.pdf',
    bbox_inches=mtransforms.Bbox([[0.48, 0.08], [0.92, 0.493]]).transformed(
        fig.transFigure - fig.dpi_scale_trans
    ),
)
