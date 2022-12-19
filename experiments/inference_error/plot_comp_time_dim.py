# %% get results
import numpy as np
import pickle
import matplotlib.pyplot as plt


FONTSIZE = 14
plt.rcParams["figure.figsize"] = (5, 2.9)
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.grid.axis"] = "y"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["xtick.labelsize"] = FONTSIZE
plt.rcParams["ytick.labelsize"] = FONTSIZE
plt.rcParams["font.size"] = FONTSIZE
plt.rc("legend", fontsize=FONTSIZE - 1)

# Load non-param results and FaDIn
file_name = "results/comp_time_dim.pkl"
open_file = open(file_name, "rb")
all_results = pickle.load(open_file)
open_file.close()


def get_results(results):
    dim_list = results[-1]["n_dim_list"]
    dt_list = results[-1]["dt_list"]
    T_list = results[-1]["T_list"]
    seeds = results[-1]["seeds"]
    n_dim = len(dim_list)
    n_dt = len(dt_list)
    n_seeds = len(seeds)
    n_T = len(T_list)

    our_results = np.zeros((n_dim, n_T, n_dt, n_seeds))
    tick_results = np.zeros((n_dim, n_T, n_dt, n_seeds))
    for i in range(n_dim):
        for j in range(n_T):
            for k in range(n_dt):
                for m in range(n_seeds):
                    idx = (
                        i * (n_T * n_dt * n_seeds)
                        + j * (n_dt * n_seeds)
                        + k * (n_seeds)
                        + m
                    )
                    our_results[i, j, k, m] = all_results[idx][0]["comp_time"]
                    tick_results[i, j, k, m] = all_results[idx][1]["comp_time"]
    return our_results, tick_results


comp_time_FaDIn, comp_time_tick = get_results(all_results)
# %%
n_dim_list = all_results[-1]['n_dim_list'][::2]
T_list = all_results[-1]['T_list']
fig, ax = plt.subplots(1, 1)

mk = ["s", "h"]
colors = ['C1', 'C0']
ls = ['-.', '-']
mksize = 8
lw = 2

for i in range(len(T_list)):
    ax.loglog(n_dim_list, comp_time_FaDIn.mean(3)[::2, i, 0], marker=mk[0],
              markevery=1, linestyle=ls[i], markersize=mksize, c=colors[0])
    ax.fill_between(n_dim_list,
                    np.percentile(comp_time_FaDIn[::2, i, 0, :], 20, axis=1),
                    np.percentile(comp_time_FaDIn[::2, i, 0, :], 80, axis=1),
                    alpha=0.1, color=colors[0]
                    )
    ax.loglog(n_dim_list, comp_time_tick.mean(3)[::2, i, 0], marker=mk[1],
              markevery=1, linestyle=ls[i], markersize=mksize, c=colors[1])
    ax.fill_between(n_dim_list,
                    np.percentile(comp_time_tick[::2, i, 0, :], 20, axis=1),
                    np.percentile(comp_time_tick[::2, i, 0, :], 80, axis=1),
                    alpha=0.1, color=colors[1])


ax.set_xlim(2, 100)
ax.set_xlabel(r'$p$')
fig.tight_layout()
plt.savefig("plots/comparison/time_comparison_nonparam_dim.pdf", bbox_inches="tight")
