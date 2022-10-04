# %% import libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

file_name = "results/comp_autodiff_800.pkl"
open_file = open(file_name, "rb")
all_results = pickle.load(open_file)
open_file.close()

# %% plot results
##
# %matplotlib inline

dt_list = all_results[-1]['dt_list']
T_list = all_results[-1]['T_list']
seeds = all_results[-1]['seeds']

def get_results(results, T_list, dt_list, seeds):
    n_dt = len(dt_list); n_seeds = len(seeds); n_T = len(T_list)

    comptime_FaDIn = np.zeros((n_T, n_dt, n_seeds))
    comptime_autodiff = np.zeros((n_T, n_dt, n_seeds))
    for j in range(n_T):
        for k in range(n_dt):
            for l in range(n_seeds):
                idx = j*(n_dt*n_seeds) + k*(n_seeds)  + l 
                comptime_FaDIn[j, k, l] = results[idx]['time_FaDIn']
                comptime_autodiff[j, k, l] = results[idx]['time_autodiff']

    return [comptime_FaDIn, comptime_autodiff]

comptime_FaDIn, comptime_autodiff = get_results(all_results, T_list, dt_list, seeds)

n_T = len(T_list)
n_dt = len(dt_list)
fontsize = 20
linestyle = [':', '--', '-.']
plt.figure(figsize=(7,6))

for i in range(n_dt):
    print(i) 
    cF = comptime_FaDIn[:, i] 
    ca = comptime_autodiff[:, i]
    plt.loglog(T_list, np.median(cF, axis=1), c='b', linestyle=linestyle[i])
    plt.loglog(T_list, np.median(ca, axis=1), c='r' ,linestyle=linestyle[i])

    q20 = np.percentile(cF, 20, axis=1)
    q80 = np.percentile(cF, 80, axis=1)
    plt.fill_between(T_list, q20, q80, alpha=0.2, color='b')

    q20 = np.percentile(ca, 20, axis=1)
    q80 = np.percentile(ca, 80, axis=1)
    plt.fill_between(T_list, q20, q80, alpha=0.2, color='r')

    plt.xlim(T_list[0], T_list[-1])
    plt.xlabel(r'$\Delta$', fontsize=fontsize)
    plt.ylabel('Time (s.)', fontsize=fontsize)

    custom_lines_T = [Line2D([0], [0], color='k', lw=3, ls=':'),
                      Line2D([0], [0], color='k', lw=3, ls='--'),
                      Line2D([0], [0], color='k', lw=3, ls='-.')]

    custom_lines_m = [Line2D([0], [0], color='b', lw=3),
                      Line2D([0], [0], color='r', lw=3)]    

    first_legend = plt.legend(custom_lines_T, 
                              ['T=10','T=100', 'T=1000'], 
                              fontsize=15, 
                              bbox_to_anchor=(0.59, 1.2), 
                              ncol=2)

    plt.gca().add_artist(first_legend)          
    plt.legend(custom_lines_m, ['FaDIn', 'L2 autodiff'], 
                fontsize=15, bbox_to_anchor=(1.01, 1.2))  

    plt.tight_layout()
    plt.savefig('plots/comptime_autodiff.pdf')
# %%

# %%
