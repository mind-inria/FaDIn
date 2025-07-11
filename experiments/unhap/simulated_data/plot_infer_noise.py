# %%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FONTSIZE = 14
plt.rcParams["figure.figsize"] = (5, 3.2)
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.grid.axis"] = "y"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rc('legend', fontsize=FONTSIZE - 1)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


colors = [matplotlib.cm.viridis(x) for x in np.linspace(0, 1, 5)][1:]
setting = 'high'

df = pd.read_csv(f'results/error_denoising_infer_unbalanced_{setting}.csv')

T = df["end_time"].unique()
T.sort()
ls = [':', '-']
models = ['jointfadin', 'mixture']

lw = 4
# %%
for j, model in enumerate(models):
    for i, t in enumerate(T):
        this_df = df.query("end_time == @t")
        curve = this_df.groupby("noise")[f"err_norm2_{model}"].quantile(
            [0.25, 0.5, 0.75]).unstack()
        plt.semilogy(
                curve.index, curve[0.5], lw=lw, c=colors[2-i],
                markersize=10, markevery=3, linestyle=ls[j]
            )

plt.xlim(0.1, 1.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('Estimation Error', labelpad=0)
plt.xlabel('Noise level', labelpad=0, size=12)
plt.savefig(f'plots/error_denoising_infer_unbalanced_{setting}.pdf')
plt.show()
# %%
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 0.5))
custom_lines_model = [
    Line2D([], [], color='k', lw=3,  linestyle=ls[i]) for i in range(2)
]
custom_lines_T = [
    Line2D([], [], color=colors[2-j], lw=3,) for j in range(3)
]

leg = ax1.legend(
    custom_lines_model,
    [f"{models[i]}" for i in range(2)],
    title="Models", loc="lower center",
    bbox_to_anchor=(-0.05, -0.3, 0.65, 0.01), ncol=3,  prop={'size': 9}
    )
leg2 = ax1.legend(
    custom_lines_T,
    [f"{T[i]}" for i in range(3)],
    title=r"$T$", loc="lower center",
    bbox_to_anchor=(0.1, -0.3, 1.2, 0.01), ncol=3, prop={'size': 9}
    )
ax1.add_artist(leg)
ax1.add_artist(leg2)
ax1.axis("off")
plt.savefig('plots/legend_infer_noise.pdf')
# %%
