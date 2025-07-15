# %%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


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

palette = [matplotlib.cm.viridis_r(x) for x in np.linspace(0, 1, 5)][1:]

df = pd.read_csv('results/error_rho_infer_5000.csv')

T = df["end_time"].unique()
T.sort()

noise = .5
_ = plt.figure()
for i, t in enumerate(T):
    df = df[df['noise'] == noise]
    this_df = df.query("end_time == @t")
    curve = this_df.groupby("alpha")["rec_score"].quantile(
        [0.35, 0.5, 0.65]).unstack()
    plt.plot(
        curve.index, curve[0.5], lw=4, c=palette[i],
        markersize=10, markevery=3, label=f'$T$ = {T[i]}'
    )
    plt.fill_between(
        curve.index, curve[0.35], curve[0.65], alpha=0.2,
        color=palette[i]
    )
plt.xlim(0, 1.47)
plt.xticks(fontsize=7)
plt.yticks(fontsize=10)
plt.ylabel(r'rc score', labelpad=0)
plt.xlabel(r'$\alpha$', labelpad=0, size=15)
plt.legend()
plt.savefig(f'plots/infer_rho_5000_{noise}.pdf')
plt.show()
# %%


df = pd.read_csv('results/error_rho_infer_5000.csv')

mk = ['x', 'o', '^']
fig, ax = plt.subplots()
noise = .1
# Assuming T is defined somewhere in your code
for i, t in enumerate(T):
    df = df[df['noise'] == noise]
    this_df = df.query("end_time == @t")
    curve1 = this_df.groupby("alpha")["pr_score"].quantile(
        [0.25, 0.5, 0.75]).unstack()
    curve2 = this_df.groupby("alpha")["rec_score"].quantile(
        [0.25, 0.5, 0.75]).unstack()

    pr = curve1[0.5].values[::2]
    rec = curve2[0.5].values[::2]
    palette = np.linspace(0, 1.5, pr.shape[0])

    # Scatter plot with specified marker and color
    im = ax.scatter(
        pr,
        rec,
        c=palette,
        marker=mk[i],
        label=f'T={t}',
        cmap='plasma',
        s=80
    )

ax.set_xlabel('Precision', size=15, labelpad=0)
ax.set_ylabel('Recall', size=15, labelpad=0)
ax.tick_params(axis='both', which='major', labelsize=7)
ax.tick_params(axis='both', which='minor', labelsize=7)

plt.savefig('plots/scatter_rho_uniform.pdf')
plt.show()
