# %%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# from tueplots.bundles.iclr2023()
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


def plot_length_analysis():
    df = pd.read_csv('results/sensitivity_length.csv')
    T = df["T"].unique()
    T.sort()

    palette = [matplotlib.cm.viridis_r(x) for x in np.linspace(0, 1, 5)][1:]

    fig1 = plt.figure()

    df['estimates'] = 'EXP'
    methods = [("EXP", "s-", None)]

    for m, ls, hatch in methods:
        for i, t in enumerate(T):
            this_df = df.query("T == @t and estimates == @m")
            curve = this_df.groupby("W")["err_norm2"].quantile(
                [0.25, 0.5, 0.75]).unstack()
            plt.loglog(
                curve.index, curve[0.5], ls, lw=4, c=palette[i],
                markersize=10, markevery=3
            )
            plt.fill_between(
                curve.index, curve[0.25], curve[0.75], alpha=0.2,
                color=palette[i], hatch=hatch
            )
            plt.xlim(1e-0, 1e2)

    fig1.tight_layout()
    plt.xlabel(r'$W$')
    plt.ylabel(r'$\ell_2$ error', fontdict={'color': 'k'})
    plt.savefig("plots/approx/sensi_w_exp_stat.pdf", bbox_inches='tight')

    fig2 = plt.figure()

    for m, ls, hatch in methods:
        for i, t in enumerate(T):
            this_df = df.query("T == @t and estimates == @m")
            curve = this_df.groupby("W")["time"].quantile(
                [0.25, 0.5, 0.75]).unstack()
            plt.loglog(
                curve.index, curve[0.5], ls, lw=4, c=palette[i],
                markersize=10, markevery=3
            )
            plt.fill_between(
                curve.index, curve[0.25], curve[0.75], alpha=0.2,
                color=palette[i], hatch=hatch
            )
            plt.xlim(1e-0, 1e2)

    fig2.tight_layout()
    plt.xlabel(r'$W$')
    plt.ylabel('Time (s.)', fontdict={'color': 'k'})
    plt.savefig("plots/approx/sensi_w_exp_time.pdf", bbox_inches='tight')


plt.close('all')


plot_length_analysis()
