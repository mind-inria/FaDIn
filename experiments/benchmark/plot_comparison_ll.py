# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FONTSIZE = 18
plt.rcParams["figure.figsize"] = (5, 3.2)
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.grid.axis"] = "y"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rc('legend', fontsize=FONTSIZE - 1)
# %%


def plot_comparison_ll(kernel):
    df = pd.read_csv(f'results/comparison_ll_{kernel}.csv')
    lw = 3
    markersize = 12
    STYLE_METHODS = {
        'FaDIn': dict(marker="s", markersize=markersize, lw=lw, c='C1'),
        'Log-Likelihood': dict(marker="*", markersize=markersize, lw=lw, c='C5'),
    }
    LINE_STYLE_dt = {0.1: "-.", 0.01: "-"}
    lw = 3
    markersize = 12
    custom_lines_m = []
    methods = ['FaDIn', 'Log-Likelihood']
    markers = ['s', '*']
    colors = ['C1', 'C5']
    for m in methods:
        custom_lines_m.append(Line2D([], [], **STYLE_METHODS[m]))
    custom_lines_dt = [
        plt.Line2D([], [], ls=ls, c='k') for ls in LINE_STYLE_dt.values()
    ]

    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 0.7))

    legend_method = ax1.legend(custom_lines_m, methods,
                               title="Methods", loc="center left", ncol=2)
    ax1.add_artist(legend_method)
    ax1.axis("off")

    fig1.savefig("plots/comparison_ll/legend_ll_methods.pdf", bbox_inches="tight")

    fig4, ax4 = plt.subplots(1, 1, figsize=(4, 0.7))

    legend_method = ax4.legend(custom_lines_dt, [0.1, 0.01],
                               title=r"$\Delta$", loc="center left", ncol=2)
    ax4.add_artist(legend_method)
    ax4.axis("off")

    fig4.savefig("plots/comparison_ll/legend_ll_dt.pdf", bbox_inches="tight")

    fig2, ax2 = plt.subplots(1, 1)
    error = ['err_fadin', 'err_ll']
    for i in range(2):
        for delta in [0.1, 0.01]:
            this_df = df.query('dt==@delta')
            curve = this_df.groupby("T")[error[i]].quantile([0.45, 0.5, 0.55]).unstack()
            ax2.loglog(curve.index, curve[0.5], lw=lw, c=colors[i],
                       marker=markers[i], markersize=markersize,
                       ls=LINE_STYLE_dt[delta])
            ax2.fill_between(curve.index, curve[0.45], curve[0.55],
                             alpha=0.2, color=colors[i])
    ax2.set_xlim(1e2, 1e5)
    ax2.set_ylabel(r"$\ell_2$-error")
    ax2.set_xlabel(r"$T$")

    fig2.savefig(f"plots/comparison_ll/stats_ll_{kernel}.pdf", bbox_inches="tight")

    fig3, ax3 = plt.subplots(1, 1)
    time = ['time_fadin', 'time_ll']
    for i in range(2):
        for delta in [0.1, 0.01]:
            this_df = df.query('dt==@delta')
            curve = this_df.groupby("T")[time[i]].quantile([0.25, 0.5, 0.75]).unstack()
            ax3.loglog(curve.index, curve[0.5], lw=lw, c=colors[i],
                       marker=markers[i], markersize=markersize,
                       ls=LINE_STYLE_dt[delta])
            ax3.fill_between(curve.index, curve[0.25], curve[0.75],
                             alpha=0.3, color=colors[i])
    ax3.set_xlim(1e2, 1e5)
    ax3.set_xlabel(r"$T$", size=FONTSIZE)
    ax3.set_ylabel("Time (s.)")

    fig3.savefig(f"plots/comparison_ll/time_ll_{kernel}.pdf", bbox_inches="tight")


plt.close('all')

plot_comparison_ll('raised_cosine')
plot_comparison_ll('truncated_gaussian')
plot_comparison_ll('truncated_exponential')
