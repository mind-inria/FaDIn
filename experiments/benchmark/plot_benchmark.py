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


def plot_benchmark(kernel='RC'):
    df = pd.read_csv(f'results/benchmark_{kernel}.csv')
    lw = 3
    markersize = 12
    STYLE_METHODS = {
        'FaDIn': dict(marker="s", markersize=markersize, lw=lw, c='C1'),
        'Non-param EM': dict(marker="h", markersize=markersize, lw=lw, c='C0'),
        'Non-param SGD': dict(marker="^", markersize=markersize, lw=lw, c="C2"),
        'Gibbs': dict(marker="<", markersize=markersize, lw=lw, c='C3'),
        'VB': dict(marker=">", markersize=markersize, lw=lw, c='C4')
    }
    lw = 3
    markersize = 12
    custom_lines_m = []
    methods = ['FaDIn', 'Non-param EM', 'Non-param SGD', 'Gibbs', 'VB']
    markers = ['s', 'h', '^', '<', '>']
    colors = ['C1', 'C0', 'C2', 'C3', 'C4']
    for m in methods:
        custom_lines_m.append(Line2D([], [], **STYLE_METHODS[m]))

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 0.7))

    legend_method = ax1.legend(custom_lines_m, methods, loc="center left", ncol=5)
    ax1.add_artist(legend_method)
    ax1.axis("off")

    fig1.savefig("plots/legend_benchmark.pdf", bbox_inches="tight")

    fig2, ax2 = plt.subplots(1, 1)
    error = ['err_fadin', 'err_nonparam_tick', 'err_nonparam', 'err_gibbs', 'err_vb']
    for i in range(5):
        curve = df.groupby("T")[error[i]].quantile([0.25, 0.5, 0.75]).unstack()
        ax2.loglog(curve.index, curve[0.5], lw=lw, c=colors[i],
                   marker=markers[i], markersize=markersize)
        ax2.fill_between(curve.index, curve[0.25], curve[0.75],
                         alpha=0.3, color=colors[i])
    ax2.set_xlim(1e3, 1e5)
    ax2.set_ylabel(r"$||\hat{\lambda} - \lambda^* ||_1 \; \;  / \; \; G$")
    ax2.set_xlabel("T")

    fig2.savefig(f"plots/stats_benchmark_{kernel}.pdf", bbox_inches="tight")

    fig3, ax3 = plt.subplots(1, 1)
    time = ['time_fadin', 'time_nonparam_tick', 'time_nonparam',
            'time_gibbs', 'time_vb']
    for i in range(5):
        curve = df.groupby("T")[time[i]].quantile([0.25, 0.5, 0.75]).unstack()
        ax3.loglog(curve.index, curve[0.5], lw=lw, c=colors[i],
                   marker=markers[i], markersize=markersize)
        ax3.fill_between(curve.index, curve[0.25], curve[0.75],
                         alpha=0.3, color=colors[i])
    ax3.set_xlim(1e3, 1e5)
    ax3.set_xlabel("T", size=FONTSIZE)
    ax3.set_ylabel("Time (s.)")

    fig3.savefig(f"plots/time_benchmark_{kernel}.pdf", bbox_inches="tight")


plt.close('all')

plot_benchmark(kernel='RC')
plot_benchmark(kernel='tg')
plot_benchmark(kernel='exp')
