# %%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


def plot_fig1_paper(kernel='TG', param='baseline', leg=False):
    """
    kernel : str
        'TG' | 'RC' | 'EXP'
    """

    df = pd.read_csv(f'results/error_discrete_{kernel}_m.csv')

    T = df["T"].unique()
    T.sort()

    is_TG = kernel == "TG"
    is_RC = kernel == "RC"
    is_EXP = kernel == "EXP"

    if is_TG:
        df['estimates'] = 'TG'
        methods = [("TG", "s-", None)]
    elif is_RC:
        df['estimates'] = 'RC'
        methods = [("RC", "s-", None)]
    elif is_EXP:
        df['estimates'] = 'EXP'
        methods = [("EXP", "s-", None)]
    else:
        raise ValueError()

    err_param = 'err_{}'.format(param)

    palette = [matplotlib.cm.viridis_r(x) for x in np.linspace(0, 1, 5)][1:]

    fig = plt.figure()

    for m, ls, hatch in methods:
        for i, t in enumerate(T):
            this_df = df.query("T == @t and estimates == @m")
            curve = this_df.groupby("dt")[err_param].quantile(
                [0.25, 0.5, 0.75]).unstack()
            plt.loglog(
                curve.index, curve[0.5], ls, lw=4, c=palette[i],
                markersize=10, markevery=3
            )
            plt.fill_between(
                curve.index, curve[0.25], curve[0.75], alpha=0.2,
                color=palette[i], hatch=hatch
            )
            plt.xlim(1e-1, 1e-3)

        custom_lines_T = [
            Line2D([], [], color=palette[i], lw=5) for i in range(4)
        ]
        if leg:
            fig3, ax3 = plt.subplots(1, 1, figsize=(8, 0.7))
            ax3.legend(
                custom_lines_T,
                [r"$10^{%d}$" % np.log10(t) for t in T],
                title="$T$", loc="center", ncol=4)
            ax3.axis("off")
            fig3.savefig(f"plots/approx/legend_{kernel}_{param}.pdf")

    fig.tight_layout()
    plt.xlabel(r'$\Delta$')
    c = 'w' if kernel == 'EM' else 'k'
    plt.ylabel(r'$\ell_2$ error', fontdict={'color': c})

    plt.savefig(f"plots/approx/fig1_{kernel}_{param}_multi.png", bbox_inches='tight')
    plt.savefig(f"plots/approx/fig1_{kernel}_{param}_multi.pdf", bbox_inches='tight')


plt.close('all')
plot_fig1_paper(leg=True)
plot_fig1_paper(kernel='TG', param='baseline')
plot_fig1_paper(kernel='TG', param='alpha')
plot_fig1_paper(kernel='TG', param='m')
plot_fig1_paper(kernel='TG', param='sigma')

plot_fig1_paper(kernel='RC', param='baseline')
plot_fig1_paper(kernel='RC', param='alpha')
plot_fig1_paper(kernel='RC', param='u')
plot_fig1_paper(kernel='RC', param='sigma')

plot_fig1_paper(kernel='EXP', param='baseline')
plot_fig1_paper(kernel='EXP', param='alpha')
plot_fig1_paper(kernel='EXP', param='decay')
