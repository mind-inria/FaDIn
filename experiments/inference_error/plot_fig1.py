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


def plot_fig1_paper(kernel='TG', leg=True):
    """
    kernel : str
        'TG' | 'EM' | 'RC' | 'EXP'
    """

    df = pd.read_csv(f'results/error_discrete_{kernel}.csv')

    T = df["T"].unique()
    T.sort()

    is_EM = kernel == "EM"
    is_TG = kernel == "TG"
    is_RC = kernel == "RC"
    is_EXP = kernel == "EXP"

    if is_EM:
        methods = [("continuous", "--", '//'), ("EM", "o-", None)]
    elif is_TG:
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

    palette = [matplotlib.cm.viridis_r(x) for x in np.linspace(0, 1, 5)][1:]

    for m, ls, hatch in methods:
        for i, t in enumerate(T):
            this_df = df.query("T == @t and estimates == @m")
            curve = this_df.groupby("dt")["err_norm2"].quantile(
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

    # Create legend

    # Add legend in 2 separated boxes
    if is_EM:
        labels_m = ["FaDIn", "EM", "Cont. EM"]
        handles_m = [
            plt.Line2D([], [], c="k", lw=3, marker='s', markersize=10),
            plt.Line2D([], [], c="k", lw=3, marker='o', markersize=10),
            plt.Line2D([], [], c="k", ls="--", lw=3),
        ]
        plt.legend(
            handles_m,
            labels_m,
            ncol=3,
            title="Method",
            bbox_to_anchor=(-.1, 1, 1, 0.01),
            loc="lower center",
        )

    elif is_TG or is_RC or is_EXP:
        custom_lines_T = [
            Line2D([], [], color=palette[i], lw=3) for i in range(2)
        ]
        if leg:
            plt.legend(
                custom_lines_T,
                [r"$10^{%d}$" % np.log10(t) for t in [1000, 10000]],
                title="$T$", loc="lower center",
                bbox_to_anchor=(0, 1, 1, 0.01), ncol=2
            )

    plt.xlabel(r'$\Delta$')
    c = 'w' if kernel == 'EM' else 'k'
    plt.ylabel(r'$\ell_2$ error', fontdict={'color': c})

    plt.savefig(f"plots/approx/fig1_{kernel}_all.png", bbox_inches='tight')
    plt.savefig(f"plots/approx/fig1_{kernel}_all.pdf", bbox_inches='tight')


plt.close('all')


plot_fig1_paper(kernel='TG')
plot_fig1_paper(kernel='RC', leg=False)
plot_fig1_paper(kernel='EXP', leg=False)
# plot_fig1_paper(kernel='EM')
# plt.show()

# %%
