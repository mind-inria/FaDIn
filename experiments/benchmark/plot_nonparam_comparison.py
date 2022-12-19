# %% get results
#

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fadin.kernels import DiscreteKernelFiniteSupport

FONTSIZE = 14
plt.rcParams["figure.figsize"] = (5, 2.8)
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.grid.axis"] = "y"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["xtick.labelsize"] = FONTSIZE
plt.rcParams["ytick.labelsize"] = FONTSIZE
plt.rcParams["font.size"] = FONTSIZE
plt.rc("legend", fontsize=FONTSIZE - 1)

CMAP = [plt.cm.viridis_r(x) for x in np.linspace(0, 1, 5)][1:]

LINE_STYLE_T = {
    1e5: "-.",
    1e6: "-",
}
STYLE_METHODS = {
    'FaDIn': dict(marker="s", markersize=8, lw=2, c='C1'),
    'Non-param': dict(marker="h", markersize=8, lw=2, c='C0'),
    'L2-Autodiff': dict(marker="^", markersize=8, lw=2, c="C2"),
    'True kernel': dict(ls="--", lw=3, c='k', zorder=10),
}

# Load non-param results and FaDIn
file_name = "results/non_param.pkl"
open_file = open(file_name, "rb")
all_results = pickle.load(open_file)
open_file.close()


def get_results(results):
    dt_list = results[-1]["dt_list"]
    T_list = results[-1]["T_list"]
    seeds = results[-1]["seeds"]
    n_dt = len(dt_list)
    n_seeds = len(seeds)
    n_T = len(T_list)

    our_results = np.zeros((2, n_T, n_dt, n_seeds))
    tick_results = np.zeros((2, n_T, n_dt, n_seeds))
    for i in range(2):
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

# Load results on autodiff solver
file_name = "results/comp_autodiff_.pkl"
open_file = open(file_name, "rb")
results_autodiff = pickle.load(open_file)
open_file.close()

dt_list_ = results_autodiff[-1]["dt_list"]
T_list_ = results_autodiff[-1]["T_list"]
seeds_ = results_autodiff[-1]["seeds"]


def get_results_(results, T_list, dt_list, seeds):
    n_dt = len(dt_list)
    n_seeds = len(seeds)
    n_T = len(T_list)

    comptime_autodiff = np.zeros((n_T, n_dt, n_seeds))
    for j in range(n_T):
        for k in range(n_dt):
            for m in range(n_seeds):
                idx = j * (n_dt * n_seeds) + k * (n_seeds) + m
                comptime_autodiff[j, k, m] = results[idx]["time_autodiff"]

    return comptime_autodiff


comptime_autodiff = get_results_(results_autodiff, T_list_, dt_list_, seeds_)


comptime_autodiff_ = []
for file_name in ["results/comp_autodiff_million.pkl",
                  "results/comp_autodiff_long_run_million_01_.pkl"]:
    with open(file_name, "rb") as f:
        result_ad_million = pickle.load(f)
    comptime_autodiff_.append(result_ad_million[0]["time_autodiff"])
comptime_autodiff_ = np.array(comptime_autodiff_)[:, None]


def mean_int(x, y, dt):
    c = y.sum()
    return (x * y).sum() / c  # come from the reparametrization


def plot_nonparam(all_results, comp_time_our, comp_time_tick,
                  comptime_autodiff, autodiff_, m, T_idx, kernel="RC"):

    dt = all_results[m][0]["dt"]
    dt_list = all_results[-1]["dt_list"]
    T_list = all_results[-1]["T_list"]
    L = int(1 / dt)
    discretization = torch.linspace(0, 1, L)
    discretisation = np.linspace(0, 1, L)
    alpha = np.array([[0.8]])
    if kernel == "RC":
        K_idx = 0
        mu = np.array([[0.5]])
        sigma = np.array([[0.3]])
        u = mu - sigma
        RC = DiscreteKernelFiniteSupport(delta=dt, n_dim=1,
                                         kernel='raised_cosine')
        true_kernel = RC.kernel_eval([torch.Tensor(u), torch.Tensor(sigma)],
                                     discretization).squeeze().numpy()
    if kernel == "TG":
        K_idx = 0
        mean = np.array([[0.5]])
        sigma = np.array([[0.3]])
        TG = DiscreteKernelFiniteSupport(delta=dt, n_dim=1,
                                         kernel='truncated_gaussian')
        true_kernel = TG.kernel_eval([torch.Tensor(mean), torch.Tensor(sigma)],
                                     discretization).squeeze().numpy()
    if kernel == "EXP":
        K_idx = 0
        decay = np.array([[0.5]])
        EXP = DiscreteKernelFiniteSupport(delta=dt, n_dim=1,
                                          kernel='truncated_exponential')
        true_kernel = EXP.kernel_eval([torch.Tensor(decay)],
                                      discretization).squeeze().numpy()
    else:
        K_idx = 1

    true_kernel *= alpha.item()

    our = comp_time_our[K_idx, T_idx, :, :]
    tick = comp_time_tick[K_idx, T_idx, :, :]
    our_ = comp_time_our[K_idx, 3, :, :]
    tick_ = comp_time_tick[K_idx, 3, :, :]

    autodiff = comptime_autodiff[0]

    our_kernel = all_results[m][0]["kernel"]
    tick_kernel = all_results[m][1]["kernel"] / 0.8
    tick_kernel = np.insert(tick_kernel, [-1], tick_kernel[-1])
    T = T_list[T_idx]
    fig1, ax1 = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 0.4))
    ax = [ax1, ax2, ax3]

    to_plot = [
        ("True kernel", true_kernel),
        ("FaDIn", our_kernel),
        ("Non-param", tick_kernel),
    ]
    for label, curve in to_plot:
        ax[0].plot(
            discretisation, curve, **STYLE_METHODS[label],
            markevery=20 if label == "Non-param" else 10)
    ax[0].set_xlim(0, 1)
    ax[0].set_xlabel("Kernel support")
    to_plot = [
        ('FaDIn', dt_list, our, 1e5),
        ('Non-param', dt_list, tick, 1e5),
        ('L2-Autodiff', [0.1, 0.01], autodiff, 1e5),
        ('FaDIn', dt_list, our_, 1e6),
        ('Non-param', dt_list, tick_, 1e6),
        ('L2-Autodiff', [0.1, 0.01], autodiff_, 1e6),
    ]
    for label, x, y, t in to_plot:
        style = STYLE_METHODS[label].copy()
        style['ls'] = LINE_STYLE_T[t]
        ax[1].loglog(x, y.mean(1), **style)

        ax[1].fill_between(
            x,
            np.percentile(y, 20, axis=1),
            np.percentile(y, 80, axis=1),
            alpha=0.1,
            color=style['c'],
        )
    ax[1].set_xlim(dt_list[0], dt_list[-1])
    ax[1].set_xlabel(r"$\Delta$")
    ax[1].set_ylabel("Time (s.)")
    ax[1].set_xticks(dt_list)

    custom_lines_m = []
    methods = ["True kernel", "FaDIn", "Non-param", "L2-Autodiff"]
    for me in methods:
        custom_lines_m.append(Line2D([], [], **STYLE_METHODS[me]))

    custom_lines_T = [
        plt.Line2D([], [], ls=ls, c='k') for ls in LINE_STYLE_T.values()
    ]

    legend_method = ax[2].legend(
        custom_lines_m, methods, loc="center left", ncol=4,
    )
    ax[2].add_artist(legend_method)
    ax[2].legend(
        custom_lines_T,
        [r"$10^{%d}$" % np.log10(t) for t in LINE_STYLE_T],
        loc="center right",
        ncol=2,
    )
    ax[2].axis("off")

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(
        "plots/comparison/ker_comparison_nonparam_T={}_dt={}_K={}.pdf".format(
            T, np.round(all_results[m][0]["dt"], 3), kernel
        ),
        # we need a bounding box in inches
        bbox_inches="tight",
    )
    fig2.savefig(
        "plots/comparison/time_comparison_nonparam_T={}_K={}.pdf".format(T, kernel),
        bbox_inches="tight",
    )
    fig3.savefig("plots/comparison/legend_fig2__T={}_K={}.pdf".format(T, kernel),
                 bbox_inches="tight")
    return 0


plt.close("all")
plot_nonparam(
    all_results, comp_time_FaDIn, comp_time_tick,
    comptime_autodiff, comptime_autodiff_, 10, 0, kernel="RC"
)
plot_nonparam(
    all_results, comp_time_FaDIn, comp_time_tick,
    comptime_autodiff, comptime_autodiff_, 70, 2, kernel="RC"
)
# Load non-param results and FaDIn
file_name = "results/non_param_tg.pkl"
open_file = open(file_name, "rb")
all_results = pickle.load(open_file)
open_file.close()
plot_nonparam(
    all_results, comp_time_FaDIn, comp_time_tick,
    comptime_autodiff, comptime_autodiff_, 13, 0, kernel="TG"
)

plot_nonparam(
    all_results, comp_time_FaDIn, comp_time_tick,
    comptime_autodiff, comptime_autodiff_, 40, 1, kernel="TG"
)

plot_nonparam(
    all_results, comp_time_FaDIn, comp_time_tick,
    comptime_autodiff, comptime_autodiff_, 70, 2, kernel="TG"
)

plot_nonparam(
    all_results, comp_time_FaDIn, comp_time_tick,
    comptime_autodiff, comptime_autodiff_, 100, 3, kernel="TG"
)

file_name = "results/non_param_exp.pkl"
open_file = open(file_name, "rb")
all_results = pickle.load(open_file)
open_file.close()

plot_nonparam(
    all_results, comp_time_FaDIn, comp_time_tick,
    comptime_autodiff, comptime_autodiff_, 10, 0, kernel="EXP"
)

plot_nonparam(
    all_results, comp_time_FaDIn, comp_time_tick,
    comptime_autodiff, comptime_autodiff_, 40, 1, kernel="EXP"
)

plot_nonparam(
    all_results, comp_time_FaDIn, comp_time_tick,
    comptime_autodiff, comptime_autodiff_, 70, 2, kernel="EXP"
)

plot_nonparam(
    all_results, comp_time_FaDIn, comp_time_tick,
    comptime_autodiff, comptime_autodiff_, 100, 3, kernel="EXP"
)
# %%
