import torch
import numpy as np
import matplotlib.pyplot as plt

from fadin.kernels import DiscreteKernelFiniteSupport


def plot(solver, plotfig=False, bl_noise=False, title=None, ch_names=None,
         savefig=None):
    """
    Plots estimated kernels and baselines of `FaDIn` solver.
    Should be called after calling the `fit` method on solver.

    Parameters
    ----------
    solver: `FaDIn` solver.
        `fit` method should be called on the solver before calling `plot`.

    plotfig: bool (default `False`)
        If set to `True`, the figure is plotted.

    bl_noise: bool (default`False`)
        Whether to plot the baseline of noisy activations.
        Only works if the solver has 'baseline_noise' attribute.

    title: `str` or `None`, default=`None`
        Title of the plot. If set to `None`, the title text is generic.

    ch_names: list of `str` (default `None`)
        Channel names for subplots. If set to `None`, will be set to
        `np.arange(solver.n_dim).astype('str')`.
    savefig: str or `None`, default=`None`
        Path for saving the figure. If set to `None`, the figure is not saved.

    Returns
    -------
    fig, axs : matplotlib.pyplot Figure
        n_dim x n_dim subplots, where subplot of coordinates (i, j) shows the
        kernel component $\\alpha_{i, j}\\phi_{i, j}$ and the baseline $\\mu_i$
        of the intensity function $\\lambda_i$.

    """
    # Recover kernel time values and y values for kernel plot
    discretization = torch.linspace(0, solver.kernel_length, 200)
    kernel = DiscreteKernelFiniteSupport(solver.delta,
                                         solver.n_dim,
                                         kernel=solver.kernel,
                                         kernel_length=solver.kernel_length)

    kappa_values = kernel.kernel_eval(solver.kernel_,
                                      discretization).detach()
    # Plot
    if ch_names is None:
        ch_names = np.arange(solver.n_dim).astype('str')
    fig, axs = plt.subplots(nrows=solver.n_dim,
                            ncols=solver.n_dim,
                            figsize=(4 * solver.n_dim, 4 * solver.n_dim),
                            sharey=True,
                            sharex=True,
                            squeeze=False)
    for i in range(solver.n_dim):
        for j in range(solver.n_dim):
            # Plot baseline
            label = (rf'$\mu_{{{ch_names[i]}}}$=' +
                     f'{round(solver.baseline_[i].item(), 2)}')
            axs[i, j].hlines(
                y=solver.baseline_[i].item(),
                xmin=0,
                xmax=solver.kernel_length,
                label=label,
                color='orange',
                linewidth=4
            )
            if bl_noise:
                # Plot noise baseline
                mutilde = round(solver.baseline_noise_[i].item(), 2)
                label = rf'$\tilde{{\mu}}_{{{ch_names[i]}}}$={mutilde}'
                axs[i, j].hlines(
                    y=solver.baseline_noise_[i].item(),
                    xmin=0,
                    xmax=solver.kernel_length,
                    label=label,
                    color='green',
                    linewidth=4
                )
            # Plot kernel (i, j)
            phi_values = solver.alpha_[i, j].item() * kappa_values[i, j, 1:]
            axs[i, j].plot(
                discretization[1:],
                phi_values,
                label=rf'$\phi_{{{ch_names[i]},{ch_names[j]}}}$',
                linewidth=4
            )
            if solver.kernel == 'truncated_gaussian':
                # Plot mean of gaussian kernel
                mean = round(solver.kernel_[0].item(), 2)
                axs[i, j].vlines(
                    x=mean,
                    ymin=0,
                    ymax=torch.max(phi_values).item(),
                    label=rf'mean={mean}',
                    color='pink',
                    linestyles='dashed',
                    linewidth=3,
                )
            # Handle text
            axs[i, j].set_xlabel('Time', size='x-large')
            axs[i, j].tick_params(
                axis='both',
                which='major',
                labelsize='x-large'
            )
            axs[i, j].set_title(
                f'{ch_names[j]}-> {ch_names[i]}',
                size='x-large'
            )
            axs[i, j].legend(fontsize='large', loc='best')
    # Plot title
    if title is None:
        fig_title = 'Hawkes influence ' + solver.kernel + ' kernel'
    else:
        fig_title = title
    fig.suptitle(fig_title, size=20)
    fig.tight_layout()
    # Save figure
    if savefig is not None:
        fig.savefig(savefig)
    # Plot figure
    if plotfig:
        fig.show()

    return fig, axs
