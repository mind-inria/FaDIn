import torch
import time
import matplotlib.pyplot as plt
import numpy as np

from fadin.utils.utils import optimizer, projected_grid, \
    init_kernel_parameters
from fadin.utils.compute_constants import compute_constants_fadin
from fadin.loss_and_gradient import compute_gradient_fadin
from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.init import init_hawkes_params



class FaDIn(object):
    """Define the FaDIn framework for estimated Hawkes processes.

    The framework is detailed in:

    Guillaume Staerman, Cédric Allain, Alexandre Gramfort, Thomas Moreau
    FaDIn: Fast Discretized Inference for Hawkes Processes with General
    Parametric Kernels
    https://arxiv.org/abs/2210.04635

    FaDIn minimizes the discretized L2 loss of Hawkes processes
    defined by the intensity  as a convolution between the kernel
    :math:`\\phi_{ij}` and the sum of Dirac functions
    :math:`z_i  := \\sum_{t^i_n \\in \\mathscr{F}^i_T} \\delta_{t^i_n}`
    located at the event occurrences :math:`t^i_n`:

    .. math::
        \\forall i \\in [1 \\dots p], \\quad
        \\lambda_i(t) = \\mu_i + \\sum_{j=1}^p \\phi_{ij} * z_j(t),
        \\quad t \\in [0, T]

    where

    * :math:`p` is the dimension of the process
    * :math:`\\mu_i` are the baseline intensities
    * :math:`\\phi_{ij}` are the kernels
    * :math:`z_j(t)` are the activation vector on the discretized grid.

    Parameters
    ----------
    n_dim : `int`
        Dimension of the underlying Hawkes process.

    kernel : `str` or `callable`
        Either define a kernel in ``{'raised_cosine' | 'truncated_gaussian' |
        'truncated_exponential'}`` or a custom kernel.

    init: `str` or `dict`, default='random'
        Initialization strategy of the parameters of the Hawkes process.
        If set to 'random', the parameters are initialized randomly.
        If set to 'moment_matching_max', the parameters are initialized
        using the moment matching method with max mode.
        If set to 'moment_matching_mean', the parameters are initialized
        using the moment matching method with mean mode.
        Otherwise, the parameters are initialized using the given dictionary,
        , which must contain the following keys:
        - 'baseline': `tensor`, shape (n_dim,): Initial baseline
        - 'alpha': `tensor`, shape (n_dim, n_dim): Initial alpha
        - 'kernel': `list` of tensors of shape (n_dim, n_dim):
        Initial kernel parameters.

    optim_mask: `dict` of `tensor` or `None`, default=`None`.
        Dictionary containing the masks for the optimization of the parameters
        of the Hawkes process. If set to `None`, all parameters are optimized.
        The dictionary must contain the following keys:
        - 'baseline': `tensor` of shape (n_dim,), or `None`.
            Tensor of same shape as the baseline vector, with values in (0, 1).
            `baseline` coordinates where then tensor is equal to 0
            will not be optimized. If set to `None`, all coordinates of
            baseline will be optimized.
        - 'alpha': `tensor` of shape (n_dim, n_dim), or `None`.
            Tensor of same shape as the `alpha` tensor, with values in (0, 1).
            `alpha` coordinates and kernel parameters where `alpha_mask` = 0
            will not be optimized. If set to `None`, all coordinates of alpha
            and kernel parameters will be optimized.

    kernel_length : `float`, `default=1.`
        Length of kernels in the Hawkes process.

    delta : `float`, `default=0.01`
        Step size of the discretization grid.

    optim : `str` in ``{'RMSprop' | 'Adam' | 'GD'}``, default='RMSprop'
        The algorithms used to optimize the Hawkes processes parameters.

    step_size : `float`, `default=1e-3`
        Learning rate of the chosen optimization algorithm.

    max_iter : `int`, `default=1000`
        Maximum number of iterations during fit.

    ztzG_approx : `boolean`, `default=True`
        If ztzG_approx is false, compute the true ztzG precomputation constant
        that is the computational bottleneck of FaDIn. if ztzG_approx is true,
        ztzG is approximated with Toeplitz matrix not taking into account
        edge effects.

    log : `boolean`, `default=False`
        Record the loss values during the optimization.

    grad_kernel : `None` or `callable`, default=None
        If kernel in ``{'raised_cosine'| 'truncated_gaussian' |
        'truncated_exponential'}`` the gradient function is implemented.
        If kernel is custom, the custom gradient must be given.

    tol : `float`, `default=1e-5`
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). If not reached the solver does 'max_iter'
        iterations.

    random_state : `int`, `RandomState` instance or `None`, `default=None`
        Set the torch seed to 'random_state'.

    Attributes
    ----------
    param_baseline : `tensor`, shape (n_dim)
        Baseline parameter of the Hawkes process.

    param_alpha : `tensor`, shape (n_dim, n_dim)
        Weight parameter of the Hawkes process.

    param_kernel : `list` of `tensor`
        list containing tensor array of kernels parameters.
        The size of the list varies depending the number of
        parameters. The shape of each tensor is `(n_dim, n_dim)`.

    v_loss : `tensor`, shape (n_iter)
        If `log=True`, compute the loss accross iterations.
        If no early stopping, `n_iter` is equal to `max_iter`.
    """
    compute_gradient = staticmethod(compute_gradient_fadin)
    precomputations = True

    def __init__(self, n_dim, kernel, init='random', optim_mask=None,
                 kernel_length=1, delta=0.01, optim='RMSprop',
                 params_optim=dict(), max_iter=2000, ztzG_approx=True,
                 log=False, grad_kernel=None,
                 tol=10e-5, random_state=None):

        # Discretization parameters
        self.delta = delta
        self.kernel_length = kernel_length
        self.L = int(kernel_length / delta)
        self.ztzG_approx = ztzG_approx

        # Optimizer parameters
        self.kernel = kernel
        self.solver = optim
        self.max_iter = max_iter
        self.log = log
        self.tol = tol
        self.n_dim = n_dim
        self.kernel_model = DiscreteKernelFiniteSupport(
            self.delta,
            self.n_dim,
            kernel,
            self.kernel_length,
            0,
            self.kernel_length,
            grad_kernel
        )
        if optim_mask is None:
            optim_mask = {'baseline': None, 'alpha': None}
        if optim_mask['baseline'] is None:
            self.baseline_mask = torch.ones([n_dim])
        else:
            assert optim_mask['baseline'].shape == torch.Size([n_dim]), \
                "Invalid baseline_mask shape, must be (n_dim,)"
            self.baseline_mask = optim_mask['baseline']
        if optim_mask['alpha'] is None:
            self.alpha_mask = torch.ones([self.n_dim, self.n_dim])
        else:
            assert optim_mask['alpha'].shape == torch.Size([n_dim, n_dim]), \
                "Invalid alpha_mask shape, must be (n_dim, n_dim)"
            self.alpha_mask = optim_mask['alpha']

        # Model parameters
        if 'moment_matching' in init:
            self.init_mode = 'moment_matching'
            self.mm_mode = init.split('_')[-1]
        elif init == 'random':
            self.init_mode = 'random'
        else:
            self.init_mode = init
            assert 'baseline' in init.keys(), \
                "baseline parameter must be given"
            assert 'alpha' in init.keys(), \
                "alpha parameter must be given"
            assert 'kernel' in init.keys(), \
                "kernel parameter must be given"

        # If the learning rate is not given, fix it to 1e-3
        if 'lr' not in params_optim.keys():
            params_optim['lr'] = 1e-3
        self.params_solver = params_optim

        # device and seed
        if random_state is None:
            torch.manual_seed(0)
        else:
            torch.manual_seed(random_state)

    def fit(self, events, end_time):
        """Learn the parameters of the Hawkes processes on a discrete grid.

        Parameters
        ----------
        events : list of array of size number of timestamps,
        list size is self.n_dim.

        end_time : int
            The end time of the Hawkes process.

        Returns
        -------
        TODO: attributes
        self : object
            Fitted parameters.
        """
        # Initialize solver parameters
        n_grid = int(1 / self.delta) * end_time + 1
        discretization = torch.linspace(0, self.kernel_length, self.L)
        events_grid = projected_grid(events, self.delta, n_grid)
        n_events = events_grid.sum(1)
        n_ground_events = [events[i].shape[0] for i in range(len(events))]
        print('number of events is:', n_ground_events)
        n_ground_events = torch.tensor(n_ground_events)
        # Initialize Hawkes parameters
        self.params_intens = init_hawkes_params(
            self,
            self.init_mode,
            events,
            n_ground_events,
            end_time
        )

        # Initialize optimizer
        self.opt = optimizer(
            self.params_intens,
            self.params_solver,
            solver=self.solver
        )

        ####################################################
        # Precomputations
        ####################################################
        start = time.time()
        self.zG, self.zN, self.ztzG = compute_constants_fadin(events_grid,
                                                              self.L,
                                                              self.ztzG_approx)
        print('precomput:', time.time() - start)

        ####################################################
        # save results
        ####################################################
        self.v_loss = torch.zeros(self.max_iter)

        self.param_baseline = torch.zeros(self.max_iter + 1, self.n_dim)
        self.param_alpha = torch.zeros(self.max_iter + 1,
                                       self.n_dim,
                                       self.n_dim)
        self.param_kernel = torch.zeros(self.n_kernel_params,
                                        self.max_iter + 1,
                                        self.n_dim, self.n_dim)
        self.param_baseline[0] = self.params_intens[0].detach()
        self.param_alpha[0] = self.params_intens[1].detach()

        for i in range(self.n_kernel_params):
            self.param_kernel[i, 0] = self.params_intens[2 + i].detach()

        ####################################################
        start = time.time()
        # Optimize parameters
        for i in range(self.max_iter):
            print(f"Fitting model... {i/self.max_iter:6.1%}\r", end='',
                  flush=True)

            self.opt.zero_grad()
            self.compute_gradient(
                self, events_grid, discretization, i, n_events, end_time
            )
            self.opt.step()

            # Save parameters
            self.params_intens[0].data = self.params_intens[0].data.clip(0) * \
                self.baseline_mask
            self.params_intens[1].data = self.params_intens[1].data.clip(0) * \
                self.alpha_mask
            self.param_baseline[i + 1] = self.params_intens[0].detach()
            self.param_alpha[i + 1] = self.params_intens[1].detach()
            for j in range(self.n_kernel_params):
                self.params_intens[2 + j].data = \
                    self.params_intens[2 + j].data.clip(0)
                self.param_kernel[j, i + 1] = \
                    self.params_intens[2 + j].detach()

            # Early stopping
            if i % 100 == 0:
                error_b = torch.abs(self.param_baseline[i + 1] -
                                    self.param_baseline[i]).max()
                error_al = torch.abs(self.param_alpha[i + 1] -
                                     self.param_alpha[i]).max()
                error_k = torch.abs(self.param_kernel[0, i + 1] -
                                    self.param_kernel[0, i]).max()

                if error_b < self.tol and error_al < self.tol \
                   and error_k < self.tol:
                    print('early stopping at iteration:', i)
                    self.param_baseline = self.param_baseline[:i + 1]
                    self.param_alpha = self.param_alpha[:i + 1]
                    for j in range(self.n_kernel_params):
                        self.param_kernel[j] = self.param_kernel[j, i + 1]
                    break
        print('iterations in ', time.time() - start)

        return self


def plot(solver, plotfig=False, bl_noise=False, title=None, ch_names=None,
         savefig=None):
    """
    Plots estimated kernels and baselines of solver.
    Should be called after calling the `fit` method on solver.

    Parameters
    ----------
    solver: |`MarkedFaDin` or `FaDIn` solver.
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

    kappa_values = kernel.kernel_eval(solver.params_intens[-2:],
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
                     f'{round(solver.baseline[i].item(), 2)}')
            axs[i, j].hlines(
                y=solver.baseline[i].item(),
                xmin=0,
                xmax=solver.kernel_length,
                label=label,
                color='orange',
                linewidth=4
            )
            if bl_noise:
                # Plot noise baseline
                mutilde = round(solver.baseline_noise[i].item(), 2)
                label = rf'$\tilde{{\mu}}_{{{ch_names[i]}}}$={mutilde}'
                axs[i, j].hlines(
                    y=solver.baseline_noise[i].item(),
                    xmin=0,
                    xmax=solver.kernel_length,
                    label=label,
                    color='green',
                    linewidth=4
                )
            # Plot kernel (i, j)
            phi_values = solver.alpha[i, j].item() * kappa_values[i, j, 1:]
            axs[i, j].plot(
                discretization[1:],
                phi_values,
                label=rf'$\phi_{{{ch_names[i]},{ch_names[j]}}}$',
                linewidth=4
            )
            if solver.kernel == 'truncated_gaussian':
                # Plot mean of gaussian kernel
                mean = round(solver.params_intens[-2][i, j].item(), 2)
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
