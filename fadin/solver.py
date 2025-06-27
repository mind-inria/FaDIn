import torch
import time

from fadin.utils.utils import optimizer, projected_grid
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

        # Initialization option for Hawkes parameters
        s = ['random', 'moment_matching_max', 'moment_matching_mean']
        if isinstance(init, str):
            assert init in s, (
                f"Invalid string init {init}. init must be a dict or in {s}."
            )
        else:
            keys = set(['baseline', 'alpha', 'kernel'])
            is_dict = isinstance(init, dict)
            assert is_dict and set(init.keys()) == keys, (
                f"If init is not a str, it should be a dict with keys {keys}. "
                f"Got {init}."
            )
        self.init = init

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
            self.init,
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
    

class UNHaP(object):
    """Define the UNHaP framework for estimated mixture of Hawkes and
    Poisson processes.

    Unhap minimizes the discretized L2 mixture loss of Hawkes and Poisson processes.

    Parameters
    ----------
    n_dim : `int`
        Dimension of the underlying Hawkes process.

    kernel : `str` or `callable`
        Either define a kernel in ``{'raised_cosine' | 'truncated_gaussian' |
        'truncated_exponential'}`` or a custom kernel.

    baseline_mask : `tensor` of shape (n_dim,), or `None`
        Tensor of same shape as the baseline vector, with values in (0, 1).
        `baseline` coordinates where `baseline_mask` is equal to 0
        will stay constant equal to zero and not be optimized.
        If set to `None`, all coordinates of baseline will be optimized.

    alpha_mask : `tensor` of shape (n_dim, n_dim), or `None`
        Tensor of same shape as the `alpha` tensor, with values in (0, 1).
        `alpha` coordinates and kernel parameters where `alpha_mask` = 0
        will not be optimized.
        If set to `None`, all coordinates of alpha will be optimized,
        and all kernel parameters will be optimized if optimize_kernel=`True`.

    kernel_length : `float`, `default=1.`
        Length of kernels in the Hawkes process.

    delta : `float`, `default=0.01`
        Step size of the discretization grid.

    optim : `str` in ``{'RMSprop' | 'Adam' | 'GD'}``, default='RMSprop'
        The algorithm used to optimize the Hawkes processes parameters.

    params_optim : dict, {'lr', ...}
        Learning rate and parameters of the chosen optimization algorithm.

    max_iter : `int`, `default=2000`
        Maximum number of iterations during fit.

    ztzG_approx : `boolean`, `default=True`
        If ztzG_approx is false, compute the true ztzG precomputation constant
        that is the computational bottleneck of FaDIn. if ztzG_approx is true,
        ztzG is approximated with Toeplitz matrix not taking into account
        edge effects.

    batch_rho : `int`
        Number of FaDIn iterations between latent variables rho updates.

    tol : `float`, `default=1e-5`
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). If not reached the solver does 'max_iter'
        iterations.

    density_hawkes : `str`, `default=linear`
        Density of the marks of the Hawkes process, in ``{'linear' | 'uniform'}``.

    density_noise : `str`, `default=reverse_linear`
        Density of the marks of the Hawkes process, in
        ``{'reverse_linear' | 'uniform'}``.

    moment_matching: boolean, `default=False`
        If set to False, baseline, alpha and kernel parameters are randomly
        chosen. If set to True, baseline, alpha and kernel parameters are
        chosen using the smart init strategy.
        The smart init strategy is only implemented
        for truncated gaussian and raised_cosine kernels.

    random_state : `int`, `RandomState` instance or `None`, `default=None`
        Set the torch seed to 'random_state'.
        If set to `None`, torch seed will be set to 0.

    Attributes
    ----------
    param_baseline : `tensor`, shape (n_dim)
        Baseline parameter of the Hawkes process.

    param_baseline_noise : `tensor`, shape (n_dim)
        Baseline parameter of the Hawkes process.

    param_alpha : `tensor`, shape (n_dim, n_dim)
        Weight parameter of the Hawkes process.

    param_kernel : `list` of `tensor`
        list containing tensor array of kernels parameters.
        The size of the list varies depending the number of
        parameters. The shape of each tensor is `(n_dim, n_dim)`.

    """
    def __init__(self, n_dim, kernel, baseline_mask=None, alpha_mask=None,
                 kernel_length=1, delta=0.01, optim='RMSprop',
                 params_optim=dict(), max_iter=2000, batch_rho=100,
                 ztzG_approx=True, tol=10e-5, density_hawkes='linear',
                 density_noise='uniform', moment_matching=False, random_state=None):

        # Set discretization parameters
        self.delta = delta
        self.kernel_length = kernel_length
        self.L = int(self.kernel_length / delta)
        self.ztzG_approx = ztzG_approx

        # Set optimizer parameters
        self.solver = optim
        self.max_iter = max_iter
        self.tol = tol
        self.batch_rho = batch_rho

        self.density_hawkes = density_hawkes
        self.density_noise = density_noise
        self.moment_matching = moment_matching

        # Set model parameters
        self.n_dim = n_dim

        self.baseline = torch.rand(self.n_dim)

        if baseline_mask is None:
            self.baseline_mask = torch.ones([n_dim])
        else:
            assert baseline_mask.shape == self.baseline.shape, \
                "Invalid baseline_mask shape, must be (n_dim,)"
            self.baseline_mask = baseline_mask
        self.baseline = (self.baseline * self.baseline_mask).requires_grad_(True)

        self.baseline_noise = torch.rand(self.n_dim).requires_grad_(True)

        self.alpha = torch.rand(self.n_dim, self.n_dim)

        if alpha_mask is None:
            self.alpha_mask = torch.ones([self.n_dim, self.n_dim])
        else:
            assert alpha_mask.shape == self.alpha.shape, \
                "Invalid alpha_mask shape, must be (n_dim, n_dim)"
            self.alpha_mask = alpha_mask
        self.alpha = (self.alpha * self.alpha_mask).requires_grad_(True)

        kernel_params_init = init_kernel_parameters(kernel,
                                                    self.kernel_length,
                                                    self.n_dim)

        self.n_kernel_params = len(kernel_params_init)

        self.kernel_model = DiscreteKernelFiniteSupport(self.delta, self.n_dim,
                                                        kernel, self.kernel_length, 0)
        self.kernel = kernel
        # Set optimizer
        self.params_intens = [self.baseline, self.baseline_noise, self.alpha]

        for i in range(self.n_kernel_params):
            self.params_intens.append(
                kernel_params_init[i].float().clip(1e-4).requires_grad_(True))

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
        events : pandas DataFrame with three columns: timestamps, marks values
            and type of events.

        end_time : int
            The end time of the Hawkes process.

        Returns
        -------
        self : object
            Fitted parameters.
        """
        n_grid = int(1 / self.delta) * end_time + 1
        discretization = torch.linspace(0, self.kernel_length, self.L)

        events_grid, marks_grid, marked_events, _ = \
            smooth_projection_marked(events, self.delta, n_grid)

        sum_marks = marks_grid.sum(1)
        print('sum of marks per channel:', sum_marks)

        # number of events per dimension
        n_ground_events = events_grid.sum(1)
        print('number of events is:', n_ground_events)

        self.sum_marks = sum_marks
        self.n_ground_events = n_ground_events

        mask_events = torch.where(events_grid > 0)
        mask_void = torch.where(events_grid == 0)

        # computation of the marked quantities
        marked_quantities = compute_marked_quantities(events_grid, marks_grid,
                                                      self.n_dim, self.density_hawkes,
                                                      self.density_noise)

        self.rho = torch.zeros(self.n_dim, n_grid)
        self.rho[mask_events] = 0.5  # Init
        self.rho = self.rho.requires_grad_(True)
        # add rho parameter at the end of params
        self.params_mixture = [self.rho]

        if self.moment_matching:
            # Smart initialization of solver parameters
            self.baseline, self.baseline_noise, self.alpha, kernel_params_init = \
                smart_init_noise(self, marked_events, n_ground_events, end_time)

            # Set optimizer with smartinit parameters
            self.params_intens = [self.baseline, self.baseline_noise, self.alpha]

            for i in range(2):  # range(self.n_params_kernel)
                self.params_intens.append(
                    kernel_params_init[i].float().clip(1e-4).requires_grad_(True)
                )

        self.opt_intens, self.opt_mixture = optimizer_unhap([self.params_intens,
                                                             self.params_mixture],
                                                            self.params_solver,
                                                            solver=self.solver)

        self.param_rho = self.params_mixture[0].detach()

        ####################################################
        # Precomputations
        ####################################################

        start = time.time()
        z_tilde = self.rho * marks_grid
        precomputations = compute_constants_unhap(z_tilde,
                                                  marks_grid,
                                                  events_grid,
                                                  self.param_rho,
                                                  marked_quantities[1],
                                                  self.L)
        print('precomput:', time.time() - start)

        ####################################################
        # save results
        ####################################################

        _, self.param_baseline, self.param_baseline_noise, \
            self.param_alpha, self.param_kernel = init_saving_params(
                self.params_intens, self.n_kernel_params, self.n_dim, self.max_iter)

        # If kernel parameters are optimized
        for i in range(self.n_kernel_params):
            self.param_kernel[i, 0] = self.params_intens[3 + i].detach()

        ####################################################
        start = time.time()
        for i in range(self.max_iter):
            print(f"Fitting model... {i/self.max_iter:6.1%}\r", end='',
                  flush=True)

            self.opt_intens.zero_grad()

            # Update kernel and grad values
            kernel, grad_eta = self.kernel_model.kernel_and_grad(
                self.params_intens[3:3 + self.n_kernel_params],
                discretization
            )

            ####################################################
            # Compute gradients
            ####################################################

            # Update baseline, baseline noise and alpha
            self.params_intens[0].grad, self.params_intens[1].grad, \
                self.params_intens[2].grad = compute_base_gradients(
                    precomputations,
                    self.params_intens,
                    kernel,
                    self.delta,
                    end_time,
                    self.param_rho,
                    marked_quantities,
                    n_ground_events
                )

            # Update kernel
            for j in range(self.n_kernel_params):
                self.params_intens[3 + j].grad = \
                    get_grad_eta_mixture(
                        precomputations,
                        self.params_intens[0],
                        self.params_intens[2],
                        kernel,
                        grad_eta[j],
                        self.delta,
                        marked_quantities[0],
                        n_ground_events)

            self.opt_intens.step()

            if i % self.batch_rho == 0:
                self.opt_mixture.zero_grad()

                z_tilde_ = marks_grid * self.params_mixture[0].data
                self.params_mixture[0].grad = get_grad_rho_mixture(
                    z_tilde_,
                    marks_grid,
                    kernel,
                    marked_quantities[0],
                    self.params_intens,
                    self.delta,
                    mask_void,
                    n_ground_events,
                    marked_quantities)

                self.opt_mixture.step()

                self.params_mixture[0].data = self.params_mixture[0].data.clip(0, 1)
                self.params_mixture[0].data[mask_void] = 0.
                # round the rho
                z_tilde = marks_grid * torch.round(self.params_mixture[0].data)
                precomputations = compute_constants_unhap(
                    z_tilde, marks_grid, events_grid,
                    self.param_rho, marked_quantities[1], self.L)

                self.param_rho = torch.round(self.params_mixture[0].detach())

            # Save and clip parameters
            self.params_intens[0].data = self.params_intens[0].data.clip(0) * \
                self.baseline_mask
            self.params_intens[1].data = self.params_intens[1].data.clip(0)
            self.params_intens[2].data = self.params_intens[2].data.clip(0) * \
                self.alpha_mask

            self.param_baseline[i+1] = self.params_intens[0].detach()
            self.param_alpha[i+1] = self.params_intens[2].detach()
            self.param_baseline_noise[i+1] = self.params_intens[1].detach()

            for j in range(self.n_kernel_params):
                self.params_intens[3+j].data = \
                    self.params_intens[3+j].data.clip(1e-3)
                self.param_kernel[j, i+1] = self.params_intens[3+j].detach()

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
    discretization = torch.linspace(0, solver.W, 200)
    kernel = DiscreteKernelFiniteSupport(solver.delta,
                                         solver.n_dim,
                                         kernel=solver.kernel,
                                         kernel_length=solver.W)

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
                xmax=solver.W,
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
                    xmax=solver.W,
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
