import torch
import time

from fadin.utils.utils import optimizer, projected_grid
from fadin.utils.compute_constants import get_zG, get_zN, get_ztzG, get_ztzG_approx
from fadin.loss_and_gradient import discrete_l2_loss_precomputation, \
    discrete_l2_loss_conv, get_grad_baseline, get_grad_alpha, get_grad_eta, \
    discrete_ll_loss_conv
from fadin.kernels import DiscreteKernelFiniteSupport


class FaDIn(object):
    """Define the FaDIn framework for estimated Hawkes processes.

    The framework is detailed in::

    Guillaume Staerman, Cédric Allain, Alexandre Gramfort, Thomas Moreau
    FaDIn: Fast Discretized Inference for Hawkes Processes with General
    Parametric Kernels
    https://arxiv.org/abs/2210.04635

    FaDIn minimizes the discretized L2 loss of Hawkes processes
    defined by the intensity  as a convolution between the kernel
    :math: `\\phi_{ij}` and the sum of Dirac functions
    :math: `z_i  := \\sum_{t^i_n \\in \\mathscr{F}^i_T} \\delta_{t^i_n}$
    located at the event occurrences $t^i_n$:

    .. math::
        \\forall i \\in [1 \\dots D], \\quad
        \\lambda_i(t) = \\mu_i + \\sum_{j=1}^p \\phi_{ij} * z_j(t),
        \\quad t \\in [0, T]

    where

    * :math:`p` is the dimension of the process
    * :math:`\\mu_i` are the baseline intensities
    * :math:`\\phi_{ij}` are the kernels
    * :math:`z_j(t)` are the activation vector on the discretized grid,
    the projection of the timestamps on this grid

    Parameters
    ----------
    n_dim: int
        Dimension of the underlying Hawkes process.

    kernel : str or callable
        Either define a kernel in ``{'raised_cosine', 'truncated_gaussian' and
        'truncated_exponential'}`` or a custom kernel.

    kernel_params_init : list of tensor of shape (n_dim, n_dim)
        Initial parameters of the kernel.

    baseline_init : tensor, shape (n_dim,)
        Initial baseline parameters of the intensity of the Hawkes process.

    alpha_init : tensor, shape (n_dim, n_dim)
        Initial alpha parameters of the intensity of the Hawkes process.

    kernel_length: int, default=1
        Length of kernels in the Hawkes process.

    delta : float, default=0.01
        Step size of the discretization grid.

    optim : str in ``{'RMSprop' | 'Adam' | 'GD'}``, default=``'RMSprop'``
        The algorithms used to optimized the parameters of the Hawkes processes.

    step_size : float, default=1e-3
        Learning rate of the chosen optimization algorithm.

    max_iter : int, default=1000
        Maximum number of iterations during fit.

    optimize_kernel : bool, default=True
        If optimize_kernel is false, kernel parameters are not optimized
        and only the baseline and alpha are optimized.

    precomputations : bool, default=True
        If precomputations is false, pytorch autodiff is applied on the loss.
        If precomputations is true, then FaDIn is computed.

    ztzG_approx : bool, default=True
        If ztzG_approx is false, compute the true ztzG precomputation constant that
        is the computational bottleneck of FaDIn. if ztzG_approx is false,
        ztzG is approximated with Toeplitz matrix not taking into account edge effects.

    device : str in ``{'cpu', 'cuda'}
        Computations done on cpu or gpu. Gpu is not implemented yet.

    log : booleen, default=False
        Record the loss values during the optimization.

    grad_kernel : None or callable, default=None
        If kernel in ``{'raised_cosine', 'truncated_gaussian' and
        'truncated_exponential'}`` the gradient function is implemented.
        If kernel is custom, the custom gradient must be given.

    criterion : str in ``{'l2' | 'll'}``, default=``'l2'``
        The criterion to minimize. if not l2, FaDIn minimize
        the Log-Likelihood loss with AutoDifferentiation.

    tol : `float`, default=1e-5
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). If not reached the solver does 'max_iter'
        iterations.

    random_state : int, RandomState instance or None, default=None
        Set the torch seed to 'random_state'.
    """

    def __init__(self, n_dim, kernel, kernel_params_init=None, baseline_init=None,
                 alpha_init=None, kernel_length=1, delta=0.01, optim='RMSprop',
                 step_size=1e-3, max_iter=2000, optimize_kernel=True,
                 precomputations=True, ztzG_approx=True, device='cpu', log=False,
                 grad_kernel=None, criterion='l2', tol=10e-5, random_state=None):
        # param discretisation
        self.delta = delta
        self.W = kernel_length
        self.L = int(self.W / delta)
        self.ztzG_approx = ztzG_approx

        # param optim
        self.solver = optim
        self.step_size = step_size
        self.max_iter = max_iter
        self.log = log
        self.tol = tol

        # params model
        self.n_dim = n_dim
        if baseline_init is None:
            self.baseline = torch.rand(self.n_dim).requires_grad_(True)
        else:
            self.baseline = baseline_init.float().requires_grad_(True)
        if alpha_init is None:
            self.alpha = torch.rand(self.n_dim, self.n_dim).requires_grad_(True)
        else:
            self.alpha = alpha_init.float().requires_grad_(True)

        if kernel_params_init is None:
            kernel_params_init = []
            if kernel == 'raised_cosine':
                temp = torch.rand(self.n_dim, self.n_dim)
                temp2 = torch.rand(self.n_dim, self.n_dim) * temp
                kernel_params_init.append(temp)
                kernel_params_init.append(temp2)
            elif kernel == 'truncated_gaussian':
                kernel_params_init.append(torch.rand(self.n_dim, self.n_dim))
                kernel_params_init.append(torch.rand(self.n_dim, self.n_dim))
            elif kernel == 'truncated_exponential':
                kernel_params_init.append(2 * torch.rand(self.n_dim,
                                                         self.n_dim))
            else:
                raise NotImplementedError('kernel initial parameters of not \
                                           implemented kernel have to be given')

        self.kernel_params_fixed = kernel_params_init

        self.n_kernel_params = len(kernel_params_init)

        self.kernel_model = DiscreteKernelFiniteSupport(self.delta, self.n_dim, kernel,
                                                        self.W, 0, self.W,
                                                        grad_kernel)
        self.kernel = kernel
        # Set l'optimizer
        self.params_optim = [self.baseline, self.alpha]

        self.optimize_kernel = optimize_kernel

        if self.optimize_kernel:
            for i in range(self.n_kernel_params):
                self.params_optim.append(
                    kernel_params_init[i].float().requires_grad_(True))

        self.precomputations = precomputations
        self.opt = optimizer(self.params_optim, lr=self.step_size, solver=optim)

        if criterion == 'll':
            self.precomputations = False

        self.criterion = criterion
        # device and seed
        if random_state is None:
            torch.manual_seed(0)
        else:
            torch.manual_seed(random_state)

        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def fit(self, events, end_time):
        """Optimize the parameters of the Hawkes processes on a discrete grid.

        Parameters
        ----------
        events : list of array of size number of timestamps, size of the list is dim

        end_time : float
            The end time of the Hawkes process.

        Returns
        -------
        self : object
            Fitted parameters.
        """
        n_grid = int(1 / self.delta) * end_time + 1
        discretization = torch.linspace(0, self.W, self.L)
        events_grid = projected_grid(events, self.delta, n_grid)
        n_events = events_grid.sum(1)

        ####################################################
        # Precomputations
        ####################################################
        if self.precomputations:
            print('number of events is:', n_events)
            start = time.time()
            zG = get_zG(events_grid.double().numpy(), self.L)
            zN = get_zN(events_grid.double().numpy(), self.L)

            if self.ztzG_approx:
                ztzG = get_ztzG_approx(events_grid.double().numpy(), self.L)
            else:
                ztzG = get_ztzG(events_grid.double().numpy(), self.L)

            zG = torch.tensor(zG).float()
            zN = torch.tensor(zN).float()
            ztzG = torch.tensor(ztzG).float()
            print('precomput:', time.time() - start)

        ####################################################
        # save results
        ####################################################
        v_loss = torch.zeros(self.max_iter)

        param_baseline = torch.zeros(self.max_iter + 1, self.n_dim)
        param_alpha = torch.zeros(self.max_iter + 1, self.n_dim, self.n_dim)
        param_kernel = torch.zeros(self.n_kernel_params, self.max_iter + 1,
                                   self.n_dim, self.n_dim)

        param_baseline[0] = self.params_optim[0].detach()
        param_alpha[0] = self.params_optim[1].detach()

        # If kernel parameters are optimized
        if self.optimize_kernel:
            for i in range(self.n_kernel_params):
                param_kernel[i, 0] = self.params_optim[2 + i].detach()

        ####################################################
        start = time.time()
        for i in range(self.max_iter):
            print(f"Fitting model... {i/self.max_iter:6.1%}\r", end='', flush=True)

            self.opt.zero_grad()
            if self.precomputations:
                if self.optimize_kernel:
                    kernel = self.kernel_model.kernel_eval(self.params_optim[2:],
                                                           discretization)
                    grad_theta = self.kernel_model.grad_eval(self.params_optim[2:],
                                                             discretization)
                else:
                    kernel = self.kernel_model.kernel_eval(self.kernel_params_fixed,
                                                           discretization)

                if self.log:
                    v_loss[i] = discrete_l2_loss_precomputation(zG, zN, ztzG,
                                                                self.params_optim[0],
                                                                self.params_optim[1],
                                                                kernel, n_events,
                                                                self.delta,
                                                                end_time).detach()

                self.params_optim[0].grad = get_grad_baseline(zG,
                                                              self.params_optim[0],
                                                              self.params_optim[1],
                                                              kernel, self.delta,
                                                              n_events, end_time)

                self.params_optim[1].grad = get_grad_alpha(zG,
                                                           zN,
                                                           ztzG,
                                                           self.params_optim[0],
                                                           self.params_optim[1],
                                                           kernel,
                                                           self.delta,
                                                           n_events)
                if self.optimize_kernel:
                    for j in range(self.n_kernel_params):
                        self.params_optim[2 + j].grad = \
                            get_grad_eta(zG,
                                         zN,
                                         ztzG,
                                         self.params_optim[0],
                                         self.params_optim[1],
                                         kernel,
                                         grad_theta[j],
                                         self.delta,
                                         n_events)

            else:
                intens = self.kernel_model.intensity_eval(self.params_optim[0],
                                                          self.params_optim[1],
                                                          self.params_optim[2:],
                                                          events_grid,
                                                          discretization)
                if self.criterion == 'll':
                    loss = discrete_ll_loss_conv(intens, events_grid, self.delta)
                else:
                    loss = discrete_l2_loss_conv(intens, events_grid, self.delta)
                loss.backward()

            self.opt.step()

            # Save parameters
            self.params_optim[0].data = self.params_optim[0].data.clip(0)
            self.params_optim[1].data = self.params_optim[1].data.clip(0)
            param_baseline[i + 1] = self.params_optim[0].detach()
            param_alpha[i + 1] = self.params_optim[1].detach()

            # If kernel parameters are optimized
            if self.optimize_kernel:
                for j in range(self.n_kernel_params):
                    if self.kernel != 'truncated_skewed_gaussian':
                        self.params_optim[2 + j].data = \
                            self.params_optim[2 + j].data.clip(0.0001)
                    else:
                        self.params_optim[2 + j].data = \
                            self.params_optim[2 + j].data
                    param_kernel[j, i + 1] = self.params_optim[2 + j].detach()

            # Early stopping
            if i % 100 == 0:
                error_b = torch.abs(param_baseline[i + 1] - param_baseline[i]).max()
                error_al = torch.abs(param_alpha[i + 1] - param_alpha[i]).max()
                error_k = torch.abs(param_kernel[0, i + 1] - param_kernel[0, i]).max()

                if error_b < self.tol and error_al < self.tol and error_k < self.tol:
                    print('early stopping at iteration:', i)
                    param_baseline = param_baseline[:i + 1]
                    param_alpha = param_alpha[:i + 1]
                    for j in range(self.n_kernel_params):
                        param_kernel[j] = param_kernel[j, i + 1]
                    break
        print('iterations in ', time.time() - start)

        return dict(v_loss=v_loss, param_baseline=param_baseline,
                    param_alpha=param_alpha, param_kernel=param_kernel)
