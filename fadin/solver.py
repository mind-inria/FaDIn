import torch
import time
import numpy as np

from fadin.utils.utils import optimizer, projected_grid
from fadin.utils.compute_constants import get_zG, get_zN, get_ztzG, get_ztzG_approx
from fadin.loss_and_gradient import discrete_l2loss_precomputation, \
    discrete_l2loss_conv, get_grad_baseline, get_grad_alpha, get_grad_theta
from fadin.kernels import DiscreteKernelFiniteSupport


class FaDIn(object):
    """Define the FaDIn framework for estimated Hawkes processes.

    The framework is detailed in::

    Guillaume Staerman, Cédric Allain, Alexandre Gramfort, Thomas Moreau
    FaDIn: Fast Discretized Inference for Hawkes Processes with General
    Parametric Kernels
    https://arxiv.org/abs/2210.04635

    Parameters
    ----------
    kernel : str or callable
        Either define a kernel in ('raised_cosine', 'truncated_gaussian' and
        'exponential') or a custom kernel.

    kernel_params_init : list of tensor of shape (n_dim, n_dim)
        Initial parameters of the kernel.

    baseline_init : tensor, shape (n_dim,)
        Initial baseline parameters of the intensity of the Hawkes process.

    alpha_init : tensor, shape (n_dim, n_dim)
        Initial alpha parameters of the intensity of the Hawkes process.

    delta : float, default=0.01
        Step size of the discretization grid.

    optim : str in {'RMSprop' | 'Adam' | 'GD'}, default='RMSprop'
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

    device : str in 'cpu' and 'cuda'
        Computations done on cpu or gpu. Gpu is not implemented yet.

    log : booleen, default=False
        Record the loss values during the optimization.

    grad_kernel : None or callable, default=None
        If kernel in ('raised_cosine', 'truncated_gaussian' and
        'exponential') the gradient function is implemented. If kernel is custom,
        the custom gradient must be given.

    random_state : int, RandomState instance or None, default=None
        Set the torch seed to ``random_state``.
    """

    def __init__(self, kernel, kernel_params_init, baseline_init, alpha_init,
                 delta=0.01, optim='RMSprop', step_size=1e-3, max_iter=1000,
                 optimize_kernel=True, precomputations=True, ztzG_approx=True,
                 device='cpu', log=False, grad_kernel=None, random_state=None):
        # param discretisation
        self.delta = delta
        self.L = int(1 / delta)
        self.ztzG_approx = ztzG_approx
        # param optim
        self.solver = optim
        self.step_size = step_size
        self.max_iter = max_iter
        self.log = log

        # params model
        self.baseline = baseline_init.float().requires_grad_(True)
        self.alpha = alpha_init.float().requires_grad_(True)
        self.kernel_params_fixed = kernel_params_init

        self.n_kernel_params = len(kernel_params_init)

        self.n_dim = baseline_init.shape[0]
        self.kernel_model = DiscreteKernelFiniteSupport(self.delta, self.n_dim,
                                                        kernel, 0, 1, grad_kernel)
        # Set l'optimizer
        self.params_optim = [self.baseline, self.alpha]

        self.optimize_kernel = optimize_kernel

        if self.optimize_kernel:
            for i in range(self.n_kernel_params):
                self.params_optim.append(
                    kernel_params_init[i].float().requires_grad_(True))

        self.precomputations = precomputations
        self.opt = optimizer(self.params_optim, lr=self.step_size, solver=optim)

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
        n_grid = self.L * end_time + 1
        discretization = torch.linspace(0, 1, int(1 / self.delta))
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
                else:
                    kernel = self.kernel_model.kernel_eval(self.kernel_params_fixed,
                                                           discretization)

                if self.optimize_kernel:
                    grad_theta = self.kernel_model.grad_eval(self.params_optim[2:],
                                                             discretization)
                if self.log:
                    v_loss[i] = discrete_l2loss_precomputation(zG, zN, ztzG,
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
                            get_grad_theta(zG,
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
                loss = discrete_l2loss_conv(intens, events_grid, self.delta)
                loss.backward()

            self.opt.step()
            self.params_optim[0].data = self.params_optim[0].data.clip(0)
            self.params_optim[1].data = self.params_optim[1].data.clip(0)
            param_baseline[i + 1] = self.params_optim[0].detach()
            param_alpha[i + 1] = self.params_optim[1].detach()

            if self.optimize_kernel:
                for j in range(self.n_kernel_params):
                    self.params_optim[2 + j].data = \
                        self.params_optim[2 + j].data.clip(0)
                    param_kernel[j, i + 1] = self.params_optim[2 + j].detach()

        print('iterations in ', time.time() - start)

        return dict(v_loss=v_loss, param_baseline=param_baseline,
                    param_alpha=param_alpha, param_kernel=param_kernel)
