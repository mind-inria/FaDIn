import torch
from fadin.utils.utils import optimizer, projected_grid
from fadin.loss_and_gradient import l2loss_precomputation, l2loss_conv
from fadin.loss_and_gradient import get_grad_mu, get_grad_alpha, get_grad_theta
from fadin.utils.compute_constants import get_zG, get_zN, get_ztzG, get_ztzG_
from fadin.kernels import DiscreteKernelFiniteSupport
import time


class FaDIn(object):
    """"""

    def __init__(self, kernel,
                 kernel_params,
                 baseline, alpha, discrete_step,
                 solver='GD', step_size=1e-3,
                 max_iter=100, log=False,
                 random_state=None, device='cpu',
                 optimize_kernel=False, precomputations=True,
                 side_effects=True):
        """
        events: list of tensor of size number of timestamps, size of the list is dim
        events_grid: tensor dim x (size_grid)
        kernel_model: class of kernel
        kernel_params: list of parameters of the kernel
        baseline: vecteur de taille dim
        alpha: matrice de taille dim x dim (alpha)
        """

        # param discretisation
        self.discrete_step = discrete_step
        self.L = int(1 / discrete_step)
        self.side_effects = side_effects
        # param optim
        self.solver = solver
        self.step_size = step_size
        self.max_iter = max_iter
        self.log = log

        # params model
        self.baseline = baseline.float().requires_grad_(True)
        self.alpha = alpha.float().requires_grad_(True)
        self.decay = kernel_params[0].float()

        self.n_kernel_params = len(kernel_params)

        self.n_dim = baseline.shape[0]
        self.kernel_model = DiscreteKernelFiniteSupport(0, 1, self.discrete_step,
                                                        kernel, self.n_dim)
        # Set l'optimizer
        self.params_optim = [self.baseline, self.alpha]

        self.optimize_kernel = optimize_kernel
        if self.optimize_kernel:
            for i in range(self.n_kernel_params):
                self.params_optim.append(kernel_params[i].float().requires_grad_(True))
        self.precomputations = precomputations
        self.opt = optimizer(
            self.params_optim,
            lr=self.step_size,
            solver=self.solver)

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
        size_grid = self.L * end_time + 1
        discretization = torch.linspace(0, 1, int(1 / self.discrete_step))
        start = time.time()
        events_grid = projected_grid(
            events, self.discrete_step, size_grid)
        print('projec grid', time.time() - start)
        start = time.time()
        n_events = events_grid.sum(1)
        print('n_events_sum', time.time() - start)
        ####################################################
        # Precomputations
        ####################################################
        if self.precomputations:
            print('number of events is:', n_events)
            start = time.time()
            zG = get_zG(events_grid.numpy(), self.L)
            zN = get_zN(events_grid.numpy(), self.L)

            if self.side_effects:
                ztzG = get_ztzG(events_grid.numpy(), self.L)
            else:
                ztzG = get_ztzG_(events_grid.numpy(), self.L)

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
                    kernel = self.kernel_model.eval(self.params_optim[2:],
                                                    discretization)
                else:
                    kernel = self.kernel_model.eval([self.decay],
                                                    discretization)

                if self.optimize_kernel:
                    grad_theta = self.kernel_model.get_grad(self.params_optim[2:],
                                                            discretization)
                if self.log:
                    v_loss[i] = l2loss_precomputation(zG, zN, ztzG,
                                                      self.params_optim[0],
                                                      self.params_optim[1],
                                                      kernel, n_events,
                                                      self.discrete_step,
                                                      end_time).detach()

                self.params_optim[0].grad = get_grad_mu(zG,
                                                        self.params_optim[0],
                                                        self.params_optim[1],
                                                        kernel, self.discrete_step,
                                                        n_events, end_time)

                self.params_optim[1].grad = get_grad_alpha(zG,
                                                           zN,
                                                           ztzG,
                                                           self.params_optim[0],
                                                           self.params_optim[1],
                                                           kernel,
                                                           self.discrete_step,
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
                                           self.discrete_step,
                                           n_events)
            else:
                intens = self.kernel_model.intensity_eval(self.params_optim[0],
                                                          self.params_optim[1],
                                                          self.params_optim[2:],
                                                          events_grid,
                                                          discretization)
                # print(intens)
                loss = l2loss_conv(intens, events_grid, self.discrete_step)
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
        return dict(
            v_loss=v_loss,
            param_baseline=param_baseline,
            param_alpha=param_alpha,
            param_kernel=param_kernel
        )
