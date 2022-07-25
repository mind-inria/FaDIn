import numpy as np
import torch
from hawkes_discret.utils.utils import optimizer, projected_grid, init_kernel
from hawkes_discret.kernels import KernelExpDiscret


class HawkesDiscretL2(object):
    """"""

    def __init__(self, kernel_name, kernel_params,
                 baseline, adjacency, discrete_step,
                 solver='GD', step_size=1e-3,
                 max_iter=100, log=False,
                 random_state=None, device='cpu'):
        """
        events: list of tensor of size number of timestamps, size of the list is dim
        events_grid: tensor dim x (size_grid)
        kernel_model: class of kernel
        kernel_param: list of parameters of the kernel
        baseline: vecteur de taille dim
        adjacency: matrice de taille dim x dim (alpha)
        """
        # events,end time dans fit
        #
        # param discretisation
        self.discrete_step = discrete_step

        # param optim
        self.solver = solver
        self.step_size = step_size
        self.max_iter = max_iter
        self.log = log

        # params model
        self.baseline = baseline.float().requires_grad_(True)
        self.adjacency = adjacency.float().requires_grad_(True)
        self.kernel_params = kernel_params.float().requires_grad_(True)

        self.kernel_model = init_kernel(self.kernel_params,
                                        1,  # upper bound of the kernel discretisation
                                        self.discrete_step,
                                        kernel_name=kernel_name)
        # Set l'optimizer
        self.params_optim = [self.baseline, self.adjacency, self.kernel_params]
        self.opt = optimizer(
            self.params_optim,
            lr=self.step_size,
            solver=self.solver)

        #device and seed
        if random_state is None:
            torch.manual_seed(0)
        else:
            torch.manual_seed(random_state)

        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'

    def grad_baseline(self):
        """Return grad w.r.t. adjacency matrix: (dim x dim x len(time))
        """
        return self.kernel_values

    def grad_adjacency(self):
        """Return grad w.r.t. adjacency matrix: (dim x dim x len(time))
        """
        return self.kernel_values

    def fit(self, events, end_time):

        self.size_grid = int(end_time / self.discrete_step)
        self.events = events
        self.end_time = end_time

        self.events_grid, self.events_loc_grid = projected_grid(
            self.events, self.discrete_step, self.size_grid)

        self.events_loc_grid_bool = self.events_loc_grid.to(torch.bool)

        for i in range(self.max_iter):
            self.opt.zero_grad()
            if log:
                loss = lossl2()

            self.baseline.grad = grad_baseline()
            self.adjacency.grad = grad_adjacency()
            self.kernel_params.grad = gradient_kernel(
            ) * self.kernel_model.grad_params()  # grad chain rules

            self.opt.step()

            if torch.isnan(
                    self.baseline.grad).any() | torch.isnan(
                    self.adjacency.grad).any() | torch.isnan(
                    self.kernel_params.grad).any():
                raise ValueError('NaNs in coeffs! Stop optimization...')
        return
