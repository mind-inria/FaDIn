from pickletools import optimize
import numpy as np
import torch
from hawkes_discret.utils.utils import optimizer, projected_grid, init_kernel
from hawkes_discret.kernels import KernelExpDiscret
from hawkes_discret.loss_and_gradient import l2loss_precomputation, l2loss_conv
from hawkes_discret.loss_and_gradient import get_grad_mu, get_grad_alpha, get_grad_theta
from hawkes_discret.utils.compute_constants_np import get_zG, get_zN, get_ztzG
import time

class HawkesDiscretL2(object):
    """"""

    def __init__(self, kernel_name, 
                 kernel_params,
                 baseline, adjacency, discrete_step,
                 solver='GD', step_size=1e-3,
                 max_iter=100, log=False,
                 random_state=None, device='cpu', optimize_kernel=False):
        """
        events: list of tensor of size number of timestamps, size of the list is dim
        events_grid: tensor dim x (size_grid)
        kernel_model: class of kernel
        kernel_params: list of parameters of the kernel
        baseline: vecteur de taille dim
        adjacency: matrice de taille dim x dim (alpha)
        """

        # param discretisation
        self.discrete_step = discrete_step
        self.n_discrete = int(1 / discrete_step)

        # param optim
        self.solver = solver
        self.step_size = step_size
        self.max_iter = max_iter
        self.log = log

        # params model
        self.baseline = baseline.float().requires_grad_(True)
        self.adjacency = adjacency.float().requires_grad_(True)
        self.decay = kernel_params[0].float()

        self.n_kernel_params = len(kernel_params)
        #self.kernel_params_1 = kernel_params_1.float().requires_grad_(True)
        #self.kernel_params_2 = kernel_params_2.float().requires_grad_(True)

        self.kernel_model = init_kernel(1,  # upper bound of the kernel discretisation
                                        self.discrete_step,
                                        kernel_name=kernel_name)
        # Set l'optimizer
        self.params_optim = [self.baseline, self.adjacency]

        self.optimize_kernel = optimize_kernel
        if self.optimize_kernel:
            for i in range(self.n_kernel_params):
                self.params_optim.append(kernel_params[i].float().requires_grad_(True))

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

    def fit(self, events, end_time):
        start = time.time()
        size_grid = self.n_discrete * end_time + 1
        n_dim = len(events)
        discretization = torch.linspace(0, 1, int(1 / self.discrete_step))
        start = time.time()
        events_grid = projected_grid(
            events, self.discrete_step, size_grid)
        print('projec grid', time.time()-start)
        start = time.time()
        n_events = events_grid.sum(1)
        print('n_events_sum', time.time()-start)
        ####################################################
        # Precomputations
        ####################################################
        start = time.time()
        zG, _ = get_zG(events_grid.numpy(), self.n_discrete)
        zN, _ = get_zN(events_grid.numpy(), self.n_discrete)
        ztzG, _ = get_ztzG(events_grid.numpy(), self.n_discrete)
        zG = torch.tensor(zG).float()
        zN = torch.tensor(zN).float()
        ztzG = torch.tensor(ztzG).float()
        print('precomput:', time.time()-start)

        ####################################################
        # save results
        ####################################################
        v_loss = torch.zeros(self.max_iter)
        #grad_baseline = torch.zeros(self.max_iter, n_dim)
        #grad_adjacency = torch.zeros(self.max_iter, n_dim, n_dim)
        #grad_u_v = torch.zeros(self.max_iter, n_dim, n_dim)
        #grad_sigma_v = torch.zeros(self.max_iter, n_dim, n_dim)

        param_baseline = torch.zeros(self.max_iter+1, n_dim)
        param_adjacency = torch.zeros(self.max_iter+1, n_dim, n_dim)
        param_kernel = torch.zeros(self.n_kernel_params, self.max_iter+1, 
                                   n_dim, n_dim)

        param_baseline[0] = self.params_optim[0].detach()
        param_adjacency[0] = self.params_optim[1].detach()
        if self.optimize_kernel:
            for i in range(self.n_kernel_params):
                param_kernel[i, 0] = self.params_optim[2+i].detach()
        ####################################################
        start = time.time()
        #self.intensity = torch.zeros(self.max_iter, 2, size_grid)
        for i in range(self.max_iter):
            print(f"Fitting model... {i/self.max_iter:6.1%}\r", end='', flush=True)
            self.opt.zero_grad()

            if self.optimize_kernel:
                kernel = self.kernel_model.eval(self.params_optim[2:], 
                                                discretization)
            else:
                 kernel = self.kernel_model.eval([self.decay], 
                                                discretization)

            if self.optimize_kernel:
                grad_theta = self.kernel_model.compute_grad_(self.params_optim[2:],
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
                    self.params_optim[2+j].grad = get_grad_theta(zG,
                                                                zN,
                                                                ztzG,
                                                                self.params_optim[0],
                                                                self.params_optim[1],
                                                                kernel,
                                                                grad_theta[j],
                                                                self.discrete_step,
                                                                n_events)                                                                                                                                                      
            #############################

            #grad_baseline[i] = self.params_optim[0].grad.detach()
            #grad_adjacency[i] = self.params_optim[1].grad.detach()
            #grad_u_v[i] = self.params_optim[2].grad.detach()
            #grad_sigma_v[i] = self.params_optim[3].grad.detach()
            self.opt.step()
            self.params_optim[0].data = self.params_optim[0].data.clip(0)
            self.params_optim[1].data = self.params_optim[1].data.clip(0)
            param_baseline[i+1] = self.params_optim[0].detach()
            param_adjacency[i+1] = self.params_optim[1].detach()
            if self.optimize_kernel:             
                for j in range(self.n_kernel_params):
                    self.params_optim[2+j].data = self.params_optim[2+j].data.clip(0)
                    param_kernel[j, i+1] = self.params_optim[2+j].detach()

        print('iterations in ', time.time()-start)
        return dict(
            v_loss=v_loss,
            #grad_baseline=grad_baseline,
            #grad_adjacency=grad_adjacency,
            #grad_u_v=grad_u_v,
            #grad_sigma_v=grad_sigma_v,
            param_baseline=param_baseline,
            param_adjacency=param_adjacency,
            param_kernel=param_kernel
        )
