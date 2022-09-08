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
                 kernel_params_1, kernel_params_2,
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
        self.n_discrete = int(1 / discrete_step)
        # param optim
        self.solver = solver
        self.step_size = step_size
        self.max_iter = max_iter
        self.log = log

        # params model
        self.baseline = baseline.float().requires_grad_(True)
        self.adjacency = adjacency.float().requires_grad_(True)
        self.kernel_params_1 = kernel_params_1.float().requires_grad_(True)
        self.kernel_params_2 = kernel_params_2.float().requires_grad_(True)

        self.kernel_model = init_kernel(1,  # upper bound of the kernel discretisation
                                        self.discrete_step,
                                        kernel_name=kernel_name)
        # Set l'optimizer
        self.params_optim = [self.baseline, self.adjacency, 
                             self.kernel_params_1, self.kernel_params_2]
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

        size_grid = self.n_discrete * end_time + 1
        n_dim = len(events)
        discretization = torch.linspace(0, 1, int(1 / self.discrete_step))
        print(events)
        print(self.discrete_step)
        print(size_grid)
        events_grid = projected_grid(
            events, self.discrete_step, size_grid)
        n_events = events_grid.sum(1)
        # precomputations
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
        grad_baseline = torch.zeros(self.max_iter, n_dim)
        grad_adjacency = torch.zeros(self.max_iter, n_dim, n_dim)
        grad_u_v = torch.zeros(self.max_iter, n_dim, n_dim)
        grad_sigma_v = torch.zeros(self.max_iter, n_dim, n_dim)

        param_baseline = torch.zeros(self.max_iter+1, n_dim)
        param_adjacency = torch.zeros(self.max_iter+1, n_dim, n_dim)
        param_u = torch.zeros(self.max_iter+1, n_dim, n_dim)
        param_sigma = torch.zeros(self.max_iter+1, n_dim, n_dim)
        param_baseline[0] = self.params_optim[0].detach()
        param_adjacency[0] = self.params_optim[1].detach()
        param_u[0] = self.params_optim[2].detach()
        param_sigma[0] = self.params_optim[3].detach()
        ####################################################
        start = time.time()
        #self.intensity = torch.zeros(self.max_iter, 2, size_grid)
        for i in range(self.max_iter):
            print(f"Fitting model... {i/self.max_iter:6.1%}\r", end='', flush=True)
            self.opt.zero_grad()

            # Optim conv discrete
            # intensity = self.kernel_model.intensity_eval(self.params_optim[0],
            #                                             self.params_optim[1],
            #                                             self.params_optim[2],
            #                                             events_grid,
            #                                             discretization)

            #loss = l2loss_conv(intensity, events_grid, self.discrete_step, self.end_time)
            #v_loss[i] = loss.data
            # loss.backward()
            #############################

            # Optim precomputation discrete
            # en construction
            #start = time.time()
            kernel = self.kernel_model.eval(
                self.params_optim[2],
                self.params_optim[3], discretization)
            #print('kernel_eval: ', time.time()-start)    
            #start = time.time()
            grad_u, grad_sigma = self.kernel_model.compute_grad(self.params_optim[2], 
                                                                self.params_optim[3],
                                                                discretization)
            #print('grad_kernel: ', time.time()-start)
            #start = time.time()                                                      
            v_loss[i] =l2loss_precomputation(zG, zN, ztzG,
                                              self.params_optim[0],
                                              self.params_optim[1],
                                              kernel, n_events,
                                              self.discrete_step,
                                              end_time).detach()
            #print('loss: ', time.time()-start)
            #start = time.time()
            self.params_optim[0].grad = get_grad_mu(zG,
                                                    self.params_optim[0],
                                                    self.params_optim[1],
                                                    kernel, self.discrete_step,
                                                    n_events, end_time)
            #print('grad_mu: ', time.time()-start)
            #start = time.time()
            self.params_optim[1].grad = get_grad_alpha(zG,
                                                       zN,
                                                       ztzG,
                                                       self.params_optim[0],
                                                       self.params_optim[1],
                                                       kernel,
                                                       self.discrete_step,
                                                       n_events)
            #print('grad_alpha: ', time.time()-start)
            #start = time.time()                                                      
            self.params_optim[2].grad = get_grad_theta(zG,
                                                       zN,
                                                       ztzG,
                                                       self.params_optim[0],
                                                       self.params_optim[1],
                                                       kernel,
                                                       grad_u,
                                                       self.discrete_step,
                                                       n_events)
            #print('grad_u: ', time.time()-start)
            #start = time.time()                                                           
            self.params_optim[3].grad = get_grad_theta(zG,
                                                       zN,
                                                       ztzG,
                                                       self.params_optim[0],
                                                       self.params_optim[1],
                                                       kernel,
                                                       grad_sigma,
                                                       self.discrete_step,
                                                       n_events)  
            #print('grad_sigma: ', time.time()-start)
            #start = time.time()                                                                                                              
            #############################

            grad_baseline[i] = self.params_optim[0].grad.detach()
            grad_adjacency[i] = self.params_optim[1].grad.detach()
            grad_u_v[i] = self.params_optim[2].grad.detach()
            grad_sigma_v[i] = self.params_optim[3].grad.detach()
            param_baseline[i+1] = self.params_optim[0].detach()
            param_adjacency[i+1] = self.params_optim[1].detach()
            param_u[i+1] = self.params_optim[2].detach()
            param_sigma[i+1] = self.params_optim[3].detach()
            self.opt.step()
            self.params_optim[0].clip(0)
            self.params_optim[1].clip(0)
            self.params_optim[2].clip(0)
            self.params_optim[3].clip(0)
        print('iterations in ', time.time()-start)
        return [v_loss, grad_baseline, grad_adjacency, grad_u_v, grad_sigma_v,
                param_baseline, param_adjacency, param_u, param_sigma]
