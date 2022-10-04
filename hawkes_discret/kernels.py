from matplotlib import cm
import torch
import numpy as np
from torch.distributions.normal import Normal
from scipy.stats import norm

class KernelTruncatedGaussianDiscret(object):
    """
    Class for truncated Gaussian distribution kernel.
    """  
    def __init__(self, lower, upper, discrete_step):
        """ 
        Parameters
        ---------- 

        lower: float, left bound of the support of the kernel
        upper: float, right bound of the support of the kernel
        discrete_step: float, step size of the grid
        """

        self.size_discrete = int(1 / discrete_step)
        self.discrete_step = discrete_step
        self.lower = lower
        self.upper = upper

    def eval(self, kernel_param, discretization):
        """Return kernel evaluate on the discretisation grid

        Parameters
        ---------- 

        Returns
        -------       
        kernel_values:  tensor of size (dim x dim x L)
        """
        m = kernel_param[0]
        sigma = kernel_param[1]
        n_dim, _ = sigma.shape

        #norm_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        kernel_values = torch.zeros(n_dim, n_dim, self.size_discrete)            
        for i in range(n_dim):
            for j in range(n_dim):
                kernel_values[i, j] = torch.exp((- torch.square(discretization - m[i, j]) /
                                                (2*torch.square(sigma[i, j]))))
                #Cij = (norm_dist.cdf((torch.tensor(self.upper)-m[i, j])/sigma[i, j]) -
                #        norm_dist.cdf((torch.tensor(self.lower)-m[i, j])/sigma[i, j])) * sigma[i, j] * np.sqrt(2*np.pi)
                #kernel_values[i, j] /= Cij

        mask_kernel = (discretization <= self.lower) | (discretization > self.upper)
        kernel_values[:, :, mask_kernel] = 0.

        kernel_values /= (kernel_values.sum(2)
                          [:, :, None] * self.discrete_step)

        return kernel_values

    def compute_grad(self, kernel_param, discretization):
        """Return kernel's gradient evaluate on the discretization grid

        Parameters
        ---------- 

        Returns
        -------
        kernel_grad:  list of tensor of size (dim x dim x L)
        """
        m = kernel_param[0]
        sigma = kernel_param[1]
        n_dim, _ = sigma.shape

        grad_m = torch.zeros(n_dim, n_dim, self.size_discrete)
        grad_sigma = torch.zeros(n_dim, n_dim, self.size_discrete)
        for i in range(n_dim):
            for j in range(n_dim):
                temp = torch.exp((- torch.square(discretization - m[i, j]) /
                                                        (2*torch.square(sigma[i, j]))))
                temp_mu = ((discretization - m[i, j]) / (torch.square(sigma[i, j])))* temp

                temp_s = (torch.square(discretization - m[i, j]) / (torch.pow(sigma[i, j], 3))) * temp

                temp[0] = 0.
                temp_mu[0] = 0.
                temp_s[0] = 0.

                temp_sum = temp.sum() * self.discrete_step
                temp_mu_sum = temp_mu.sum() * self.discrete_step
                temp_s_sum = temp_s.sum() * self.discrete_step

                grad_m[i, j] = (temp_mu*temp_sum - temp*temp_mu_sum) / (temp_sum**2)
                grad_sigma[i, j] = (temp_s*temp_sum - temp*temp_s_sum) / (temp_sum**2)

        return [grad_m, grad_sigma]
"""
C = torch.zeros(n_dim, n_dim)
Cm = torch.zeros(n_dim, n_dim)
Cs = torch.zeros(n_dim, n_dim)
for i in range(n_dim):
    for j in range(n_dim):

        #Computation of Cm
        temp1_low = torch.exp(-(0.5*torch.square(self.lower - m[i, j])) / torch.square(sigma[i, j])) 
        temp1_upper = torch.exp(-(0.5*torch.square(self.upper - m[i, j])) / torch.square(sigma[i, j]))           
        Cm[i, j] = temp1_low - temp1_upper
        

        ## Computation of C
        C[i, j] = sigma[i, j] * np.sqrt(2 * np.pi)
        C[i, j] *= norm.cdf((self.upper-m[i, j]) / sigma[i, j]) \
                    - norm.cdf((self.lower-m[i, j]) / sigma[i, j])

        #Computation of Cs
        Cs[i, j] = (self.lower - m[i, j]) / sigma[i, j] * \
                torch.exp(- torch.square(self.lower-m[i, j]) / (2 * torch.square(sigma[i, j])))
        Cs[i, j] -= (self.upper - m[i, j]) / sigma[i, j] * \
                torch.exp(- torch.square(self.upper-m[i, j]) / (2 * torch.square(sigma[i, j])))
        Cs[i, j] += C[i, j] / sigma[i, j]
    
        grad_m[i, j] = ((discretization-m[i, j])/torch.square(sigma[i, j]) - (Cm[i, j ]/C[i, j]) )\
                        * kernel_values[i, j]
        grad_sigma[i, j] = (torch.square(discretization-m[i, j])/torch.pow(sigma[i, j], 3) - (Cs[i, j ]/C[i, j]) )\
                        * kernel_values[i, j]
"""
        

class KernelExpDiscret(object):
    """
    Class for truncated exponential distribution kernel.
    """

    def __init__(self, lower, upper, discrete_step):
        """ 
        Parameters
        ---------- 

        upper: float, right bound of the support of the kernel
        discrete_step: float, step size of the grid
        """

        self.size_discrete = int(1 / discrete_step)
        self.discrete_step = discrete_step
        self.lower = lower
        self.upper = upper

    def eval(self, kernel_param, discretization):
        """Return kernel evaluate on the discretisation grid

        Parameters
        ---------- 

        Returns
        -------       
        kernel_values:  tensor of size (dim x dim x L)
        """
        decay = kernel_param[0]
        kernel_values = decay.unsqueeze(
            2) * torch.exp(-decay.unsqueeze(2) * discretization)
        mask_kernel = (discretization <= 0) | (discretization > self.upper)
        kernel_values += 0
        kernel_values[:, :, mask_kernel] = 0.
        kernel_values /= (kernel_values.sum(2)
                          [:, :, None] * self.discrete_step)

        return kernel_values

    def compute_grad(self, kernel_param, discretization):
        """Return kernel's gradient evaluate on the discretization grid

        Parameters
        ---------- 

        Returns
        -------
        kernel_grad:  tensor of size (dim x dim x L)
        """
        decay = kernel_param[0]
        temp1 = decay.unsqueeze(2) * torch.exp(-decay.unsqueeze(2)
                                               * discretization)

        temp2 = (1 - decay.unsqueeze(
            2)*discretization) * torch.exp(-decay.unsqueeze(2)
                                           * discretization)
        temp1[:, :, 0] = 0.
        temp2[:, :, 0] = 0.
        temp1_sum = temp1.sum(2)[:, :, None] * self.discrete_step
        temp2_sum = temp2.sum(2)[:, :, None] * self.discrete_step
        kernel_grad = (temp2*temp1_sum - temp1*temp2_sum) / (temp1_sum**2)

        return [kernel_grad]

    def compute_grad_(self, kernel_param, discretization):
        """Return kernel's gradient evaluate on the discretization grid

        Parameters
        ---------- 

        Returns
        -------
        kernel_grad:  tensor of size (dim x dim x L)
        """
        decay = kernel_param[0]
        kernel_grad = (1 - 
                        decay.unsqueeze(2)*discretization) * torch.exp(
                                                            -decay.unsqueeze(2)
                                                            * discretization)

        return [kernel_grad]
        
    def intensity_eval(self, baseline, adjacency,
                       kernel_param, events_grid, discretization):
        """ return the evaluation of the intensity in each point of the grid
        Parameters
        ---------- 

        Returns
        -------
        intensity: tensor of size (dim x size_grid)
        """
        decay = kernel_param[0]
        kernel_values = self.eval(kernel_param, discretization)
        n_dim, _, _ = kernel_values.shape

        size_grid = events_grid[0].shape[0]
        kernel_values_adj = kernel_values * adjacency[:, :, None]

        intensity_temp = torch.zeros(n_dim, n_dim, size_grid)
        for i in range(n_dim):
            intensity_temp[i, :, :] = torch.conv_transpose1d(
                events_grid[i].view(1, size_grid),
                kernel_values_adj[:, i].view(1, n_dim,
                                             self.size_discrete))[:, :-self.size_discrete + 1]

        intensity = intensity_temp.sum(0) + baseline.unsqueeze(1)

        return intensity  


class KernelRaisedCosineDiscret(object):
    """
    Class for raised cosine distribution kernel.
    """

    def __init__(self, discrete_step):
        """ 
        """

        self.size_discrete = int(1 / discrete_step)
        self.discrete_step = discrete_step

    def eval(self, kernel_param, discretization):
        """Return kernel evaluate on the discretisation grid: time
        kernel_values:  tensor de taille (dim x dim x len(time))"""

        u, sigma = kernel_param
        n_dim, _ = sigma.shape
        kernel = torch.zeros(n_dim, n_dim, self.size_discrete)
        for i in range(n_dim):
            for j in range(n_dim):
                kernel[i, j] = (1 + torch.cos((discretization - u[i, j]) / sigma[i, j] * np.pi - np.pi)) #\
                    #/ (2 * sigma[i, j])
                mask_kernel = (discretization < u[i, j]) | (
                    discretization > (u[i, j] + 2*sigma[i, j]))
                kernel[i, j, mask_kernel] = 0.

        return kernel


    def compute_grad(self, kernel_param, discretization):
        """Return kernel's gradient evaluate on the discretization grid

        Parameters
        ---------- 

        Returns
        -------
        kernel_grad:  tensor of size (dim x dim x L)
        """
        u, sigma = kernel_param
        n_dim, _ = u.shape   
        grad_u = torch.zeros(n_dim, n_dim, self.size_discrete)
        grad_sigma = torch.zeros(n_dim, n_dim, self.size_discrete)

        for i in range(n_dim):
            for j in range(n_dim):    
                
                temp_1 = ((discretization - u[i, j]) / sigma[i, j])
                temp_2 =  temp_1 * np.pi - np.pi
                temp_3 = 2 * (sigma[i, j]**2)
                #reparam
                grad_u[i, j] = np.pi*torch.sin(temp_2) / sigma[i, j]#/ temp_3
                grad_sigma[i, j] = ( np.pi * temp_1  # temp_3 à la place de sigma 
                / sigma[i, j] ) * torch.sin(temp_2) #- (1 + 
                 #torch.cos(temp_2)) / temp_3 
                
                mask_grad = (discretization < u[i, j]) | (
                        discretization > (u[i, j] + 2*sigma[i, j]))     
                grad_u[i, j, mask_grad] = 0.
                grad_sigma[i, j, mask_grad] = 0.        

        return [grad_u, grad_sigma]

    def intensity_eval(self, baseline, adjacency,
                       kernel_param, events_grid, discretization):
        """ return the evaluation of the intensity in each point of the grid
            vector of size: dim x size_grid
        """
        u, sigma = kernel_param
        kernel_values = self.eval(kernel_param, discretization)
        n_dim, _, _ = kernel_values.shape

        size_grid = events_grid[0].shape[0]
        kernel_values_adj = kernel_values * adjacency[:, :, None]

        intensity_temp = torch.zeros(n_dim, n_dim, size_grid)
        for i in range(n_dim):
            intensity_temp[i, :, :] = torch.conv_transpose1d(
                events_grid[i].view(1, size_grid),
                kernel_values_adj[:, i].view(1, n_dim,
                                             self.size_discrete))[:, :-self.size_discrete + 1]

        intensity = intensity_temp.sum(0) + baseline.unsqueeze(1)

        return intensity
