import torch
import numpy as np


class KernelExpDiscret(object):
    """
    Class for truncated exponential distribution kernel.
    """

    def __init__(self, upper, discrete_step):
        """ 
        Parameters
        ---------- 

        upper: float, right bound of the support of the kernel
        discrete_step: float, step size of the grid
        """

        self.size_discrete = int(1 / discrete_step)
        self.discrete_step = discrete_step
        self.upper = upper

    def eval(self, decay, discretization):
        """Return kernel evaluate on the discretisation grid

        Parameters
        ---------- 

        Returns
        -------       
        kernel_values:  tensor of size (dim x dim x L)
        """

        kernel_values = decay.unsqueeze(
            2) * torch.exp(-decay.unsqueeze(2) * discretization)
        mask_kernel = (discretization <= 0) | (discretization > self.upper)
        kernel_values += 0
        kernel_values[:, :, mask_kernel] = 0.
        kernel_values /= (kernel_values.sum(2)
                          [:, :, None] * self.discrete_step)

        return kernel_values

    def compute_grad(self, decay, discretization):
        """Return kernel's gradient evaluate on the discretization grid

        Parameters
        ---------- 

        Returns
        -------
        kernel_grad:  tensor of size (dim x dim x L)
        """
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

        return kernel_grad

    def intensity_eval(self, baseline, adjacency,
                       decay, events_grid, discretization):
        """ return the evaluation of the intensity in each point of the grid
        Parameters
        ---------- 

        Returns
        -------
        intensity: tensor of size (dim x size_grid)
        """
        kernel_values = self.eval(decay, discretization)
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

        return intensity  # .clip(0) #à voir si le clip est nécessaire


class KernelRaisedCosineDiscret(object):
    """
    Class for raised cosine distribution kernel.
    """

    def __init__(self, discrete_step):
        """ 
        """

        self.size_discrete = int(1 / discrete_step)
        self.discrete_step = discrete_step

    def eval(self, u, sigma, discretization):
        """Return kernel evaluate on the discretisation grid: time
        kernel_values:  tensor de taille (dim x dim x len(time))"""
        n_dim, _ = sigma.shape

        kernel = torch.zeros(n_dim, n_dim, self.size_discrete)
        for i in range(n_dim):
            for j in range(n_dim):
                kernel[i, j] = (1 + torch.cos((discretization - u[i, j]) / sigma[i, j] * np.pi - np.pi)) \
                    / (2 * sigma[i])
                mask_kernel = (discretization < u[i, j]) | (
                    discretization > (u[i, j] + 2*sigma[i, j]))
                kernel[i, j, mask_kernel] = 0.

        return kernel

    def intensity_eval(self, baseline, adjacency,
                       decay, events_grid, discretization):
        """ return the evaluation of the intensity in each point of the grid
            vector of size: dim x size_grid
        """
        kernel_values = self.eval(decay, discretization)
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
