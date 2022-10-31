import torch
import numpy as np
from fadin.utils.utils import check_params, kernel_normalization, \
    kernel_deriv_norm, kernel_normalized, grad_kernel_callable


class DiscreteKernelFiniteSupport(object):
    """
    Class for general discretized kernels with finite support.
    """
    def __init__(self, lower, upper, delta, kernel, n_dim, grad_kernel=None):
        """
        Parameters
        ----------

        lower: float, left bound of the support of the kernel
        upper: float, right bound of the support of the kernel
        delta: float, step size of the grid
        """

        self.L = int(1 / delta)
        self.delta = delta
        self.lower = lower
        self.upper = upper
        self.kernel = kernel
        self.n_dim = n_dim
        self.grad_kernel = grad_kernel

    def eval(self, kernel_params, time_values):
        """Return kernel evaluated on 'time_values'

        Parameters
        ----------
        Returns
        -------
        kernel_values:  tensor of size (n_dim x n_dim x L)
        """
        if self.kernel == 'RaisedCosine':
            kernel_values = RaisedCosine(kernel_params, time_values)
        elif self.kernel == 'TruncatedGaussian':
            kernel_values = TruncatedGaussian(kernel_params, time_values,
                                              self.delta, self.lower, self.upper)
        elif self.kernel == 'Exponential':
            kernel_values = Exponential(kernel_params, time_values,
                                        self.delta, self.upper)
        elif callable(self.kernel):
            kernel_values = kernel_normalized(self.kernel, kernel_params, time_values,
                                              self.delta, self.lower, self.upper)
        else:
            raise NotImplementedError("Not implemented kernel. \
                                       Kernel must be a callable or a str in \
                                       RaisedCosine | TruncatedGaussian | Exponential")
        return kernel_values

    def get_grad(self, kernel_params, time_values):
        """Return kernel's gradient evaluated on 'time_values'

        Parameters
        ----------
        Returns
        ----------
        grad_values:  tensor of size (n_dim x n_dim x L)
        """
        if self.kernel == 'RaisedCosine':
            grad_values = grad_RaisedCosine(kernel_params, time_values, self.L)
        elif self.kernel == 'TruncatedGaussian':
            grad_values = grad_TruncatedGaussian(kernel_params, time_values, self.L)
        elif self.kernel == 'Exponential':
            grad_values = grad_Exponential(kernel_params, time_values, self.L)
        elif callable(self.kernel) and callable(self.grad_kernel):
            grad_values = grad_kernel_callable(self.kernel, self.grad_kernel,
                                               kernel_params, time_values, self.L,
                                               self.lower, self.upper, self.n_dim)
        else:
            raise NotImplementedError("Not implemented kernel. \
                                       Kernel and grad_kernel must be callables or \
                                       kernel has to be  in RaisedCosine | \
                                       TruncatedGaussian | Exponential")
        return grad_values

    def intensity_eval(self, baseline, alpha, kernel_param,
                       events_grid, discretization):
        """ return the evaluation of the intensity in each point of the grid
        Parameters
        ----------
        Returns
        ----------
        intensity: tensor of size (dim x size_grid)
        """
        kernel_values = self.eval(kernel_param, discretization)
        # print("calcul kernel", torch.isnan(kernel_values).any())
        size_grid = events_grid[0].shape[0]
        kernel_values_alp = kernel_values * alpha[:, :, None]
        # print("alpha fois kernel", torch.isnan(kernel_values_alp).any())
        intensity_temp = torch.zeros(self.n_dim, self.n_dim, size_grid)
        for i in range(self.n_dim):
            intensity_temp[i, :, :] = torch.conv_transpose1d(
                events_grid[i].view(1, size_grid),
                kernel_values_alp[:, i].view(1, self.n_dim, self.L))[
                    :, :-self.L + 1]
        # print("intens calcul", torch.isnan(intensity_temp).any())
        intensity_values = intensity_temp.sum(0) + baseline.unsqueeze(1)
        # print("baseline add intens", torch.isnan(intensity_temp).any())
        return intensity_values


def RaisedCosine(kernel_params, time_values):
    # reparam: alpha= alpha' / (2*sigma)
    check_params(kernel_params, 2)
    u, sigma = kernel_params
    n_dim, _ = u.shape

    values = torch.zeros(n_dim, n_dim, len(time_values))
    for i in range(n_dim):
        for j in range(n_dim):
            values[i, j] = (1 + torch.cos(((time_values - u[i, j]) / sigma[i, j]
                                          * np.pi) - np.pi))

            mask_kernel = (time_values < u[i, j]) | (
                time_values > (u[i, j] + 2 * sigma[i, j]))
            values[i, j, mask_kernel] = 0.

    return values


def grad_RaisedCosine(kernel_params, time_values, L):
    u, sigma = kernel_params
    n_dim, _ = u.shape
    grad_u = torch.zeros(n_dim, n_dim, L)
    grad_sigma = torch.zeros(n_dim, n_dim, L)

    for i in range(n_dim):
        for j in range(n_dim):
            temp_1 = ((time_values - u[i, j]) / sigma[i, j])
            temp_2 = temp_1 * np.pi - np.pi
            grad_u[i, j] = np.pi * torch.sin(temp_2) / sigma[i, j]
            grad_sigma[i, j] = (np.pi * temp_1 / sigma[i, j]**2) * torch.sin(temp_2)
            mask_grad = (time_values < u[i, j]) | (
                time_values > (u[i, j] + 2 * sigma[i, j]))
            grad_u[i, j, mask_grad] = 0.
            grad_sigma[i, j, mask_grad] = 0.

    return [grad_u, grad_sigma]


def TruncatedGaussian(kernel_params, time_values, delta, lower=0, upper=1):
    check_params(kernel_params, 2)
    m, sigma = kernel_params
    n_dim, _ = sigma.shape

    values = torch.zeros(n_dim, n_dim, len(time_values))
    for i in range(n_dim):
        for j in range(n_dim):
            values[i, j] = torch.exp((- torch.square(time_values - m[i, j])
                                     / (2 * torch.square(sigma[i, j]))))

    kernel_norm = kernel_normalization(values, time_values, delta,
                                       lower=lower, upper=upper)

    return kernel_norm


def grad_TruncatedGaussian(kernel_params, time_values, L):
    delta = 1 / L
    m, sigma = kernel_params
    n_dim, _ = sigma.shape

    grad_m = torch.zeros(n_dim, n_dim, L)
    grad_sigma = torch.zeros(n_dim, n_dim, L)
    for i in range(n_dim):
        for j in range(n_dim):
            function = torch.exp((- torch.square(time_values - m[i, j]) /
                                 (2 * torch.square(sigma[i, j]))))

            grad_function_mu = ((time_values - m[i, j]) / (torch.square(sigma[i, j]))) \
                * function

            grad_function_s = (torch.square(time_values - m[i, j]) /
                               (torch.pow(sigma[i, j], 3))) * function

            grad_m[i, j] = kernel_deriv_norm(function, grad_function_mu, delta)
            grad_sigma[i, j] = kernel_deriv_norm(function, grad_function_s, delta)

    return [grad_m, grad_sigma]


def Exponential(kernel_params, time_values, delta, upper):
    check_params(kernel_params, 1)
    decay = kernel_params[0]

    values = decay.unsqueeze(2) * torch.exp(-decay.unsqueeze(2) * time_values)

    kernel_norm = kernel_normalization(values, time_values, delta,
                                       lower=0, upper=upper)

    return kernel_norm


def grad_Exponential(kernel_params, time_values, L):
    delta = 1 / L
    decay = kernel_params[0]
    function = decay.unsqueeze(2) * torch.exp(-decay.unsqueeze(2) * time_values)

    grad_function = (1 - decay.unsqueeze(
        2) * time_values) * torch.exp(-decay.unsqueeze(2) * time_values)

    function[:, :, 0] = 0.
    grad_function[:, :, 0] = 0.
    function_sum = function.sum(2)[:, :, None] * delta
    grad_function_sum = grad_function.sum(2)[:, :, None] * delta
    kernel_grad = (grad_function * function_sum -
                   function * grad_function_sum) / (function_sum**2)

    return [kernel_grad]


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

        # norm_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        kernel_values = torch.zeros(n_dim, n_dim, self.size_discrete)
        for i in range(n_dim):
            for j in range(n_dim):
                kernel_values[i, j] = torch.exp((- torch.square(
                                                discretization - m[i, j])
                    / (2 * torch.square(sigma[i, j]))))

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
                                                (2 * torch.square(sigma[i, j]))))

                temp_mu = ((discretization - m[i, j]) / (torch.square(sigma[i, j]))) \
                    * temp

                temp_s = (torch.square(discretization - m[i, j]) /
                          (torch.pow(sigma[i, j], 3))) * temp

                temp[0] = 0.
                temp_mu[0] = 0.
                temp_s[0] = 0.

                temp_sum = temp.sum() * self.discrete_step
                temp_mu_sum = temp_mu.sum() * self.discrete_step
                temp_s_sum = temp_s.sum() * self.discrete_step

                grad_m[i, j] = (temp_mu * temp_sum - temp * temp_mu_sum) / (temp_sum**2)
                grad_sigma[i, j] = (temp_s * temp_sum - temp * temp_s_sum) \
                    / (temp_sum**2)

        return [grad_m, grad_sigma]


"""
C = torch.zeros(n_dim, n_dim)
Cm = torch.zeros(n_dim, n_dim)
Cs = torch.zeros(n_dim, n_dim)
for i in range(n_dim):
    for j in range(n_dim):

        #Computation of Cm
        temp1_low = torch.exp(-(0.5*torch.square(self.lower - m[i, j]))
                            / torch.square(sigma[i, j]))
        temp1_upper = torch.exp(-(0.5*torch.square(self.upper - m[i, j]))
        / torch.square(sigma[i, j]))
        Cm[i, j] = temp1_low - temp1_upper
        ## Computation of C
        C[i, j] = sigma[i, j] * np.sqrt(2 * np.pi)
        C[i, j] *= norm.cdf((self.upper-m[i, j]) / sigma[i, j]) \
                    - norm.cdf((self.lower-m[i, j]) / sigma[i, j])

        #Computation of Cs
        Cs[i, j] = (self.lower - m[i, j]) / sigma[i, j] * \
                torch.exp(- torch.square(self.lower-m[i, j])
                 / (2 * torch.square(sigma[i, j])))
        Cs[i, j] -= (self.upper - m[i, j]) / sigma[i, j] * \
                torch.exp(- torch.square(self.upper-m[i, j])
                / (2 * torch.square(sigma[i, j])))
        Cs[i, j] += C[i, j] / sigma[i, j]
        grad_m[i, j] = ((discretization-m[i, j])/torch.square(sigma[i, j])
        - (Cm[i, j ]/C[i, j]) )\
                        * kernel_values[i, j]
        grad_sigma[i, j] = (torch.square(discretization-m[i, j])
        /torch.pow(sigma[i, j], 3) - (Cs[i, j ]/C[i, j]) )\
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
        ----------
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
        temp1 = decay.unsqueeze(2) * torch.exp(-decay.unsqueeze(2) * discretization)

        temp2 = (1 - decay.unsqueeze(
            2) * discretization) * torch.exp(-decay.unsqueeze(2) * discretization)

        temp1[:, :, 0] = 0.
        temp2[:, :, 0] = 0.
        temp1_sum = temp1.sum(2)[:, :, None] * self.discrete_step
        temp2_sum = temp2.sum(2)[:, :, None] * self.discrete_step
        kernel_grad = (temp2 * temp1_sum - temp1 * temp2_sum) / (temp1_sum**2)

        return [kernel_grad]

    def compute_grad_(self, kernel_param, discretization):
        """Return kernel's gradient evaluate on the discretization grid

        Parameters
        ----------
        Returns
        ----------
        kernel_grad:  tensor of size (dim x dim x L)
        """
        decay = kernel_param[0]
        kernel_grad = ((1 - decay.unsqueeze(2) * discretization)
                       * torch.exp(-decay.unsqueeze(2) * discretization))

        return [kernel_grad]

    def intensity_eval(self, baseline, adjacency,
                       kernel_param, events_grid, discretization):
        """ return the evaluation of the intensity in each point of the grid
        Parameters
        ----------
        Returns
        ----------
        intensity: tensor of size (dim x size_grid)
        """
        kernel_values = self.eval(kernel_param, discretization)
        n_dim, _, _ = kernel_values.shape

        size_grid = events_grid[0].shape[0]
        kernel_values_adj = kernel_values * adjacency[:, :, None]

        intensity_temp = torch.zeros(n_dim, n_dim, size_grid)
        for i in range(n_dim):
            intensity_temp[i, :, :] = torch.conv_transpose1d(
                events_grid[i].view(1, size_grid),
                kernel_values_adj[:, i].view(1, n_dim,
                                             self.size_discrete))[
                                                 :, :-self.size_discrete + 1]

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
                kernel[i, j] = (1 + torch.cos((discretization - u[i, j]) /
                                              sigma[i, j] * np.pi - np.pi))
                # / (2 * sigma[i, j])
                mask_kernel = (discretization < u[i, j]) | (
                    discretization > (u[i, j] + 2 * sigma[i, j]))
                kernel[i, j, mask_kernel] = 0.

        return kernel

    def compute_grad(self, kernel_param, discretization):
        """Return kernel's gradient evaluate on the discretization grid

        Parameters
        ----------
        Returns
        ----------
        kernel_grad:  tensor of size (dim x dim x L)
        """
        u, sigma = kernel_param
        n_dim, _ = u.shape
        grad_u = torch.zeros(n_dim, n_dim, self.size_discrete)
        grad_sigma = torch.zeros(n_dim, n_dim, self.size_discrete)

        for i in range(n_dim):
            for j in range(n_dim):
                temp_1 = ((discretization - u[i, j]) / sigma[i, j])
                temp_2 = temp_1 * np.pi - np.pi
                grad_u[i, j] = np.pi * torch.sin(temp_2) / sigma[i, j]  # / temp_3
                grad_sigma[i, j] = (np.pi * temp_1 / sigma[i, j]) * torch.sin(temp_2)
                mask_grad = (discretization < u[i, j]) | (
                    discretization > (u[i, j] + 2 * sigma[i, j]))
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
                                             self.size_discrete))[
                                                 :, :-self.size_discrete + 1]

        intensity = intensity_temp.sum(0) + baseline.unsqueeze(1)

        return intensity
