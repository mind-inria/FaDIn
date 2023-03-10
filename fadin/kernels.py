import torch
import numpy as np

from fadin.utils.utils import check_params, kernel_normalization, \
    kernel_deriv_norm, kernel_normalized, grad_kernel_callable


class DiscreteKernelFiniteSupport(object):
    """
    A class for general discretized kernels with finite support.

    Parameters
    ----------
    delta : float
        Step size of the discretization.

    n_dim : int
        Dimension of the Hawkes process associated to this kernel class.

    kernel : str or callable
        Either define a kernel in ('raised_cosine', 'truncated_gaussian' and
        'truncated_exponential') or a custom kernel.

    kernel_length: int, default=1
        Length of kernel.

    lower : float, default=0
        Left bound of the support of the kernel. It should be between [0, 1].

    upper : float, default=1
        Right bound of the support of the kernel. It should be between [0, 1].

    grad_kernel : None or callable, default=None
        If kernel in ('raised_cosine', 'truncated_gaussian' and
        'truncated_exponential') the gradient function is implemented.
        If kernel is custom, the custom gradient must be given.

    Attributes
    ----------
    L: int
        Size of the kernel discretization.
    """
    def __init__(self, delta, n_dim, kernel, kernel_length=1,
                 lower=0., upper=1., grad_kernel=None):
        self.L = int(kernel_length / delta)
        self.delta = delta
        self.lower = lower
        self.upper = upper
        self.n_dim = n_dim
        self.kernel = kernel
        self.grad_kernel = grad_kernel

    def kernel_eval(self, kernel_params, time_values):
        """Return kernel evaluated on the given discretization.

        Parameters
        ----------
        kernel_params : list of tensor of shape (n_dim, n_dim)
            Parameters of the kernel.

        time_values : tensor, shape (L,)
            Given discretization.

        Returns
        -------
        kernel_values :  tensor, shape (n_dim, n_dim, L)
            Kernels evaluated on ``time_values``.
        """
        if self.kernel == 'raised_cosine':
            kernel_values = raised_cosine(kernel_params, time_values)
        elif self.kernel == 'truncated_gaussian':
            kernel_values = truncated_gaussian(kernel_params, time_values,
                                               self.delta, self.lower, self.upper)
        elif self.kernel == 'truncated_exponential':
            kernel_values = truncated_exponential(kernel_params, time_values,
                                                  self.delta, self.upper)
        elif self.kernel == 'kumaraswamy':
            kernel_values = kumaraswamy(kernel_params, time_values)
        elif callable(self.kernel):
            kernel_values = kernel_normalized(self.kernel, kernel_params, time_values,
                                              self.delta, self.lower, self.upper)
        else:
            raise NotImplementedError("Not implemented kernel. \
                                       Kernel must be a callable or a str in \
                                       raised_cosine | truncated_gaussian |  \
                                       truncated_exponential")
        return kernel_values

    def grad_eval(self, kernel_params, time_values):
        """Return kernel's gradient evaluated on the given discretization.

        Parameters
        ----------
        kernel_params : list of tensor of shape (n_dim, n_dim)
            Parameters of the kernel.

        time_values : tensor, shape (L,)
            Given discretization.

        Returns
        ----------
        grad_values :  tensor, shape (n_dim, n_dim, L)
            Gradients evaluated on ``time_values``.
        """
        if self.kernel == 'raised_cosine':
            grad_values = grad_raised_cosine(kernel_params, time_values, self.L)
        elif self.kernel == 'truncated_gaussian':
            grad_values = grad_truncated_gaussian(kernel_params, time_values, self.L)
        elif self.kernel == 'truncated_exponential':
            grad_values = grad_truncated_exponential(kernel_params, time_values, self.L)
        elif self.kernel == 'kumaraswamy':
            grad_values = grad_kumaraswamy(kernel_params, time_values, self.L)
        elif callable(self.kernel) and callable(self.grad_kernel):
            grad_values = grad_kernel_callable(self.kernel, self.grad_kernel,
                                               kernel_params, time_values, self.L,
                                               self.lower, self.upper, self.n_dim)
        else:
            raise NotImplementedError("Not implemented kernel. \
                                       Kernel and grad_kernel must be callables or \
                                       kernel has to be  in raised_cosine | \
                                       truncated_gaussian | truncated_exponential")
        return grad_values

    def intensity_eval(self, baseline, alpha, kernel_params,
                       events_grid, time_values):
        """Return the intensity function evaluated on the entire grid.

        Parameters
        ----------
        baseline : tensor, shape (n_dim,)
            Baseline parameter of the intensity of the Hawkes process.

        alpha : tensor, shape (n_dim, n_dim)
            Alpha parameter of the intensity of the Hawkes process.

        kernel_params : list of tensor of shape (n_dim, n_dim)
            Parameters of the kernel.

        events_grid : tensor, shape (n_dim, n_grid)
            Events projected on the pre-defined grid.

        time_values : tensor, shape (L,)
            Given discretization.

        Returns
        ----------
        intensity_values : tensor, shape (dim, n_grid)
            The intensity function evaluated on the grid.
        """
        kernel_values = self.kernel_eval(kernel_params, time_values)
        n_grid = events_grid[0].shape[0]
        kernel_values_alp = kernel_values * alpha[:, :, None]
        intensity_temp = torch.zeros(self.n_dim, self.n_dim, n_grid)
        for i in range(self.n_dim):
            intensity_temp[i, :, :] = torch.conv_transpose1d(
                events_grid[i].view(1, n_grid),
                kernel_values_alp[:, i].view(1, self.n_dim, self.L).float())[
                    :, :-self.L + 1]
        intensity_values = intensity_temp.sum(0) + baseline.unsqueeze(1)

        return intensity_values


def kumaraswamy(kernel_params, time_values):
    """Kumaraswamy kernel.

    Parameters
    ----------
    kernel_params : list of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: u and sigma.

    time_values : tensor, shape (L,)
        Given discretization.

    Returns
    ----------
    values : tensor, shape (n_dim, n_dim, L)
        Kernels evaluated on ``time_values``.
    """
    check_params(kernel_params, 2)
    a, b = kernel_params
    n_dim, _ = a.shape

    values = torch.zeros(n_dim, n_dim, len(time_values))
    for i in range(n_dim):
        for j in range(n_dim):
            pa = a[i, j] - 1
            pb = b[i, j] - 1
            values[i, j] = (a[i, j] * b[i, j] * (time_values**pa)
                            * ((1 - time_values**a[i, j]) ** pb))
            mask_kernel = (time_values <= 0.) | (time_values >= 1.)
            values[i, j, mask_kernel] = 0.

    return values


def grad_kumaraswamy(kernel_params, time_values, L):
    """Gradients of the Kumaraswamy kernel.

    Parameters
    ----------
    kernel_params : list of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: u and sigma.

    time_values : tensor, shape (L,)
        Given discretization.

    L : int
        Size of the kernel discretization.

    Returns
    ----------
    grad_list : list of two tensor of shape (n_dim, n_dim, L)
        Kernels evaluated on ``time_values``.
    """
    a, b = kernel_params
    n_dim, _ = a.shape
    kernel_values = kumaraswamy(kernel_params, time_values)
    b_minusone = b - 1
    kernel_params_ = [a, b_minusone]
    kernel_values_ = kumaraswamy(kernel_params_, time_values)

    grad_a = torch.zeros(n_dim, n_dim, L)
    grad_b = torch.zeros(n_dim, n_dim, L)
    for i in range(n_dim):
        for j in range(n_dim):
            grad_a[i, j] = kernel_values[i, j] * (1 / a[i, j] + torch.log(time_values))\
                - kernel_values_[i, j] * torch.log(time_values) * time_values**a[i, j]
            grad_b[i, j] = kernel_values[i, j] * (1 / b[i, j]
                                                  + torch.log(1 - time_values**a[i, j]))
            mask_kernel = (time_values <= 0.) | (time_values >= 1.)
            grad_a[i, j, mask_kernel] = 0.
            grad_b[i, j, mask_kernel] = 0.

    grad_list = [grad_a, grad_b]

    return grad_list


def raised_cosine(kernel_params, time_values):
    """Raised Cosine kernel.

    Parameters
    ----------
    kernel_params : list of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: u and sigma.

    time_values : tensor, shape (L,)
        Given discretization.

    Returns
    ----------
    values : tensor, shape (n_dim, n_dim, L)
        Kernels evaluated on ``time_values``.
    """
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


def grad_raised_cosine(kernel_params, time_values, L):
    """Gradients of the Raised Cosine kernel.

    Parameters
    ----------
    kernel_params : list of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: u and sigma.

    time_values : tensor, shape (L,)
        Given discretization.

    L : int
        Size of the kernel discretization.

    Returns
    ----------
    grad_list : list of two tensor of shape (n_dim, n_dim, L)
        Kernels evaluated on ``time_values``.
    """
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

    grad_list = [grad_u, grad_sigma]

    return grad_list


def truncated_gaussian(kernel_params, time_values, delta, lower=0., upper=1.):
    """Truncated Gaussian kernel normalized on [0, 1].

    Parameters
    ----------
    kernel_params : list of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: m and sigma.

    time_values : tensor, shape (L,)
        Given discretization.

    delta : float
        Step size of the discretization.

    lower : float, default=0
        Left bound of the support of the kernel. It should be between [0, 1].

    upper : float, default=1
        Right bound of the support of the kernel. It should be between [0, 1].
    Returns
    ----------
    values : tensor, shape (n_dim, n_dim, L)
        Kernels evaluated on ``time_values``.
    """
    check_params(kernel_params, 2)
    m, sigma = kernel_params
    n_dim, _ = sigma.shape

    values_ = torch.zeros(n_dim, n_dim, len(time_values))
    for i in range(n_dim):
        for j in range(n_dim):
            values_[i, j] = torch.exp((- torch.square(time_values - m[i, j])
                                       / (2 * torch.square(sigma[i, j]))))

    values = kernel_normalization(values_, time_values, delta,
                                  lower=lower, upper=upper)

    return values


def grad_truncated_gaussian(kernel_params, time_values, L):
    """Gradients of the Truncated Gaussian kernel.

    Parameters
    ----------
    kernel_params : list of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: m and sigma.

    time_values : tensor, shape (L,)
        Given discretization.

    L : int
        Size of the kernel discretization.

    Returns
    ----------
    grad_list : list of two tensor of shape (n_dim, n_dim, L)
        Kernels evaluated on ``time_values``.
    """
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

    grad_list = [grad_m, grad_sigma]

    return grad_list


def truncated_exponential(kernel_params, time_values, delta, upper=1.):
    """truncated_exponential kernel normalized on [0, 1].

    Parameters
    ----------
    kernel_params : list of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernel: decay.

    time_values : tensor, shape (L,)
        Given discretization.

    delta : float
        Step size of the discretization.

    upper : float, default=1
        Right bound of the support of the kernel. It should be between [0, 1].
    Returns
    ----------
    values : tensor, shape (n_dim, n_dim, L)
        Kernels evaluated on ``time_values``.
    """
    check_params(kernel_params, 1)
    decay = kernel_params[0]

    values_ = decay.unsqueeze(2) * torch.exp(-decay.unsqueeze(2) * time_values)

    values = kernel_normalization(values_, time_values, delta,
                                  lower=0., upper=upper)

    return values


def grad_truncated_exponential(kernel_params, time_values, L):
    """Gradients of the truncated_exponential kernel.

    Parameters
    ----------
    kernel_params : list of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: decay.

    time_values : tensor, shape (L,)
        Given discretization.

    L : int
        Size of the kernel discretization.

    Returns
    ----------
    grad_list : list of two tensor of shape (n_dim, n_dim, L)
        Kernels evaluated on ``time_values``.
    """
    delta = 1 / L
    decay = kernel_params[0]
    function = decay.unsqueeze(2) * torch.exp(-decay.unsqueeze(2) * time_values)

    grad_function = (1 - decay.unsqueeze(
        2) * time_values) * torch.exp(-decay.unsqueeze(2) * time_values)

    function[:, :, 0] = 0.
    grad_function[:, :, 0] = 0.
    function_sum = function.sum(2)[:, :, None] * delta
    grad_function_sum = grad_function.sum(2)[:, :, None] * delta
    grad_decay = (grad_function * function_sum -
                  function * grad_function_sum) / (function_sum**2)

    grad_list = [grad_decay]

    return grad_list


# def truncated_skewed_gaussian(kernel_params, time_values, delta,
#                               lower=0., upper=3., sigma=0.1):
#     check_params(kernel_params, 2)
#     beta, xi = kernel_params
#     n_dim, _ = xi.shape

#     values_ = torch.zeros(n_dim, n_dim, len(time_values))
#     for i in range(n_dim):
#         for j in range(n_dim):
#             z = (time_values - xi[i, j].item()) / sigma
#             values_[i, j] = torch.tensor(2 * norm.pdf(z) *
#                                          norm.cdf(beta[i, j].item() * z)) / sigma

#     values = kernel_normalization(values_, time_values, delta,
#                                   lower=lower, upper=upper)
#     return values


# def grad_truncated_skewed_gaussian(kernel_params, time_values, L, sigma=0.1):
#     delta = 1 / L
#     beta, xi = kernel_params
#     n_dim, _ = beta.shape

#     beta = beta.detach().numpy()
#     xi = xi.detach().numpy()
#     grad_beta = torch.zeros(n_dim, n_dim, L)
#     grad_xi = torch.zeros(n_dim, n_dim, L)
#     for i in range(n_dim):
#         for j in range(n_dim):
#             z = (time_values.numpy() - xi[i, j]) / sigma
#             deriv_gauss = - 2 * z * np.exp(- (z**2) / 2) / np.sqrt(2 * np.pi)
#             f = torch.tensor(2 * norm.pdf(z) * norm.cdf(beta[i, j] * z)) / sigma
#             grad_f_beta = 2 * z * norm.pdf(z) * norm.pdf(beta[i, j] * z) / sigma
#             grad_f_xi = - 2 * (beta[i, j] * norm.pdf(beta[i, j] * z) * norm.pdf(z)
#                                + norm.cdf(beta[i, j] * z) * deriv_gauss) / (sigma**2)
#             f = torch.tensor(f)
#             grad_f_beta = torch.tensor(grad_f_beta)
#             grad_f_xi = torch.tensor(grad_f_xi)

#             grad_beta[i, j] = kernel_deriv_norm(f, grad_f_beta, delta)
#             grad_xi[i, j] = kernel_deriv_norm(f, grad_f_xi, delta)

#     grad_list = [torch.tensor(grad_beta), torch.tensor(grad_xi)]

#     return grad_list
