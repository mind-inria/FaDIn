import torch

from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.utils.utils import check_random_state


def tg_def(kernel_params, time_values, lower=0, upper=1):
    m, sigma = kernel_params
    n_dim, _ = sigma.shape

    values = torch.zeros(n_dim, n_dim, len(time_values))
    for i in range(n_dim):
        for j in range(n_dim):
            values[i, j] = torch.exp((- torch.square(time_values - m[i, j])
                                     / (2 * torch.square(sigma[i, j]))))
    return values


def tg_grad(kernel_params, time_values, L, lower=0, upper=1):
    m, sigma = kernel_params
    n_dim, _ = sigma.shape

    grad_function_mu = torch.zeros(n_dim, n_dim, L)
    grad_function_s = torch.zeros(n_dim, n_dim, L)
    for i in range(n_dim):
        for j in range(n_dim):
            function = torch.exp((- torch.square(time_values - m[i, j]) /
                                 (2 * torch.square(sigma[i, j]))))

            grad_function_mu[i, j] = ((time_values - m[i, j]) / (
                torch.square(sigma[i, j]))) * function

            grad_function_s[i, j] = (torch.square(time_values - m[i, j]) /
                                     (torch.pow(sigma[i, j], 3))) * function

    return [grad_function_mu, grad_function_s]


def test_eval_grad():
    """Check if the grad of Truncated Gaussian
    implemented in closed form is equal to the
    generalized implementation with pre defined functions
    """
    random_state = 0
    rng = check_random_state(random_state)

    m = torch.tensor(rng.randn(2, 2))
    sigma = torch.tensor(rng.randn(2, 2))
    kernel_params = [m, sigma]

    kernel = "truncated_gaussian"
    L = 100
    n_dim = 2
    delta = 1 / L
    t = torch.linspace(0, 1, L)
    lower = 0.
    upper = 1
    TG = DiscreteKernelFiniteSupport(delta, n_dim, kernel, lower=lower, upper=upper)
    closedform = TG.kernel_eval(kernel_params, t)

    TG = DiscreteKernelFiniteSupport(delta, n_dim, tg_def, lower=lower, upper=upper)
    callabled = TG.kernel_eval(kernel_params, t)

    for i in range(len(kernel_params)):
        assert torch.allclose(closedform[i], callabled[i])

    TG = DiscreteKernelFiniteSupport(delta, n_dim, kernel, lower=lower, upper=upper)
    closedform_grad = TG.grad_eval(kernel_params, t)

    TG = DiscreteKernelFiniteSupport(delta, n_dim, tg_def, lower=lower,
                                     upper=upper, grad_kernel=tg_grad)
    callabled_grad = TG.grad_eval(kernel_params, t)

    for i in range(len(kernel_params)):
        assert torch.allclose(closedform_grad[i], callabled_grad[i])
