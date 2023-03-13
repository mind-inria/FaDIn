import numpy as np
import torch


def kernel_normalization(kernel_values, time_values, delta, lower=0, upper=1):
    """Normalize the given kernel on the given discrete grid.
    """
    kernel_norm = kernel_values.clone()
    mask_kernel = (time_values <= lower) | (time_values > upper)
    kernel_norm[:, :, mask_kernel] = 0.

    kernel_norm /= (kernel_norm.sum(2)[:, :, None] * delta)

    return kernel_norm


def kernel_normalized(kernel, kernel_params, time_values, delta, lower, upper):
    """Normalize the given kernel on the given discrete grid.
    """
    values = kernel(kernel_params, time_values, lower, upper)
    return kernel_normalization(values, time_values, delta, lower, upper)


def kernel_deriv_norm(function, grad_function_param, delta):
    """Normalize the given gradient kernels on the given discrete grid.
    """
    function[0] = 0.
    grad_function_param[0] = 0.

    function_sum = function.sum() * delta
    grad_function_param_sum = grad_function_param.sum() * delta

    return (function_sum * grad_function_param -
            function * grad_function_param_sum) / (function_sum**2)


def grad_kernel_callable(kernel, grad_kernel, kernel_params,
                         time_values, L, lower, upper, n_dim):
    """Transform the callables ``kernel and ``grad_kernel`` into
       gradient list of parameters.
    """
    delta = 1 / L
    n_param = len(kernel_params)
    function = kernel(kernel_params, time_values, lower, upper)
    grad_function = grad_kernel(kernel_params, time_values, L, lower, upper)

    grad = []
    for i in range(n_param):
        grad_temp = torch.zeros(n_dim, n_dim, L)
        for k in range(n_dim):
            for m in range(n_dim):
                grad_temp[k, m] = kernel_deriv_norm(function[k, m],
                                                    grad_function[i][k, m],
                                                    delta)
        grad.append(grad_temp)

    return grad


def check_params(list_params, number_params):
    """Check if the list of parameters is equal to the number of parameters.
    """
    if len(list_params) != number_params:
        raise Exception("The number of parameters for this kernel\
                         should be equal to {}".format(number_params))
    return 0


def projected_grid(events, grid_step, n_grid):
    """Project the events on the defined grid.
    """
    n_dim = len(events)
    size_discret = int(1 / grid_step)
    events_grid = torch.zeros(n_dim, n_grid)
    for i in range(n_dim):
        ei_torch = torch.tensor(events[i])
        temp = torch.round(ei_torch / grid_step) * grid_step
        temp2 = torch.round(temp * size_discret).long()
        idx, data = np.unique(temp2, return_counts=True)
        events_grid[i, idx] += torch.tensor(data)

    return events_grid


def optimizer(param, params_optim, solver='RMSprop'):
    """Set the Pytorch optimizer.

    Parameters
    ----------
    param : XXX
    lr : float
        learning rate
    solver : str
        solver name, possible values are 'GD', 'RMSProp', 'Adam'
        or 'CG'
    Returns
    -------
    XXX
    """
    if solver == 'GD':
        return torch.optim.SGD(param, **params_optim)
    elif solver == 'RMSprop':
        return torch.optim.RMSprop(param, **params_optim)
    elif solver == 'Adam':
        return torch.optim.Adam(param, **params_optim)
    else:
        raise NotImplementedError(
            "solver must be 'GD', 'RMSProp', 'Adam'," f"got '{solver}'")


def l2_error(x, a):
    return torch.sqrt(((x - a)**2).sum()).item()


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
