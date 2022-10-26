import numpy as np
import torch
import torch.optim as optim
# from scipy.sparse import csr_array


def kernel_normalization(kernel_values, time_values, delta, lower=0, upper=1):
    kernel_norm = kernel_values.clone()
    mask_kernel = (time_values <= lower) | (time_values > upper)
    kernel_norm[:, :, mask_kernel] = 0.

    kernel_norm /= (kernel_norm.sum(2)[:, :, None] * delta)

    return kernel_norm


def kernel_normalized(kernel, kernel_params, time_values, delta, lower, upper):
    values = kernel(kernel_params, time_values, lower, upper)
    return kernel_normalization(values, time_values, delta, lower, upper)


def kernel_deriv_norm(function, grad_function_param, delta):
    function[0] = 0.
    grad_function_param[0] = 0.

    function_sum = function.sum() * delta
    grad_function_param_sum = grad_function_param.sum() * delta

    return (function_sum * grad_function_param -
            function * grad_function_param_sum) / (function_sum**2)


def grad_kernel_callable(kernel, grad_kernel, kernel_params,
                         time_values, L, lower, upper, n_dim):
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
    if len(list_params) != number_params:
        raise Exception("The number of parameters for this kernel\
                         should be equal to {}".format(number_params))
    return 0


def projected_grid(events, grid_step, size_grid):
    n_dim = len(events)
    size_discret = int(1 / grid_step)

    events_grid = torch.zeros(n_dim, size_grid)
    for i in range(n_dim):
        ei_torch = torch.tensor(events[i])
        temp = torch.round(ei_torch / grid_step) * grid_step
        temp2 = torch.round(temp * size_discret).long()
        idx, data = np.unique(temp2, return_counts=True)
        events_grid[i, idx] += torch.tensor(data)
        #events_grid_sparse = csr_array((data, idx, [0, len(data)]))

    return events_grid# , events_grid_sparse


def optimizer(param, lr, solver='GD'):
    """
    Parameters
    ----------
    param : XXX
    lr : float
        learning rate
    solver : str
        solver name, possible values are 'GD', 'RMSProp', 'Adam', 'LBFGS'
        or 'CG'
    Returns
    -------
    XXX
    """
    if solver == 'GD':
        return optim.SGD(param, lr=lr)
    elif solver == 'RMSprop':
        return optim.RMSprop(param, lr=lr)
    elif solver == 'Adam':
        return optim.Adam(param, lr=lr, betas=(0.5, 0.999))
    elif solver == 'LBFGS':
        return optim.LBFGS(param, lr=lr)
    elif solver == 'CG':
        # XXX add conjugate gradient (not available in torch)
        raise ValueError(
            "Conjugate gradient solver is not yet implemented."
        )
    else:
        raise ValueError(
            "solver must be 'GD', 'RMSProp', 'Adam', 'LBFGS' or 'CG',"
            f"got '{solver}'"
        )


def shift(x, shift):
    p = np.roll(x, shifts=shift)
    p[:shift] = 0.
    return p
