import numpy as np
import torch
import torch.optim as optim
import hawkes_discret.kernels as kernels
from scipy.sparse import csr_array


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

        events_grid_sparse = csr_array((data, idx, [0, len(data)]))

        # for j in range(ei_torch.shape[0]):
        #    timestamps_loc[i, temp2[j]] += 1.

    return events_grid, events_grid_sparse


def optimizer(param, lr, solver="GD"):
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
    if solver == "GD":
        return optim.SGD(param, lr=lr)
    elif solver == "RMSprop":
        return optim.RMSprop(param, lr=lr)
    elif solver == "Adam":
        return optim.Adam(param, lr=lr, betas=(0.5, 0.999))
    elif solver == "LBFGS":
        return optim.LBFGS(param, lr=lr)
    elif solver == "CG":
        # XXX add conjugate gradient (not available in torch)
        raise ValueError(f"Conjugate gradient solver is not yet implemented.")
    else:
        raise ValueError(
            f"solver must be 'GD', 'RMSProp', 'Adam', 'LBFGS' or 'CG',"
            " got '{solver}'"
        )


def init_kernel(lower, upper, discret_step, kernel_name="KernelExpDiscret"):

    if kernel_name == "KernelExpDiscret":
        kernel_model = kernels.KernelExpDiscret(lower, upper, discret_step)
    elif kernel_name == "RaisedCosine":
        kernel_model = kernels.KernelRaisedCosineDiscret(discret_step)

    elif kernel_name == "TruncatedGaussian":
        kernel_model = kernels.KernelTruncatedGaussianDiscret(
            lower, upper, discret_step
        )

    else:
        raise NotImplementedError("This kernel is not implemented")

    return kernel_model


def shift(x, shift):
    p = np.roll(x, shifts=shift)
    p[:shift] = 0.0
    return p
