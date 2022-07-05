#import numpy as np
import torch
import torch.optim as optim
import hawkes_discret.kernels as kernels

def projected_grid(events, grid_step, size_grid):
    dim = len(events)
    size_discret = 1 / grid_step
    projected_timestamps = []

    timestamps_loc = torch.zeros(dim, size_grid)

    for i in range(dim):
        temp = torch.round(events[i]/grid_step) * grid_step      
        timestamps_loc[i, torch.round(temp * size_discret).long()] = 1.     
        projected_timestamps.append(temp)

    return projected_timestamps, timestamps_loc

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
            f"Conjugate gradient solver is not yet implemented."
        )
    else:
        raise ValueError(
            f"solver must be 'GD', 'RMSProp', 'Adam', 'LBFGS' or 'CG',"
            " got '{solver}'"
        )
    
def init_kernel(kernel_params, upper, discret_step,  kernel_name='KernelExpDiscret'):

    if kernel_name == 'KernelExpDiscret':
        kernel_model = kernels.KernelExpDiscret(kernel_params, 
                                                upper, 
                                                discret_step)
    return kernel_model