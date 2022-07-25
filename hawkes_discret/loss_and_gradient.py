import numpy as np
import torch


def l2loss_conv(intensity, events, delta, end_time):
    """Compute the value of the objective function using convolutions

    Parameters
    ----------
    intensity : tensor, shape (n_dim, n_grid)
        The values of the intensity function on the grid.
    events : tensor, shape (n_trials, n_channels, n_times)
        The current reconstructed signal.
    delta : float
            step size of the discretization grid.
    end_time : float
        The end time of grid.
    """
    return (((intensity**2).sum(1) * 0.5 * delta -
             (intensity * events).sum(1)).sum()) / end_time


def l2loss_precomputation(zG, zN, ztzG, 
                          baseline, adjacency,
                          kernel, n_events,
                          delta, end_time):
    """Compute the value of the objective function using precomputations

    Parameters
    ----------
    zG : tensor, shape (n_dim, n_discrete)      
    zN : tensor, shape (n_dim, n_discrete)
    ztzG : tensor, shape (n_dim, n_dim, n_discrete, n_discrete)
    kernel : tensor, shape (n_dim, n_dim, n_discrete)
    adjacency : tensor, shape (n_dim, n_dim)
    n_events : tensor, shape (n_dim)
        Number of events for each dimension.
    delta : float, 
    end_time : float
    """
    const = const_loss(delta, end_time)

    term_1 = term1(baseline)

    term_2 = 2 * const * term2(zG, baseline,
                               adjacency, kernel)

    term_3 = const * term3(ztzG, adjacency, kernel)

    term_4 = (2 / delta) * const * term4(zN, 
                                         baseline,
                                         adjacency,
                                         kernel,
                                         n_events)
    loss_precomp = term_1 + term_2 + term_3 - term_4

    return torch.tensor(loss_precomp, dtype=torch.float64)


def const_loss(delta, end_time):
    return (0.5 * delta) / end_time


def term1(baseline):
    """Compute the value of the first term of the 
    objective function using precomputations

    Parameters
    ----------
    baseline : tensor, shape (n_dim)
    """
    return (0.5 * (torch.linalg.norm(baseline, 2)**2)).item()


def term2(zG, baseline, adjacency, kernel):
    """Compute the value of the second term of the 
    objective function using precomputations

    Parameters
    ----------
    zG : tensor, shape (n_dim, n_discrete)  
    baseline : tensor, shape (n_dim)
    adjacency : tensor, shape (n_dim)
    kernel : tensor, shape (n_dim, n_dim, n_discrete)
    """
    n_dim, _ = zG.shape

    res = 0
    for i in range(n_dim):
        temp = 0
        for j in range(n_dim):
            temp += adjacency[i, j] * (zG[j] @ kernel[i, j])
        res += baseline[i] * temp

    return res.item()


def term3(ztzG, adjacency, kernel):
    """Compute the value of the third term of the 
    objective function using precomputations

    Parameters
    ----------
    ztzG : tensor, shape (n_dim, n_dim, n_discrete, n_discrete)  
    adjacency : tensor, shape (n_dim)
    kernel : tensor, shape (n_dim, n_dim, n_discrete)
    """
    n_dim, _, n_discrete = kernel.shape

    res = 0
    for i in range(n_dim):
        for k in range(n_dim):
            for j in range(n_dim):
                temp = adjacency[i, j] * adjacency[i, k]
                temp2 = 0
                for tau in range(n_discrete):
                    for tau_p in range(n_discrete):
                        temp2 += (kernel[i, j, tau] * kernel[i,
                                  k, tau_p]) * ztzG[j, k, tau, tau_p]
                res += temp * temp2

    return res.item()


def term4(zN, baseline, adjacency, kernel, n_events):
    """Compute the value of the 4th term of the 
    objective function using precomputations

    Parameters
    ----------
    zN : tensor, shape (n_dim, n_dim, n_discrete)  
    baseline : tensor, shape (n_dim)
    adjacency : tensor, shape (n_dim)
    kernel : tensor, shape (n_dim, n_dim, n_discrete)
    n_events : tensor, shape (n_dim)
    """
    n_dim, _, _ = kernel.shape

    res = 0
    for i in range(n_dim):
        res += baseline[i] * n_events[i]
        for j in range(n_dim):
            temp = zN[i, j] @ kernel[i, j]
            res += temp * adjacency[i, j]

    return res.item()
