import torch


def l2loss_conv(intensity, events, delta):
    """Compute the value of the objective function using convolutions

    Parameters
    ----------
    intensity : tensor, shape (n_dim, n_grid)
        The values of the intensity function on the grid.
    events : tensor, shape (n_trials, n_channels, n_times)
    delta : float
            step size of the discretization grid.
    """
    return 2 * (((intensity**2).sum(1) * 0.5 * delta -
                 (intensity * events).sum(1)).sum()) / events.sum()


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

    term_1 = end_time * term1(baseline)

    term_2 = 2 * delta * term2(zG, baseline, adjacency, kernel)

    term_3 = delta * term3(ztzG, adjacency, kernel)

    term_4 = 2 * term4(zN, baseline, adjacency, kernel, n_events)

    loss_precomp = term_1 + term_2 + term_3 - term_4

    return loss_precomp / n_events.sum()


def term1(baseline):
    """Compute the value of the first term of the
    objective function using precomputations

    Parameters
    ----------
    baseline : tensor, shape (n_dim)
    """
    return torch.linalg.norm(baseline, ord=2)**2


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

    return res


def term3(ztzG, adjacency, kernel):
    """Compute the value of the third term of the
    objective function using precomputations

    Parameters
    ----------
    ztzG : tensor, shape (n_dim, n_dim, n_discrete, n_discrete)
    adjacency : tensor, shape (n_dim)
    kernel : tensor, shape (n_dim, n_dim, n_discrete)
    """
    n_dim, _, L = kernel.shape

    res = 0
    for i in range(n_dim):
        for k in range(n_dim):
            for j in range(n_dim):
                temp = adjacency[i, j] * adjacency[i, k]
                # temp2 = kernel[i, j].view(1, L) *
                # (ztzG[j, k] * kernel[i, k].view(L, 1)).sum(0)
                temp2 = kernel[i, k].view(1, L) * (ztzG[j, k]
                                                   * kernel[i, j].view(L, 1)).sum(0)
                # for tau in range(n_discrete):
                #    for tau_p in range(n_discrete):
                #        temp2 += (kernel[i, j, tau] * kernel[i,
                #                  k, tau_p]) * ztzG[j, k, tau, tau_p]

                res += temp * temp2.sum()

    return res


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

    return res


def get_grad_mu(zG, baseline, adjacency, kernel,
                delta, n_events, end_time):
    """ return the gradient of the parameter mu
    Parameters
    ----------
    Returns
    ----------
    grad_mu: tensor of size (dim)
    """
    n_dim, _, _ = kernel.shape

    grad_mu = torch.zeros(n_dim)
    for k in range(n_dim):
        temp = 0
        for j in range(n_dim):
            temp += adjacency[k, j] * (zG[j] @ kernel[k, j])
        grad_mu[k] = delta * temp
        grad_mu[k] += end_time * baseline[k]
        grad_mu[k] -= n_events[k]

    return 2 * (grad_mu / n_events.sum())


def get_grad_alpha(zG, zN, ztzG, baseline, adjacency,
                   kernel, delta, n_events):
    """ return the gradient of the parameter alpha
    Parameters
    ----------
    Returns
    ----------
    grad_alpha: tensor of size (dim x dim)
    """
    n_dim, _, _ = kernel.shape

    grad_alpha = torch.zeros(n_dim, n_dim)
    for k in range(n_dim):
        dk = delta * baseline[k]
        for n in range(n_dim):
            temp = 0
            for j in range(n_dim):
                temp += adjacency[k, j] * (torch.outer(kernel[k, n], kernel[k, j]) *
                                           ztzG[n, j]).sum()
            grad_alpha[k, n] += delta * temp
            grad_alpha[k, n] += dk * kernel[k, n] @ zG[n]
            grad_alpha[k, n] -= zN[k, n] @ kernel[k, n]

    return 2 * (grad_alpha / n_events.sum())


def get_grad_theta(zG, zN, ztzG, baseline,
                   adjacency, kernel,
                   grad_kernel, delta, n_events):
    """ return the gradient of the parameter theta
    Parameters
    ----------
    Returns
    ----------
    grad_theta: tensor of size (dim x dim)
    """
    n_dim, _, L = kernel.shape

    grad_theta = torch.zeros(n_dim, n_dim)
    for m in range(n_dim):
        cst = 2 * delta * baseline[m]
        for n in range(n_dim):
            grad_theta[m, n] = cst * adjacency[m, n] * (
                grad_kernel[m, n] @ zG[n])
            grad_theta[m, n] -= 2 * adjacency[m, n] * (
                grad_kernel[m, n] @ zN[m, n])
            temp = 0
            for k in range(n_dim):
                cst2 = adjacency[m, n] * adjacency[m, k]
                temp_ = 0
                temp_ += 2 * (kernel[m, k].view(1, L)
                              * (ztzG[n, k] * grad_kernel[m, n].view(L, 1)).sum(0))
                # temp_ += (grad_kernel[m, n].view(1, L)
                #          * (ztzG[k, n] * kernel[m, k].view(L, 1)).sum(0))
                # for tau in range(L):
                #   for taup in range(L):
                #       temp_ += (grad_kernel[m, n, tau]
                #                 * kernel[m, k, taup]
                #              * ztzG[n, k, tau, taup])
                #       temp_ += (grad_kernel[m, n, taup]
                #                 * kernel[m, k, tau]
                #              * ztzG[k, n, tau, taup])
                temp += cst2 * temp_.sum()

            grad_theta[m, n] += delta * temp

    return grad_theta / n_events.sum()
