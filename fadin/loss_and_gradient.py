import torch


def discrete_l2loss_conv(intensity, events_grid, delta):
    """Compute the l2 discrete loss using convolutions.

    Parameters
    ----------
    intensity : tensor, shape (n_dim, n_grid)
        Values of the intensity function evaluated  on the grid.

    events_grid : tensor, shape (n_dim, n_grid)
        Events projected on the pre-defined grid.

    delta : float
        Step size of the discretization grid.
    """
    return 2 * (((intensity**2).sum(1) * 0.5 * delta -
                 (intensity * events_grid).sum(1)).sum()) / events_grid.sum()


def discrete_l2loss_precomputation(zG, zN, ztzG, baseline, alpha, kernel,
                                   n_events, delta, end_time):
    """Compute the l2 discrete loss using precomputation terms.

    Parameters
    ----------
    zG : tensor, shape (n_dim, L)

    zN : tensor, shape (n_dim, L)

    ztzG : tensor, shape (n_dim, n_dim, L, L)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L)
        Kernel values on the discretization.

    n_events : tensor, shape (n_dim,)
        Number of events for each dimension.

    delta : float
        Step size of the discretization grid.

    end_time : float
        The end time of the Hawkes process.
    """

    term_1 = end_time * term1(baseline)

    term_2 = 2 * delta * term2(zG, baseline, alpha, kernel)

    term_3 = delta * term3(ztzG, alpha, kernel)

    term_4 = 2 * term4(zN, baseline, alpha, kernel, n_events)

    loss_precomp = term_1 + term_2 + term_3 - term_4

    return loss_precomp / n_events.sum()


def term1(baseline):
    """Compute the value of the first term of the
    discrete l2 loss using precomputations

    Parameters
    ----------
    baseline : tensor, shape (n_dim,)
    """
    return torch.linalg.norm(baseline, ord=2)**2


def term2(zG, baseline, alpha, kernel):
    """Compute the value of the second term of the
    discrete l2 loss using precomputations

    Parameters
    ----------
    zG : tensor, shape (n_dim, L)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L)
        Kernel values on the discretization.
    """
    n_dim, _ = zG.shape

    res = 0
    for i in range(n_dim):
        temp = 0
        for j in range(n_dim):
            temp += alpha[i, j] * (zG[j] @ kernel[i, j])
        res += baseline[i] * temp

    return res


def term3(ztzG, alpha, kernel):
    """Compute the value of the third term of the
    discrete l2 loss using precomputations

    Parameters
    ----------
    ztzG : tensor, shape (n_dim, n_dim, L, L)

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L)
        Kernel values on the discretization.
    """
    n_dim, _, L = kernel.shape

    res = 0
    for i in range(n_dim):
        for k in range(n_dim):
            for j in range(n_dim):
                temp = alpha[i, j] * alpha[i, k]
                # temp2 = kernel[i, j].view(1, L) *
                # (ztzG[j, k] * kernel[i, k].view(L, 1)).sum(0)
                temp2 = kernel[i, k].view(1, L) * (ztzG[j, k]
                                                   * kernel[i, j].view(L, 1)).sum(0)
                # for tau in range(L):
                #    for tau_p in range(L):
                #        temp2 += (kernel[i, j, tau] * kernel[i,
                #                  k, tau_p]) * ztzG[j, k, tau, tau_p]

                res += temp * temp2.sum()

    return res


def term4(zN, baseline, alpha, kernel, n_events):
    """Compute the value of the 4th term of the
    discrete l2 loss using precomputations

    Parameters
    ----------
    zN : tensor, shape (n_dim, n_dim, L)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L)
        Kernel values on the discretization.

    n_events : tensor, shape (n_dim,)
        Number of events for each dimension.
    """
    n_dim, _, _ = kernel.shape

    res = 0
    for i in range(n_dim):
        res += baseline[i] * n_events[i]
        for j in range(n_dim):
            temp = zN[i, j] @ kernel[i, j]
            res += temp * alpha[i, j]

    return res


def get_grad_baseline(zG, baseline, alpha, kernel,
                      delta, n_events, end_time):
    """Return the gradient of the discrete l2 loss w.r.t. the baseline.

    Parameters
    ----------
    zG : tensor, shape (n_dim, L)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L)
        Kernel values on the discretization.

    delta : float
        Step size of the discretization grid.

    n_events : tensor, shape (n_dim,)
        Number of events for each dimension.

    end_time : float
        The end time of the Hawkes process.

    Returns
    ----------
    grad_baseline: tensor, shape (dim,)
    """
    n_dim, _, _ = kernel.shape

    grad_baseline = torch.zeros(n_dim)
    for k in range(n_dim):
        temp = 0
        for j in range(n_dim):
            temp += alpha[k, j] * (zG[j] @ kernel[k, j])
        grad_baseline[k] = delta * temp
        grad_baseline[k] += end_time * baseline[k]
        grad_baseline[k] -= n_events[k]

    return 2 * grad_baseline / n_events.sum()


def get_grad_alpha(zG, zN, ztzG, baseline, alpha, kernel, delta, n_events):
    """Return the gradient of the discrete l2 loss w.r.t. alpha.

    Parameters
    ----------

    zG : tensor, shape (n_dim, L)

    zN : tensor, shape (n_dim, L)

    ztzG : tensor, shape (n_dim, n_dim, L, L)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L)
        Kernel values on the discretization.

    delta : float
        Step size of the discretization grid.

    n_events : tensor, shape (n_dim,)
        Number of events for each dimension.

    Returns
    ----------
    grad_alpha : tensor, shape (n_dim, n_dim)
    """
    n_dim, _, _ = kernel.shape

    grad_alpha_ = torch.zeros(n_dim, n_dim)
    for k in range(n_dim):
        dk = delta * baseline[k]
        for n in range(n_dim):
            temp = 0
            for j in range(n_dim):
                temp += alpha[k, j] * (torch.outer(kernel[k, n], kernel[k, j]) *
                                       ztzG[n, j]).sum()
            grad_alpha_[k, n] += delta * temp
            grad_alpha_[k, n] += dk * kernel[k, n] @ zG[n]
            grad_alpha_[k, n] -= zN[k, n] @ kernel[k, n]

    grad_alpha = 2 * grad_alpha_ / n_events.sum()

    return grad_alpha


def get_grad_theta(zG, zN, ztzG, baseline, alpha, kernel, grad_kernel, delta, n_events):
    """Return the gradient of the discrete l2 loss w.r.t. one kernel parameters.

    Parameters
    ----------

    zG : tensor, shape (n_dim, L)

    zN : tensor, shape (n_dim, L)

    ztzG : tensor, shape (n_dim, n_dim, L, L)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L)
        Kernel values on the discretization.

    grad_kernel : list of tensor of shape (n_dim, n_dim, L)
        Gradient values on the discretization.

    delta : float
        Step size of the discretization grid.

    n_events : tensor, shape (n_dim,)
        Number of events for each dimension.

    Returns
    ----------
    grad_theta : tensor, shape (n_dim, n_dim)
    """
    n_dim, _, L = kernel.shape

    grad_theta_ = torch.zeros(n_dim, n_dim)
    for m in range(n_dim):
        cst = 2 * delta * baseline[m]
        for n in range(n_dim):
            grad_theta_[m, n] = cst * alpha[m, n] * (grad_kernel[m, n] @ zG[n])
            grad_theta_[m, n] -= 2 * alpha[m, n] * (grad_kernel[m, n] @ zN[m, n])
            temp = 0
            for k in range(n_dim):
                cst2 = alpha[m, n] * alpha[m, k]
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

            grad_theta_[m, n] += delta * temp

    grad_theta = grad_theta_ / n_events.sum()

    return grad_theta
