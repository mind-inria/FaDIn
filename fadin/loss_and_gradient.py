import torch


def discrete_l2_loss_conv(intensity, events_grid, delta):
    """Compute the l2 discrete loss using convolutions.

    .. math::
        \\frac{1}{N_T}\\sum_{i=1}^{p}  (\\Delta\\sum_{s \\in [|0, G |]
        (\\tilde{\\lambda}_{i}[s]^2 - 2\\sum_{\\tilde{t}_n^i \\in
        \\widetilde{\\mathscr{F}}_T^i}\\tilde{\\lambda}_{i}
        (\\frac{\\tilde{t}_n^{i}}{\\Delta})

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


def discrete_ll_loss_conv(intensity, events_grid, delta):
    """Compute the LL discrete loss using convolutions.

    Parameters
    ----------
    intensity : tensor, shape (n_dim, n_grid)
        Values of the intensity function evaluated  on the grid.

    events_grid : tensor, shape (n_dim, n_grid)
        Events projected on the pre-defined grid.

    delta : float
        Step size of the discretization grid.
    """
    mask = events_grid > 0
    intens = torch.log(intensity[mask])
    return (intensity.sum(1) * delta -
            intens.sum()).sum() / events_grid.sum()


def discrete_l2_loss_precomputation(zG, zN, ztzG, baseline, alpha, kernel,
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

    comp_1 = end_time * squared_compensator_1(baseline)

    comp_2 = 2 * delta * squared_compensator_2(zG, baseline, alpha, kernel)

    comp_3 = delta * squared_compensator_3(ztzG, alpha, kernel)

    intens_ev = 2 * intens_events(zN, baseline, alpha, kernel, n_events)

    loss_precomp = comp_1 + comp_2 + comp_3 - intens_ev

    return loss_precomp / n_events.sum()


def squared_compensator_1(baseline):
    """Compute the value of the first term of the
    discrete l2 loss using precomputations

    .. math::
        ||\\mu||_2^2

    Parameters
    ----------
    baseline : tensor, shape (n_dim,)
    """
    return torch.linalg.norm(baseline, ord=2) ** 2


def squared_compensator_2(zG, baseline, alpha, kernel):
    """Compute the value of the second term of the
    discrete l2 loss using precomputations

    .. math::
        \\sum_{i=1}^{p}\\mu_i \\sum_{j=1}^{p} \\sum_{\tau=1}^{L}
        \\phi_{ij}^\\Delta[\\tau] \\left(\\sum_{s=1}^{G} z_{j}[s-\\tau]
        \\right)

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

    if n_dim > 1:
        prod_zG_ker = torch.einsum('ju,iju->ij', zG, kernel)
        alpha_prod = alpha * prod_zG_ker
        res_matrix = baseline.view(n_dim, 1) * alpha_prod
        res = res_matrix.sum()
    else:
        res = 0
        for i in range(n_dim):
            temp = 0
            for j in range(n_dim):
                temp += alpha[i, j] * (zG[j] @ kernel[i, j])
            res += baseline[i] * temp

    return res


def squared_compensator_3(ztzG, alpha, kernel):
    """Compute the value of the third term of the
    discrete l2 loss using precomputations

    .. math::
        \\sum_{i=1}^{p} \\sum_{k=1}^{p} \\sum_{j=1}^{p}
        \\sum_{\\tau=1}^{L} \\sum_{\\tau'=1}^{L} \\phi_{ij}^\\Delta[\\tau]
        \\phi_{ik}^\\Delta[\\tau'] \\left( \\sum_{s=1}^{G}
        z_{j}[s-\\tau]~z_{k}[s-\\tau '] \\right)

    Parameters
    ----------
    ztzG : tensor, shape (n_dim, n_dim, L, L)

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L)
        Kernel values on the discretization.
    """
    n_dim, _, L = kernel.shape

    if n_dim > 1:
        alpha_prod = torch.einsum('ij,ik->ijk', alpha, alpha)
        prod_ztzG_ker = torch.einsum('nkuv,mnu->nkmv', ztzG, kernel)
        prod_ker_ztzG = torch.einsum('mkv,nkmv->mknv', kernel, prod_ztzG_ker)
        res = (alpha_prod * prod_ker_ztzG.sum(3)).sum()
    else:
        res = 0
        for i in range(n_dim):
            for k in range(n_dim):
                for j in range(n_dim):
                    alpha_prod_ijk = alpha[i, j] * alpha[i, k]
                    temp2 = kernel[i, k].view(1, L) * (
                        ztzG[j, k] * kernel[i, j].view(L, 1)
                    ).sum(0)
                    res += alpha_prod_ijk * temp2.sum()

    return res


def intens_events(zN, baseline, alpha, kernel, n_events):
    """Compute the value of the 4th term of the
    discrete l2 loss using precomputations, i.e.
    the intensity function values evaluated in the events.

    .. math::
        \\Biggl(\\sum_{i=1}^{p} N_T^i \\mu_i + \\sum_{i=1}^{p}
        \\sum_{j=1}^{p}\\sum_{\tau=1}^{L} \\phi_{ij}^\\Delta[\tau]
        \\Biggl(\\sum_{\\tilde{t}_n^i \\in \\widetilde{\\mathscr{F}}_T^i}
        z_{j}(\\frac{\\tilde{t}_n^i}{\\Delta}-\\tau}\\Biggl)

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

    if n_dim > 1:
        prod_zN_ker = torch.einsum('iju,iju->ij', zN, kernel)
        alpha_prod_dot = torch.tensordot(alpha, prod_zN_ker)
        base_ev = torch.dot(baseline, n_events)
        res = base_ev + alpha_prod_dot
    else:
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

    .. math::
        N_T\\frac{\\partial\\mathcal{L}_G}{\\partial \\mu_{m}} =
        2 T \\mu_m -  2 N_T^m + 2 \\Delta\\sum_{j=1}^{p} \\sum_{\\tau=1}^{L}
        \\phi_{mj}^\\Delta[\\tau]\\Phi_{j}(\\tau; G)

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

    if n_dim > 1:
        cst1 = end_time * baseline
        cst2 = 0.5 * n_events.sum()

        dot_kernel = torch.einsum('kju,ju->kj', kernel, zG)
        dot_kernel_ = (dot_kernel * alpha).sum(1)

        grad_baseline = (dot_kernel_ * delta + cst1 - n_events) / cst2
    else:
        grad_baseline_ = torch.zeros(n_dim)
        for k in range(n_dim):
            temp = 0
            for j in range(n_dim):
                temp += alpha[k, j] * (zG[j] @ kernel[k, j])
            grad_baseline_[k] = delta * temp
            grad_baseline_[k] += end_time * baseline[k]
            grad_baseline_[k] -= n_events[k]
        grad_baseline = 2 * grad_baseline_ / n_events.sum()

    return grad_baseline


def get_grad_alpha(zG, zN, ztzG, baseline, alpha, kernel, delta, n_events):
    """Return the gradient of the discrete l2 loss w.r.t. alpha.

    .. math::
        N_T\\frac{\\partial\\mathcal{L}_G}{\\partial \\alpha_{ml}} =
        2\\Delta \\mu_m  \\sum_{\\tau=1}^{L} \\frac{\\partial
        \\phi_{m,l}^\\Delta[\\tau]}{\\partial \\alpha_{m,l}} \\Phi_l(\\tau; G)
        + 2 \\Delta \\sum_{k=1}^{p} \\sum_{\\tau=1}^{L} \\sum_{\\tau'=1}^{L}
        \\phi_{mk}^\\Delta[\\tau'] \\frac{\\partial \\phi_{m,l}^\\Delta[\\tau]}
        {\\partial \\alpha_{m,l}} \\Psi_{l,k}(\\tau, \\tau'; G)

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

    if n_dim > 2:
        cst1 = delta * baseline.view(n_dim, 1)

        dot_kernel = torch.einsum('njuv,knu->njkv', ztzG, kernel)
        ker_ztzg = torch.einsum('kju,njku->knj', kernel, dot_kernel)

        term1 = torch.einsum('knj,kj->kn', ker_ztzg, alpha)
        term2 = torch.einsum('knu,nu->kn', kernel, zG)
        term3 = torch.einsum('knu,knu->kn', zN, kernel)

        grad_alpha_ = term1 * delta + cst1 * term2 - term3
    else:
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


def get_grad_eta(zG, zN, ztzG, baseline, alpha, kernel,
                 grad_kernel, delta, n_events):
    """Return the gradient of the discrete l2 loss w.r.t. one kernel
    parameter.

    .. math::
        N_T\\frac{\\partial\\mathcal{L}_G}{\\partial \\eta{ml}} =
        2\\Delta \\mu_m  \\sum_{\\tau=1}^{L} \\frac{\\partial
        \\phi_{m,l}^\\Delta[\\tau]}{\\partial \\eta{m,l}} \\Phi_l(\\tau; G)
        + 2 \\Delta \\sum_{k=1}^{p} \\sum_{\\tau=1}^{L} \\sum_{\\tau'=1}^{L}
        \\phi_{mk}^\\Delta[\\tau'] \\frac{\\partial \\phi_{m,l}^\\Delta[\\tau]}
        {\\partial \\eta{m,l}} \\Psi_{l,k}(\\tau, \\tau'; G)

    Parameters
    ----------
    zG : tensor, shape (n_dim, L)

    zN : tensor, shape (n_dim, n_dim, L)

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

    if n_dim > 2:

        cst1 = 2 * alpha
        cst2 = delta * baseline.view(n_dim, 1) * cst1
        cst3 = torch.einsum('mn,mk->mkn', alpha, alpha)
        temp1 = torch.einsum('mnk,nk->mn', grad_kernel, zG)
        temp2 = torch.einsum('mnk,mnk->mn', grad_kernel, zN)
        temp3 = torch.einsum('nkuv,mnu->nkmv', ztzG, grad_kernel)
        temp4 = torch.einsum('mkv,nkmv->mknv', kernel, temp3)

        grad_theta_ = cst2 * temp1 - cst1 * temp2 + \
            2 * delta * (cst3 * temp4.sum(3)).sum(1)
    else:
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

                    temp += cst2 * temp_.sum()

                grad_theta_[m, n] += delta * temp

    grad_theta = grad_theta_ / n_events.sum()

    return grad_theta
