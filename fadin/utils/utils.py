import numpy as np
import torch
import matplotlib.pyplot as plt


def convert_float_tensor(x):
    return torch.tensor(x).float()


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


def init_kernel_parameters(kernel, kernel_length, n_dim):
    kernel_params_init = []
    if kernel == 'raised_cosine':
        temp = 0.5 * kernel_length * torch.rand(n_dim, n_dim)
        temp2 = 0.5 * kernel_length * torch.rand(n_dim, n_dim)
        kernel_params_init.append(temp)
        kernel_params_init.append(temp2)
    elif kernel == 'truncated_gaussian':
        temp = 0.25 * kernel_length * torch.rand(n_dim, n_dim)
        temp2 = 0.5 * kernel_length * torch.rand(n_dim, n_dim)
        kernel_params_init.append(temp)
        kernel_params_init.append(temp2)
    elif kernel == 'truncated_exponential':
        kernel_params_init.append(2 * torch.rand(n_dim, n_dim))
    else:
        raise NotImplementedError('kernel initial parameters of not \
                                    implemented kernel have to be given')
    return kernel_params_init


def smart_init_kernel(solver, events, n_ground_events):
    """Smart initialization of kernel parameters.
    """
    kernel_params_init = [torch.ones(solver.n_dim, solver.n_dim),
                          torch.ones(solver.n_dim, solver.n_dim)]
    for i in range(solver.n_dim):
        for j in range(solver.n_dim):
            # Mean, std of time delta of [i, j] kernel
            delta_t = torch.zeros(int(n_ground_events[i].item()))
            for n in range(int(n_ground_events[i])):
                t_n_i = events[i][n][0]
                t_n_j = torch.max(
                    torch.where(torch.tensor(events[j][:, 0]) < t_n_i,
                                torch.tensor(events[j][:, 0]),
                                0.)
                    )
                delta_t[n] = t_n_i - t_n_j
            avg = torch.mean(delta_t)
            std = torch.std(delta_t)

            # Parameters of [i, j] kernel
            if solver.kernel == 'truncated_gaussian':
                kernel_params_init[0][i, j] = avg
            if solver.kernel == 'raised_cosine':
                u = max(avg - std, 0)
                kernel_params_init[0][i, j] = u
            kernel_params_init[1][i, j] = std
    return kernel_params_init


def smart_init(solver, events, n_ground_events, end_time):
    """Smart initialization of baseline and alpha.
    $\\mu_i^s = \frac{\\#\\mathscr{F}^i_T}{(D+1)T} \forall i \\in [1, D]$
    """
    assert solver.kernel in ['truncated_gaussian', 'raised_cosine'], (
        f"Smart initialization not implemented for kernel {solver.kernel}"
    )
    # Baseline init

    baseline = n_ground_events / (end_time * (solver.n_dim + 1))
    baseline = (baseline * solver.baseline_mask).requires_grad_(True)

    # Alpha init
    alpha = torch.ones(solver.n_dim, solver.n_dim) / (solver.n_dim + 1)
    alpha = (alpha * solver.alpha_mask).requires_grad_(True)

    # Kernel parameters init
    if solver.optimize_kernel:
        kernel_params_init = smart_init_kernel(solver, events, n_ground_events)

    return baseline, alpha, kernel_params_init


def smart_init_noise(solver, events, n_ground_events, end_time):
    """Smart initialization of baseline and alpha, with an extra baseline term for
    noisy events.
    $\\mu_i^{s, noise} = \\frac{\\#\\mathscr{F}^i_T}{(D+2)T} \\forall i \\in [1, D]$
    $\\mu_i^s = \frac{\\#\\mathscr{F}^i_T}{(D+2)T} \forall i \\in [1, D]$
    $\\alpha_{i, j}^s = \\frac{1}{D+2} \\forall i,j \\in [1, D]
    """
    assert solver.kernel in ['truncated_gaussian', 'raised_cosine'], (
        f"Smart initialization not implemented for kernel {solver.kernel}"
    )
    # Baseline init
    baseline = n_ground_events / (end_time * (solver.n_dim + 2))
    baseline = (baseline * solver.baseline_mask).requires_grad_(True)

    bl_noise = n_ground_events / (end_time * (solver.n_dim + 2))
    bl_noise = (bl_noise * solver.baseline_mask).requires_grad_(True)

    # Alpha init
    alpha = torch.ones(solver.n_dim, solver.n_dim) / (solver.n_dim + 2)
    alpha = (alpha * solver.alpha_mask).requires_grad_(True)

    # Kernel parameters init
    kernel_params_init = smart_init_kernel(solver, events, n_ground_events)

    return baseline, bl_noise, alpha, kernel_params_init


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
    # size_discret = int(1 / grid_step)
    events_grid = torch.zeros(n_dim, n_grid)
    for i in range(n_dim):
        ei_torch = torch.tensor(events[i])
        temp = torch.round(ei_torch / grid_step).long()  # * grid_step
        # temp2 = torch.round(temp * size_discret)
        idx, data = np.unique(temp, return_counts=True)
        events_grid[i, idx] += torch.tensor(data)

    return events_grid


def smooth_projection_marked(marked_events, delta, n_grid):
    """Project events on the grid and remove duplica in both events grid and
    the original events lists. Return also the marks on the grid."""
    n_dim = len(marked_events)
    events_grid = torch.zeros((n_dim, n_grid))
    marks_grid = torch.zeros((n_dim, n_grid))
    marked_events_smooth = [None] * n_dim
    id_unique = []
    for i in range(n_dim):
        ev_i = marked_events[i][:, 0]
        marks_i = marked_events[i][:, 1]
        idx = np.round(ev_i / delta).astype(np.int64)
        events_grid[i, idx] += 1

        z = np.round(ev_i / delta) * delta
        ev_unique, index_unique = np.unique(z, return_index=True)
        marked_events_smooth[i] = np.concatenate(
            (ev_unique[:, None], marks_i[index_unique][:, None]), axis=1)

        #  idx_grid = #np.round(ev_unique * L)
        marks_grid[i, idx[index_unique]] += torch.tensor(marks_i[index_unique])
        id_unique.append(index_unique)
    #  marks_grid = projected_grid_marked(marked_events_smooth, delta, n_grid)
    return events_grid, marks_grid, marked_events_smooth, id_unique


def optimizer_fadin(param, params_optim, solver='RMSprop'):
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


def optimizer_unhap(param, params_optim, solver='RMSprop'):
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
        return torch.optim.SGD(param[0], **params_optim), \
            torch.optim.SGD(param[1], **params_optim)
    elif solver == 'RMSprop':
        return torch.optim.RMSprop(param[0], **params_optim), \
            torch.optim.RMSprop(param[1], **params_optim)
    elif solver == 'Adam':
        return torch.optim.Adam(param[0], **params_optim), \
            torch.optim.Adam(param[1], **params_optim)
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


def momentmatching_kernel_nomark(solver, events, n_ground_events,
                                 plot_delta=False, mode='max'):
    """Moment matching initialization of kernel parameters. Implemented for
    'truncated_gaussian' and 'raised_cosine' kernels.
    For the truncated gaussian kernel, the means $m_{i,j}$ and std
    $\\sigma_{i,j}$ are:
    $m_{i, j} =
    \\frac{1}{N_{g_i}(T)}\\sum_{t_n^i \\in \\mathscr{F}_T^i}
    \\delta t^{i, j}_n$
    $\\sigma_{i, j} =
    \\sqrt{\\dfrac{
        \\sum_{t_n^i \\in \\mathscr{F}_T^i} (\\delta t^{i, j}_n - m_{i, j})^2
    }{N_{g_i}(T) - 1}}.
    For the raised cosine kernel, the parameters $u_{i,j}$ and $s_{i,j} are:
    $u^{\\text{m}}_{i, j} =
    \\max{(0, m^{\\text{m}}_{i, j} - \\sigma^{\\text{m}}_{i, j})}$
    $s^{\\text{m}}_{i, j} = \\sigma_{i, j}^{\\text{m}}$

    Parameters
    ----------
    solver :  `FaDIn` or `MarkedFaDIn` object
        The solver object
    events : list of torch.tensor
        List of events for each dimension
    n_ground_events : torch.tensor
        Number of ground events for each dimension
    plot_delta : bool, default=False
        Whether to plot the delta_t distribution
    mode : str, default='max'
        Mode to compute the delta_t distribution. Supported values are 'max'
        and 'mean'.

    Returns
    -------
    list of torch.tensor
        List of kernel parameters

    """
    kernel_params_init = [torch.ones(solver.n_dim, solver.n_dim),
                          torch.ones(solver.n_dim, solver.n_dim)]
    for i in range(solver.n_dim):
        for j in range(solver.n_dim):
            # Mean, std of time delta of [i, j] kernel
            if mode == 'max':
                delta_t = torch.zeros(int(n_ground_events[i].item()))
                for n in range(int(n_ground_events[i])):
                    t_n_i = events[i][n]
                    t_n_j = torch.max(
                        torch.where(torch.tensor(events[j]) < t_n_i,
                                    torch.tensor(events[j]),
                                    0.)
                        )
                    delta_t[n] = t_n_i - t_n_j
                avg = torch.mean(delta_t)
                std = torch.std(delta_t)
            if mode == 'mean':
                delta_t = []
                for n in range(int(n_ground_events[i])):
                    t_n_i = events[i][n]
                    for t_n_j in events[j]:
                        if t_n_j < t_n_i - solver.kernel_length:
                            continue
                        if t_n_j >= t_n_i:
                            break
                        delta_t.append(t_n_i - t_n_j)
                avg = torch.mean(torch.tensor(delta_t))
                std = torch.std(torch.tensor(delta_t))
            # Plot delta_t distribution
            if plot_delta:
                fig_delta, ax_delta = plt.subplots()
                ax_delta.hist(delta_t, bins=20)
                ax_delta.set_xlim([0, solver.kernel_length])
                ax_delta.set_xlabel('Time')
                ax_delta.set_ylabel('Histogram')
                fig_delta.suptitle('Moment Matching delta_t')
                fig_delta.show()
            # Parameters of [i, j] kernel
            if solver.kernel == 'truncated_gaussian':
                kernel_params_init[0][i, j] = avg
            if solver.kernel == 'raised_cosine':
                u = max(avg - std, 0)
                kernel_params_init[0][i, j] = u
            kernel_params_init[1][i, j] = std
    return kernel_params_init


def momentmatching_nomark(solver, events, n_ground_events, end_time,
                          mode='max'):
    """Moment matching initialization of baseline, alpha and kernel parameters.
    $\\mu_i^s = \frac{\\#\\mathscr{F}^i_T}{(D+1)T} \forall i \\in [1, D]$
    $\\alpha_{i, j}^s = \\frac{1}{D+1} \\forall i,j \\in [1, D]$
    Kernel parameters are initialized by `momentmatching_kernel`.
    """
    assert solver.kernel in ['truncated_gaussian', 'raised_cosine'], (
        f"Smart initialization not implemented for kernel {solver.kernel}"
    )
    # Baseline init
    baseline = n_ground_events / (end_time * (solver.n_dim + 1))
    baseline = (baseline * solver.baseline_mask).requires_grad_(True)

    # Alpha init
    alpha = torch.ones(solver.n_dim, solver.n_dim) / (solver.n_dim + 1)
    alpha = (alpha * solver.alpha_mask).requires_grad_(True)

    # Kernel parameters init
    kernel_params_init = momentmatching_kernel_nomark(
        solver, events, n_ground_events, mode=mode
    )
    return baseline, alpha, kernel_params_init


def init_saving_params(params_intens, n_kernel_params, n_dim, max_iter):

    v_loss = torch.zeros(max_iter)

    param_baseline = torch.zeros(max_iter + 1, n_dim)
    param_baseline_noise = torch.zeros(max_iter + 1, n_dim)
    param_alpha = torch.zeros(max_iter + 1, n_dim, n_dim)
    param_kernel = torch.zeros(n_kernel_params, max_iter + 1, n_dim, n_dim)

    param_baseline[0] = params_intens[0].detach()
    param_baseline_noise[0] = params_intens[1].detach()
    param_alpha[0] = params_intens[2].detach()

    return v_loss, param_baseline, param_baseline_noise, param_alpha, param_kernel
