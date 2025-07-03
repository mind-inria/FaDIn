import torch
import matplotlib.pyplot as plt


def momentmatching_kernel(solver, events, n_ground_events,
                          plot_delta=False, mode='max'):
    """Moment matching initialization of kernel parameters.

    Implemented for 'truncated_gaussian' and 'raised_cosine' kernels.
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
                    if events[i].ndim == 1:
                        # unmarked case
                        t_n_i = events[i][n]
                        j_iter = torch.tensor(events[j])
                    if events[i].ndim == 2:
                        # marked case
                        t_n_i = events[i][n][0]
                        j_iter = torch.tensor(events[j][:, 0])
                    t_n_j = torch.max(
                        torch.where(j_iter < t_n_i, j_iter, 0.)
                        )
                    delta_t[n] = t_n_i - t_n_j
                avg = torch.mean(delta_t)
                std = torch.std(delta_t)
            if mode == 'mean':
                delta_t = []
                for n in range(int(n_ground_events[i])):
                    if events[i].ndim == 1:
                        # unmarked case
                        t_n_i = events[i][n]
                        j_iter = events[j]
                    if events[i].ndim == 2:
                        # marked case
                        t_n_i = events[i][n][0]
                        j_iter = events[j][:, 0]
                    # t_n_i = events[i][n]
                    for t_n_j in j_iter:
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


def init_hawkes_params_fadin(solver, init, events, n_ground_events, end_time):
    """
    Computes the initial Hawkes parameters for the FaDIn solver.

    The function supports three modes of initialization:
    - 'random': random initialization of parameters.
    - 'moment_matching': moment matching initialization of parameters.
    - given: parameters are given by user.

    Parameters
    ----------
    solver : FaDIn
        FaDIn solver.
    init: `str` or `dict`
        Mode of initialization. Supported values are 'random',
        'moment_matching', and a dictionary with keys 'baseline', 'alpha' and
        'kernel'.
    events: list of array of size number of timestamps,
        list size is self.n_dim.
    n_ground_events : torch.tensor
        Number of ground events for each dimension
    end_time: float
        End time of the events time series.

    Returns:
    --------
    params_intens: list
        List of parameters of the Hawkes process.
        [baseline, alpha, kernel_params]
        baseline : `tensor`, shape `(solver.n_dim)`
            Baseline parameter of the Hawkes process.
        alpha : `tensor`, shape `(solver.n_dim, n_dim)`
            Weight parameter of the Hawkes process.
        kernel_params : `list` of `tensor`
            list containing tensor array of kernels parameters.
            The size of the list varies depending the number of
            parameters. The shape of each tensor is
            `(solver.n_dim, solver.n_dim)`.
    """
    # Compute initial Hawkes parameters
    if 'moment_matching' in init:
        mm_mode = init.split('_')[-1]
        baseline, alpha, kernel_params_init = momentmatching_fadin(
            solver,
            events,
            n_ground_events,
            end_time,
            mm_mode
        )

    elif init == 'random':
        baseline, alpha, kernel_params_init = random_params_fadin(solver)
    else:
        baseline = init['baseline'].float()
        alpha = init['alpha'].float()
        kernel_params_init = init['kernel']

    # Format initial parameters for optimization
    solver.baseline = (baseline * solver.baseline_mask).requires_grad_(True)
    solver.alpha = (alpha * solver.alpha_mask).requires_grad_(True)
    params_intens = [solver.baseline, solver.alpha]
    solver.n_kernel_params = len(kernel_params_init)
    for i in range(solver.n_kernel_params):
        kernel_param = kernel_params_init[i].float().clip(1e-4)
        kernel_param.requires_grad_(True)
        params_intens.append(kernel_param)

    return params_intens


def momentmatching_fadin(solver, events, n_ground_events, end_time,
                         mode='max'):
    """Moment matching initialization of baseline, alpha and kernel parameters.

    $\\mu_i^s = \frac{\\#\\mathscr{F}^i_T}{(D+1)T} \forall i \\in [1, D]$
    $\\alpha_{i, j}^s = \\frac{1}{D+1} \\forall i,j \\in [1, D]$
    Kernel parameters are initialized by function
    momentmatching_kernel_nomark`.

    Parameters
    ----------
    solver: FaDIn
        FaDIn solver.
    events: list of array of size number of timestamps,
        list size is self.n_dim.
    n_ground_events : torch.tensor
        Number of ground events for each dimension
    end_time: float
        End time of the events time series.
    mode: `str`
        Mode to compute the delta_t distribution. Supported values are 'max'
        and 'mean'.

    Returns
    -------
    baseline: torch.tensor
        Baseline parameter of the Hawkes process.
    alpha: torch.tensor
        Weight parameter of the Hawkes process.
    kernel_params_init: list of torch.tensor
        List of kernel parameters.
    """
    assert solver.kernel in ['truncated_gaussian', 'raised_cosine'], (
        f"Smart initialization not implemented for kernel {solver.kernel}"
    )
    # Baseline init
    baseline = n_ground_events / (end_time * (solver.n_dim + 1))

    # Alpha init
    alpha = torch.ones(solver.n_dim, solver.n_dim) / (solver.n_dim + 1)

    # Kernel parameters init
    kernel_params_init = momentmatching_kernel(
        solver, events, n_ground_events, mode=mode
    )
    return baseline, alpha, kernel_params_init


def init_hawkes_params_unhap(solver, init, events, n_ground_events,
                             end_time):
    """
    Computes the initial Hawkes parameters for the UNHaP solver.

    The function supports three modes of initialization:
    - 'random': random initialization of parameters.
    - 'moment_matching': moment matching initialization of parameters.
    - given: parameters are given by user.

    Parameters
    ----------
    solver : UNHaP
        instance of UNHaP solver.
    init: `str` or `dict`
        Mode of initialization. Supported values are 'random',
        'moment_matching', and a dictionary with keys 'baseline', 'baseline_noise',
        'alpha' and 'kernel'.
    events: list of array of size number of timestamps,
        list size is self.n_dim.
    n_ground_events : torch.tensor
        Number of ground events for each dimension
    end_time: float
        End time of the events time series.

    Returns:
    --------
    params_intens: list
        List of parameters of the Hawkes process.
        [baseline, baseline_noise, alpha, kernel_params]
        baseline : `tensor`, shape `(solver.n_dim)`
            Baseline parameter of the Hawkes process.
        baseline_noise : `tensor`, shape `(solver.n_dim)`
            Noise baseline parameter of the Hawkes process.
        alpha : `tensor`, shape `(solver.n_dim, n_dim)`
            Weight parameter of the Hawkes process.
        kernel_params : `list` of `tensor`
            list containing tensor array of kernels parameters.
            The size of the list varies depending the number of
            parameters. The shape of each tensor is
            `(solver.n_dim, solver.n_dim)`.
    """
    # Compute initial Hawkes parameters
    if 'moment_matching' in init:
        mm_mode = init.split('_')[-1]
        baseline, baseline_noise, alpha, kernel_params_init = momentmatching_unhap(
            solver,
            events,
            n_ground_events,
            end_time,
            mm_mode
        )

    elif init == 'random':
        baseline, baseline_noise, alpha, kernel_params_init = random_params_unhap(
            solver)
    else:
        baseline = init['baseline'].float()
        baseline_noise = init['baseline_noise'].float()
        alpha = init['alpha'].float()
        kernel_params_init = init['kernel']

    # Format initial parameters for optimization
    solver.baseline = (baseline * solver.baseline_mask).requires_grad_(True)
    solver.baseline_noise = (baseline_noise * solver.bl_noise_mask).requires_grad_(True)
    solver.alpha = (alpha * solver.alpha_mask).requires_grad_(True)
    params_intens = [solver.baseline, solver.baseline_noise, solver.alpha]
    solver.n_kernel_params = len(kernel_params_init)
    for i in range(solver.n_kernel_params):
        kernel_param = kernel_params_init[i].float().clip(1e-4)
        kernel_param.requires_grad_(True)
        params_intens.append(kernel_param)
    return params_intens


def momentmatching_unhap(solver, events, events_grid, n_ground_events,
                         end_time, mode='max', plot_delta=False):
    """Smart initialization of baseline and alpha, with an extra baseline term
    for noisy events.
    $\\mu_i^{s, noise} = \\frac{\\#\\mathscr{F}^i_T}{(D+2)T} \\forall i \\in [1, D]$
    $\\mu_i^s = \frac{\\#\\mathscr{F}^i_T}{(D+2)T} \forall i \\in [1, D]$
    $\\alpha_{i, j}^s = \\frac{1}{D+2} \\forall i,j \\in [1, D]
    """ 
    assert solver.kernel in ['truncated_gaussian', 'raised_cosine'], (
        f"Smart initialization not implemented for kernel {solver.kernel}"
    )
    # Baseline, noise baseline and alpha init
    baseline = n_ground_events / (end_time * (solver.n_dim + 2))
    baseline_noise = n_ground_events / (end_time * (solver.n_dim + 2))
    alpha = torch.ones(solver.n_dim, solver.n_dim) / (solver.n_dim + 2)

    # Kernel parameters init
    if solver.optimize_kernel:
        kernel_params_init = momentmatching_kernel(
            solver, events, events_grid, n_ground_events, plot_delta, mode
        )
    return baseline, baseline_noise, alpha, kernel_params_init


def random_params_fadin(solver):
    """Random initialization of baseline, alpha, and kernel parameters.

    Hawkes parameters for FaDIn solver are initialized using a random
    distribution.

    Parameters
    ----------
    solver : FaDIn
        FaDIn solver.

    Returns
    -------
    baseline: torch.tensor
        Baseline parameter of the Hawkes process.
    alpha: torch.tensor
        Weight parameter of the Hawkes process.
    kernel_params_init: list of torch.tensor
        List of kernel parameters.
    """
    # Baseline init
    baseline = torch.rand(solver.n_dim)

    # Alpha init
    alpha = torch.rand(solver.n_dim, solver.n_dim)

    # Kernel parameters init
    kernel_params_init = []
    if solver.kernel == 'raised_cosine':
        temp = 0.5 * solver.kernel_length * \
            torch.rand(solver.n_dim, solver.n_dim)
        temp2 = 0.5 * solver.kernel_length * \
            torch.rand(solver.n_dim, solver.n_dim)
        kernel_params_init.append(temp)
        kernel_params_init.append(temp2)
    elif solver.kernel == 'truncated_gaussian':
        temp = 0.25 * solver.kernel_length * \
            torch.rand(solver.n_dim, solver.n_dim)
        temp2 = 0.5 * solver.kernel_length * \
            torch.rand(solver.n_dim, solver.n_dim)
        kernel_params_init.append(temp)
        kernel_params_init.append(temp2)
    elif solver.kernel == 'truncated_exponential':
        kernel_params_init.append(
            2 * torch.rand(solver.n_dim, solver.n_dim)
        )
    else:
        raise NotImplementedError(
            'kernel initial parameters of not \
                implemented kernel have to be given'
        )
    return baseline, alpha, kernel_params_init


def random_params_unhap(solver):
    """Random initialization of baseline, alpha, and kernel parameters.

    Hawkes parameters for UNHaP solver are initialized using a random
    distribution.

    Parameters
    ----------
    solver : UNHaP
        UNHaP solver.

    Returns
    -------
    baseline: torch.tensor
        Baseline parameter of the Hawkes process.
    baseline_noise: torch.tensor
        Noise baseline parameter of the Hawkes process.
    alpha: torch.tensor
        Weight parameter of the Hawkes process.
    kernel_params_init: list of torch.tensor
        List of kernel parameters.
    """

    # Noise baseline init
    baseline_noise = torch.rand(solver.n_dim)

    # Other parameters are initiated same as in FaDIn
    baseline, alpha, kernel_params_init = random_params_fadin(solver)
    return baseline, baseline_noise, alpha, kernel_params_init
