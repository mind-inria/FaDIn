import torch
import matplotlib.pyplot as plt


def init_hawkes_params(solver, init_mode, events, n_ground_events, end_time):
    """
    Computes the initial Hawkes parameters for the FaDIn solver.

    The function supports three modes of initialization:
    - 'random': Random initialization of parameters.
    - 'moment_matching': Moment matching initialization of parameters.
    - given: parameters are given by user.

    Parameters
    ----------
    solver : FaDIn
        FaDIn solver.
    init_mode: `str` or `dict`
        Mode of initialization. Supported values are 'random', 'moment_matching', and 
        a dictionary with keys 'baseline', 'alpha' and 'kernel'.
    events: list of array of size number of timestamps,
        list size is self.n_dim.
    n_ground_events : torch.tensor
        Number of ground events for each dimension
    end_time: float
        End time of the events time series.

    Returns:
    --------
    params_intens: list
        List of parameters of the Hawkes process. [baseline, alpha, kernel_params]
        baseline : `tensor`, shape `(solver.n_dim)`
            Baseline parameter of the Hawkes process.
        alpha : `tensor`, shape `(solver.n_dim, n_dim)`
            Weight parameter of the Hawkes process.
        kernel_params : `list` of `tensor`
            list containing tensor array of kernels parameters.
            The size of the list varies depending the number of
            parameters. The shape of each tensor is `(solver.n_dim, solver.n_dim)`.
    """
    # Compute initial Hawkes parameters
    if init_mode == 'moment_matching':
        baseline, alpha, kernel_params_init = momentmatching_nomark(
            solver,
            events,
            n_ground_events,
            end_time,
            solver.mm_mode
        )
    elif init_mode == 'random':
        baseline, alpha, kernel_params_init = random_params(solver)
    else:
        baseline = init_mode['baseline'].float()
        alpha = init_mode['alpha'].float()
        kernel_params_init = init_mode['kernel']

    # Format initial parameters for optimization
    baseline = (baseline * solver.baseline_mask).requires_grad_(True)
    alpha = (alpha * solver.alpha_mask).requires_grad_(True)
    params_intens = [baseline, alpha]
    solver.n_kernel_params = len(kernel_params_init)
    for i in range(solver.n_kernel_params):
        kernel_param = kernel_params_init[i].float().clip(1e-4)
        kernel_param.requires_grad_(True)
        params_intens.append(kernel_param)

    return params_intens


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

    # Alpha init
    alpha = torch.ones(solver.n_dim, solver.n_dim) / (solver.n_dim + 1)

    # Kernel parameters init
    kernel_params_init = momentmatching_kernel_nomark(
        solver, events, n_ground_events, mode=mode
    )
    return baseline, alpha, kernel_params_init


def random_params(solver):
    """Random initialization of baseline, alpha and kernel parameters.
    Baseline and alpha are initialized with uniform distribution.
    Kernel parameters are initialized with uniform distribution.
    """
    # Baseline init
    baseline = torch.rand(solver.n_dim)

    # Alpha init
    alpha = torch.rand(solver.n_dim, solver.n_dim)

    # Kernel parameters init
    kernel_params_init = []
    if solver.kernel == 'raised_cosine':
        temp = 0.5 * solver.kernel_length * torch.rand(solver.n_dim, solver.n_dim)
        temp2 = 0.5 * solver.kernel_length * torch.rand(solver.n_dim, solver.n_dim)
        kernel_params_init.append(temp)
        kernel_params_init.append(temp2)
    elif solver.kernel == 'truncated_gaussian':
        temp = 0.25 * solver.kernel_length * torch.rand(solver.n_dim, solver.n_dim)
        temp2 = 0.5 * solver.kernel_length * torch.rand(solver.n_dim, solver.n_dim)
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
