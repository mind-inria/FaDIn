import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import scipy.stats as stats

from fadin.utils.functions import identity, linear_zero_one
from fadin.utils.functions import reverse_linear_zero_one, truncated_gaussian


def find_max(intensity_function, duration):
    """Find the maximum intensity of a function."""
    res = minimize_scalar(
        lambda x: -intensity_function(x),
        bounds=(0, duration)
    )
    return -res.fun


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


def kernel_norm(x, kernel, kernel_length, params_kernel=dict()):
    """Normalize the given kernel on a finite support between 0 and
    kernel_length"""
    cdf_normalization = kernel.cdf(kernel_length, **params_kernel) \
        - kernel.cdf(0, **params_kernel)
    return kernel.pdf(x, **params_kernel) / cdf_normalization


def compute_intensity(events, s, baseline, alpha, kernel,
                      kernel_length, params_kernel=dict()):
    """Compute the intensity function at events s giving the history of events
    """
    diff = []
    n_dim = len(events)
    for i in range(n_dim):
        diff.append(s - events[i])

    contrib_dims = [0] * n_dim
    for i in range(n_dim):
        contrib_dims[i] = 0
        for j in range(n_dim):
            # param du kernel ij to extend
            contrib_dims[i] += alpha[i, j] *\
                kernel_norm(
                    diff[j], kernel, kernel_length, **params_kernel
                ).sum()
    intens = baseline + contrib_dims
    return np.array(intens)


def custom_density(density, params=dict(), size=1, kernel_length=None):
    """Sample elements from custom or scipy-defined distributions"""
    if callable(density):
        if kernel_length is None:
            kernel_length = 1
        distrib = custom_distribution(custom_density=density,
                                      params=params,
                                      kernel_length=kernel_length)
    elif isinstance(density, str):
        distrib = getattr(stats, density)
    else:
        raise TypeError('density has to be a str or a callable')

    return distrib.rvs(size=size)


class custom_distribution(stats.rv_continuous):
    """Construct finite support density and allows efficient scipy sampling"""
    def __init__(self, custom_density, params=dict(), kernel_length=10):
        super().__init__()
        # init our variance divergence
        self.custom_density = custom_density
        self.params = params
        self.kernel_length = kernel_length
        # init our cdf and ppf functions
        self.cdf_func, self.ppf_func = self.create_cdf_ppf()

    # function to normalise the pdf over chosen domain

    def normalisation(self, x):
        return simpson(self.pdf(x), x=x)

    def create_cdf_ppf(self):
        # define normalization support with the given kernel length
        discrete_time = np.linspace(0, self.kernel_length, 1001)
        # normalise our pdf to sum to 1 so it satisfies a distribution
        norm_constant_time = self.normalisation(discrete_time)
        # compute pdfs to be summed to form cdf
        my_pdfs = self.pdf(discrete_time) / norm_constant_time
        # cumsum to form cdf
        my_cdf = np.cumsum(my_pdfs)
        # make sure cdf bounded on [0,1]
        my_cdf = my_cdf / my_cdf[-1]
        # create cdf and ppf
        func_cdf = interp1d(discrete_time, my_cdf)
        func_ppf = interp1d(my_cdf, discrete_time, fill_value='extrapolate')
        return func_cdf, func_ppf

    # pdf function for averaged normals
    def _pdf(self, x):
        # custom * pdf_kernel
        return self.custom_density(x, **self.params)

    # cdf function
    def _cdf(self, x):
        return self.cdf_func(x)

    # inverse cdf function
    def _ppf(self, x):
        return self.ppf_func(x)


def simu_poisson(end_time, intensity, upper_bound=None, random_state=None):
    """ Simulate univariate Poisson processes on [0, end_time] with
    the Ogata's modified thinning algorithm.
    If the intensity is a numerical value, simulate a Homegenous Poisson
    Process.
    If the intensity is a function, simulate an Inhomogenous Poisson Process.

    Parameters
    ----------
    end_time : int | float
        Duration of the Poisson process.

    intensity: callable, int or float
        the intensity function of the underlying Poisson process.
        If callable, a inhomogenous Poisson process is simulated.
        If int or float, an homogenous Poisson process is simulated.

    upper_bound : int, float or None, default=None
        Upper bound of the intensity function. If None,
        the maximum of the function is taken onto a finite discrete grid.

    random_state : int, RandomState instance or None, default=None
        Set the numpy seed to 'random_state'.

    Returns
    -------
    events : array
        The timestamps of the point process' events.

    """
    rng = check_random_state(random_state)

    if not callable(intensity):
        assert isinstance(intensity, (int, float))
        n_events = rng.poisson(lam=intensity*end_time, size=1)
        events = np.sort(rng.uniform(0, end_time, size=n_events))
        return events

    if upper_bound is None:
        upper_bound = find_max(intensity, end_time)

    #  Simulate a homogenous Poisson process on [0, end_time]
    n_events = rng.poisson(lam=upper_bound*end_time, size=1)
    ev_x = rng.uniform(low=0.0, high=end_time, size=n_events)
    ev_y = rng.uniform(low=0.0, high=upper_bound, size=n_events)

    # ogata's thinning algorithm
    accepted = intensity(ev_x) > ev_y
    events = np.sort(ev_x[accepted])
    return events


def simu_multi_poisson(end_time, intensity, upper_bound=None,
                       random_state=None):
    """Simulate multivariate Poisson processes on [0, end_time] with
    the Ogata's modified thinning algorithm by superposition of univariate
    processes. If the intensity is a numerical value, simulate a Homegenous
    Poisson Process. If the intensity is a function, simulate an
    Inhomogenous Poisson Process.

    Parameters
    ----------
    end_time : int | float
        Duration of the Poisson process.

    intensity: list of callable, list of int or float
        the intensity function of the underlying Poisson process.
        If callable, a inhomogenous Poisson process is simulated.
        If int or float, an homogenous Poisson process is simulated.

    upper_bound : int, float or None, default=None
        Upper bound of the intensity function. If None,
        the maximum of the function is taken onto a finite discrete grid.

    random_state : int, RandomState instance or None, default=None
        Set the numpy seed to 'random_state'.

    Returns
    -------
    events : list of array
        The timestamps of the point process' events.
    """
    rng = check_random_state(random_state)

    events = []
    n_dim = len(intensity)
    # Homogenous case
    if not callable(intensity[0]):
        assert isinstance(intensity[0], (int, float))
        n_events = rng.poisson(lam=np.array(intensity)*end_time, size=n_dim)
        for i in range(n_dim):
            evs = np.sort(rng.uniform(0, end_time, size=n_events[i]))
            events.append(evs)
        return events

    if upper_bound is None:
        upper_bound = np.zeros(n_dim)
        for i in range(n_dim):
            upper_bound[i] = find_max(intensity[i], end_time)

    n_events = rng.poisson(lam=np.array(upper_bound)*end_time, size=n_dim)
    for i in range(n_dim):
        ev_xi = rng.uniform(low=0.0, high=end_time, size=n_events[i])
        ev_yi = rng.uniform(low=0.0, high=upper_bound[i], size=n_events[i])
        accepted_i = intensity[i](ev_xi) > ev_yi

        events.append(np.sort(ev_xi[accepted_i]))

    return events


def simu_hawkes_cluster(end_time, baseline, alpha, kernel,
                        params_kernel=dict(), kernel_length=None,
                        upper_bound=None, random_state=None):
    """ Simulate a multivariate Hawkes process following an immigration-birth
    procedure. Edge effects may be reduced according to the second
    references below.

    References:

    Møller, J., & Rasmussen, J. G. (2006). Approximate simulation of Hawkes
    processes. Methodology and Computing in Applied Probability, 8, 53-64.

    Møller, J., & Rasmussen, J. G. (2005). Perfect simulation of Hawkes
    processes. Advances in applied probability, 37(3), 629-646.

    Parameters
    ----------
    end_time : int | float
        Duration of the Poisson process.

    baseline : array of float of size (n_dim,)
        Baseline parameter of the Hawkes process.

    alpha : array of float of size (n_dim, n_dim)
        Weight parameter associated to the kernel function.

    kernel: str or callable
        The choice of the kernel for the simulation.
        String kernel available are probability distribution from scipy.stats
        module.
        A custom kernel can be implemented with the form kernel(x, **params).

    params_kernel: dict
        Parameters of the kernel used to simulate the process.
        It must follow parameters associated to scipy.stats distributions.

    kernel_length: float or None, default=None
        If the custom kernel has finite support, fixe the limit of the support.
        The support need to be high enough such that probability mass between
        zero and kernel_length is strictly higher than zero.

    upper_bound : int, float or None, default=None
        Upper bound of the baseline function. If None,
        the maximum of the baseline is taken onto a finite discrete grid.

    random_state : int, RandomState instance or None, default=None
        Set the numpy seed to 'random_state'.

    Returns
    -------
    events : list of arrays
        The timestamps of the point process' events.
    """
    rng = check_random_state(random_state)

    n_dim = baseline.shape[0]
    immigrants = simu_multi_poisson(end_time, baseline,
                                    upper_bound=upper_bound,
                                    random_state=random_state)
    gen = dict(gen0=immigrants)
    events = immigrants.copy()

    it = 0
    while len(gen[f'gen{it}']):
        print(f"Simulate generation {it}\r")
        Ck = gen[f'gen{it}']
        Dk = [[0] * n_dim for _ in range(n_dim)]
        C = [[0] * n_dim for _ in range(n_dim)]
        F = []
        s = 0
        for i in range(n_dim):
            Fi = []
            for j in range(n_dim):
                Dk[i][j] = rng.poisson(lam=alpha[i, j], size=len(Ck[j]))
                nij = Dk[i][j].sum()
                C[i][j] = np.repeat(Ck[j], repeats=Dk[i][j])
                Eij = custom_density(
                    kernel,
                    params_kernel,
                    size=nij,
                    kernel_length=kernel_length
                )
                Fij = C[i][j] + Eij
                Fi.append(Fij)
                s += Fij.shape[0]
            F.append(np.hstack(Fi))

        if s > 0:
            gen[f'gen{it+1}'] = F
            for i in range(n_dim):
                events[i] = np.concatenate((events[i], F[i]))
        else:
            for i in range(n_dim):
                valid_events = events[i] < end_time
                events[i] = np.sort(events[i][valid_events])
            break

        it += 1

    return events


def simu_hawkes_thinning(end_time, baseline, alpha, kernel,
                         kernel_length, params_kernel=dict(),
                         random_state=None):
    """ Simulate a multivariate Hawkes process with finite support kernels
         following a Thinning procedure. Note that kernels have to be the same
         for each dimension, i.e. phi = phi_ij for all i,j.

    References:

    Ogata, Y. (1981). On Lewis' simulation method for point processes.
    IEEE transactions on information theory, 27(1), 23-31.

    Parameters
    ----------
    end_time : int | float
        Duration of the Poisson process.

    baseline : array of float of size (n_dim,)
        Baseline parameter of the Hawkes process.

    alpha : array of float of size (n_dim, n_dim)
        Weight parameter associated to the kernel function.

    kernel: str
        The choice of the kernel for the simulation.
        Kernel available are probability distribution from scipy.stats module.
        Note that this function will automatically convert the scipy kernel to a
        finite support kernel of size 'kernel_length'.

    kernel_length: int
        Length of kernels in the Hawkes process.

    params_kernel: dict
        Parameters of the kernel used to simulate the process.
        It must follow parameters associated to scipy.stats distributions.

    random_state : int, RandomState instance or None, default=None
        Set the numpy seed to 'random_state'.

    Returns
    -------
    events : list of arrays
        The timestamps of the point process' events.
    """
    rng = check_random_state(random_state)

    n_dim, _ = alpha.shape
    kernel = getattr(stats, kernel)
    # Initialise history
    events = []
    for i in range(n_dim):
        events.append([])

    t = 0
    while (t < end_time):

        # upper bound for thinning sampling
        # assuming that kernels are alpha times a density function
        bound_contrib = alpha.max(1).sum()
        n_events = 0
        for i in range(n_dim):
            n_events += np.sum(np.array(events[i]) > (t - kernel_length))

        intensity_sup = baseline.sum() + n_events * bound_contrib

        r = stats.expon.rvs(scale=1 / intensity_sup)
        t += r

        lambda_t = compute_intensity(events, t, baseline, alpha,
                                     kernel, kernel_length, **params_kernel)
        sum_intensity = lambda_t.sum()

        if np.random.rand() * intensity_sup <= sum_intensity:
            k = list(rng.multinomial(1, list(lambda_t / sum_intensity))).index(1)
            events[k].append(t)

    # Delete points outside of [0, end_time]
    for i in range(n_dim):
        if (len(events[i]) > 0) and (events[i][-1] > end_time):
            del events[i][-1]
        events[i] = np.array(events[i])

    return events


def simu_marked_hawkes_cluster(end_time, baseline, alpha, time_kernel, marks_kernel,
                               marks_density, params_time_kernel=dict(),
                               params_marks_kernel=dict(),
                               params_marks_density=dict(),
                               time_kernel_length=None,
                               marks_kernel_length=None,
                               random_state=None):
    """ Simulate a multivariate marked Hawkes process following
        an immigration-birth procedure.

    Parameters
    ----------
    end_time : int | float
        Duration of the Poisson process.

    baseline : array of float of size (n_dim,)
        Baseline parameter of the Hawkes process.

    alpha : array of float of size (n_dim, n_dim)
        Weight parameter associated to the kernel function.

    time_kernel: str or callable
        The choice of the time kernel for the simulation.
        String kernel available are probability distribution from scipy.stats module.
        A custom kernel can be implemented with the form time_kernel(x, **params).

    marks_density: str
        The choice of the kernel for the simulation.
        String kernel available are probability distribution from scipy.stats module.

    params_time_kernel: dict
        Parameters of the kernel used to simulate the process.
        It must follow parameters associated to scipy.stats distributions or the custom
        given callable.

    params_marks_density: dict
        Parameters of the kernel used to simulate the process.
        It must follow parameters associated to scipy.stats distributions or the custom
        given callable.

    kernel_length: float or None, default=None
        If the custom kernel has finite support, fixe the limit of the support.
        The support need to be high enough such that probability mass between
        zero and kernel_length is strictly higher than zero.

    random_state : int, RandomState instance or None, default=None
        Set the numpy seed to 'random_state'.

    Returns
    -------
    events : list of array-like with shape (n_events, 2)
        The timestamps and the marks of the point process' events.
        Timestamps are first coordinate.
    """

    if random_state is None:
        np.random.seed(np.random.randint(0, 100))
    else:
        np.random.seed(random_state)

    n_dim = baseline.shape[0]
    immigrants = simu_multi_poisson(end_time, baseline,
                                    upper_bound=None,
                                    random_state=random_state)

    immigrants_marks = [custom_density(
        marks_density, params_marks_density, size=immigrants[i].shape[0],
        kernel_length=marks_kernel_length) for i in range(n_dim)]

    gen_time = dict(gen0=immigrants)
    gen_mark = dict(gen0=immigrants_marks)
    gen_labels = dict()

    events = []
    labels = [0] * n_dim
    for i in range(n_dim):
        temp = np.vstack([immigrants[i], immigrants_marks[i]]).T
        events.append(temp)

    it = 0
    while len(gen_time[f'gen{it}']):
        print(f"Simulate generation {it}\r")
        time_ev_k = gen_time[f'gen{it}']
        marks_k = gen_mark[f'gen{it}']
        induced_ev = [[0] * n_dim for _ in range(n_dim)]
        time_ev = [[0] * n_dim for _ in range(n_dim)]
        sim_ev = []
        sim_marks = []
        sim_labels = []
        s = 0
        for i in range(n_dim):
            sim_ev_i = []
            sim_marks_i = []
            sim_labels_i = []
            for j in range(n_dim):
                induced_ev[i][j] = np.random.poisson(
                    lam=alpha[i, j]*marks_k[j], size=len(time_ev_k[j]))

                no_exitations = np.where(induced_ev[i][j] == 0)[0]
                gen_labels_ij = np.ones(len(time_ev_k[j]))
                if it == 0:
                    gen_labels_ij[no_exitations] = 0.
                else:
                    gen_labels_ij[no_exitations] = 2.

                nij = induced_ev[i][j].sum()

                time_ev[i][j] = np.repeat(
                    time_ev_k[j], repeats=induced_ev[i][j]
                )

                inter_arrival_ij = custom_density(
                    time_kernel, params_time_kernel, size=nij,
                    kernel_length=time_kernel_length)

                sim_ev_ij = time_ev[i][j] + inter_arrival_ij
                sim_ev_i.append(sim_ev_ij)

                # custom distrib on mark density
                sim_marks_ij = custom_density(
                    marks_density, params_marks_density, size=nij,
                    kernel_length=marks_kernel_length)

                assert (sim_marks_ij > 1).sum() == 0
                sim_marks_i.append(sim_marks_ij)
                sim_labels_i.append(gen_labels_ij)

                s += sim_ev_ij.shape[0]
            sim_ev.append(np.hstack(sim_ev_i))
            sim_marks.append(np.hstack(sim_marks_i))
            sim_labels.append(np.hstack(sim_labels_i))

        if s > 0:
            gen_time[f'gen{it+1}'] = sim_ev
            gen_mark[f'gen{it+1}'] = sim_marks
            gen_labels[f'gen{it}'] = sim_labels
            for i in range(n_dim):
                temp = np.vstack([sim_ev[i], sim_marks[i]]).T
                events[i] = np.concatenate((events[i], temp))
                if it > 0:
                    labels[i] = np.concatenate((labels[i], sim_labels[i]))
                else:
                    labels[i] = sim_labels[i]
        else:
            for i in range(n_dim):
                if it > 0:
                    labels[i] = np.concatenate((labels[i], sim_labels[i]))
                else:
                    labels[i] = sim_labels[i]
                valid_events = events[i][:, 0] < end_time
                temp_ev = events[i][valid_events]
                temp_lab = labels[i][valid_events]

                sorting = np.argsort(temp_ev[:, 0])
                events[i] = temp_ev[sorting]
                labels[i] = temp_lab[sorting]
            break

        it += 1

    return events, labels


def simulate_marked_data(baseline, baseline_noise, alpha, end_time, mu,
                         sigma, seed=0):
    """ Simulate a marked Hawkes process with noise.

    The marked Hawkes process has a truncated_gaussian kernel. The marks
    densities are `linear_zero_ones` for the Hawkes process and
    `reverse_linear_zero_ones` for the noisy Poisson process.
    Parameters
    ----------
    baseline : array of float of size (n_dim,)
        Baseline parameter of the Hawkes process.
    baseline_noise : float
        Baseline parameter of the noisy Poisson process.
    alpha : array of float of size (n_dim, n_dim)
        Weight parameter associated to the kernel function.
    end_time : int | float
        Duration of the event time segment.
    mu : float
        Mean of the truncated Gaussian kernel.
    sigma : float
        Standard deviation of the truncated Gaussian kernel.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    events_cat : list of arrays
        The timestamps and the marks of the point process' events.
        Timestamps are first coordinate.
    noisy_marks : list of arrays
        The marks of the noisy Poisson process.
    true_rho : list of arrays
        The labels of the events, where 0 indicates noise and 1 indicates
        marked Hawkes events.
    """
    n_dim = len(baseline)

    marks_kernel = identity
    marks_density = linear_zero_one
    time_kernel = truncated_gaussian

    params_marks_density = dict()
    params_marks_kernel = dict(slope=1.2)
    params_time_kernel = dict(mu=mu, sigma=sigma)

    marked_events, _ = simu_marked_hawkes_cluster(
        end_time,
        baseline,
        alpha,
        time_kernel,
        marks_kernel,
        marks_density,
        params_marks_kernel=params_marks_kernel,
        params_marks_density=params_marks_density,
        time_kernel_length=None,
        marks_kernel_length=None,
        params_time_kernel=params_time_kernel,
        random_state=seed,
    )

    noisy_events_ = simu_multi_poisson(end_time, [baseline_noise])

    random_marks = [
        np.random.rand(noisy_events_[i].shape[0]) for i in range(n_dim)
    ]
    noisy_marks = [
        custom_density(
            reverse_linear_zero_one,
            dict(),
            size=noisy_events_[i].shape[0],
            kernel_length=1.0,
        )
        for i in range(n_dim)
    ]
    noisy_events = [
        np.concatenate(
            (noisy_events_[i].reshape(-1, 1), random_marks[i].reshape(-1, 1)),
            axis=1
        )
        for i in range(n_dim)
    ]

    events = [
        np.concatenate((noisy_events[i], marked_events[i]), axis=0)
        for i in range(n_dim)
    ]

    events_cat = [events[i][events[i][:, 0].argsort()] for i in range(n_dim)]

    labels = [
        np.zeros(marked_events[i].shape[0] + noisy_events_[i].shape[0])
        for i in range(n_dim)
    ]
    labels[0][-marked_events[0].shape[0]:] = 1.0
    true_rho = [labels[i][events[i][:, 0].argsort()] for i in range(n_dim)]

    return events_cat, noisy_marks, true_rho
