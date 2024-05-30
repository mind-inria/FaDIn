import numpy as np
import pandas as pd

import scipy
from scipy import stats


def find_max(intensity_function, duration):
    """Find the maximum intensity of a function."""
    res = scipy.optimize.minimize_scalar(
        lambda x: -intensity_function(x), bounds=(0, duration)
    )
    return -res.fun


def to_pandas(events):

    return pd.DataFrame({
        'time': np.concatenate(events),
        'type': np.concatenate([
            [i] * len(evi) for i, evi in enumerate(events)
        ])
    })


def from_pandas(events):
    assert 'mark' not in events.columns
    return [
        events.query("type == @i")['time'].values
        for i in events['type'].unique()
    ]


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
    "Normalize the given kernel on a finite support between 0 and kernel_length"
    cdf_normalization = kernel.cdf(kernel_length, **params_kernel) \
        - kernel.cdf(0, **params_kernel)
    return kernel.pdf(x, **params_kernel) / cdf_normalization


def compute_intensity(events, s, baseline, alpha, kernel,
                      kernel_length, params_kernel=dict()):
    "Compute the intensity function at events s giving the history of events"
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
                kernel_norm(diff[j], kernel, kernel_length, **params_kernel).sum()
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
        return scipy.integrate.simps(self.pdf(x), x)

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
        func_cdf = scipy.interpolate.interp1d(discrete_time, my_cdf)
        func_ppf = scipy.interpolate.interp1d(
            my_cdf, discrete_time, fill_value='extrapolate'
        )
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
    If the intensity is a numerical value, simulate a Homegenous Poisson Process,
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
        return to_pandas(events)

    if upper_bound is None:
        upper_bound = find_max(intensity, end_time)

    #  Simulate a homogenous Poisson process on [0, end_time]
    n_events = rng.poisson(lam=upper_bound*end_time, size=1)
    ev_x = rng.uniform(low=0.0, high=end_time, size=n_events)
    ev_y = rng.uniform(low=0.0, high=upper_bound, size=n_events)

    # ogata's thinning algorithm
    accepted = intensity(ev_x) > ev_y
    events = np.sort(ev_x[accepted])
    return to_pandas(events)


def simu_multi_poisson(end_time, intensity, upper_bound=None, random_state=None):
    """Simulate multivariate Poisson processes on [0, end_time] with
    the Ogata's modified thinning algorithm by superposition of univariate processes.
    If the intensity is a numerical value, simulate a Homegenous Poisson Process,
    If the intensity is a function, simulate an Inhomogenous Poisson Process.

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
        return to_pandas(events)

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

    return to_pandas(events)


def simu_hawkes_cluster(end_time, baseline, alpha, kernel,
                        params_kernel=dict(), kernel_length=None,
                        upper_bound=None, random_state=None):
    """ Simulate a multivariate Hawkes process following an immigration-birth procedure.
        Edge effects may be reduced according to the second references below.

    References:

    Møller, J., & Rasmussen, J. G. (2006). Approximate simulation of Hawkes processes.
    Methodology and Computing in Applied Probability, 8, 53-64.

    Møller, J., & Rasmussen, J. G. (2005). Perfect simulation of Hawkes processes.
    Advances in applied probability, 37(3), 629-646.

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
        String kernel available are probability distribution from scipy.stats module.
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
    immigrants = from_pandas(immigrants)
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
                    kernel, params_kernel, size=nij, kernel_length=kernel_length)
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

    return to_pandas(events)


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
        Note that this function will automatically convert the scipy kernel to
        a finite support kernel of size 'kernel_length'.

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

    return to_pandas(events)
