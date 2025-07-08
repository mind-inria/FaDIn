import numpy as np
# from fadin.kernels import truncated_gaussian
from fadin.utils.functions import identity, linear_zero_one
from fadin.utils.functions import reverse_linear_zero_one, truncated_gaussian

from fadin.utils.utils_simu import simu_marked_hawkes_cluster, custom_density
from fadin.utils.utils_simu import simu_multi_poisson
from fadin.solver import UNHaP

baseline = np.array([0.3])
baseline_noise = np.array([0.05])
alpha = np.array([[1.45]])
mu = np.array([[0.4]])
sigma = np.array([[0.1]])

delta = 0.01
end_time = 1000
seed = 0
max_iter = 20
batch_rho = 200


# Create the simulating function
def simulate_marked_data(baseline, baseline_noise, alpha, end_time, seed=0):
    n_dim = len(baseline)

    marks_kernel = identity
    marks_density = linear_zero_one
    time_kernel = truncated_gaussian

    params_marks_density = dict()
    # params_marks_density = dict(scale=1)
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

    random_marks = [np.random.rand(noisy_events_[i].shape[0]) for i in range(n_dim)]
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
            (noisy_events_[i].reshape(-1, 1), random_marks[i].reshape(-1, 1)), axis=1
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


ev, noisy_marks, true_rho = simulate_marked_data(
    baseline, baseline_noise.item(), alpha, end_time, seed=0
)


def test_unhap_init():
    unhap_mmmax = UNHaP(
        n_dim=1,
        kernel="truncated_gaussian",
        init="moment_matching_max",
        max_iter=max_iter
    )
    unhap_mmmax.fit(ev, end_time)

    unhap_mmtimean = UNHaP(
        n_dim=1,
        kernel="truncated_gaussian",
        init="moment_matching_mean",
        max_iter=max_iter
    )
    unhap_mmtimean.fit(ev, end_time)

    unhap_random = UNHaP(
        n_dim=1,
        kernel="truncated_gaussian",
        init="random",
        max_iter=max_iter
    )
    unhap_random.fit(ev, end_time)
    assert unhap_mmmax is not None, "UNHaP moment matching max failed"
    assert unhap_mmtimean is not None, "UNHaP moment matching mean failed"
    assert unhap_random is not None, "UNHaP random initialization failed"
    return None
