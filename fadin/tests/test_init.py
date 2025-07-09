import numpy as np

from fadin.utils.utils_simu import simulate_marked_data
from fadin.solver import UNHaP


def test_unhap_init():

    baseline = np.array([0.3])
    baseline_noise = np.array([0.05])
    alpha = np.array([[1.45]])
    mu = np.array([[0.4]])
    sigma = np.array([[0.1]])

    end_time = 1000
    seed = 0
    max_iter = 20
    batch_rho = 5

    ev, noisy_marks, true_rho = simulate_marked_data(
        baseline, baseline_noise.item(), alpha, end_time, mu, sigma, seed=seed
    )
    unhap_mmmax = UNHaP(
        n_dim=1,
        kernel="truncated_gaussian",
        init="moment_matching_max",
        max_iter=max_iter,
        batch_rho=batch_rho
    )
    unhap_mmmax.fit(ev, end_time)

    unhap_mmtimean = UNHaP(
        n_dim=1,
        kernel="truncated_gaussian",
        init="moment_matching_mean",
        max_iter=max_iter,
        batch_rho=batch_rho
    )
    unhap_mmtimean.fit(ev, end_time)

    unhap_random = UNHaP(
        n_dim=1,
        kernel="truncated_gaussian",
        init="random",
        max_iter=max_iter,
        batch_rho=batch_rho
    )
    unhap_random.fit(ev, end_time)
    assert unhap_mmmax is not None, "UNHaP moment matching max failed"
    assert unhap_mmtimean is not None, "UNHaP moment matching mean failed"
    assert unhap_random is not None, "UNHaP random initialization failed"
    return None
