import numpy as np
from fadin.solver import FaDIn, UNHaP
from fadin.utils.utils_simu import simu_hawkes_cluster, simulate_marked_data


def test_fadin_attr():
    events = simu_hawkes_cluster(
        end_time=1000,
        baseline=np.array([0.4]),
        alpha=np.array([[0.9]]),
        kernel='expon',
        params_kernel={'scale': 1/4.},
        random_state=0
    )
    solver = FaDIn(
        n_dim=1,
        kernel="truncated_exponential",
        kernel_length=1,
        delta=0.01,
        optim="RMSprop",
        params_optim={'lr': 1e-3},
        max_iter=20
    )
    solver.fit(events, 1000)
    assert hasattr(solver, 'baseline_'), \
        "FaDIn should have baseline_ attribute"
    assert hasattr(solver, 'alpha_'), "FaDIn should have alpha_ attribute"
    assert hasattr(solver, 'kernel_'), "FaDIn should have kernel_ attribute"


def test_unhap_attr():
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

    solver = UNHaP(
        n_dim=1,
        kernel="truncated_gaussian",
        init="random",
        max_iter=max_iter,
        batch_rho=batch_rho
    )
    solver.fit(ev, end_time)
    assert hasattr(solver, 'baseline_'), \
        "UNHaP should have baseline_ attribute"
    assert hasattr(solver, 'baseline_noise_'), \
        "UNHaP should have baseline_noise_ attribute"
    assert hasattr(solver, 'alpha_'), "UNHaP should have alpha_ attribute"
    assert hasattr(solver, 'kernel_'), "UNHaP should have kernel_ attribute"
