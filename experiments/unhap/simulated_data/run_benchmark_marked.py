"""Warning: the pyhawkes and ttpp packages are difficult to install, install
and run at your own risk :) !"""

# %% Import libraries
import itertools
import time
import numpy as np

from joblib import Parallel, delayed
import pandas as pd
import torch
from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.solver import FaDIn


from pyhawkes.models import (
    DiscreteTimeNetworkHawkesModelGammaMixture,
    DiscreteTimeStandardHawkesModel,
)
from pyhawkes.internals.network import StochasticBlockModel

from ttpp.data import load_dataset
from ttpp.training import train_tripp, train_rnn

from fadin.utils.utils import projected_grid, projected_grid_marked
from fadin.utils.utils_simu import simu_marked_hawkes_cluster, \
    simu_multi_poisson
from fadin.solver import UNHaP
from fadin.loss_and_gradient import discrete_ll_loss_conv


def identity(x, **param):
    return x


def linear_zero_one(x, **params):
    temp = 2 * x
    mask = x > 1
    temp[mask] = 0.0
    return temp


def reverse_linear_zero_one(x, **params):
    temp = 2 - 2 * x
    mask = x > 1
    temp[mask] = 0.0
    return temp


def truncated_gaussian(x, **params):
    rc = DiscreteKernelFiniteSupport(
        delta=0.01, n_dim=1, kernel="truncated_gaussian"
    )
    mu = params["mu"]
    sigma = params["sigma"]
    kernel_values = rc.kernel_eval(
        [torch.Tensor(mu), torch.Tensor(sigma)], torch.tensor(x)
    )

    return kernel_values.double().numpy()


def evaluate_intensity(baseline, alpha, mean, sigma, delta, events_grid):
    L = int(1 / delta)
    TG = DiscreteKernelFiniteSupport(
        delta, n_dim=1, kernel="truncated_gaussian", lower=0, upper=1
    )

    intens = TG.intensity_eval(
        torch.tensor(baseline),
        torch.tensor(alpha),
        [torch.Tensor(mean), torch.Tensor(sigma)],
        events_grid,
        torch.linspace(0, 1, L),
    )
    return intens


def simulate_data(baseline, baseline_noise, alpha, delta, end_time, seed=0):

    n_dim = len(baseline)

    time_kernel = truncated_gaussian
    params_time_kernel = dict(mu=mu, sigma=sigma)

    marks_kernel = identity
    marks_density = linear_zero_one
    time_kernel = truncated_gaussian

    params_marks_density = dict(scale=1)
    params_marks_kernel = dict(slope=1.2)
    params_time_kernel = dict(mu=mu, sigma=sigma)

    marked_events, _ = simu_marked_hawkes_cluster(
        end_time,
        baseline,
        alpha,
        time_kernel,
        marks_kernel,
        marks_density,
        params_time_kernel=params_time_kernel,
        params_marks_kernel=params_marks_kernel,
        params_marks_density=params_marks_density,
        time_kernel_length=None,
        marks_kernel_length=None,
        upper_bound=None,
        random_state=None,
    )

    noisy_events_ = simu_multi_poisson(end_time, baseline_noise)

    random_marks = [
        np.random.rand(noisy_events_[i].shape[0]) / 5.0 for i in range(n_dim)
    ]

    noisy_events = [
        np.concatenate(
            (noisy_events_[i].reshape(-1, 1), random_marks[i].reshape(-1, 1)),
            axis=1
        )
        for i in range(n_dim)
    ]
    # marked Hawkes concatenated with marked Poisson
    mev_cat = [
        np.concatenate((noisy_events[i], marked_events[i]), axis=0)
        for i in range(n_dim)
    ]

    # marked events concatenated and sorted
    mev_cat_sorted = [mev_cat[i][mev_cat[i][:, 0].argsort(0)]
                      for i in range(n_dim)]
    # events concatenated and sorted
    ev_cat_sorted = [mev_cat_sorted[i][:, 0] for i in range(n_dim)]

    L = int(1 / delta)
    events_grid = projected_grid(ev_cat_sorted, delta, L * end_time + 1)
    _, events_grid_clean = projected_grid_marked(
        marked_events, delta, L * end_time + 1
    )
    intens = evaluate_intensity(baseline, alpha, mu, sigma, delta, events_grid)

    return (
        ev_cat_sorted,
        mev_cat_sorted,
        intens,
        events_grid,
        marked_events,
        events_grid_clean,
    )


def run_vb(S, events_grid_test, size_grid, dt, seed):
    np.random.seed(seed)
    init_len = size_grid
    start = time.time()
    init_model = DiscreteTimeStandardHawkesModel(
        K=1, dt=dt, dt_max=1, B=5, alpha=1.0, beta=1.0
    )
    init_model.add_data(S[:init_len, :])

    init_model.initialize_to_background_rate()
    init_model.fit_with_bfgs()

    ###########################################################
    # Create a test weak spike-and-slab model
    ###########################################################

    test_network = StochasticBlockModel(K=1, C=1)
    test_model = DiscreteTimeNetworkHawkesModelGammaMixture(
        K=1, dt=dt, dt_max=1, B=5, network=test_network
    )
    test_model.add_data(S)

    # Initialize with the standard model parameters
    if init_model is not None:
        test_model.initialize_with_standard_model(init_model)

    ###########################################################
    # Fit the test model with variational Bayesian inference
    ###########################################################
    # VB coordinate descent

    N_iters = 1000
    vlbs = []
    samples = []
    for itr in range(N_iters):
        vlbs.append(test_model.meanfield_coordinate_descent_step())

        # Resample from variational distribution and plot
        test_model.resample_from_mf()
        samples.append(test_model.copy_sample())

    results = dict(
        intens_vb=test_model.compute_rate(),
        ll=test_model.heldout_log_likelihood(
            events_grid_test.T.numpy().astype(np.int64)
        ),
        time=time.time() - start,
    )

    return results


def run_experiment(baseline, baseline_noise, alpha, end_time, delta, seed):
    ev_cat_sorted, mev_cat_sorted, true_intens, events_grid, \
        marked_events, events_grid_clean = simulate_data(
            baseline,
            baseline_noise,
            alpha,
            delta,
            end_time=end_time,
            seed=seed
        )
    ev_cat_sorted_test, mev_cat_sorted_test, true_intens_test, _, \
        marked_events_test, events_grid_clean_test = \
        simulate_data(
            baseline,
            baseline_noise,
            alpha,
            delta,
            end_time=end_time,
            seed=seed
        )
    print(events_grid.sum())
    print(events_grid_clean_test.sum())
    true_LL = discrete_ll_loss_conv(true_intens_test, events_grid_clean_test,
                                    delta, end_time)

    ##########################################################
    max_iter = 10_000
    start = time.time()
    solver = UNHaP(
        n_dim=1,
        kernel="truncated_gaussian",
        kernel_length=1.0,
        delta=delta,
        optim="RMSprop",
        params_optim={"lr": 1e-3},
        max_iter=max_iter,
        batch_rho=100,
        init='moment_matching_mean'
    )

    solver.fit(mev_cat_sorted, end_time)
    comp_time_mix = time.time() - start
    baseline_mix = solver.param_baseline[-10:].mean().item()
    alpha_mix = solver.param_alpha[-10:].mean().item()
    mu_mix = solver.param_kernel[0][-10:].mean().item()
    sigma_mix = solver.param_kernel[1][-10:].mean().item()
    print(baseline_mix)
    print(alpha_mix)
    print(mu_mix)
    print(sigma_mix)
    mixture_intens = evaluate_intensity(
        [baseline_mix],
        [[alpha_mix]],
        [[mu_mix]],
        [[sigma_mix]],
        delta,
        events_grid_clean_test,
    )

    err_mixture = np.absolute(
        mixture_intens.numpy() - true_intens.numpy()
    ).mean()

    ll_mixture = discrete_ll_loss_conv(
        mixture_intens, events_grid_clean_test, delta, end_time
    )

    results = dict(
        err_mixture=err_mixture,
        comp_time_mixture=comp_time_mix,
        ll_mixture=ll_mixture.item(),
    )

    ##########################################################
    start = time.time()
    # StocUNHaP
    solver = UNHaP(
        n_dim=1,
        kernel="truncated_gaussian",
        kernel_length=1.0,
        delta=delta,
        optim="RMSprop",
        params_optim={"lr": 1e-3},
        max_iter=max_iter,
        batch_rho=100,
        init='moment_matching_mean',
        stoc_classif=True
    )
    solver.fit(mev_cat_sorted, end_time)
    comp_time_stocunhap = time.time() - start
    baseline_stocunhap = solver.param_baseline[-10:].mean().item()
    alpha_stocunhap = solver.param_alpha[-10:].mean().item()
    mu_stocunhap = solver.param_kernel[0][-10:].mean().item()
    sigma_stocunhap = solver.param_kernel[1][-10:].mean().item()
    print(baseline_stocunhap)
    print(alpha_stocunhap)
    print(mu_stocunhap)
    print(sigma_stocunhap)
    stocunhap_intens = evaluate_intensity(
        [baseline_stocunhap],
        [[alpha_stocunhap]],
        [[mu_stocunhap]],
        [[sigma_stocunhap]],
        delta,
        events_grid_clean_test,
    )

    err_stocunhap = np.absolute(
        stocunhap_intens.numpy() - stocunhap_intens.numpy()
    ).mean()

    ll_stocunhap = discrete_ll_loss_conv(
        stocunhap_intens, events_grid_clean_test, delta, end_time
    )
    results["err_stocunhap"] = err_stocunhap
    results["comp_time_stocunhap"] = comp_time_stocunhap
    results["ll_stocunhap"] = ll_stocunhap.item()

    results["end_time"] = end_time
    results["baseline_noise"] = baseline_noise.item()

    ##########################################################
    start = time.time()
    solver = FaDIn(
        n_dim=1,
        kernel="truncated_gaussian",
        kernel_length=1.0,
        delta=delta,
        optim="RMSprop",
        params_optim={"lr": 1e-3},
        max_iter=max_iter
    )
    # Change the mark to one to use FaDIn
    mev_cat_sorted_ = [mev_cat_sorted[i].copy() for i in range(1)]
    for i in range(1):
        mev_cat_sorted_[i][:, 1] = 1.0
    solver.fit(mev_cat_sorted_, end_time)
    comp_time_fadin = time.time() - start
    baseline_fadin = solver.param_baseline[-10:].mean().item()
    alpha_fadin = solver.param_alpha[-10:].mean().item()
    mu_fadin = solver.param_kernel[0][-10:].mean().item()
    sigma_fadin = solver.param_kernel[1][-10:].mean().item()
    print(baseline_fadin)
    print(alpha_fadin)
    print(mu_fadin)
    print(sigma_fadin)
    fadin_intens = evaluate_intensity(
        [baseline_fadin],
        [[alpha_fadin]],
        [[mu_fadin]],
        [[sigma_fadin]],
        delta,
        events_grid_clean_test,
    )

    err_fadin = np.absolute(fadin_intens.numpy() - true_intens.numpy()).mean()

    ll_fadin = discrete_ll_loss_conv(
        fadin_intens, events_grid_clean_test, delta, end_time
    )
    results["err_fadin"] = err_fadin
    results["comp_time_fadin"] = comp_time_fadin
    results["ll_fadin"] = ll_fadin.item()
    results["end_time"] = end_time
    results["baseline_noise"] = baseline_noise.item()

    ##########################################################
    S = events_grid.T.numpy().astype(np.int64)
    size_grid = events_grid.shape[1]
    dic_res = run_vb(S, events_grid_clean_test, size_grid, delta, seed)

    results["err_vb"] = np.absolute(
        true_intens.numpy() - dic_res["intens_vb"].T
    ).mean()
    results["ll_vb"] = -dic_res["ll"] / end_time
    results["comp_time_vb"] = dic_res["time"]
    ##########################################################
    # TRIPP
    ##########################################################
    events_format = []
    mean = 0
    n_dim = 1
    for i in range(n_dim):
        events_format.append({"arrival_times": ev_cat_sorted[i]})
        mean += ev_cat_sorted[i].shape[0]
    mean /= n_dim
    data = {"sequences": [], "t_max": end_time, "mean_number_items": mean}
    data["sequences"] = events_format

    dset = load_dataset("None", data)
    for i in range(n_dim):
        events_format.append({"arrival_times": ev_cat_sorted_test[i]})
        mean += ev_cat_sorted_test[i].shape[0]
    mean /= n_dim
    data = {"sequences": [], "t_max": end_time, "mean_number_items": mean}
    data["sequences"] = events_format

    dset_test = load_dataset("None", data)
    d_train, _, _ = dset.train_val_test_split(
        train_size=1, val_size=0, test_size=0
    )
    start = time.time()
    tripp = train_tripp(
        d_train, d_train,
        T=end_time,
        n_knots=20,
        block_size=16,
        n_blocks=4,
        n_iter=500
    )
    comp_time_tripp = time.time() - start
    start = time.time()
    rnn = train_rnn(
        d_train, d_train, T=end_time, n_knots=20, hidden_size=32, n_iter=1000
    )
    comp_time_rnn = time.time() - start
    d_test = torch.utils.data.DataLoader(
        dset_test, batch_size=1, shuffle=False
    )

    for x, mask in d_test:
        nll_loss_tripp = -(tripp.log_prob(x, mask) / end_time)

    for x, mask in d_test:
        nll_loss_rnn = -(rnn.log_prob(x, mask) / end_time)

    results["ll_tripp"] = nll_loss_tripp.detach().item()
    results["ll_rnn"] = nll_loss_rnn.detach().item()

    results["comp_time_tripp"] = comp_time_tripp
    results["comp_time_rnn"] = comp_time_rnn

    results["true_ll"] = true_LL.item()
    return results


baseline = np.array([0.1])
mu = np.array([[0.5]])
sigma = np.array([[0.1]])
alpha = np.array([[1.0]])
delta = 0.01
end_time_list = [100, 500, 1_000]
baseline_noise_list = [np.array([1.0])]
seeds = np.arange(10)


n_jobs = 70
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, baseline_noise,
                            alpha, end_time,
                            delta, seed=seed)
    for end_time, baseline_noise, seed in itertools.product(
        end_time_list, baseline_noise_list, seeds
    )
)

# save results
df = pd.DataFrame(all_results)
df.to_csv(
    f"results/benchmark_marked_bl_noise{baseline_noise_list}.csv",
    index=False
)
