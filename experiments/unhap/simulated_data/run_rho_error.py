# %% import stuff
import numpy as np
import itertools
from joblib import Parallel, delayed
import pandas as pd
import torch
import time
from sklearn.metrics import precision_score, recall_score

from fadin.kernels import DiscreteKernelFiniteSupport

from unhap.utils.utils import smooth_projection_marked
from unhap.utils.utils_simu import simu_marked_hawkes_cluster, simu_multi_poisson
from unhap.solver import UNHaP


def simulate_data(baseline, baseline_noise, alpha, end_time, seed=0):
    n_dim = len(baseline)

    def identity(x, **param):
        return x

    def linear_zero_one(x, **params):
        temp = 2 * x
        mask = x > 1
        temp[mask] = 0.
        return temp

    def truncated_gaussian(x, **params):
        rc = DiscreteKernelFiniteSupport(delta=0.01, n_dim=1,
                                         kernel='truncated_gaussian')
        mu = params['mu']
        sigma = params['sigma']
        kernel_values = rc.kernel_eval(
            [torch.Tensor(mu), torch.Tensor(sigma)], torch.tensor(x))

        return kernel_values.double().numpy()

    marks_kernel = identity
    marks_density = linear_zero_one
    time_kernel = truncated_gaussian

    params_marks_density = dict()
    # params_marks_density = dict(scale=1)
    params_marks_kernel = dict(slope=1.2)
    params_time_kernel = dict(mu=mu, sigma=sigma)

    marked_events, y_true = simu_marked_hawkes_cluster(
        end_time, baseline, alpha, time_kernel, marks_kernel, marks_density,
        params_marks_kernel=params_marks_kernel,
        params_marks_density=params_marks_density,
        time_kernel_length=None, marks_kernel_length=None, upper_bound=None,
        params_time_kernel=params_time_kernel, random_state=seed)

    noisy_events_ = simu_multi_poisson(end_time, baseline_noise)

    random_marks = [
        np.random.rand(noisy_events_[i].shape[0]) for i in range(n_dim)]
    noisy_events = [
        np.concatenate((noisy_events_[i].reshape(-1, 1),
                        random_marks[i].reshape(-1, 1)), axis=1) for i in range(n_dim)]

    events = [
        np.concatenate(
            (noisy_events[i], marked_events[i]), axis=0) for i in range(n_dim)]

    events_cat = [events[i][events[i][:, 0].argsort()] for i in range(n_dim)]
    n_grid = int(1 / 0.01) * end_time + 1
    events_grid, _, marked_events_smooth, index_unique = \
        smooth_projection_marked(events_cat, 0.01, n_grid)

    ##########################################
    # Compute rho for the original events
    labels = []
    false_events = np.where(y_true == 0)[0]
    for i in range(n_dim):
        a = marked_events[i].shape[0]
        b = noisy_events_[i].shape[0]
        labels_i = np.zeros(a+b)
        labels_i[b:] = 1.
        labels_i[false_events] = 0.
        labels.append(labels_i)
    true_rho = [labels[i][events[i][:, 0].argsort()] for i in range(n_dim)]

    # select the rho for the smoothed events
    true_rho_smooth = [torch.tensor(
        true_rho[i][index_unique]).float() for i in range(n_dim)]
    loc_events = torch.where(events_grid > 0)
    rho_star = events_grid.clone()
    rho_star[loc_events] = torch.vstack(true_rho_smooth)

    return marked_events_smooth, rho_star, loc_events, labels, a


def run_experiment(baseline, baseline_noise, alpha, end_time, delta, seed):

    marked_events_smooth, rho_star, loc_events, _, _ = simulate_data(
        baseline, baseline_noise, alpha, end_time=end_time, seed=seed)
    start = time.time()
    max_iter = 10000
    solver = UNHaP(
        n_dim=1,
        kernel="truncated_gaussian",
        kernel_length=1.,
        delta=delta, optim="RMSprop",
        params_optim={'lr': 1e-3},
        max_iter=max_iter, criterion='l2',
        optimize_kernel=True,
        optimize_alpha=True, optimize_rho=True,
        batch_rho=200,
    )

    solver.fit(marked_events_smooth, end_time)
    comp_time = time.time() - start
    mean_rho_error = torch.abs(
        solver.param_rho[loc_events] - rho_star[loc_events]).mean()

    y_pred = solver.param_rho[loc_events]

    y_true = rho_star[loc_events]

    rec_score = recall_score(y_true, y_pred)
    pr_score = precision_score(y_true, y_pred)

    results = dict(mean_rho_error=mean_rho_error.item(),
                   rec_score=rec_score,
                   pr_score=pr_score)

    results["seed"] = seed
    results['alpha'] = alpha.item()
    results["end_time"] = end_time
    results["delta"] = delta
    results["noise"] = baseline_noise.item()
    results["n_events"] = marked_events_smooth[0].shape[0]
    results["comp_time"] = comp_time

    return results


baseline = np.array([.4])
mu = np.array([[0.5]])
sigma = np.array([[0.1]])
delta = 0.01

end_time_list = [100, 1000, 10_000]
alpha_list = [0, 0.2,  0.3, 0.5, 0.7,  0.8, 0.9, 1.1, 1.2, 1.3, 1.47]
baseline_noise_list = [np.array([0.1]), np.array([0.5]), np.array([1.])]
seeds = np.arange(100)

n_jobs = 70
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, baseline_noise,
                            np.array([[alpha]]), end_time,
                            delta, seed=seed)
    for end_time, alpha, baseline_noise, seed in itertools.product(
        end_time_list, alpha_list, baseline_noise_list, seeds
    )
)

# save results
df = pd.DataFrame(all_results)

df.to_csv('results/error_rho_infer_5000.csv', index=False)
