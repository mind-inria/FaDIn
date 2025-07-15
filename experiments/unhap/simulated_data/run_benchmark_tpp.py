# %% Import libraries
import itertools
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import torch

from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.loss_and_gradient import discrete_ll_loss_conv
from fadin.utils.utils import optimizer_fadin, projected_grid_marked
from fadin.utils.utils import projected_grid
from fadin.utils.utils_simu import simu_marked_hawkes_cluster
from fadin.utils.utils_simu import simu_multi_poisson, custom_density
from fadin.utils.functions import identity, linear_zero_one
from fadin.utils.functions import reverse_linear_zero_one


class TPPSelect():
    def __init__(self, n_dim, delta=0.01, k=0.3, n_candidates=10, reg=0):
        self.n_dim = n_dim
        self.k = k
        self.n_candidates = n_candidates
        self.reg = reg
        self.delta = delta

        self.kernel = DiscreteKernelFiniteSupport(
            delta, n_dim=self.n_dim, kernel="truncated_gaussian",
            lower=0, kernel_length=1.0
        )
        self.ker_grid = torch.arange(0, 1.00001, delta)
        self.L = len(self.ker_grid)

    def intensity(self, events_grid):
        """Compute the intensity of the Hawkes process on a grid.

        Parameters:
        -----------
        events_grid : torch.Tensor, shape (n_dim, n_grid)
            History of the events on a grid as a sparse vector.

        """
        n_dim, n_grid = events_grid.shape

        alpha = self.params[n_dim:n_dim*(n_dim+1)]
        kernel_params = self.params[n_dim*(n_dim+1):].reshape(2, 1, 1)

        # Evaluate of the kernel on the grid, shape (n_dim, n_dim, ker_grid)
        kernel_val = self.kernel.kernel_eval(kernel_params, self.ker_grid)
        kernel_val = kernel_val * alpha.reshape(n_dim, n_dim, 1)

        return torch.conv_transpose1d(
            events_grid, kernel_val.transpose(0, 1).float()
        )[:, :-self.L + 1]

    def loss(self, events_grid, rho):
        n_dim, n_grid = events_grid.shape

        intensity = self.intensity(events_grid)
        n_events = events_grid.sum(1)
        n_clustered = rho.sum(1)

        baseline = self.params[:n_dim]
        integral = torch.norm(
            intensity + baseline[:, None], p='fro'
        )**2 * self.delta
        contrib_cluster = torch.dot(
            (events_grid * rho).ravel(), intensity.ravel()
        )
        contrib_exo = baseline * (n_events - n_clustered)
        return integral - 2 * (contrib_cluster + contrib_exo)

    def inner_min(self, events_grid, rho, max_iter=20):
        self.opt = optimizer_fadin(
            [self.params], {'lr': 1e-3}, solver='RMSprop'
        )
        for i in range(20):
            self.opt.zero_grad()
            loss = self.loss(events_grid, rho)
            loss.backward()
            self.opt.step()
        return self.loss(events_grid, rho)

    def fit(self, events, end_time):

        n_events = len(events[0])
        k = self.k if isinstance(self.k, int) else int(self.k * n_events)
        print(f"Looking for {n_events - k} structured events.")

        n_grid = int(1 / self.delta * end_time) + 1
        events_grid, events_grid_wm = projected_grid_marked(
            events, self.delta, n_grid
        )

        # Smart initialization of solver parameters
        n_dim = self.n_dim
        baseline = n_events / (end_time * (n_dim + 1))
        alpha = 1 / (n_dim + 1)
        kernel_params = torch.ones(2) * 0.5
        self.params = torch.tensor([
            baseline, alpha, *kernel_params
        ]).requires_grad_(True)

        rho = (events_grid > 0).float()
        self.inner_min(events_grid_wm, rho, max_iter=200)
        for i in range(k):
            print(f"Fitting...{(i+1) / k:6.1%}\r", end='', flush=True)
            candidate_events = torch.multinomial(
                rho[0], self.n_candidates, replacement=False
            )
            F, idx_select = torch.inf, -1
            for idx in candidate_events:
                rho[0, idx] = 0
                LL = self.inner_min(events_grid_wm, rho)
                rho[0, idx] = 1
                if LL < F:
                    F, idx_select = LL, idx
            rho[0, idx_select] = 0

        self.param_baseline = self.params[:1].detach()
        self.param_alpha = self.params[1:2].detach()
        self.param_kernel = self.params[2:].detach().reshape(2, 1)
        self.param_rho = rho
        print("Fitting...  done")


def truncated_gaussian(x, **params):
    rc = DiscreteKernelFiniteSupport(delta=0.01, n_dim=1,
                                     kernel='truncated_gaussian')
    mu = params['mu']
    sigma = params['sigma']
    kernel_values = rc.kernel_eval(
        [torch.Tensor(mu), torch.Tensor(sigma)], torch.tensor(x))

    return kernel_values.double().numpy()


def evaluate_intensity(baseline, alpha, mean, sigma, delta, events_grid):
    L = int(1 / delta)
    TG = DiscreteKernelFiniteSupport(
        delta,
        n_dim=1,
        kernel='truncated_gaussian',
        lower=0
    )

    intens = TG.intensity_eval(torch.tensor(baseline),
                               torch.tensor([[0.97]]),
                               [torch.Tensor(mean),
                                torch.Tensor(sigma)],
                               events_grid, torch.linspace(0, 1, L))
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
        end_time, baseline, alpha, time_kernel, marks_kernel, marks_density,
        params_time_kernel=params_time_kernel,
        params_marks_kernel=params_marks_kernel,
        params_marks_density=params_marks_density, time_kernel_length=None,
        marks_kernel_length=None, random_state=seed)

    noisy_events_ = simu_multi_poisson(
        end_time, baseline_noise, random_state=seed
    )
    np.random.seed(seed)
    random_marks = [
        np.random.rand(noisy_events_[i].shape[0]) for i in range(n_dim)]
    random_marks = [
        custom_density(
            reverse_linear_zero_one,
            dict(),
            size=noisy_events_[i].shape[0],
            kernel_length=1.
        ) for i in range(n_dim)
    ]

    noisy_events = [
        np.concatenate(
            (noisy_events_[i].reshape(-1, 1), random_marks[i].reshape(-1, 1)),
            axis=1
        ) for i in range(n_dim)
    ]
    # marked Hawkes concatenated with marked Poisson
    mev_cat = [np.concatenate(
            (noisy_events[i], marked_events[i]), axis=0) for i in range(n_dim)]

    # marked events concatenated and sorted
    mev_cat_sorted = [
        mev_cat[i][mev_cat[i][:, 0].argsort(0)] for i in range(n_dim)
    ]
    # events concatenated and sorted
    ev_cat_sorted = [mev_cat_sorted[i][:, 0] for i in range(n_dim)]

    L = int(1 / delta)
    events_grid = projected_grid(ev_cat_sorted, delta, L * end_time + 1)
    _, events_grid_clean = projected_grid_marked(
        marked_events, delta, L * end_time + 1
    )
    intens = evaluate_intensity(baseline, alpha, mu, sigma, delta, events_grid)

    return ev_cat_sorted, mev_cat_sorted, intens, events_grid, \
        marked_events, events_grid_clean


def run_experiment(baseline, baseline_noise, alpha, end_time, delta, mu, sigma,
                   seed):

    ev_cat_sorted, mev_cat_sorted, true_intens, events_grid, \
        marked_events, events_grid_clean = simulate_data(
            baseline,
            baseline_noise,
            alpha,
            delta=delta,
            end_time=end_time,
            seed=seed
        )
    ev_cat_sorted_test, mev_cat_sorted_test, true_intens_test, _, \
        marked_events_test, events_grid_clean_test = \
        simulate_data(
            baseline,
            baseline_noise,
            alpha,
            delta=delta,
            end_time=end_time,
            seed=seed
        )

    true_LL = discrete_ll_loss_conv(true_intens_test, events_grid_clean_test,
                                    delta, end_time)

    start = time.time()

    solver = TPPSelect(n_dim=1, k=0.2, delta=delta)

    solver.fit(mev_cat_sorted, end_time)
    comp_time_mfadin = time.time() - start
    baseline_mfadin = solver.param_baseline[-10:].mean().item()
    alpha_mfadin = solver.param_alpha[-10:].mean().item()
    mu_mfadin = solver.param_kernel[0][-10:].mean().item()
    sigma_mfadin = solver.param_kernel[1][-10:].mean().item()

    mfadin_intens = evaluate_intensity(
        [baseline_mfadin], [[alpha_mfadin]], [[mu_mfadin]],
        [[sigma_mfadin]], delta, events_grid_clean_test)
    err_mfadin = np.absolute(
        mfadin_intens.numpy() - true_intens.numpy()
    ).mean()

    ll_mfadin = discrete_ll_loss_conv(mfadin_intens.clip(1e-5),
                                      events_grid_clean_test,
                                      delta, end_time)

    results = dict(err_mfadin=err_mfadin,
                   comp_time_mfadin=comp_time_mfadin,
                   ll_mfadin=ll_mfadin.item(),
                   true_ll=true_LL.item(),
                   end_time=end_time,
                   baseline_noise=baseline_noise.item(),
                   seed=seed
                   )

    return results


baseline = np.array([.1])
mu = np.array([[0.5]])
sigma = np.array([[0.1]])
alpha = np.array([[1.0]])
delta = 0.01
end_time_list = [100, 500, 1_000]
baseline_noise = [np.array([1.])]
seeds = np.arange(10)


n_jobs = 10  # run with 70 to go faster
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, baseline_noise,
                            alpha, end_time,
                            delta, mu, sigma, seed=seed)
    for end_time, baseline_noise, seed in itertools.product(
        end_time_list, baseline_noise, seeds
    )
)

# save results
df = pd.DataFrame(all_results)
df.to_csv('results/benchmark_rebutall_tpp_select.csv', index=False)

print('\n Mean \n:', df[df['baseline_noise'] == 1.].groupby('end_time').mean())
print('\n std \n:', df[df['baseline_noise'] == 1.].groupby('end_time').std())
