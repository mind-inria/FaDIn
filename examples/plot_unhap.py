# %% Imports
import numpy as np
import matplotlib.pyplot as plt

from fadin.utils.utils_simu import simu_marked_hawkes_cluster, custom_density
from fadin.utils.utils_simu import simu_multi_poisson
from fadin.solver import UNHaP
from fadin.utils.functions import identity, linear_zero_one
from fadin.utils.functions import reverse_linear_zero_one, truncated_gaussian
from fadin.utils.vis import plot


# %% Fixing the parameter of the simulation setting

baseline = np.array([.3])
baseline_noise = np.array([.05])
alpha = np.array([[1.45]])
mu = np.array([[0.5]])
sigma = np.array([[0.1]])

delta = 0.01
end_time = 10000
seed = 0
max_iter = 20000
batch_rho = 200

# %% Create the simulating function


def simulate_data(baseline, baseline_noise, alpha, end_time, seed=0):
    n_dim = len(baseline)

    marks_kernel = identity
    marks_density = linear_zero_one
    time_kernel = truncated_gaussian

    params_marks_density = dict()
    # params_marks_density = dict(scale=1)
    params_marks_kernel = dict(slope=1.2)
    params_time_kernel = dict(mu=mu, sigma=sigma)

    marked_events, _ = simu_marked_hawkes_cluster(
        end_time, baseline, alpha, time_kernel, marks_kernel, marks_density,
        params_marks_kernel=params_marks_kernel,
        params_marks_density=params_marks_density,
        time_kernel_length=None, marks_kernel_length=None,
        params_time_kernel=params_time_kernel, random_state=seed)

    noisy_events_ = simu_multi_poisson(end_time, [baseline_noise])

    # random_marks = [
    #     np.random.rand(noisy_events_[i].shape[0]) for i in range(n_dim)]
    noisy_marks = [custom_density(
                    reverse_linear_zero_one, dict(), size=noisy_events_[i].shape[0],
                    kernel_length=1.) for i in range(n_dim)]
    noisy_events = [
        np.concatenate((noisy_events_[i].reshape(-1, 1),
                        noisy_marks[i].reshape(-1, 1)), axis=1) for i in range(n_dim)]

    events = [
        np.concatenate(
            (noisy_events[i], marked_events[i]), axis=0) for i in range(n_dim)]

    events_cat = [events[i][events[i][:, 0].argsort()] for i in range(n_dim)]

    labels = [np.zeros(marked_events[i].shape[0]
              + noisy_events_[i].shape[0]) for i in range(n_dim)]
    labels[0][-marked_events[0].shape[0]:] = 1.
    true_rho = [labels[i][events[i][:, 0].argsort()] for i in range(n_dim)]
    # put the mark to one to test the impact of the marks
    # events_cat[0][:, 1] = 1.

    return events_cat, noisy_marks, true_rho


ev, noisy_marks, true_rho = simulate_data(baseline, baseline_noise.item(),
                                          alpha, end_time, seed=0)
# %% Apply UNHAP

solver = UNHaP(n_dim=1,
               kernel="truncated_gaussian",
               kernel_length=1.,
               delta=delta, optim="RMSprop",
               params_optim={'lr': 1e-3},
               max_iter=max_iter,
               batch_rho=batch_rho,
               density_hawkes='linear',
               density_noise='reverse_linear',
               moment_matching=True
               )
solver.fit(ev, end_time)

# %% Print estimated parameters

print('Estimated baseline is: ', solver.param_baseline[-10:].mean().item())
print('Estimated alpha is: ', solver.param_alpha[-10:].mean().item())
print('Estimated kernel mean is: ', (solver.param_kernel[0][-10:].mean().item()))
print('Estimated kernel sd is: ', solver.param_kernel[1][-10:].mean().item())
print('Estimated noise baseline is: ', solver.param_baseline_noise[-10:].mean().item())
# error on params
error_baseline = (solver.param_baseline[-10:].mean().item() - baseline.item()) ** 2
error_alpha = (solver.param_alpha[-10:].mean().item() - alpha.item()) ** 2
error_mu = (solver.param_kernel[0][-10:].mean().item() - 0.5) ** 2
error_sigma = (solver.param_kernel[1][-10:].mean().item() - 0.1) ** 2
sum_error = error_baseline + error_alpha + error_mu + error_sigma
error_params = np.sqrt(sum_error)

print('L2 square errors of the vector of parameters is:', error_params)
# %% Plot estimated parameters
fig, axs = plot(
    solver,
    plotfig=False,
    bl_noise=True,
    title='UNHaP fit',
    savefig=None
)
plt.show(block=True)
# %%
