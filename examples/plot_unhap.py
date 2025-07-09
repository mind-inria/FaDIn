"""
==============================================
UNHaP on simulated univariate Hawkes processes
==============================================

This example demonstrates inference performed
by UNHaP on univariate Hawkes processes simulated
with a truncated gaussian kernel.
"""

# Authors: Guillaume Staerman <guillaume.staerman@inria.fr>
#          Virginie Loison <virginie.loison@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>
#
# License: MIT

###############################################################################
# Let us first define the parameters of our model.
# %% Imports
import numpy as np
import matplotlib.pyplot as plt

from fadin.utils.utils_simu import simulate_marked_data
from fadin.solver import UNHaP
from fadin.utils.vis import plot


# %% Fix the simulation and solver parameters

baseline = np.array([0.3])
baseline_noise = np.array([0.05])
alpha = np.array([[1.45]])
mu = np.array([[0.4]])
sigma = np.array([[0.1]])

delta = 0.01
end_time = 1000
seed = 0
max_iter = 2000
batch_rho = 200

# %% Simulate Hawkes Process with truncated Gaussian kernel and Poisson noise

ev, noisy_marks, true_rho = simulate_marked_data(
    baseline, baseline_noise.item(), alpha, end_time, mu, sigma, seed=0
)
# %% Initiate and fit UNHAP to the simulated events

solver = UNHaP(
    n_dim=1,
    kernel="truncated_gaussian",
    kernel_length=1.0,
    init='moment_matching_mean',
    delta=delta,
    optim="RMSprop",
    params_optim={"lr": 1e-3},
    max_iter=max_iter,
    batch_rho=batch_rho,
    density_hawkes="linear",
    density_noise="uniform",
)
solver.fit(ev, end_time)

# %% Print estimated parameters

print("Estimated baseline is: ", solver.param_baseline[-10:].mean().item())
print("Estimated alpha is: ", solver.param_alpha[-10:].mean().item())
print("Estimated kernel mean is: ", (solver.param_kernel[0][-10:].mean().item()))
print("Estimated kernel sd is: ", solver.param_kernel[1][-10:].mean().item())
print("Estimated noise baseline is: ", solver.param_baseline_noise[-10:].mean().item())
# error on params
error_baseline = (solver.param_baseline[-10:].mean().item() - baseline.item()) ** 2
error_baseline_noise = (
    solver.param_baseline_noise[-10:].mean().item() - baseline_noise.item()
) ** 2
error_alpha = (solver.param_alpha[-10:].mean().item() - alpha.item()) ** 2
error_mu = (solver.param_kernel[0][-10:].mean().item() - 0.5) ** 2
error_sigma = (solver.param_kernel[1][-10:].mean().item() - 0.1) ** 2
sum_error = error_baseline + error_baseline_noise + error_alpha + error_mu + error_sigma
error_params = np.sqrt(sum_error)

print("L2 square error of the vector of parameters is:", error_params)

# %% Plot estimated parameters
fig, axs = plot(
    solver,
    plotfig=False,
    bl_noise=True,
    title="UNHaP fit",
    savefig=None
)
plt.show(block=True)
