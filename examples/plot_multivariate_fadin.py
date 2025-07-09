"""
==============================================
FaDIn on simulated 2-d Hawkes processes
==============================================

This example demonstrates inference performed
by FaDIn on multivariate Hawkes processes simulated
with specific kernels.
"""

# Authors: Guillaume Staerman <guillaume.staerman@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>
#
# License: MIT

###############################################################################
# Let us first define the parameters of our model.

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from fadin.utils.utils_simu import simu_hawkes_cluster
from fadin.solver import FaDIn
from fadin.kernels import DiscreteKernelFiniteSupport

n_dim = 2
dt = 0.01
T = 1_000_000
kernel_length = 5
L = int(1 / dt)
size_grid = int(T / dt) + 1
discretization = torch.linspace(0, kernel_length, L)

###############################################################################
# Here, we set the parameters of a Hawkes process with a Exponential(1)
# distribution.

baseline = np.array([.1, .5])
alpha = np.array([[0.6, 0.3], [0.25, 0.7]])

###############################################################################
# Here, we simulate the data

kernel = 'expon'

events = simu_hawkes_cluster(T, baseline, alpha, kernel)

###############################################################################
# Here, we initiate FaDIn and fit it to the simulated data.

solver = FaDIn(
    n_dim=n_dim,
    kernel="truncated_exponential",
    kernel_length=kernel_length,
    delta=dt, optim="RMSprop",
    params_optim={'lr': 1e-3},
    max_iter=10000
)
solver.fit(events, T)

# We can now access the estimated parameters of the model.

estimated_baseline = solver.param_baseline_[-10:].mean(0)
estimated_alpha = solver.param_alpha_[-10:].mean(0)
param_kernel = [solver.param_kernel_[0][-10:].mean(0)]

print('Estimated baseline is:', solver.baseline_)
print('Estimated alpha is:', solver.alpha_)
print('Estimated parameters of the truncated Exponential kernel is:',
      solver.kernel_)

###############################################################################
# Here, we plot the values of the estimated kernels with FaDIn.

kernel = DiscreteKernelFiniteSupport(dt, n_dim, kernel='truncated_exponential',
                                     kernel_length=kernel_length)
kernel_values = kernel.kernel_eval(solver.kernel_, discretization)

plt.subplots(figsize=(12, 8))
for i in range(n_dim):
    for j in range(n_dim):
        plt.subplot(n_dim, n_dim, i*n_dim + j + 1)
        plt.plot(discretization[1:], kernel_values[i, j, 1:]/kernel_length)
        plt.plot(discretization[1:], torch.exp(-discretization[1:]), c='k')
        plt.ylabel(rf'$\phi_{{{i}{j}}}$', size=20)

black_patch = mpatches.Patch(facecolor='k')
blue_patch = mpatches.Patch(facecolor='C0')
labels = ['FaDIn\' estimated kernel', 'True kernel']
colors = [blue_patch, black_patch]
plt.legend(handles=colors, labels=labels, loc="center right",
           borderaxespad=0.1, fontsize='x-large')
plt.suptitle('Hawkes influence kernels', size=20)
plt.show()
