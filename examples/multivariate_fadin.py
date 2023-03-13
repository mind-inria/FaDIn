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

import torch

n_dim = 2
dt = 0.01
T = 1_000_000
kernel_length = 1
L = int(1 / dt)
size_grid = int(T / dt) + 1
discretization = torch.linspace(0, kernel_length, L)

###############################################################################
# Here, we set the parameters of a Hawkes process with a Truncated Gaussian kernel

import numpy as np

baseline = np.array([.1, .8])
alpha = np.array([[0.4, 0.5], [0.5, 0.4]])
m = np.array([[0.4, 0.6], [0.55, 0.6]])
sigma = np.array([[0.3, 0.3], [0.25, 0.3]])


###############################################################################
# Here, we simulate the data

from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc

from fadin.kernels import DiscreteKernelFiniteSupport


kernel = DiscreteKernelFiniteSupport(dt, n_dim, kernel='truncated_gaussian')

kernel_values = kernel.kernel_eval([torch.Tensor(m), torch.Tensor(sigma)],
                                   discretization)
kernel_values = kernel_values * alpha[:, :, None]

discretization_np = discretization.double().numpy()

tf = []
for i in range(n_dim):
    for j in range(n_dim):
        k = kernel_values[i, j].double().numpy()
        tf.append(HawkesKernelTimeFunc(t_values=discretization_np, y_values=k))

kernels = [[tf[i+j*2] for i in range(n_dim)] for j in range(n_dim)]

hawkes = SimuHawkes(
    baseline=baseline, kernels=kernels, end_time=T, verbose=False, seed=0
)

hawkes.simulate()
events = hawkes.timestamps

###############################################################################
# Here, we visualize the kernel shape

import matplotlib.pyplot as plt

plt.figure()
for i in range(n_dim):
    for j in range(n_dim):
        plt.plot(discretization_np, kernel_values[i, j], label=f'{i},{j}')
plt.legend()
plt.show()


###############################################################################
# Here, we apply FaDIn

from fadin.solver import FaDIn

solver = FaDIn(n_dim=n_dim,
               kernel="truncated_gaussian",
               kernel_length=kernel_length,
               delta=dt, optim="RMSprop",
               params_optim={'lr': 1e-3},
               max_iter=10000, criterion='l2'
               )
solver.fit(events, T)

# We average on the 10 last values of the optimization
estimated_baseline = solver.param_baseline[-10:].mean(0)
estimated_alpha = solver.param_alpha[-10:].mean(0)
param_kernel = [solver.param_kernel[0][-10:].mean(0),
                solver.param_kernel[1][-10:].mean(0)]

print('Estimated baseline is:', estimated_baseline)
print('Estimated baseline is:', estimated_alpha)
print('Estimated m parameter of the truncated Gaussian kernel is:', param_kernel[0])
print('Estimated sigma parameter of the truncated Gaussian kernel is:', param_kernel[1])

# %%
