"""
==============================================
FaDIn on simulated univariate Hawkes processes
==============================================

This example demonstrates inference performed
by FaDIn on univariate Hawkes processes simulated
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

n_dim = 1
dt = 0.01
T = 10_000
kernel_length = 1
L = int(1 / dt)
size_grid = int(T / dt) + 1
discretization = torch.linspace(0, kernel_length, L)

###############################################################################
# Here, we set the parameters of a Hawkes process with a Raised Cosine kernel

import numpy as np

baseline = np.array([.1])
alpha = np.array([[0.8]])
u = np.array([[.2]])
sigma = np.array([[.3]])


###############################################################################
# Here, we simulate the data

from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc

from fadin.kernels import DiscreteKernelFiniteSupport

kernel = DiscreteKernelFiniteSupport(dt, n_dim, kernel='raised_cosine')
kernel_values = kernel.kernel_eval([torch.Tensor(u), torch.Tensor(sigma)],
                                   discretization)
kernel_values = kernel_values * alpha[:, :, None]

discretization_np = discretization.double().numpy()
kernel_values_np = kernel_values.squeeze().double().numpy()

tf = HawkesKernelTimeFunc(t_values=discretization_np, y_values=kernel_values_np)
kernels = [[tf]]
hawkes = SimuHawkes(
    baseline=baseline, kernels=kernels, end_time=T, verbose=False, seed=0
)

hawkes.simulate()
events = hawkes.timestamps

###############################################################################
# Here, we visualize the kernel shape

import matplotlib.pyplot as plt

plt.figure()
plt.plot(discretization_np, kernel_values_np)
plt.show()

###############################################################################
# Here, we apply FaDIn

from fadin.solver import FaDIn

solver = FaDIn(n_dim=1,
               kernel="raised_cosine",
               kernel_length=kernel_length,
               delta=dt, optim="RMSprop",
               step_size=1e-3, max_iter=10000, criterion='l2'
               )
results = solver.fit(events, T)

# We average on the 10 last values of the optimization
estimated_baseline = results['param_baseline'][-10:].mean().item()
estimated_alpha = results['param_alpha'][-10:].mean().item()
param_kernel = [results['param_kernel'][0][-10:].mean().item(),
                results['param_kernel'][1][-10:].mean().item()]

print('Estimated baseline is:', estimated_baseline)
print('Estimated baseline is:', estimated_alpha)
print('Estimated u parameter of the raised cosine kernel is:', param_kernel[0])
print('Estimated sigma parameter of the raised cosine kernel is:', param_kernel[1])
