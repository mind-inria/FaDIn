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
# %%
import numpy as np
import torch

from fadin.utils.utils_simu import simu_hawkes_cluster
from fadin.solver import FaDIn

n_dim = 1
dt = 0.01
T = 1_000_000
kernel_length = 5
L = int(1 / dt)
size_grid = int(T / dt) + 1
discretization = torch.linspace(0, kernel_length, L)

###############################################################################
# Here, we set the parameters of a Hawkes process with a Raised Cosine kernel

baseline = np.array([.4])
alpha = np.array([[0.8]])


# %%
###############################################################################
# Here, we simulate the data

# standard parameter is beta equal to one
kernel = 'expon' 

events = simu_hawkes_cluster(T, baseline, alpha, kernel)

###############################################################################
# Here, we apply FaDIn

solver = FaDIn(n_dim=1,
               kernel="truncated_exponential",
               kernel_length=kernel_length,
               delta=dt, optim="RMSprop",
               params_optim={'lr': 1e-3},
               max_iter=10000, criterion='l2'
               )
solver.fit(events, T)

# We average on the 10 last values of the optimization
estimated_baseline = solver.param_baseline[-10:].mean().item()
estimated_alpha = solver.param_alpha[-10:].mean().item()
param_kernel = [solver.param_kernel[0][-10:].mean().item()]

print('Estimated baseline is:', estimated_baseline)
print('Estimated alpha is:', estimated_alpha)
print('Estimated beta parameter of the exponential kernel is:', param_kernel[0])


# %%
