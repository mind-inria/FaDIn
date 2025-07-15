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

import numpy as np
import torch
import matplotlib.pyplot as plt

from fadin.utils.utils_simu import simu_hawkes_cluster
from fadin.solver import FaDIn
from fadin.kernels import DiscreteKernelFiniteSupport

n_dim = 1
dt = 0.01
T = 10_000
kernel_length = 5
L = int(1 / dt)
size_grid = int(T / dt) + 1
discretization = torch.linspace(0, kernel_length, L)

###############################################################################
# Here, we set the parameters of a Hawkes process with an Exponential(1) distribution.

baseline = np.array([.4])
alpha = np.array([[0.9]])
beta = 0.8
###############################################################################
# Here, we simulate the data.

# standard parameter is beta, the parameter of the exponential distribution,
# equal to one.
kernel = 'expon'
events = simu_hawkes_cluster(T, baseline, alpha, kernel,
                             params_kernel={'scale': 1 / beta})

###############################################################################
# Let's take a closer look at the events
print('Type of events object', type(events))
# events is a list of numpy arrays, one for each dimension
print('Number of events', len(events[0]))
print('First 10 events timestamps', events[0][:10])
# For each event, its occurence time (timestamp) is stored in the numpy array.

# `events`` is a list. Its elements are numpy arrays containing the timestamps
# of the events. This is the expected data format for FaDIn.

###############################################################################
# Here, we apply FaDIn.

solver = FaDIn(n_dim=1,
               kernel="truncated_exponential",
               kernel_length=kernel_length,
               delta=dt, optim="RMSprop",
               params_optim={'lr': 1e-3},
               max_iter=2000
               )
solver.fit(events, T)

###############################################################################
# Here, we print the estimated parameters of the Hawkes process.

print('Estimated baseline is:', solver.baseline_.item())
print('Estimated alpha is:', solver.alpha_.item())
print('Estimated beta parameter of the exponential kernel is:', solver.kernel_)


###############################################################################
# Here, we plot the values of the estimated kernel with FaDIn.

kernel = DiscreteKernelFiniteSupport(dt, n_dim, kernel='truncated_exponential',
                                     kernel_length=kernel_length)
kernel_values = kernel.kernel_eval(solver.kernel_, discretization)

plt.plot(discretization[1:], kernel_values.squeeze()[1:]/kernel_length,
         label='FaDIn\' estimated kernel')
plt.plot(discretization[1:], beta * torch.exp(-discretization[1:]*beta),
         label='True kernel', c='k')
plt.title('Hawkes influence kernel', size=20)
plt.xlabel('Time', size=20)
plt.ylabel(r'$\phi(t)$', size=25)
plt.legend(fontsize='x-large')
plt.show()
