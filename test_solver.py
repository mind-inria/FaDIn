import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from tick.hawkes import SimuHawkesExpKernels
from tick.hawkes import HawkesSumExpKern, HawkesExpKern
from tick.hawkes import HawkesKernelTimeFunc
from tick.hawkes import SimuHawkes, SimuHawkesMulti


from hawkes_discret.kernels import KernelExpDiscret
from hawkes_discret.hawkes_discret_l2 import HawkesDiscretL2
from hawkes_discret.utils.utils import projected_grid

import pickle

kernel_model = 'KernelExpDiscret'


######## Data Simulation and problem parameters ########
end_time_1 = 10000
discrete_step_1 = 0.01
L = int(1 / discrete_step_1)
size_grid_1 = int(end_time_1 / discrete_step_1) +1
discretization = torch.linspace(0, 1, L, dtype=torch.float64)

baseline_true_1 = torch.tensor([1.1], dtype=torch.float64)
adjacency_true_1 = torch.tensor([[0.4]], dtype=torch.float64)
kernel_params_true_1 = torch.tensor([[1.]], dtype=torch.float64)

Kernel = KernelExpDiscret(1, discrete_step_1)
kernel_values = Kernel.eval(kernel_params_true_1, discretization)
kernel_values *= adjacency_true_1 

tf = HawkesKernelTimeFunc(t_values=discretization.numpy(), 
                           y_values=kernel_values.squeeze().detach().numpy())
hawkes = SimuHawkes(kernels=[[tf]],
                    baseline=baseline_true_1.detach().numpy(), 
                    end_time=end_time_1,
                    verbose=False, seed=0)
hawkes.simulate()
events_1 = hawkes.timestamps

################################################################


max_iter_1 = 10
baseline_init_1 = torch.tensor([0.8], dtype=torch.float64)
adjacency_init_1 = torch.tensor([[0.6]], dtype=torch.float64)
kernel_params_init_1 = torch.tensor([[1.2]], dtype=torch.float64)
#adjacency_reparam_init_1 =  adjacency_init_1*kernel_params_init_1


solver_1 = HawkesDiscretL2(kernel_model, 
                         kernel_params_init_1, 
                         baseline_init_1, 
                         adjacency_init_1, 
                         discrete_step_1,
                         solver='RMSprop', 
                         step_size=1e-3,
                         max_iter=max_iter_1, log=False,
                         random_state=0, device='cpu')

results_1 = solver_1.fit(events_1, end_time_1)


#####################Plot results ###############################
import matplotlib 

loss = results_1[0]
grad_baseline = results_1[1]
grad_adjacency = results_1[2]
grad_decay = results_1[3]
param_baseline_e = torch.abs(results_1[4] - baseline_true_1)
param_adjacency_e = torch.abs(results_1[5] - adjacency_true_1)
param_decay_e = torch.abs(results_1[6] - kernel_params_true_1)
epochs = torch.arange(max_iter_1)
epochss = torch.arange(max_iter_1+1)

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
lw=5
fig, axs = plt.subplots(2,3, figsize=(25, 25))

axs[0, 0].plot(epochs, grad_baseline, lw=lw)
axs[0, 0].set_title('grad_baseline', size=35)

axs[0, 1].plot(epochs, grad_adjacency.squeeze(), lw=lw)
axs[0, 1].set_title('grad_alpha', size=35)

axs[0, 2].plot(epochs, grad_decay.squeeze(), lw=lw)
axs[0, 2].set_title('grad_decay', size=35)

axs[1, 0].plot(epochss, param_baseline_e, lw=lw)
axs[1, 0].set_title('absolute error baseline', size=35)

axs[1, 1].plot(epochss, param_adjacency_e.squeeze(), lw=lw)
axs[1, 1].set_title('absolute error alpha', size=35)

axs[1, 2].plot(epochss, param_decay_e.squeeze(), lw=lw)
axs[1, 2].set_title('absolute error decay', size=35)

plt.figure(figsize=(12,4))
plt.plot(epochs, loss, lw=lw)
plt.title('Loss', size=20)



################################################################