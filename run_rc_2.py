# %% import stuff
## import libraries
from hawkes_discret.kernels import KernelRaisedCosineDiscret
from hawkes_discret.hawkes_discret_l2 import HawkesDiscretL2

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from tick.hawkes import SimuHawkes,  HawkesKernelTimeFunc

################################ 
## Meta parameters
################################ 
dt = 0.01
T = 100_000
L = int(1/dt)
size_grid = int(T/dt) +1
discretization = torch.linspace(0, 1, L)

# %% simulate data
## 
################################ 
#### Simulated data
################################ 
RC = KernelRaisedCosineDiscret(dt)

baseline = torch.tensor([1.1], dtype=torch.float64)
alpha = torch.tensor([[0.8]], dtype=torch.float64)
mu = torch.tensor([[0.5]], dtype=torch.float64)
sigma = torch.tensor([[0.3]], dtype=torch.float64)
u = (mu.clone() - sigma.clone())

kernel_values = RC.eval(u, sigma, discretization) #* dt
kernel_values = kernel_values * alpha[:, :, None]

t_values = discretization.double().numpy()
k = kernel_values[0, 0].double().numpy()

tf = HawkesKernelTimeFunc(t_values=t_values, y_values=k)
kernels = [[tf]]

hawkes = SimuHawkes(baseline=[1.1], kernels=kernels, end_time=T,
                    verbose=False, seed=0)

hawkes.simulate()
events = hawkes.timestamps


# %% solver
## 
v = 0.15
baseline_init = baseline + v
alpha_init = alpha + v
mu_init = mu + v
sigma_init = sigma + v
u_init = u + v


start = time.time()
max_iter = 1000
solver_1 = HawkesDiscretL2('RaisedCosine', 
                         u_init, 
                         sigma_init,
                         baseline_init, 
                         alpha_init, 
                         dt,
                         solver='RMSprop', 
                         step_size=1e-3,
                         max_iter=max_iter, log=False,
                         random_state=0, device='cpu')

results_1 = solver_1.fit(events, T)
print(time.time() - start)


file_name = "test2.pkl"
open_file = open(file_name, "wb")
pickle.dump(results_1, open_file)
open_file.close()

# %% name
loss = results_1[0]
grad_baseline = results_1[1]
grad_adjacency = results_1[2]
grad_u = results_1[3]
grad_sigma = results_1[4]
param_baseline_e = torch.abs(results_1[5])# - baseline)
param_adjacency_e = torch.abs(results_1[6])# - adjacency)
param_u_e = torch.abs(results_1[7])# - u)
param_sigma_e = torch.abs(results_1[8])# - sigma)
epochs = torch.arange(max_iter)
epochss = torch.arange(max_iter+1)

# %% plot
#% matplotlib inline
import matplotlib
matplotlib.rc('xtick', labelsize=13) 
matplotlib.rc('ytick', labelsize=13) 
lw = 5
fontsize = 18
n_dim=1
fig, axs = plt.subplots(2,4, figsize=(15, 10))

for i in range(n_dim):
    axs[0, 0].plot(epochs, grad_baseline, lw=lw)
    axs[0, 0].set_title('grad_baseline', size=fontsize)
    
    axs[1, 0].plot(epochss, param_baseline_e, lw=lw)
    axs[1, 0].set_title('mu', size=fontsize)

    for j in range(n_dim):
        axs[0, 1].plot(epochs, grad_adjacency[:, i, j], lw=lw, label=(i, j))
        axs[0, 1].set_title('grad_alpha', size=fontsize)
        axs[0, 1].legend(fontsize=fontsize-5)
        
        axs[0, 2].plot(epochs, grad_u[:, i, j], lw=lw)
        axs[0, 2].set_title('grad_u', size=fontsize)
        
        axs[0, 3].plot(epochs, grad_sigma[:, i, j], lw=lw)
        axs[0, 3].set_title('grad_sigma', size=fontsize)
        
        axs[1, 1].plot(epochss, param_adjacency_e[:, i, j], lw=lw)
        axs[1, 1].set_title('alpha', size=fontsize)

        axs[1, 2].plot(epochss, param_u_e[:, i, j], lw=lw)
        axs[1, 2].set_title('u', size=fontsize)

        axs[1, 3].plot(epochss, param_sigma_e[:, i, j], lw=lw)
        axs[1, 3].set_title('sigma', size=fontsize)
        
plt.figure(figsize=(12,4))

plt.tight_layout()
plt.plot(epochs, loss, lw=10)
plt.title('Loss', size=fontsize)
# %%
