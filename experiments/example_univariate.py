# %% import stuff
# import libraries
import time
import numpy as np
import torch
from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc

from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.solver import FaDIn

################################
# Meta parameters
################################
dt = 0.01
T = 100000
size_grid = int(T / dt) + 1

# mem = Memory(location=".", verbose=2)

# %% Experiment
################################


# @mem.cache
def simulate_data(baseline, alpha, mu, sigma, T, dt, seed=0):
    L = int(1 / dt)
    discretization = torch.linspace(0, 1, L)
    u = mu - sigma
    n_dim = u.shape[0]
    RC = DiscreteKernelFiniteSupport(0, 1, dt, kernel='RaisedCosine', n_dim=n_dim)
    kernel_values = RC.eval(
        [torch.Tensor(u), torch.Tensor(sigma)], discretization
    )
    kernel_values = kernel_values * alpha[:, :, None]

    t_values = discretization.double().numpy()
    k = kernel_values[0, 0].double().numpy()

    tf = HawkesKernelTimeFunc(t_values=t_values, y_values=k)
    kernels = [[tf]]
    hawkes = SimuHawkes(
        baseline=baseline, kernels=kernels, end_time=T, verbose=False, seed=int(seed)
    )

    hawkes.simulate()
    events = hawkes.timestamps
    return events

# %% solver
##


# @mem.cache
def run_solver(events, u_init, sigma_init, baseline_init, alpha_init, dt, T, seed=0):
    start = time.time()
    max_iter = 2000
    solver = FaDIn("RaisedCosine",
                             [torch.tensor(u_init),
                              torch.tensor(sigma_init)],
                             torch.tensor(baseline_init),
                             torch.tensor(alpha_init),
                             dt, solver="RMSprop",
                             step_size=1e-3,
                             max_iter=max_iter,
                             log=False,
                             random_state=0,
                             device="cpu",
                             optimize_kernel=True)
    print(time.time() - start)
    results = solver.fit(events, T)
    results_ = dict(param_baseline=results['param_baseline'][-10:].mean().item(),
                    param_alpha=results['param_alpha'][-10:].mean().item(),
                    param_kernel=[results['param_kernel'][0][-10:].mean().item(),
                                  results['param_kernel'][1][-10:].mean().item()])
    results_["time"] = time.time() - start
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = dt
    return results_

# %% Test


baseline = np.array([1.1])
alpha = np.array([[0.8]])
mu = np.array([[0.5]])
sigma = np.array([[0.3]])
u = mu - sigma

events = simulate_data(baseline, alpha, mu, sigma, T, dt, seed=0)

v = 0.2
baseline_init = baseline + v
alpha_init = alpha + v
mu_init = mu 
sigma_init = sigma + v
u_init = mu_init - sigma_init
results = run_solver(events, u_init, sigma_init,
                     baseline_init, alpha_init,
                     dt, T, seed=0)

print(np.abs(results['param_baseline'] - baseline))
print(np.abs(results['param_alpha'] - alpha))
print(np.abs(results['param_kernel'][0] - u))
print(np.abs(results['param_kernel'][1] - sigma))

# %% try kernel functions
from scipy.linalg import toeplitz
import numpy as np
import time
def _get_ztzG(events, n_discrete):
    """
    events.shape = n_dim, n_grid
    ztzG.shape = n_dim, n_dim, n_discrete, n_discrete
    """
    n_dim, _ = events.shape
    ztzG = np.zeros(shape=(n_dim, n_dim, n_discrete, n_discrete))

    for i in range(n_dim):
        ei = events[i]
        for j in range(n_dim):
            ej = events[j]
            for tau in range(n_discrete):
                for tau_p in range(tau + 1):
                    if tau_p == 0:
                        if tau == 0:
                            ztzG[i, j, tau, tau_p] = ei @ ej
                        else:
                            ztzG[i, j, tau, tau_p] = ei[:-tau] @ ej[tau:]
                    else:
                        diff = tau - tau_p
                        ztzG[i, j, tau, tau_p] = ei[:-tau] @ ej[diff:-tau_p]
    return ztzG


def get_ztzG(events, n_discrete):
    """
    events.shape = n_dim, n_grid
    ztzG.shape = n_dim, n_dim, n_discrete, n_discrete
    zLtzG.shape = n_dim, n_dim, n_discrete
    """
    ztzG = _get_ztzG(events, n_discrete)
    idx = np.arange(n_discrete)
    ztzG_nodiag = ztzG.copy()
    ztzG_nodiag[:, :, idx, idx] = 0.0
    ztzG_ = np.transpose(ztzG_nodiag, axes=(1, 0, 3, 2)) + ztzG
    # zLtzG = ztzG_.sum(3)
    return ztzG_
def get_ztzG2(events, n_discrete):
    n_dim, _ = events.shape

    ztzG = np.zeros(shape=(n_dim, n_dim,
                           n_discrete,
                           n_discrete))
    for i in range(n_dim):
        ei = events[i]
        for j in range(n_dim):
            ej = events[j]
            ztzG[i, j, 0, 0] = ei @ ej
            for tau in range(1, n_discrete):
                ztzG[i, j, tau, 0] = ei[:-tau] @ ej[tau:]
                ztzG[i, j, 0, tau] = ei[tau:] @ ej[:-tau] #le terme en tau_p
                for tau_p in range(1, n_discrete):
                    if (tau_p == tau):
                        ztzG[i, j, tau, tau] = ei[:-tau] @ ej[:-tau]
                    elif (tau > tau_p):
                        diff = tau - tau_p
                        ztzG[i, j, tau, tau_p] = ei[:-tau] @ ej[diff:-tau_p]
                    elif (tau < tau_p):
                        diff_ = tau_p - tau
                        ztzG[i, j, tau, tau_p] = ei[diff_:-tau] @ ej[:-tau_p]

    return ztzG


L = 100
T = 100
G = L * T
zj = np.random.randint(3, size=G)
zk = np.random.randint(4, size=G)
events = np.concatenate((zj.reshape(1, G), zk.reshape(1, G)), axis=0)
true = get_ztzG2(events, L)

diff_tau = np.zeros(L)
diff_tau[0] = zj @ zk

start = time.time()
ztzG = np.zeros((2, 2, L, L))
#ztzG_ = np.zeros((2, 2, L, L))
diff_tau = np.zeros((2, 2, L))
for i in range(2):
    ei = events[i]
    for j in range(2):
        ej = events[j]
        
        #  diff_tau2 = np.zeros(L)
        diff_tau[i, j, 0] = ei @ ej
        for tau in range(1, L):
            diff_tau[i, j, tau] = ei[:-tau] @ ej[tau:]
            #  diff_tau2[tau] = ej[:-tau] @ ei[tau:]
        # print(diff_tau == diff_tau2)
        ztzG[i, j] = toeplitz(diff_tau[i, j])
        #ztzG_[i, j] = toeplitz(diff_tau2)
print(time.time() - start)


start = time.time()
us = get_ztzG(events, L)
print(time.time() - start)
start = time.time()
true = get_ztzG2(events, L)
print(time.time() - start)


# %%
