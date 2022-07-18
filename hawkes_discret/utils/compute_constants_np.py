import numba
import numpy as np

def shift(x, shift):
    p = np.roll(x, shift=shift)
    p[:shift] = 0.
    return p

#@numba.jit(nopython=True, cache=True)
def compute_un_(events, size_discret):
    """events: list of array of size G
    constant_un: list of array of size size_discret"""

    constant_un = []
    n_dim = len(events)
    
    for i in range(n_dim):

        n_events_i = events[i].sum()
        sum_shift = np.zeros(size_discret) + n_events_i

        #Compute cumsum at the end of the vector of timestamps of size L
        # tau = 0:L-1
        sum_shift[1:] -= np.cumsum(np.flip(events[i][-size_discret+1:]))

        constant_un.append(sum_shift)
    
    return constant_un

#lent, à optimiser:
def compute_un_prime_(events, size_discret):
    """events: list of tensor of size G
    constant_un: list of tensor of taille size_discret"""

    constant_un_prime = []
    n_dim = len(events)

    for i in range(n_dim):

        timestamps = np.where(events[i] > 0)[0]
        sum_shift = np.zeros(size_discret)
        sum_shift[0] = events[i].sum()

        for tau in range(1, size_discret):

            shifted_events = shift(events[i], tau)
            sum_shift[tau] = shifted_events[timestamps].sum()


            
        constant_un_prime.append(sum_shift)
  
    return constant_un_prime

#numba ralentit le code
#@numba.jit((numba.float32[:, :], numba.int64), nopython=True, cache=True)
def compute_deux_(events, size_discret):
    """events: list of tensor of size G
    constant_udeux: list of tensor of taille size_discret"""
    
    n_dim = len(events)
    constant_deux = np.zeros((n_dim, n_dim, size_discret, size_discret))

    for i in range(n_dim):
        for j in range(n_dim):
            for tau in range(size_discret):
                for tau_p in range(tau+1):
                    if tau_p == 0:
                        if tau == 0:
                            constant_deux[i, j, tau, tau_p] = (
                            events[i]  * events[j]).sum()                                   
                        else:
                            constant_deux[i, j, tau, tau_p] = (
                            events[i][:-tau]  * events[j][tau:]).sum()
                    else:                       
                        diff = tau - tau_p
                        constant_deux[i, j, tau, tau_p] = (
                        events[i][:-tau]  * events[j][diff:-tau_p]).sum()
    return constant_deux


def compute_trois_(events, size_discret):
    constant_trois = []

    constant_un = compute_un_(events, size_discret)

    n_dim = len(events)

    for i in range(n_dim):
        constant_trois.append(constant_un[i].sum())

    return constant_trois

def compute_trois_prime_(events, size_discret):
    constant_trois_prime = []

    constant_un = compute_un_prime_(events, size_discret)

    n_dim = len(events)

    for i in range(n_dim):
        constant_trois_prime.append(constant_un[i].sum())

    return constant_trois_prime

def compute_quatre_(events, size_discret):

    constant_deux = compute_deux_(events, size_discret)
    constant_quatre = constant_deux.sum(3)

    return constant_quatre