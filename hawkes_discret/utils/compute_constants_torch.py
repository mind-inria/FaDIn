#import numba
import numpy as np
import torch
import time
#from numba import jit


def shift(x, shift):
    p = torch.roll(x, shifts=shift)
    p[:shift] = 0.
    return p

#@jit(nopython=True, cache=True)
def shift_(x, shift):
    p = np.roll(x, shifts=shift)
    p[:shift] = 0.
    return p
    

def compute_un(events, size_discret):
    """events: list of tensor of size G
    constant_un: list of tensor of taille size_discret"""

    constant_un = []
    n_dim = len(events)
    
    for i in range(n_dim):

        n_events_i = events[i].sum()
        sum_shift = torch.zeros(size_discret) + n_events_i

        #Compute cumsum at the end of the vector of timestamps of size L
        # tau = 0:L-1
        sum_shift[1:] -= torch.cumsum(torch.flip(events[i][-size_discret+1:], dims=[0]), dim=0)

        constant_un.append(sum_shift)
    
    return constant_un

#lent, à optimiser:
def compute_un_prime(events, size_discret):
    """events: list of tensor of size G
    constant_un: list of tensor of taille size_discret"""

    constant_un = []
    n_dim = len(events)

    for i in range(n_dim):

        timestamps = np.where(events[i] > 0)[0]
        sum_shift = torch.zeros(size_discret)
        sum_shift[0] = events[i].sum()

        for tau in range(1, size_discret):

            shifted_events = shift(events[i], tau)
            sum_shift[tau] = shifted_events[timestamps].sum()


            
        constant_un.append(sum_shift)
  
    return constant_un

def compute_deux(events, size_discret):
    """events: list of tensor of size G
    constant_udeux: list of tensor of taille size_discret"""
    
    n_dim = len(events)
    constant_deux = torch.zeros(n_dim, n_dim, size_discret, size_discret)

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

def compute_trois(events, size_discret):
    constant_trois = []

    constant_un = compute_un(events, size_discret)

    n_dim = len(events)

    for i in range(n_dim):
        constant_trois.append(constant_un[i].sum())

    return constant_trois

def compute_trois_prime(events, size_discret):
    constant_trois_prime = []

    constant_un = compute_un_prime(events, size_discret)

    n_dim = len(events)

    for i in range(n_dim):
        constant_trois_prime.append(constant_un[i].sum())

    return constant_trois_prime

def compute_quatre(events, size_discret):

    constant_deux = compute_deux(events, size_discret)
    constant_quatre = constant_deux.sum(3)

    return constant_quatre