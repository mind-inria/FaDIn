import numba
import numpy as np
from scipy.linalg import toeplitz

"""
@numba.jit(nopython=True, cache=True)
def get_constants(events, L):
    n_dim, _ = events.shape

    zG = np.zeros(shape=(n_dim, L))
    zN = np.zeros(shape=(n_dim, n_dim, L))
    ztzG = np.zeros(shape=(n_dim, n_dim,
                           L,
                           L))
    zLG = np.zeros(n_dim)

    for i in range(n_dim):
        ei = events[i]
        n_ei = ei.sum()
        zG[i] = n_ei
        zG[i, 1:] -= np.cumsum(np.flip(ei[-L + 1:]))
        for j in range(n_dim):
            ej = events[j]
            #zN[i, j, 0] = ej @ ei useless in the solver since kernel[i,j, 0] = 0.
            for tau in range(L):
                if tau > 0:
                    zN[i, j, tau] = ej[:-tau] @ ei[tau:]
                for tau_p in range(tau + 1):
                    if tau_p == 0:
                        if tau == 0:
                            ztzG[i, j, tau, tau_p] = ei @ ej
                        else:
                            ztzG[i, j, tau, tau_p] = ei[:-tau] @ ej[tau:]
                    else:
                        diff = tau - tau_p
                        ztzG[i, j, tau, tau_p] = ei[:-tau] @ ej[diff:-tau_p]
        zLG[i] = zG[i].sum()
    #idx = np.arange(L)
    #ztzG_nodiag = ztzG.copy()
    #ztzG_nodiag[:, :, idx, idx] = 0.
    #ztzG_ = np.transpose(ztzG_nodiag, axes=(1, 0, 3, 2)) + ztzG
    #return zG, zN, ztzG, zLG, zN.sum(2), ztzG.sum(3)
"""


@numba.jit(nopython=True, cache=True)
def get_zG(events, L):
    """
    events.shape = n_dim, n_grid
    zG.shape =  n_dim, L
    zLG.shape = n_dim
    """
    n_dim, _ = events.shape

    zG = np.zeros(shape=(n_dim, L))
    for i in range(n_dim):
        ei = events[i]
        n_ei = ei.sum()
        zG[i] = n_ei
        # Compute cumsum at the end of the vector of timestamps of size L
        # tau = 0:L-1
        zG[i, 1:] -= np.cumsum(np.flip(ei[-L + 1:]))

    return zG


@numba.jit(nopython=True, cache=True)
def get_zN(events, L):
    """
    events.shape = n_dim, n_grid
    zN.shape = n_dim, n_dim, L
    zLN.shape = n_dim, n_dim
    """
    n_dim, _ = events.shape

    zN = np.zeros(shape=(n_dim, n_dim, L))
    for i in range(n_dim):
        ei = events[i]
        for j in range(n_dim):
            ej = events[j]
            zN[i, j, 0] = ej @ ei  # useless in the solver since kernel[i,j, 0] = 0.
            for tau in range(1, L):
                zN[i, j, tau] = ej[:-tau] @ ei[tau:]
    # zLN = zN.sum(2)

    return zN


# @numba.jit(nopython=True, cache=True)
def _get_ztzG(events, L):
    """
    events.shape = n_dim, n_grid
    ztzG.shape = n_dim, n_dim, L, L
    """
    n_dim, _ = events.shape
    ztzG = np.zeros(shape=(n_dim, n_dim, L, L))

    for i in range(n_dim):
        ei = events[i]
        for j in range(n_dim):
            ej = events[j]
            for tau in range(L):
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


def get_ztzG(events, L):
    """
    events.shape = n_dim, n_grid
    ztzG.shape = n_dim, n_dim, L, L
    zLtzG.shape = n_dim, n_dim, L
    """
    ztzG = _get_ztzG(events, L)
    idx = np.arange(L)
    ztzG_nodiag = ztzG.copy()
    ztzG_nodiag[:, :, idx, idx] = 0.0
    ztzG_ = np.transpose(ztzG_nodiag, axes=(1, 0, 3, 2)) + ztzG

    return ztzG_


def get_ztzG_(events, L):
    """
    events.shape = n_dim, n_grid
    ztzG.shape = n_dim, n_dim, L, L
    """
    n_dim, _ = events.shape
    ztzG = np.zeros(shape=(n_dim, n_dim, L, L))

    diff_tau = np.zeros(shape=(n_dim, n_dim, L))
    for i in range(n_dim):
        ei = events[i]
        for j in range(n_dim):
            ej = events[j]
            diff_tau[i, j, 0] = ei @ ej
            for tau in range(1, L):
                diff_tau[i, j, tau] = ei[:-tau] @ ej[tau:]
            ztzG[i, j] = toeplitz(diff_tau[i, j])
    return ztzG


"""
def get_ztzG2(events, L):
    n_dim, _ = events.shape

    ztzG = np.zeros(shape=(n_dim, n_dim,
                           L,
                           L))
    for i in range(n_dim):
        ei = events[i]
        for j in range(n_dim):
            ej = events[j]
            ztzG[i, j, 0, 0] = ei @ ej
            for tau in range(1, L):
                ztzG[i, j, tau, 0] = ei[:-tau] @ ej[tau:]
                ztzG[i, j, 0, tau] = ei[tau:] @ ej[:-tau] #le terme en tau_p
                for tau_p in range(1, L):
                    if (tau_p == tau):
                        ztzG[i, j, tau, tau] = ei[:-tau] @ ej[:-tau]
                    elif (tau > tau_p):
                        diff = tau - tau_p
                        ztzG[i, j, tau, tau_p] = ei[:-tau] @ ej[diff:-tau_p]
                    elif (tau < tau_p):
                        diff_ = tau_p - tau
                        ztzG[i, j, tau, tau_p] = ei[diff_:-tau] @ ej[:-tau_p]

    return ztzG
"""


def get_zLG(events, L):
    """
    events.shape = n_dim, n_grid
    zG.shape = n_dim, L
    zLG.shape = n_dim
    """
    n_dim, _ = events.shape

    zLG = np.zeros(n_dim)
    zG = get_zG(events, L)
    for i in range(n_dim):
        zLG[i] = zG[i].sum()

    return zLG


def get_zLN(events, L):
    """
    events.shape = n_dim, n_grid
    zN.shape = n_dim, n_dim, L
    zLN.shape = n_dim, n_dim
    """
    zN = get_zN(events, L)
    zLN = zN.sum(2)

    return zLN


def get_zLtzG(events, L):
    """
    events.shape = n_dim, n_grid
    ztzG.shape = n_dim, n_dim, L, L
    zLtzG.shape = n_dim, n_dim, L
    """
    ztzG = get_ztzG(events, L)
    zLtzG = ztzG.sum(3)

    return zLtzG
