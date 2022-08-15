import numba
import numpy as np

"""
@numba.jit(nopython=True, cache=True)
def get_constants(events, n_discrete):
    n_dim, _ = events.shape

    zG = np.zeros(shape=(n_dim, n_discrete))
    zN = np.zeros(shape=(n_dim, n_dim, n_discrete))
    ztzG = np.zeros(shape=(n_dim, n_dim,
                           n_discrete,
                           n_discrete))
    zLG = np.zeros(n_dim)

    for i in range(n_dim):
        ei = events[i]
        n_ei = ei.sum()
        zG[i] = n_ei
        zG[i, 1:] -= np.cumsum(np.flip(ei[-n_discrete + 1:]))
        for j in range(n_dim):
            ej = events[j]
            #zN[i, j, 0] = ej @ ei useless in the solver since kernel[i,j, 0] = 0.
            for tau in range(n_discrete):
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
    #idx = np.arange(n_discrete)
    #ztzG_nodiag = ztzG.copy()
    #ztzG_nodiag[:, :, idx, idx] = 0.
    #ztzG_ = np.transpose(ztzG_nodiag, axes=(1, 0, 3, 2)) + ztzG  

    return zG, zN, ztzG, zLG, zN.sum(2), ztzG.sum(3)
"""

@numba.jit(nopython=True, cache=True)
def get_zG(events, n_discrete):
    """
    events.shape = n_dim, n_grid
    zG.shape =  n_dim, n_discrete
    """
    n_dim, _ = events.shape

    zG = np.zeros(shape=(n_dim, n_discrete))
    zLG = np.zeros(n_dim)
    for i in range(n_dim):
        ei = events[i]
        n_ei = ei.sum()
        zG[i] = n_ei
        # Compute cumsum at the end of the vector of timestamps of size L
        # tau = 0:L-1
        zG[i, 1:] -= np.cumsum(np.flip(ei[-n_discrete + 1:]))
        zLG[i] = zG[i].sum()

    return zG, zLG


@numba.jit(nopython=True, cache=True)
def get_zN(events, n_discrete):
    """
    events.shape = n_dim, n_grid
    zN.shape = n_dim, n_dim, n_discrete
    """
    n_dim, _ = events.shape

    zN = np.zeros(shape=(n_dim, n_dim, n_discrete))
    for i in range(n_dim):
        ei = events[i]
        for j in range(n_dim):
            ej = events[j]
            zN[i, j, 0] = ej @ ei #useless in the solver since kernel[i,j, 0] = 0.
            for tau in range(1, n_discrete):
                zN[i, j, tau] = ej[:-tau] @ ei[tau:]

    return zN, zN.sum(2)


@numba.jit(nopython=True, cache=True)
def _get_ztzG(events, n_discrete):
    """
    events.shape = n_dim, n_grid
    ztzG.shape = n_dim, n_dim, n_discrete, n_discrete
    """
    n_dim, _ = events.shape

    ztzG = np.zeros(shape=(n_dim, n_dim,
                           n_discrete,
                           n_discrete))
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
    ztzG = _get_ztzG(events, n_discrete)
    idx = np.arange(n_discrete)
    ztzG_nodiag = ztzG.copy()
    ztzG_nodiag[:, :, idx, idx] = 0.
    ztzG_ = np.transpose(ztzG_nodiag, axes=(1, 0, 3, 2)) + ztzG

    return ztzG_, ztzG_.sum(3)

def get_zLG(events, n_discrete):
    """
    events.shape = n_dim, n_grid
    zG.shape = n_dim, n_discrete
    zLG.shape = n_dim
    """
    n_dim, _ = events.shape

    zLG = np.zeros(n_dim)
    zG = get_zG(events, n_discrete)
    for i in range(n_dim):
        zLG[i] = zG[i].sum()

    return zLG


def get_zLN(events, n_discrete):
    """
    events.shape = n_dim, n_grid
    zN.shape = n_dim, n_dim, n_discrete
    zLN.shape = n_dim, n_dim
    """
    zN = get_zN(events, n_discrete)
    zLN = zN.sum(2)

    return zLN


def get_zLtzG(events, n_discrete):
    """
    events.shape = n_dim, n_grid
    ztzG.shape = n_dim, n_dim, n_discrete, n_discrete
    zLtzG.shape = n_dim, n_dim, n_discrete
    """
    ztzG = get_ztzG(events, n_discrete)
    zLtzG = ztzG.sum(3)

    return zLtzG
