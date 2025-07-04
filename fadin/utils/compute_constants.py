import numba
import numpy as np
import torch
from scipy.linalg import toeplitz

from fadin.utils.utils import convert_float_tensor


@numba.jit(nopython=True, cache=True)
def get_zG(events_grid, L):
    """
    events_grid.shape = n_dim, n_grid
    zG.shape = n_dim, L
    zLG.shape = n_dim
    """
    n_dim, _ = events_grid.shape

    zG = np.zeros(shape=(n_dim, L))
    for i in range(n_dim):
        ei = events_grid[i]
        n_ei = ei.sum()
        zG[i] = n_ei
        # Compute cumsum at the end of the vector of timestamps of size L
        # tau = 0:L-1
        zG[i, 1:] -= np.cumsum(np.flip(ei[-L + 1 :]))

    return zG


@numba.jit(nopython=True, cache=True)
def get_zN(events_grid, L):
    """
    events_grid.shape = n_dim, n_grid
    zN.shape = n_dim, n_dim, L
    zLN.shape = n_dim, n_dim
    """
    n_dim, _ = events_grid.shape

    zN = np.zeros(shape=(n_dim, n_dim, L))
    for i in range(n_dim):
        ei = events_grid[i]
        for j in range(n_dim):
            ej = events_grid[j]
            zN[i, j, 0] = ej @ ei  # useless in the solver since kernel[i, j, 0] = 0.
            for tau in range(1, L):
                zN[i, j, tau] = ej[:-tau] @ ei[tau:]

    return zN


@numba.jit(nopython=True, cache=True)
def _get_ztzG(events_grid, L):
    """
    events_grid.shape = n_dim, n_grid
    ztzG.shape = n_dim, n_dim, L, L
    """
    n_dim, n_grid = events_grid.shape
    ztzG = np.zeros(shape=(n_dim, n_dim, L, L))

    for i in range(n_dim):
        ei = events_grid[i]
        for j in range(n_dim):
            ej = events_grid[j]
            for tau in range(L):
                for tau_p in range(tau + 1):
                    # if tau_p == 0:
                    #     if tau == 0:
                    #         ztzG[i, j, tau, tau_p] = ei @ ej
                    #     else:
                    #         ztzG[i, j, tau, tau_p] = ei[:-tau] @ ej[tau:]
                    # else:
                    diff = tau - tau_p
                    ztzG[i, j, tau, tau_p] = (
                        ei[: n_grid - tau] @ ej[diff : n_grid - tau_p]
                    )
    return ztzG


def get_ztzG(events_grid, L):
    """
    events_grid.shape = n_dim, n_grid
    ztzG.shape = n_dim, n_dim, L, L
    zLtzG.shape = n_dim, n_dim, L
    """
    ztzG = _get_ztzG(events_grid, L)
    idx = np.arange(L)
    ztzG_nodiag = ztzG.copy()
    ztzG_nodiag[:, :, idx, idx] = 0.0
    ztzG_ = np.transpose(ztzG_nodiag, axes=(1, 0, 3, 2)) + ztzG

    return ztzG_


def get_ztzG_approx(events_grid, L):
    """
    events_grid.shape = n_dim, n_grid
    ztzG.shape = n_dim, n_dim, L, L
    """
    n_dim, _ = events_grid.shape
    ztzG = np.zeros(shape=(n_dim, n_dim, L, L))
    diff_tau = np.zeros(shape=(n_dim, n_dim, L))

    if n_dim > 5:  # Avoid to do inner product in the dimension' loops.
        diff_tau[:, :, 0] = events_grid @ events_grid.T
        for tau in range(1, L):
            diff_tau[:, :, tau] = events_grid[:, :-tau] @ events_grid[:, tau:].T
        for i in range(n_dim):
            for j in range(n_dim):
                ztzG[i, j] = toeplitz(diff_tau[i, j])
    else:
        for i in range(n_dim):
            ei = events_grid[i]
            for j in range(n_dim):
                ej = events_grid[j]
                diff_tau[i, j, 0] = ei @ ej
                for tau in range(1, L):
                    diff_tau[i, j, tau] = ei[:-tau] @ ej[tau:]
                ztzG[i, j] = toeplitz(diff_tau[i, j])

    return ztzG


def get_ztzG_approx_(events_grid, L):
    """
    events_grid.shape = n_dim, n_grid
    ztzG.shape = n_dim, n_dim, L, L
    """
    n_dim, _ = events_grid.shape
    ztzG = np.zeros(shape=(n_dim, n_dim, L, L))

    diff_tau = np.zeros(shape=(n_dim, n_dim, L))
    diff_tau[:, :, 0] = events_grid @ events_grid.T
    for tau in range(1, L):
        diff_tau[:, :, tau] = events_grid[:, :-tau] @ events_grid[:, tau:].T
    for i in range(n_dim):
        for j in range(n_dim):
            ztzG[i, j] = toeplitz(diff_tau[i, j])
    return ztzG


def compute_constants_fadin(events_grid, L, ztzG_approx=True):
    """Compute all precomputations"""
    zG = get_zG(events_grid.double().numpy(), L)
    zN = get_zN(events_grid.double().numpy(), L)

    if ztzG_approx:
        ztzG = get_ztzG_approx(events_grid.double().numpy(), L)
    else:
        ztzG = get_ztzG(events_grid.double().numpy(), L)

    zG = torch.tensor(zG).float()
    zN = torch.tensor(zN).float()
    ztzG = torch.tensor(ztzG).float()

    return zG, zN, ztzG


def compute_marked_quantities(
    events_grid, marks_grid, n_dim, density_hawkes, density_noise
):
    if density_hawkes == "linear":
        square_int_hawkes = torch.tensor([4 / 3 for _ in range(n_dim)])
        marks_grid_hawkes = 2 * marks_grid
    elif density_hawkes == "uniform":
        square_int_hawkes = torch.tensor([1.0 for _ in range(n_dim)])
        marks_grid_hawkes = events_grid
    else:
        raise NotImplementedError(
            "this density is not implemented \
                                    must be in linear | uniform "
        )

    if density_noise == "reverse_linear":
        square_int_noise = torch.tensor([4 / 3 for _ in range(n_dim)])
        marks_grid_noise = 2 - 2 * marks_grid
    elif density_noise == "uniform":
        square_int_noise = torch.tensor([1.0 for _ in range(n_dim)])
        marks_grid_noise = events_grid
    else:
        raise NotImplementedError(
            "this density is not implemented \
                                    must be in reverse_linear | uniform "
        )

    return square_int_hawkes, marks_grid_hawkes, square_int_noise, marks_grid_noise


def get_phi_tilde(z_tilde, L):
    """
    marks_grid.shape = n_dim, n_grid
    rho.shape = n_dim, n_grid
    phi_tilde.shape = n_dim, L

    NB: events grid are considered with the distribution of influence of the mark
    """
    phi_tilde = get_zG(z_tilde.detach().numpy(), L)

    return phi_tilde


def get_xi_tilde(marks_grid, z_tilde, rho, L):
    """
    marks_grid.shape = n_dim, n_grid
    z_tilde.shape = n_dim, n_grid
    rho.shape = n_dim, n_grid
    phi_tilde.shape = n_dim, L

    NB: events grid are considered with the distribution of influence of the mark
    """
    vec = (marks_grid**2) * rho - z_tilde**2
    phi_tilde = get_zG(vec.detach().numpy(), L)

    return phi_tilde


def get_phi_tilde_events(z_tilde, events_ground_grid, L, rho, marks_grid_hawkes):
    """
    z_tilde.shape = n_dim, n_grid
    zLN.shape = n_dim, n_dim, L
    """
    n_dim, n_grid = z_tilde.shape

    phi_tilde_events = np.zeros(shape=(n_dim, n_dim, L))
    prod = marks_grid_hawkes * rho
    for i in range(n_dim):
        #  ei = events_ground_grid[i]  # Count drived marks
        prod_i = prod[i]
        for j in range(n_dim):
            z_tilde_j = z_tilde[j]
            for tau in range(L):
                phi_tilde_events[i, j, tau] = (
                    (z_tilde_j[: n_grid - tau]) * prod_i[tau:]
                ).sum()

    return phi_tilde_events


def get_psi_tilde_approx(z_tilde, L):
    """
    marks_grid.shape = n_dim, n_grid
    loc_events.shape = list of n_dim of vector of size Number of events
    rho.shape = list of n_dim of vector of size Number of events
    ztzG.shape = n_dim, n_dim, L, L
    """
    psi_tilde = get_ztzG_approx(z_tilde.detach().numpy(), L)

    return psi_tilde


def compute_constants_unhap(
    z_tilde, marks_grid, events_grid, param_rho, marks_grid_hawkes, L
):
    """Compute all precomputations terms."""
    phi_tilde = get_phi_tilde(z_tilde, L)
    phi_tilde_events = get_phi_tilde_events(
        z_tilde, events_grid, L, param_rho, marks_grid_hawkes
    )
    psi_tilde = get_psi_tilde_approx(z_tilde, L)
    xi_tilde = get_xi_tilde(marks_grid, z_tilde, param_rho, L)

    phi_tilde = convert_float_tensor(phi_tilde)
    phi_tilde_events = convert_float_tensor(phi_tilde_events)
    psi_tilde = convert_float_tensor(psi_tilde)
    xi_tilde = convert_float_tensor(xi_tilde)

    return phi_tilde, phi_tilde_events, psi_tilde, xi_tilde
