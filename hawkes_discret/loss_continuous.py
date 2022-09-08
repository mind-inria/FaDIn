import numpy as np
import torch
import numba


def l2loss_cont(events, baseline, adjacency,
                Dg, Dg2, C, E, end_time):
    n_dim = len(events)
    value = 0

    n_events = np.zeros(n_dim)
    for i in range(n_dim):
        value += baseline[i] * baseline[i] * end_time
        n_events[i] = len(events[i])
        temp1 = 0
        temp2 = 0
        temp3 = 0
        temp4 = 0
        for j in range(n_dim):
            temp1 += adjacency[i, j] * Dg[i, j]
            temp2 += adjacency[i, j] * adjacency[i,  j] * Dg2[i, j]
            temp3 += adjacency[i, j] * C[i, j]
            for j1 in range(n_dim):
                temp4 += adjacency[i, j] * adjacency[i, j1] * E[i, j, j1]

        value += 2 * baseline[i] * temp1 + temp2 - 2 * \
            temp3 + 2 * temp4 - 2 * baseline[i] * n_events[i]

    return value / n_events.sum()


def l2loss_cont_dec(events, baseline, adjacency,
                    Dg, Dg2, C, E, end_time):
    n_dim = len(events)

    n_events = np.zeros(n_dim)
    for i in range(n_dim):
        n_events[i] = events[i].shape[0]

    t1 = term1_(baseline, end_time)
    t2 = term2_(baseline, adjacency, Dg)
    t3 = term3_(adjacency, Dg2)
    t4 = term4_(adjacency, E)
    t5 = term5_(baseline, events)
    t6 = term6_(adjacency, C)

    return (t1+t2+t3+t4+t5+t6) / n_events.sum()


def term1_(baseline, end_time):
    return end_time * (torch.linalg.norm(baseline)**2)


def term2_(baseline, adjacency, Dg):
    n_dim, _ = adjacency.shape

    res = 0
    for i in range(n_dim):
        temp = 0
        for j in range(n_dim):
            temp += adjacency[i, j] * Dg[i, j]
        res += baseline[i] * temp
    return 2*res


def term3_(adjacency, Dg2):
    n_dim, _ = adjacency.shape
    res = 0
    for i in range(n_dim):
        for j in range(n_dim):
            res += (adjacency[i, j]**2) * Dg2[i, j]
    return res


def term4_(adjacency, E):
    n_dim, _ = adjacency.shape
    res = 0
    for i in range(n_dim):
        for j in range(n_dim):
            for k in range(n_dim):
                res += adjacency[i, j] * (adjacency[i, k]
                                          * E[i, j, k])
    return 2*res


def term5_(baseline, events):
    n_dim = len(events)

    res = 0
    for i in range(n_dim):
        res += baseline[i] * len(events[i])
    return -2*res


def term6_(adjacency, C):
    n_dim, _ = adjacency.shape

    res = 0
    for i in range(n_dim):
        for j in range(n_dim):
            res += adjacency[i, j] * C[i, j]
    return -2*res


@numba.jit(nopython=True, cache=True)
def compute_constants_exp(events, decays, end_time):

    n_dim = len(events)

    Dg = np.zeros((n_dim, n_dim))
    Dg2 = np.zeros((n_dim, n_dim))
    C = np.zeros((n_dim, n_dim))
    E = np.zeros((n_dim, n_dim, n_dim))

    for i in range(n_dim):
        H = np.zeros((n_dim, n_dim))
        Dg_i = np.zeros(n_dim)
        Dg2_i = np.zeros(n_dim)
        C_i = np.zeros(n_dim)

        timestamps_i = events[i]
        N_i_size = len(timestamps_i)
        for j in range(n_dim):
            realization_j = events[j]
            N_j_size = len(realization_j)
            betaij = decays[i, j]
            ij = 0
            for k in range(N_i_size):
                if (k > 0):
                    for j1 in range(n_dim):
                        beta_j1_j = decays[j1, j]
                        H[j1, j] *= np.exp(-beta_j1_j *
                                           (timestamps_i[k] - timestamps_i[k - 1]))

                while ((ij < N_j_size) and (realization_j[ij] < timestamps_i[k])):
                    for j1 in range(n_dim):
                        beta_j1_j = decays[j1, j]
                        H[j1, j] += beta_j1_j * np.exp(
                            -beta_j1_j * (timestamps_i[k] - realization_j[ij]))

                    Dg_i[j] += (1 - np.exp(-betaij *
                                (end_time - realization_j[ij])))
                    Dg2_i[j] += betaij * (
                        1 - np.exp(-2 * betaij * (end_time - realization_j[ij]))) / 2
                    ij += 1

                C_i[j] += H[i, j]

                # Here we compute E(j1,i,j)
                for j1 in range(n_dim):
                    beta_j1_i = decays[j1, i]
                    beta_j1_j = decays[j1, j]
                    r = beta_j1_i / (beta_j1_i + beta_j1_j)
                    E[j1, i, j] += r * (
                        1 - np.exp(-(end_time - timestamps_i[k]) *
                                   (beta_j1_i + beta_j1_j))) * H[j1, j]
            if (ij < N_j_size):
                while (ij < N_j_size):
                    Dg2_i[j] += betaij * (
                        1 - np.exp(-2 * betaij * (end_time - realization_j[ij]))) / 2
                    Dg_i[j] += (1 - np.exp(-betaij *
                                (end_time - realization_j[ij])))
                    ij += 1
        Dg[i] = Dg_i
        Dg2[i] = Dg2_i
        C[i] = C_i

    return C, Dg, Dg2, E
