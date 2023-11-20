import numpy as np


###############################################################################
# General
###############################################################################

def block_process_1d(a, blocksize):
    """For a given array a, returns an array of same size b, 
    constructed by keeping maximum values of a within blocks of given size.

    Parameters
    ----------
    a : numpy.array
        Array to process.

    blocksize : int
        Size of the block to process a with.

    Returns
    -------
    b : numpy.array
        Processed array, of same shape of input array a.

    Examples
    --------
    >>> a = numpy.array([0, 1, 0, 0, 1, 3, 0])
    >>> blocksize = 2
    >>> block_process_1d(a, blocksize)
    numpy.array([0, 1, 0, 0, 0, 3, 0])
    """
    if len(a) < blocksize:
        return a

    b = np.zeros(a.shape)
    a_len = a.shape[0]
    for i in range(a_len):
        block = a[int(max(i-blocksize+1, 0)): int(min(i+blocksize, a_len))]
        if np.max(block) == a[i]:
            b[i] = a[i]

    return b


###############################################################################
# On driver events
###############################################################################

def proprocess_tasks(tasks, events_timestamps):
    if isinstance(tasks, int) or isinstance(tasks, str):
        tt = np.sort(events_timestamps[tasks])
    elif isinstance(tasks, list):
        tt = np.r_[events_timestamps[tasks[0]]]
        for i in tasks[1:]:
            tt = np.r_[tt, events_timestamps[i]]
        tt = np.sort(tt)
    return tt


###############################################################################
# On stochastic process' activations
###############################################################################

def filter_activation(acti, atom_to_filter='all', sfreq=150.,
                      time_interval=0.01):
    """For an array of atoms activations values, only keeps maximum values
    within a given time intervalle.

    In other words, we apply a filter in order to have a minimum time
    interval between two consecutives activations, and only keeping the
    maximum values

    Parameters
    ----------
    acti : numpy.array

    atom_to_filter : 'all' | int | array-like of int
        Ids of atoms to apply the filter on. If 'all', then applied on every
        atom in input `acti`. Defaults to 'all'.

    sfreq = float
        Sampling frequency, allow to transform `time_interval` into a number of
        timestamps. Defaults to 150.

    time_interval : float
        In second, the time interval within which we would like to keep the
        maximum activation values. Defaults to 0.01

    Returns
    -------
    acti : numpy.array
        Same as input, but with only maximum values within the given time
        intervalle.
    """

    blocksize = round(time_interval * sfreq)
    print("Filter activation on {} atoms using a sliding block of {} "
          "timestamps.".format(
              atom_to_filter, blocksize))

    if isinstance(atom_to_filter, str) and atom_to_filter == 'all':
        acti = np.apply_along_axis(block_process_1d, 1, acti, blocksize)
    elif isinstance(atom_to_filter, (list, np.ndarray)):
        for aa in atom_to_filter:
            acti[aa] = block_process_1d(acti[aa], blocksize)
    elif isinstance(atom_to_filter, int):
        acti[atom_to_filter] = block_process_1d(
            acti[atom_to_filter], blocksize)

    return acti


def get_atoms_timestamps(acti, sfreq=None, info=None, threshold=0,
                         percent=False, per_atom=True):
    """Get atoms' activation timestamps, using a threshold on the activation
    values to filter out unsignificant values.

    Parameters
    ----------
    acti : numpy.array of shape (n_atoms, n_timestamps)
        Sparse vector of activations values for each of the extracted atoms.

    sfreq : float
        Sampling frequency used in CDL. If None, will search for an "sfreq" key
        in the info dictionary. Defaults to None.

    info : dict
        Similar to mne.Info instance. Defaults to None.

    threshold : int | float
        Threshold value to filter out unsignificant ativation values. Defaults
        to 0.

    percent : bool
        If True, threshold is treated as a percentage: e.g., threshold = 5
        indicates that 5% of the activations will be removed, either per atom,
        or globally. Defaults to False.

    per_atom : bool
        If True, the threshold as a percentage will be applied per atom, e.g., 
        threshold = 5 will remove 5% of the activation of each atom. If false,
        the thresholding will be applied to all the activations. Defaults to
        True.

    Returns
    -------
    atoms_timestamps : numpy array
        array of timestamps
    """

    assert (sfreq is not None) or ('sfreq' in info.keys()), \
        "Please give an info dict that has a 'sfreq' key."

    if sfreq is None:
        sfreq = info['sfreq']

    n_atoms = acti.shape[0]
    if percent and per_atom:
        acti_nan = acti.copy()
        acti_nan[acti_nan == 0] = np.nan
        mask = acti_nan >= np.nanpercentile(
            acti_nan, threshold, axis=1, keepdims=True)
        atoms_timestamps = [acti_nan[i][mask[i]] / sfreq
                            for i in range(n_atoms)]
        return atoms_timestamps

    if percent and not per_atom:
        # compute the q-th percentile over all positive values
        threshold = np.percentile(acti[acti > 0], threshold)

    atoms_timestamps = [np.where(acti[i] > threshold)[0] / sfreq
                        for i in range(n_atoms)]

    return atoms_timestamps
