import json
import numpy as np
import copy
import torch
from fadin.solver import FaDIn, plot
from fadin.utils.utils_meg import proprocess_tasks, filter_activation, \
    get_atoms_timestamps

# Load CDL output
with open("experiments/meg/cdl_sample.json", "r") as fp:
    dict_cdl = json.load(fp)


BL_MASK = torch.Tensor([1, 1, 1])
ALPHA_MASK = torch.Tensor([[0, 0, 0], [0, 0, 0], [1, 1, 0]])
OPTIM_MASK = {'baseline': BL_MASK, 'alpha': ALPHA_MASK}


def fit_fadin_sample(list_tasks, atom, cdl_dict, filter_interval, thresh_acti,
                     kernel, optim_mask,
                     plotfig=False, figtitle=None, savefig=None,
                     **fadin_init):
    """
    Loads driver events and atom activations on sample dataset. Fits a FaDIn
    solver to the timestamps and plots estimated kernels and baselines.
    Returns kernel plot and estimated parameters of the intensity function.

    Parameters
    ----------
    list_tasks: `Tuple` of lists
        One element of this tuple is a list of driver activations to group
        in the same driver channel.

    atom: `int`
        Coordinate of the atom whose activations will be used for *
        Hawkes process parameterization.

    cdl_dict: `dict` of `dict`
        Output of dictionary learning applied to `sample` dataset,
        with keys as follows:
        'dict_cdl_params' : `dict`
            Value of GreedyCDL's parameters.

        'dict_other_params' : `dict`
            Value of all other parameters, such as data source, sfreq, etc.

        'dict_cdl_fit_res' : `dict` of `numpy.array`
            Results of the cdl.fit(), with u_hat_, v_hat_ and z_hat.

        'dict_pair_up' : `dict`
            Pre-process of results that serve as input in Hawkes process
            fitting algorithm.

    filter_interval: `float` or `None`
        Time interval for block filtering of atom activations:
        in each time window of width `filter_interval`,
        only the timestamp with max activation will be kept.
        If set to `None`, no block filtering is done on atom activations.

    thresh_acti: `float` or `None`
        Threshold applied on atom activations.
        Only timestamps of activations > `thresh_acti` will be kept.

    kernel: `str`
        Name of kernel to use in the FaDIn solver.
        Supported values are `truncated_exponential`, `truncated_gaussian` and
        `raised_cosine`.

    baseline_mask: `Tensor` of shape 3 or `None`
        argument of FaDIn solver.

    alpha_mask: `Tensor` of shape 3,3 or `None`
        argument of FaDIn solver.

    plotfig: bool (default `False`)
        Whether the fitted kernels and baselines figure is plotted.

    figtitle: `str` or `None` (default `None`)
        Title for plot of fitted kernels and baselines.
        If set to `None`, the figure title is generic.

    savefig: str or `None` (default `None`)
        Path for saving the plot of fitted kernels and baselines.
        If set to `None`, the figure is not saved.

    **fadin_init: `dict`
        extra arguments of FaDIn solver.

    Returns:
    ----------
    fig, params_intens
        fig: matplotlib.pyplot.Figure
            Plot of estimated kernels and baselines
        params_intens: list
            Parameters of the intensity function, i.e the baseline
            and kernel parameters.
    """
    # Pro-process CDL output for Hawkes process fitting
    dict_events_acti = cdl_dict['dict_pair_up']
    T = int(dict_events_acti['T']) + 1
    sfreq = cdl_dict['dict_other_params']['sfreq']

    # Driver events
    events_timestamps = copy.deepcopy(dict_events_acti['events_timestamps'])
    events_acti_tt = [proprocess_tasks(tasks, events_timestamps)
                      for tasks in list_tasks]

    # Atom activations
    acti = np.array([copy.deepcopy(dict_events_acti['acti_shift'][atom])])
    if filter_interval is not None:
        acti = filter_activation(acti,
                                 atom_to_filter='all',
                                 sfreq=sfreq,
                                 time_interval=filter_interval)
    if thresh_acti is not None:
        acti = get_atoms_timestamps(acti=acti,
                                    sfreq=sfreq,
                                    threshold=thresh_acti)
    events_acti_tt.append(acti)

    # Fit Hawkes process to data
    solver = FaDIn(n_dim=len(events_acti_tt), kernel=kernel,
                   optim_mask=optim_mask,
                   **fadin_init)
    solver.fit(events_acti_tt, T)
    # Return results
    fig, _ = plot(solver, plotfig=plotfig, title=figtitle, savefig=savefig,
                  ch_names=['STIM audio', 'STIM visual', f'acti {atom}']
                  )
    params_intens = solver.params_intens
    del solver
    return fig, params_intens


##############################################################################
# REPRODUCE SAMPLE RESULTS IN FaDIn PAPER
##############################################################################

fig, tg_atom6_mask = fit_fadin_sample(
    list_tasks=(['1', '2'], ['3', '4']),
    atom=6,
    cdl_dict=dict_cdl,
    filter_interval=0.01,
    thresh_acti=0.6e-10,
    kernel='truncated_gaussian',
    optim_mask=OPTIM_MASK,
    kernel_length=0.5,
    delta=0.02,
    optim='RMSprop',
    params_optim={'lr': 1e-3},
    max_iter=10000,
    ztzG_approx=False,
    figtitle='Masked FaDIn with TG kernels, atom 6 filt and thresh actis',
    savefig='fit_fadin_sample_plots/nomarked_masked_tg_6_practi.png',
    plotfig=True
)
print('Truncated gaussian, atom 6, with mask')
print('Estimated baseline:', tg_atom6_mask[0])
print('Estimated alpha:', tg_atom6_mask[1])
print('Estimated kernel parameters', tg_atom6_mask[2:])


fig, tg_atom3_allmask = fit_fadin_sample(
    list_tasks=(['1', '2'], ['3', '4']),
    atom=3,
    cdl_dict=dict_cdl,
    filter_interval=0.01,
    thresh_acti=0.6e-10,
    kernel='truncated_gaussian',
    optim_mask=OPTIM_MASK,
    kernel_length=0.5,
    delta=0.02,
    optim='RMSprop',
    params_optim={'lr': 1e-3},
    max_iter=10000,
    ztzG_approx=False,
    figtitle='Masked FaDIn with TG kernels, atom 3 filt and thresh actis',
    savefig='fit_fadin_sample_plots/nomarked_masked_tg_3_practi.png',
    plotfig=True
)
print('Truncated gaussian, atom 3, with mask')
print('Estimated baseline:', tg_atom3_allmask[0])
print('Estimated alpha:', tg_atom3_allmask[1])
print('Estimated kernel parameters', tg_atom3_allmask[2:])

fig, rc_atom3_mask = fit_fadin_sample(
    list_tasks=(['1', '2'], ['3', '4']),
    atom=3,
    cdl_dict=dict_cdl,
    filter_interval=0.01,
    thresh_acti=0.6e-10,
    kernel='raised_cosine',
    optim_mask=OPTIM_MASK,
    kernel_length=0.5,
    delta=0.02,
    optim='RMSprop',
    params_optim={'lr': 1e-3},
    max_iter=20000,
    ztzG_approx=False,
    figtitle='Masked FaDIn with RC kernels, atom 3 filt and thresh actis',
    savefig='fit_fadin_sample_plots/nomarked_masked_rc_3_practi.png',
    plotfig=True
)
print('Raised_cosine, atom 3, with mask')
print('Estimated baseline:', rc_atom3_mask[0])
print('Estimated alpha:', 2 * rc_atom3_mask[3] * rc_atom3_mask[1])
print('Estimated kernel parameters u and s', rc_atom3_mask[2:])

fig, rc_atom6_mask = fit_fadin_sample(
    list_tasks=(['1', '2'], ['3', '4']),
    atom=6,
    cdl_dict=dict_cdl,
    filter_interval=0.01,
    thresh_acti=0.6e-10,
    kernel='raised_cosine',
    optim_mask=OPTIM_MASK,
    kernel_length=0.5,
    delta=0.02,
    optim='RMSprop',
    params_optim={'lr': 1e-3},
    max_iter=20000,
    ztzG_approx=False,
    figtitle='Masked FaDIn with RC kernels, atom 6 filt and thresh actis',
    savefig='fit_fadin_sample_plots/nomarked_masked_rc_6_practi.png',
    plotfig=True
)

print('Raised_cosine, atom 6, with mask')
print('Estimated baseline:', rc_atom6_mask[0])
print('Estimated alpha:', 2 * rc_atom6_mask[3] * rc_atom6_mask[1])
print('Estimated kernel parameters u and s', rc_atom6_mask[2:])
