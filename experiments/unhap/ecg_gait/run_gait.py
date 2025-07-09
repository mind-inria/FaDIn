"""
Adapted from
https://alphacsc.github.io/stable/auto_examples/dicodile/plot_gait.html#sphx-glr-auto-examples-dicodile-plot-gait-py
Code to benchmark on gait data:
- CDL + UNHaP
- CDL + FaDIn
- Neurokit
- pyHRV
"""
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import neurokit2 as nk
import pyhrv.time_domain as td
from biosppy.signals.ecg import ecg
from dicodile.data.gait import get_gait_data

try:
    from alphacsc import BatchCDL
    from alphacsc.utils.convolution import construct_X_multi
except ImportError:
    raise ImportError(
        'Please install the package alphacsc to run this example.\n'
        'pip install alphacsc'
    )

from fadin.solver import FaDIn, UNHaP
from fadin.utils.vis import plot

from fadin.utils.utils import projected_grid_marked
from utils_ecg import process_cdl_actis
from utils_gait import SLOTS
from utils_gait import save_errors
##############################################################################
# PARAMETERS
##############################################################################

RUN_BASELINES = True

# DATA PARAMETERS
SFREQ = 100

# CDL PARAMETERS
N_ATOMS = 1
# Set individual atom (patch) size.
N_TIMES_ATOM = 150  # samples
N_ITERATIONS = 20
REG = 0.5

# UNHAP AND FADIN PARAMETERS
MODEL_LIST = [FaDIn, UNHaP, 'StocUNHaP']
KERNEL_LENGTH = 2
DELTA = 0.01
LR = 1e-3
L = int(1 / DELTA)
KERNEL_TYPE = 'truncated_gaussian'
N_ITERATIONS_FADIN = 20_000
RANDOM_STATE = 40  # Only used if 'moment_matching' in INIT
BATCH_RHO = 1000
INIT = 'moment_matching_mean'

# PLOT PARAMETERS
PLOTFIG = True  # Whether plots are drawn.
SHOWFIG = False  # Whether plots are shown at each iteration.
SAVEFIG = True  # Whether plots are saved. Only read if PLOTFIG is True.
RESPATH = Path('results') / 'Gait'
RESPATH.mkdir(parents=True, exist_ok=True)

# DataFrames to save errors
ABS_ERROR = pd.DataFrame(
    data=100.,
    index=SLOTS.keys(),
    columns=[str(m) for m in MODEL_LIST] + ['pyHRV'] + ['Neurokit']
)
REL_ABS_ERROR = pd.DataFrame(
    data=1.,
    index=SLOTS.keys(),
    columns=[str(m) for m in MODEL_LIST] + ['pyHRV'] + ['Neurokit']
)

for KEY in SLOTS.keys():

    SUBJECT = SLOTS[KEY][0]
    TRIAL = SLOTS[KEY][1]
    print(f'\nsubject {SUBJECT}, trial {TRIAL}')

    # Save results regularly
    if TRIAL == '1':
        print('Saving DataFrames')
        ABS_ERROR.to_csv(RESPATH / f'abs_error{MODEL_LIST}')
        REL_ABS_ERROR.to_csv(RESPATH / f'rel_abs_error{MODEL_LIST}')

    if SLOTS[KEY] in [['4', '5']]:
        continue
    # LOAD DATA
    trial = get_gait_data(subject=SUBJECT, trial=TRIAL)  # dictionary
    data = trial['data']

    ##########################################################################
    # GROUND TRUTH
    ##########################################################################
    step_start = np.array(trial['RightFootActivity'])[:, 0]
    step_delta = np.diff(step_start)
    gt_mean = np.mean(step_delta) / SFREQ
    print('GT inter-step interval', round(gt_mean, 3), 'seconds')
    # Data keys:
    # First letter: Left or Right foot.
    # Second letter : A = acceleration, R = angular velocity.
    # Third letter : axis (X, Y, Z or V).
    # V is aligned with gravity, others are in the capteur referential.
    X = trial['data']['RAV'].to_numpy()

    # Plot histogram of ground truth step delta
    if PLOTFIG:
        fig_gt, ax_gt = plt.subplots(layout='tight')
        fig_gt.suptitle(
            f's{SUBJECT} t{TRIAL} ground truth of inter-step interval'
        )
        ax_gt.hist(step_delta / SFREQ)
        ax_gt.set_xlim([0, KERNEL_LENGTH])
        ax_gt.set_xlabel('Time (s)')
        ax_gt.set_ylabel('Histogram')
        if SAVEFIG:
            fig_gt.savefig(RESPATH / f's{SUBJECT}t{TRIAL}_GT.svg')

    ##########################################################################
    # ECG METHODS
    ##########################################################################
    if RUN_BASELINES:
        # PYHRV
        # Get R-peaks series using biosppy
        t, _, rpeaks = ecg(X, sampling_rate=SFREQ, show=False)[:3]
        # Compute inter-step interval using R-peak series
        nn_stats = td.nni_parameters(rpeaks=t[rpeaks])
        mean_interval_pyhrv = nn_stats['nni_mean'] / 1000  # seconds

        save_errors(
            mean_interval_pyhrv,
            gt_mean,
            KEY,
            ABS_ERROR,
            REL_ABS_ERROR,
            method='pyHRV'
        )

        # Neurokit
        nk_start = time.time()
        signals, info = nk.ecg_process(
            X,
            sampling_rate=SFREQ,
            method='pantompkins1985'
        )
        mean_interval_nk = (60. / signals['ECG_Rate']).mean()  # seconds
        save_errors(
            mean_interval_nk,
            gt_mean,
            KEY,
            ABS_ERROR,
            REL_ABS_ERROR,
            method='Neurokit'
        )
        nk_time = time.time() - nk_start
        print('Neurokit runtime:', round(nk_time, 3), 'seconds')

    ##########################################################################
    # CDL + UNHaP
    ##########################################################################
    # Reshape X to (n_trials, n_channels, n_times) for alphaCSC
    X = X.reshape(1, 1, *X.shape)
    # DICTIONARY LEARNING
    # Inialize CDL solver
    cdl = BatchCDL(
        # Shape of the dictionary
        N_ATOMS,
        N_TIMES_ATOM,
        rank1=False,
        n_iter=N_ITERATIONS,
        n_jobs=4,
        window=True,
        sort_atoms=True,
        random_state=20,
        lmbd_max='fixed',
        reg=REG,
        verbose=1
    )
    # Fit cdl solver to data
    res = cdl.fit(X)
    D_hat = res._D_hat

    # RECONSTRUCTED SIGNAL
    z_hat = res._z_hat
    X_hat = construct_X_multi(z_hat, D_hat)

    if PLOTFIG:
        # Plot learned atom
        fig_atom, ax_atom = plt.subplots(layout='tight')
        ax_atom.plot(np.arange(D_hat.shape[2]) / SFREQ, D_hat.squeeze())
        ax_atom.set_xlabel('Time (s)')
        ax_atom.set_title('Temporal atom of CDL')
        if SAVEFIG:
            fig_atom.savefig(
                RESPATH / f's{SUBJECT}t{TRIAL}_temp_atom.svg'
            )

    # POINT PROCESSES
    # FaDIn
    if FaDIn in MODEL_LIST:
        # Initialize solver
        solver_fadin = FaDIn(
            n_dim=1,
            kernel=KERNEL_TYPE,
            kernel_length=KERNEL_LENGTH,
            params_optim={'lr': LR},
            delta=DELTA,
            max_iter=N_ITERATIONS_FADIN,
            random_state=RANDOM_STATE,
        )
        # Convert CDL events (# z_hat is normalized in process_cdl_actis)
        events_nm = process_cdl_actis(z_hat, freq=SFREQ, reg=REG, marked=False)
        T = int(len(z_hat.squeeze()) / SFREQ) + 1
        # fit solver to events
        solver_fadin.fit(events_nm, T)
        # Compute and save errors
        mean_interval_fadin = solver_fadin.kernel_[0].item()
        save_errors(
            mean_interval_fadin,
            gt_mean,
            KEY,
            ABS_ERROR,
            REL_ABS_ERROR,
            str(FaDIn)
        )
        # Plot learned kernel
        if PLOTFIG:
            plot(
                solver_fadin,
                plotfig=True,
                bl_noise=False,
                title=f'FaDIn subject {SUBJECT}, trial {TRIAL}',
                savefig=RESPATH / f's{SUBJECT}t{TRIAL}_fadin_kernel.svg'
            )
        del solver_fadin
    # UNHaP
    if UNHaP in MODEL_LIST or 'StocUNHaP' in MODEL_LIST:
        # Initialize solver
        solver_unhap = UNHaP(
            n_dim=1,
            kernel=KERNEL_TYPE,
            kernel_length=KERNEL_LENGTH,
            params_optim={'lr': LR},
            delta=DELTA,
            max_iter=N_ITERATIONS_FADIN,
            random_state=RANDOM_STATE,
            batch_rho=BATCH_RHO,
            init=INIT
        )
        # Convert CDL events (# z_hat is normalized in process_cdl_actis)
        events = process_cdl_actis(z_hat, freq=SFREQ, reg=REG, marked=True)
        T = len(z_hat.squeeze()) / SFREQ
        # Fit solver to events
        solver_unhap.fit(events, T)
        # Compute and save errors
        mean_interval_unhap = solver_unhap.kernel_[0].item()
        save_errors(
            mean_interval_unhap,
            gt_mean,
            KEY,
            ABS_ERROR,
            REL_ABS_ERROR,
            str(UNHaP)
        )
        if PLOTFIG:
            # Plot learned kernel
            solver_unhap.kernel_length = KERNEL_LENGTH
            fig_kernel, ax_kernel = plot(
                solver_unhap,
                plotfig=True,
                bl_noise=True,
                title=f'UNHaP subject {SUBJECT}, trial {TRIAL}',
                savefig=RESPATH / f's{SUBJECT}t{TRIAL}_kernel.svg'
            )
            # TODO: Twin x-axis for independent y-axes
            # axes = [ax_kernel, ax_kernel.twinx()]
            # axes[1].hist(step_delta / SFREQ)

            # Compute PP intensity function and rho on data
            n_grid = int(1 / solver_unhap.delta * T) + 1
            events_grid, events_grid_wm = projected_grid_marked(
                    events, solver_unhap.delta, n_grid
            )
            discretization = torch.linspace(
                0, solver_unhap.kernel_length, solver_unhap.L
            )
            intens = solver_unhap.kernel_model.intensity_eval(
                solver_unhap.baseline_,
                solver_unhap.alpha_,
                solver_unhap.kernel_,
                events_grid,
                discretization
            ).detach().numpy().squeeze()
            rho = solver_unhap.rho_.numpy().squeeze()

            # Plot signal, CDL activations, Hawkes intensity
            fig_hat, ax_hat = plt.subplots(
                3, figsize=(12, 8), sharex='all', layout='tight'
            )
            fig_hat.suptitle(f'subject {SUBJECT}, trial {TRIAL}')
            # Original and constructed signal
            ax_hat[0].plot(X.squeeze(), label='ORIGINAL')
            ax_hat[0].plot(X_hat.squeeze(), label='RECONSTRUCTED')
            ax_hat[0].set_xlabel('time (x10ms)')
            ax_hat[0].legend(loc='best')
            ax_hat[0].set_title('Signal')
            # Activations
            ax_hat[1].stem(
                z_hat[0][0], linefmt='red', label='noise activations'
            )
            ax_hat[1].stem(
                z_hat.squeeze() * rho[:len(z_hat.squeeze())],
                linefmt='green',
                label='PP activations'
            )
            ax_hat[1].legend(loc='best')
            ax_hat[1].set_title('CDL activations')
            # Intensity
            ax_hat[2].plot(
                DELTA * SFREQ * np.arange(len(intens)),
                rho * intens, label='Intensity function'
            )
            ax_hat[2].plot(
                DELTA * SFREQ * np.arange(len(rho)),
                rho, label='rho'
            )
            ax_hat[2].set_title('Intensity function')
            ax_hat[2].legend(loc='best')
            if SAVEFIG:
                fig_hat.savefig(
                    RESPATH / f's{SUBJECT}t{TRIAL}_actis_intens.svg'
                )
        del solver_unhap, cdl, res

    if 'StocUNHaP' in MODEL_LIST:
        # Initialize StocUNHaP solver
        solver_stoc_unhap = UNHaP(
            n_dim=1,
            kernel=KERNEL_TYPE,
            kernel_length=KERNEL_LENGTH,
            params_optim={'lr': LR},
            delta=DELTA,
            max_iter=N_ITERATIONS_FADIN,
            random_state=RANDOM_STATE,
            init=INIT,
            batch_rho=BATCH_RHO,
            stoc_classif=True
        )
        # Convert CDL events (# z_hat is normalized in process_cdl_actis)
        events = process_cdl_actis(z_hat, freq=SFREQ, reg=REG, marked=True)
        T = len(z_hat.squeeze()) / SFREQ
        # Fit solver to events
        solver_stoc_unhap.fit(events, T)
        # Compute and save errors
        mean_interval_stoc_unhap = solver_stoc_unhap.kernel_[0].item()
        save_errors(
            mean_interval_stoc_unhap,
            gt_mean,
            KEY,
            ABS_ERROR,
            REL_ABS_ERROR,
            'StocUNHaP'
        )
        if PLOTFIG:
            # Plot learned kernel
            fig_kernel, ax_kernel = plot(
                solver_stoc_unhap,
                plotfig=True,
                bl_noise=True,
                title=f'StocUNHaP subject {SUBJECT}, trial {TRIAL}',
                savefig=RESPATH / f's{SUBJECT}t{TRIAL}_stoc_kernel.svg'
            )
    if SHOWFIG:
        plt.show(block=True)

# Compute error statistics
mae = ABS_ERROR.mean(axis=0)
mrae = REL_ABS_ERROR.mean(axis=0)
std_ae = ABS_ERROR.std(axis=0)
std_rae = REL_ABS_ERROR.std(axis=0)
print('Mean Absolute Error', mae)
print('std Absolute Error', std_ae)

# Save metrics
ABS_ERROR.to_csv(RESPATH / f'abs_error{MODEL_LIST}.csv')
REL_ABS_ERROR.to_csv(RESPATH / f'rel_abs_error{MODEL_LIST}.csv')
