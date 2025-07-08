"""Apply Hawkes processes solvers to ECG data.
Done following vitaldb example notebook:
https://github.com/vitaldb/examples/blob/master/vitaldb_python_library.ipynb .

Iteratively runs a set of chosen solvers on 20 5-minute ECG intervals.
Computes the mean absolute error and mean relative absolute error between
the heart rate estimated by each solver and the ground truth heart rate.

"""
import time
from pathlib import Path


import numpy as np
import pandas as pd
from joblib import Memory

from pyhrv.hrv import hrv
import neurokit2 as nk

# %% Imports
from fadin.solver import FaDIn, UNHaP

from utils_ecg import dataload, RESAMP_FREQ, ECG_CH_NAME
from utils_ecg import peak_events, plot_signal_rho
from utils_ecg import process_cdl_actis, fadin_fit, cdl

memory = Memory(location='__cache__', verbose=10)

# Segment time intervals for ECG data from vitaldb
SLOTS = {
    0: ['https://api.vitaldb.net/0010.vital', 300, 305],
    2: ['https://api.vitaldb.net/0001.vital', 35, 40],
    1: ['https://api.vitaldb.net/0001.vital', 20, 25],
    3: ['https://api.vitaldb.net/0001.vital', 55, 60],
    4: ['https://api.vitaldb.net/0002.vital', 55, 60],
    5: ['https://api.vitaldb.net/0002.vital', 120, 125],  # 125, 130
    6: ['https://api.vitaldb.net/0002.vital', 150, 155],
    7: ['https://api.vitaldb.net/0003.vital', 45, 50],
    8: ['https://api.vitaldb.net/0004.vital', 55, 60],
    9: ['https://api.vitaldb.net/0004.vital', 105, 110],
    10: ['https://api.vitaldb.net/0004.vital', 120, 125],
    12: ['https://api.vitaldb.net/0006.vital', 10, 15],
    13: ['https://api.vitaldb.net/0006.vital', 25, 30],
    14: ['https://api.vitaldb.net/0007.vital', 45, 50],
    15: ['https://api.vitaldb.net/0007.vital', 50, 55],
    16: ['https://api.vitaldb.net/0007.vital', 115, 120],
    17: ['https://api.vitaldb.net/0010.vital', 120, 125],
    18: ['https://api.vitaldb.net/0010.vital', 175, 180],
    19: ['https://api.vitaldb.net/0010.vital', 230, 235],
}

# FaDIn solver hyperparameters
KERNEL_LENGTH = 1.5
DELTA = 0.01
LR = 1e-3
L = int(1 / DELTA)
KERNEL_TYPE = 'truncated_gaussian'
N_ITERATIONS_FADIN = 50_000
RANDOM_STATE = 40  # Only used if model=FaDIn or smart_init=False

# Dictionary learning hyperparameters
N_ATOMS = 1
N_TIMES_ATOMS = int(RESAMP_FREQ)  # int(0.7 * RESAMP_FREQ)
N_ITERATIONS = 40
SORT_ATOMS = True  # Return the atoms in decreasing signal explanability
WINDOW = True  # Whether the atoms are forced to be = 0 on their edges
RANDOM_SEED = 0
REG = 2.5  # reg parameter of BatchCDL function 1.5, 3

# Module choice
MULTI_START = False
CDL_OR_PEAKS = 'cdl'  # 'cdl' or 'peaks'
RUN_BASELINES = False
RUN_SOLVER = True

# Change following line to choose which solver models will be used.
MODEL_LIST = [UNHaP, 'StocUNHaP']
BATCH_RHO = 1000
INIT = 'moment_matching_max'  # 'moment_matching_mean', 'moment_matching_max', 'random'

# CDL activations processing parameters:
MARKED_EVENTS = True
ACTI_THRESH = 0.
SPARSIFY_ACTIS = False

# Path for saving figure. If set to None, figures are not saved.
SAVEFIG = None  # 'ECG_plots/august_2024/'
RESPATH = Path('results') / 'ECG' / CDL_OR_PEAKS
RESPATH.mkdir(parents=True, exist_ok=True)

if MULTI_START:
    # indexes = np.squeeze(
    #   np.repeat(list(SLOTS), 4).reshape((1, -1), order='F')
    # )
    # print('indexes', indexes)
    df_runs = pd.DataFrame(
        columns=['slot', 'init_mode', 'log_likelihood', 'abs_error',
                 'rel_abs_error', 'bl_noise', 'bl_hawkes', 'alpha_hawkes',
                 'mean', 'std'],
    )


# Functions

# Initialize Dataframes to save errors
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
RUNTIME = pd.DataFrame(
    data=1.,
    index=SLOTS.keys(),
    columns=[str(m) for m in MODEL_LIST] + ['pyHRV'] + ['Neurokit']
)
##############################################################################
# PYHRV AND NEUROKIT
##############################################################################
if RUN_BASELINES:
    for KEY in SLOTS.keys():
        FILEURL = SLOTS[KEY][0]
        TIME_START = SLOTS[KEY][1]  # minutes
        TIME_STOP = SLOTS[KEY][2]  # minutes
        print('File', FILEURL)
        print(TIME_START, '-', TIME_STOP, 'time interval')

        # Load data
        T, clean_df, cleandf_hr = dataload(
            FILEURL,
            time_start=TIME_START,
            time_stop=TIME_STOP,
            verbose=False,
            plotfig=False
        )
        print('Ground truth heart rate:', np.mean(cleandf_hr))
        # pyHRV
        pyhrv_start = time.time()
        hrv_stats = hrv(
            signal=clean_df[ECG_CH_NAME].to_numpy(), sampling_rate=RESAMP_FREQ
        )
        mean_hr_pyhrv = hrv_stats['hr_mean']
        print('pyHRV Mean Heart rate:', mean_hr_pyhrv)

        ae_pyhrv = np.abs(mean_hr_pyhrv - np.mean(cleandf_hr))
        rae_pyhrv = ae_pyhrv / np.mean(cleandf_hr)
        ABS_ERROR.loc[KEY, 'pyHRV'] = ae_pyhrv
        REL_ABS_ERROR.loc[KEY, 'pyHRV'] = rae_pyhrv
        pyhrv_time = time.time() - pyhrv_start
        print('pyHRV time:', pyhrv_time)
        RUNTIME.loc[KEY, 'pyHRV'] = pyhrv_time
        # Neurokit
        nk_start = time.time()
        signals, info = nk.ecg_process(
            clean_df[ECG_CH_NAME].to_numpy(dtype=np.float32),
            sampling_rate=RESAMP_FREQ,
            method='neurokit'
        )
        mean_hr_nk = signals['ECG_Rate'].mean()
        print('Neurokit Mean Heart rate:', mean_hr_nk)

        ae_nk = np.abs(mean_hr_nk - np.mean(cleandf_hr))
        rae_nk = ae_nk / np.mean(cleandf_hr)
        ABS_ERROR.loc[KEY, 'Neurokit'] = ae_nk
        REL_ABS_ERROR.loc[KEY, 'Neurokit'] = rae_nk
        nk_time = time.time() - nk_start
        print('Neurokit time:', nk_time)
        RUNTIME.loc[KEY, 'Neurokit'] = nk_time

    ABS_ERROR.to_csv(f'{RESPATH}abs_error_pyHRV_neurokit.csv')
    REL_ABS_ERROR.to_csv(f'{RESPATH}rel_abs_error_pyHRV_neurokit.csv')
    RUNTIME.to_csv(f'{RESPATH}runtime_pyHRV_neurokit.csv')

    print('pyHRV mean absolute error:', ABS_ERROR['pyHRV'].mean())
    print('pyHRV absolute error std:', ABS_ERROR['pyHRV'].std())
    print('pyHRV mean relative absolute error:', REL_ABS_ERROR['pyHRV'].mean())
    print('pyHRV relative absolute error std:', REL_ABS_ERROR['pyHRV'].std())
    print('Neurokit mean absolute error:', ABS_ERROR['Neurokit'].mean())
    print('Neurokit absolute error std:', ABS_ERROR['Neurokit'].std())
    print(
        'Neurokit mean relative absolute error:',
        REL_ABS_ERROR['Neurokit'].mean()
    )
    print(
        'Neurokit relative absolute error std:',
        REL_ABS_ERROR['Neurokit'].std()
    )
    print('PyHRV mean runtime:', RUNTIME['pyHRV'].mean())
    print('PyHRV std runtime:', RUNTIME['pyHRV'].std())
    print('Neurokit mean runtime:', RUNTIME['Neurokit'].mean())
    print('Neurokit std runtime:', RUNTIME['Neurokit'].std())
##############################################################################
# EVENT DETECTION + UNHAP
##############################################################################
for KEY in SLOTS.keys():
    FILEURL = SLOTS[KEY][0]
    TIME_START = SLOTS[KEY][1]  # minutes
    TIME_STOP = SLOTS[KEY][2]  # minutes
    print('File', FILEURL)
    print(TIME_START, '-', TIME_STOP, 'time interval')

    # Load data
    T, clean_df, cleandf_hr = dataload(
        FILEURL,
        time_start=TIME_START,
        time_stop=TIME_STOP,
        verbose=False,
        plotfig=False
    )
    model_start = time.time()
    # Naive event detection
    # events_t = naive_events(
    #     clean_df, time_start=TIME_START, thresh=0.5, plotfig=False
    #     )

    if CDL_OR_PEAKS == 'peaks':
        # Event detection using peaks
        acti_events = peak_events(clean_df, RESAMP_FREQ, marked=MARKED_EVENTS)

    if CDL_OR_PEAKS == 'cdl':
        # Event detection using convolutional dictionary learning
        cdl_actis = cdl(clean_df, REG, KEY, plotfig=False, save_fig=SAVEFIG)

        # Process activations
        acti_events = process_cdl_actis(
            cdl_actis,
            freq=RESAMP_FREQ,
            reg=REG,
            threshold=ACTI_THRESH,
            sparsify=SPARSIFY_ACTIS,
            marked=MARKED_EVENTS,
            save_fig=SAVEFIG
        )

    # Run solver on events
    for model in MODEL_LIST:
        if not RUN_SOLVER:
            break
        if MULTI_START:
            df_runs.to_csv(f'results/ECG/{MODEL_LIST}df_runs_ecg.csv')
        assert model != FaDIn or not MARKED_EVENTS, (
            "FaDIn cannot be fed marked events. Change MARKED_EVENTS to False."
        )
        print(str(model))
        if model in [UNHaP, 'StocUNHaP']:
            stoc_classif = model == 'StocUNHaP'
            if not MULTI_START:
                s, ae, rae = fadin_fit(
                    cleandf_hr,
                    TIME_START,
                    TIME_STOP,
                    model=UNHaP,
                    events=acti_events,
                    T=T,
                    figtitle=f'{model} on ECG CDL raw actis',
                    figname=f'{FILEURL[-8:-6]}{model}_raw_{TIME_START}-'
                            f'{TIME_STOP}min_lmbd{REG}_smartinit',
                    verbose=True,
                    init=INIT,
                    batch_rho=BATCH_RHO,
                    stoc_classif=stoc_classif,
                )
            if MULTI_START:
                # Run multiple starts:
                # moment matching mean, moment matching max, and random init.
                s, ae, rae, ll = fadin_fit(
                    cleandf_hr,
                    TIME_START,
                    TIME_STOP,
                    model=model,
                    events=acti_events,
                    T=T,
                    figtitle=f'{model} on ECG CDL raw actis',
                    figname=f'{FILEURL[-8:-6]}{model}_raw_{TIME_START}-'
                            f'{TIME_STOP}min_lmbd{REG}_smartinit',
                    verbose=True,
                    init='moment_matching_mean',
                    batch_rho=BATCH_RHO,
                    return_ll=True
                )
                results_mmmean = pd.DataFrame.from_dict({
                    'slot': [KEY],
                    'init_mode': ['moment_matching_mean'],
                    'log_likelihood': [ll],
                    'abs_error': [ae],
                    'rel_abs_error': [rae],
                    'bl_noise': [s.baseline_noise.item()],
                    'bl_hawkes': [s.baseline.item()],
                    'alpha_hawkes': [s.alpha.item()],
                    'mean': [s.kernel_params_fixed[0].item()],
                    'std': [s.kernel_params_fixed[1].item()]
                }
                )

                s, ae, rae, ll = fadin_fit(
                    cleandf_hr,
                    TIME_START,
                    TIME_STOP,
                    model=model,
                    events=acti_events,
                    T=T,
                    figtitle=f'{model} on ECG CDL raw actis',
                    figname=f'{FILEURL[-8:-6]}{model}_raw_{TIME_START}-'
                            f'{TIME_STOP}min_lmbd{REG}_smartinit',
                    verbose=True,
                    init='moment_matching_max',
                    batch_rho=BATCH_RHO,
                    return_ll=True
                )
                results_mmmax = pd.DataFrame.from_dict({
                    'slot': [KEY],
                    'init_mode': [INIT],
                    'log_likelihood': [ll],
                    'abs_error': [ae],
                    'rel_abs_error': [rae],
                    'bl_noise': [s.baseline_noise.item()],
                    'bl_hawkes': [s.baseline.item()],
                    'alpha_hawkes': [s.alpha.item()],
                    'mean': [s.kernel_params_fixed[0].item()],
                    'std': [s.kernel_params_fixed[1].item()]
                }
                )

                s, ae, rae, ll = fadin_fit(
                    cleandf_hr,
                    TIME_START,
                    TIME_STOP,
                    model=model,
                    events=acti_events,
                    T=T,
                    figtitle=f'{model} on ECG CDL raw actis',
                    figname=f'{FILEURL[-8:-6]}{model}_raw_{TIME_START}-'
                            f'{TIME_STOP}min_lmbd{REG}_smartinit',
                    verbose=True,
                    init='random',
                    random_state=1,
                    batch_rho=BATCH_RHO,
                    return_ll=True
                )
                results_random1 = pd.DataFrame(data={
                    'slot': [KEY],
                    'init_mode': ['random1'],
                    'log_likelihood': [ll],
                    'abs_error': [ae],
                    'rel_abs_error': [rae],
                    'bl_noise': [s.baseline_noise.item()],
                    'bl_hawkes': [s.baseline.item()],
                    'alpha_hawkes': [s.alpha.item()],
                    'mean': [s.kernel_params_fixed[0].item()],
                    'std': [s.kernel_params_fixed[1].item()]
                }
                )
                s, ae, rae, ll = fadin_fit(
                    cleandf_hr,
                    TIME_START,
                    TIME_STOP,
                    model=model,
                    events=acti_events,
                    T=T,
                    figtitle=f'{model} on ECG CDL raw actis',
                    figname=f'{FILEURL[-8:-6]}{model}_raw_{TIME_START}-'
                            f'{TIME_STOP}min_lmbd{REG}_smartinit',
                    verbose=True,
                    init='random',
                    random_state=12,
                    batch_rho=BATCH_RHO,
                    return_ll=True
                )
                results_random2 = pd.DataFrame(data={
                    'slot': [KEY],
                    'init_mode': ['random2'],
                    'log_likelihood': [ll],
                    'abs_error': [ae],
                    'rel_abs_error': [rae],
                    'bl_noise': [s.baseline_noise.item()],
                    'bl_hawkes': [s.baseline.item()],
                    'alpha_hawkes': [s.alpha.item()],
                    'mean': [s.kernel_params_fixed[0].item()],
                    'std': [s.kernel_params_fixed[1].item()]
                }
                )
                df_runs = pd.concat([df_runs, results_mmmean, results_mmmax,
                                     results_random1, results_random2],
                                    ignore_index=True)

                ae = np.min(
                    [results_mmmean['abs_error'],
                     results_mmmax['abs_error'],
                     results_random1['abs_error'],
                     results_random2['abs_error']]
                )
                rae = np.min(
                    [results_mmmean['rel_abs_error'],
                     results_mmmax['rel_abs_error'],
                     results_random1['rel_abs_error'],
                     results_random2['rel_abs_error']]
                )

        else:
            # FaDIn
            s, ae, rae = fadin_fit(
                cleandf_hr,
                TIME_START,
                TIME_STOP,
                model=model,
                events=acti_events,
                T=T,
                figtitle=f'{model} on ECG CDL raw actis',
                figname=f'{FILEURL[-8:-6]}{model}_raw_{TIME_START}-'
                        f'{TIME_STOP}min_lmbd{REG}_smartinit',
                verbose=True,
                init='random',
            )
        model_time = time.time()
        print('Model time:', model_time)
        ABS_ERROR.loc[KEY, str(model)] = ae
        REL_ABS_ERROR.loc[KEY, str(model)] = rae
        RUNTIME.loc[KEY, str(model)] = model_time
        if model == UNHaP and KEY == 0:
            plot_signal_rho(
                s, begin_s=206, end_s=209,
                figtitle='figsignalrhozoom2', save_fig=SAVEFIG
            )
    # plt.show(block=True)
    # if KEY == 0:
    #     plot_signal_cdl(cdl_actis, clean_df, begin_s=206, end_s=209,
    #                     figtitle='figsignalactizoom2', save_fig=SAVEFIG)
    #     plt.show(block=True)

mae = ABS_ERROR.mean(axis=0)
mrae = REL_ABS_ERROR.mean(axis=0)
std_ae = ABS_ERROR.std(axis=0)
std_rae = REL_ABS_ERROR.std(axis=0)
ABS_ERROR.to_csv(f'{RESPATH}abs_error{MODEL_LIST}.csv')
REL_ABS_ERROR.to_csv(f'{RESPATH}rel_abs_error{MODEL_LIST}.csv')
RUNTIME.to_csv(f'{RESPATH}runtime{MODEL_LIST}.csv')
print('Absolute error: \n', ABS_ERROR)
print('Relative absolute error: \n', REL_ABS_ERROR)
print('Stats on absolute error: \n', ABS_ERROR.describe())
print('Stats on relative absolute error: \n', REL_ABS_ERROR.describe())
print('Mean Absolute Error: \n', mae, )
print('std of Absolute Error: \n', std_ae, '\n')
print('Mean Relative Absolute Error: \n', mrae)
print('std of Relative Absolute Error: \n', std_rae, '\n')
