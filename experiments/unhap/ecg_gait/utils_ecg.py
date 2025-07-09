import vitaldb
import pandas as pd
import numpy as np
import alphacsc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from joblib import Memory
import torch

from fadin.solver import FaDIn, UNHaP
from fadin.loss_and_gradient import discrete_ll_loss_conv
from fadin.utils.utils import projected_grid_marked
from fadin.kernels import DiscreteKernelFiniteSupport

memory = Memory()

# Dataload parameters
ECG_CH_NAME = "SNUADC/ECG_II"  # Name of ECG vitaldb track
HR_CH_NAME = "Solar8000/HR"  # Name of heart rate vitaldb track
RESAMP_FREQ = 200  # Frequency (Hz) of data downsampling 200

# FaDIn solver hyperparameters
KERNEL_LENGTH = 1.5
DELTA = 0.01
LR = 1e-3
L = int(1 / DELTA)
KERNEL_TYPE = "truncated_gaussian"
N_ITERATIONS_FADIN = 50_000
RANDOM_STATE = 40  # Only used if model=FaDIn or smart_init=False

# Dictionary learning hyperparameters
N_ATOMS = 1
N_TIMES_ATOMS = int(RESAMP_FREQ)  # int(0.7 * RESAMP_FREQ)
N_ITERATIONS = 40
SORT_ATOMS = True  # Return the atoms in decreasing signal explanability
WINDOW = True  # Whether the atoms are forced to be = 0 on their edges
RANDOM_SEED = 0

SLOTS = {
    0: ["https://api.vitaldb.net/0010.vital", 300, 305],
    2: ["https://api.vitaldb.net/0001.vital", 35, 40],
    1: ["https://api.vitaldb.net/0001.vital", 20, 25],
    3: ["https://api.vitaldb.net/0001.vital", 55, 60],
    4: ["https://api.vitaldb.net/0002.vital", 55, 60],
    5: ["https://api.vitaldb.net/0002.vital", 120, 125],  # 125, 130
    6: ["https://api.vitaldb.net/0002.vital", 150, 155],
    7: ["https://api.vitaldb.net/0003.vital", 45, 50],
    8: ["https://api.vitaldb.net/0004.vital", 55, 60],
    9: ["https://api.vitaldb.net/0004.vital", 105, 110],
    10: ["https://api.vitaldb.net/0004.vital", 120, 125],
    12: ["https://api.vitaldb.net/0006.vital", 10, 15],
    13: ["https://api.vitaldb.net/0006.vital", 25, 30],
    14: ["https://api.vitaldb.net/0007.vital", 45, 50],
    15: ["https://api.vitaldb.net/0007.vital", 50, 55],
    16: ["https://api.vitaldb.net/0007.vital", 115, 120],
    17: ["https://api.vitaldb.net/0010.vital", 120, 125],
    18: ["https://api.vitaldb.net/0010.vital", 175, 180],
    19: ["https://api.vitaldb.net/0010.vital", 230, 235],
}


def plot_signal_rho(solver, begin_s, end_s, figtitle="", save_fig=None):
    rho = solver.rho_[0]
    print("rho", rho)
    print("shape rho", rho.shape)
    begin_rho = int(begin_s / solver.delta)
    end_rho = int(end_s / solver.delta)
    subrho = rho[begin_rho:end_rho]
    figrho, axrho = plt.subplots(1, 1, figsize=(4, 3), squeeze=True)
    # Rho
    axrho.stem(
        subrho,
        label="rho",
        markerfmt="go",
        linefmt="g-",
        basefmt="",
    )
    axrho.tick_params(axis="both", which="major", labelsize="xx-large")
    axrho.legend(loc="best", fontsize="xx-large")
    axrho.set_xlabel("Time (seconds)", size="xx-large")
    axrho.set_title("rho", size="xx-large")
    if save_fig is not None:
        figrho.savefig(f"{save_fig}signal-rho{figtitle}stem.svg")
    figrho.tight_layout()
    figrho.show()


def plot_signal_cdl(
    z_hat, clean_df, begin_s=None, end_s=None, figtitle="", save_fig=None
):
    if begin_s is None:
        begin_s = 0
    if end_s is None:
        end_s = len(clean_df) // RESAMP_FREQ
    subdf = clean_df.iloc[
        begin_s * RESAMP_FREQ: end_s * RESAMP_FREQ
    ]
    subzhat = z_hat[0][0, begin_s * RESAMP_FREQ: end_s * RESAMP_FREQ]
    fig1, ax1 = plt.subplots(1, 1, figsize=(4, 3), squeeze=True)
    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4), squeeze=True)
    # Original signal
    ax2.plot(
        subdf["Time"],
        subdf[ECG_CH_NAME],
        label="ECG",
        linewidth=3,
        color="black"
    )
    # CDL activations
    pos_index = np.argwhere(subzhat > 0).squeeze()
    ax1.stem(
        subdf.iloc[pos_index]["Time"],
        subzhat[pos_index],
        markerfmt="bo",
        linefmt="b-",
        basefmt="",
        label="CDL activations",
    )
    ax1.hlines(
        y=0,
        xmin=subdf.iloc[0]["Time"],
        xmax=subdf.iloc[len(subdf) - 1]["Time"]
    )
    ax1.tick_params(axis="both", which="major", labelsize="xx-large")
    ax1.legend(loc="best", fontsize="xx-large")
    ax1.set_xlabel("Time (seconds)", size="xx-large")
    ax1.set_title("Dictionary activations events", size="xx-large")
    fig1.savefig(f"{save_fig}signal-actis{figtitle}stem.svg")
    fig1.tight_layout()
    fig1.show()

    ax2.tick_params(axis="both", which="major", labelsize="xx-large")
    ax2.legend(loc="best", fontsize="xx-large")
    ax2.set_xlabel("Time (seconds)", size="xx-large")
    ax2.set_ylabel("mV", size="xx-large")
    ax2.set_title("ECG signal", size="xx-large")
    fig2.savefig(f"{save_fig}signal{figtitle}.svg")
    fig2.tight_layout()
    fig2.show()


def dataload(fileurl, time_start, time_stop, verbose=False, plotfig=False):
    """Load vitaldb recording, extract [time_start, time_stop] of ECG and
    heart rate channels.
    Parameters
    ----------

    fileurl: str
        URL of file loaded by `vitaldb.VitalFile`.

    time_start: float
        Beginning of time interval to load, in minutes.
        If `time_start` and `time_end` are set to None, the entire recording
        is loaded.

    time_end: float
        End of time interval to load, in minutes.
        If `time_start` and `time_end` are set to None, the entire recording
        is loaded.

    verbose: bool, default=`False`

    plot: bool, default=`False`
        If set to True, the loaded data is plotted.

    Returns:
    T, clean_df, cleandf_hr
    T (float) is the time interval length in seconds
    clean_df (pandas DataFrame) is a DataFrame with time index, and 2 columns:
    ECG and heart rate.
    cleandf_hr (pandas DataFrame) is a DataFrame with time index and 1 column:
    heart rate with no NaN value.
    """

    # Read vitaldb file
    vf = vitaldb.VitalFile(fileurl)  # 'https://api.vitaldb.net/0001.vital'
    #  Load data in DataFrame format
    df = vf.to_pandas(
        [ECG_CH_NAME, HR_CH_NAME],
        interval=1 / RESAMP_FREQ,
        return_datetime=True
    )
    # Select clean time interval for event detection
    if time_start is not None and time_stop is not None:
        T = int(60 * (time_stop - time_start))
        clean_df = df.iloc[
            int(60 * RESAMP_FREQ * time_start): int(
                RESAMP_FREQ * (60 * time_start + T)
            )
        ]
    else:
        T = int(len(df) / RESAMP_FREQ)
        clean_df = df
    cleandf_hr = clean_df.loc[pd.notna(clean_df[HR_CH_NAME])][HR_CH_NAME]

    if verbose:
        # Print file metadata
        print(f"Recording duration: {vf.dtend - vf.dtstart:.3f} seconds")
        print(f"{len(vf.trks)} tracks")
        print("Recording tracks", vf.get_track_names())

        # ECG metadata
        track = vf.trks[ECG_CH_NAME]
        print("ECG Channel name", track.name)
        print(f"Heart rate track type: {track.type} (1:wav, 2:num, 5:str)")
        print("ECG sampling rate", track.srate, "Hz")
        print("Channel unit", track.unit)

        # Heart rate medatada
        track_hr = vf.trks[HR_CH_NAME]
        print("Heart rate channel name:", track_hr.name)
        print(f"Heart rate track type: {track_hr.type} (1:wav, 2:num, 5:str)")
        print("Heart rate unit", track_hr.unit)

    if plotfig:
        subdf_hr = df.loc[pd.notna(df[HR_CH_NAME])]  # for visual plot
        fig_hr = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig_hr.add_trace(
            go.Scattergl(
                x=subdf_hr["Time"],
                y=subdf_hr[HR_CH_NAME],
                name="Heart rate",
            ),
            row=1,
            col=1,
        )
        fig_hr.add_trace(
            go.Scattergl(
                x=clean_df["Time"],
                y=clean_df[ECG_CH_NAME],
                name="ECG"
            ),
            row=2,
            col=1,
        )
        fig_hr.update_layout(title=fileurl)
        fig_hr.show(renderer="browser")

    return T, clean_df, cleandf_hr


def peak_events(clean_df, freq, marked):
    """Detect heartbeats on ECG using scipy.find_peaks.

    Each peak detected in the ECG signal is considered as a event.
    The mark of each event is its peak amplitude.

    Parameters
    ----------
    clean_df: pandas DataFrame
        Signal DataFrame. index=timestamps, column ECG_NAME contain ECG values.
    freq: float
        Sampling frequency of the ECG signal.
    marked: boolean
        If set to True, output is a numpy array containing
        detected events timestamps and their associated marks.
        If set to False, output is a numpy array containing
        detected events timestamps.

    Returns
    -------
    events_t: numpy array
        Timestamps of detected events in seconds.
        Activation events on which TPP solver will be fitted.
        If `marked=True`, events are listed as their timestamps.
        If `marked=False`, events are listed as [timestamp, mark].

    """
    # Detect ECG peaks
    peaks_ind, props = find_peaks(clean_df[ECG_CH_NAME], height=0.1)

    if marked:
        marks = props["peak_heights"]
        return [
            np.array(
                [[peaks_ind[i] / freq, marks[i]]
                 for i in range(len(peaks_ind))]
            )
        ]
    else:
        # Convert index to time in seconds
        events_t = np.reshape(peaks_ind / freq, (1, len(peaks_ind)))
        return events_t


def naive_events(clean_df, time_start, thresh=0.5, plotfig=False):
    """Detect heartbeats on ECG using a naive thresholding method.
    Each time block where signal > thresh is considered as a heartbeat.

    Parameters
    ----------
    clean_df: pandas DataFrame
        Signal DataFrame. index=timestamps, column ECG_NAME contain ECG values.

    time_start: float
        Beginning time of events time interval, in minutes.

    thresh: float, default=0.5
        Threshold value for heartbeat detection, in mV.

    plot: boolean, default=False
        If set to True, ECG and detected events are plotted.

    Returns
    -------
    events_t: numpy array
        Timestamps of detected events in seconds.
    """
    # Detect ECG peaks, ie where ECG > threshold
    clean_df["discretized"] = np.int32(
        np.where(clean_df[ECG_CH_NAME] > thresh, True, False)
    )
    # Mark beginning of ECG peaks for FaDIn events
    # When contiguous indexes are < threshold, only keep first index of block
    clean_df["events"] = np.roll(
        np.int32(
            np.where(
                np.diff(clean_df["discretized"], append=np.array([0])) == 1.0,
                True,
                False,
            )
        ),
        1,
    )
    # Retrieve timestamps of ECG peaks
    events_ind = clean_df.loc[clean_df["events"] == 1].index.to_numpy()
    events_t = np.reshape(
        events_ind / RESAMP_FREQ - 60 * time_start, (1, len(events_ind))
    )  # seconds

    if plotfig:
        # Plot ECG and detected events
        fig2 = go.Figure(
            go.Scattergl(
                x=clean_df["Time"],
                y=clean_df[ECG_CH_NAME],
                name=ECG_CH_NAME
            )
        )
        fig2.add_trace(
            go.Scattergl(
                x=clean_df["Time"],
                y=clean_df["discretized"],
                name="discretized"
            )
        )
        fig2.add_trace(
            go.Scattergl(
                x=clean_df["Time"],
                y=clean_df["events"],
                name="events",
                line_color="red",
            )
        )
        fig2.show(renderer="browser")

    return events_t


def process_cdl_actis(
    z_hat,
    freq,
    reg,
    threshold=0.0,
    sparsify=False,
    marked=False,
    plothist=False,
    save_fig=None,
):
    """Turns activations into arguments for FaDIn solvers.
    Events with activation < `threshold` will be discarded.
    If sparsify=`True`, consecutive activation blocks will be discarded and the
    first activation of each activation block will be kept.

    Parameters
    ----------
    z_hat: numpy array
        activation vector of learned dictionary.

    freq: `float` or `int`
        sampling frequency of z_hat.

    reg: float
        Regularization parameter of CDL.

    threshold: `float` (default 0)
        Threshold to apply to activations.
        Only activations > threshold will be kept.

    sparsify: `bool` (default `False`)
        How to handle blocks of consecutive non-zero activations.
        if set to `True`, only the first activation of each block is kept.
        If set to `False`, blocks are left untouched.

    marked: `bool` (default `False`)
        If set to `False`, output is a numpy array containing
        processed activations timestamps, suitable for FaDIn.
        If set to `True`, output is a numpy array containing
        processed activations timestamps and their associated marks,
        suitable for MarkedFaDIn.

    plothist: `bool` (default `False`)
        Whether activation blocks length histogram is plotted. The length
        in this histogram are computed after thresholding and before
        sparsifying.

    save_fig: string or None, default=None.
        Path where to save histogram plot.
        If set to None, histogram is not saved.

    Returns
    -------
    events: numpy array of dimension (1, n_events)
        Activation events on which FaDIn solver will be fitted.
        If `marked=True`, events are listed as their timestamps.
        If `marked=False`, events are listed as [timestamp, mark].
    """
    z = z_hat[0][0].copy()
    # Normalize activations by max absolute value to have marks in [0, 1]
    z = np.abs(z) / np.abs(z).max()

    # Threshold activations
    actis_bool = np.int32(z > threshold)
    not_actis = np.where(actis_bool == 0)[0]
    acti_blocks_len = np.diff(not_actis) - 1
    figactis, axactis = plt.subplots()
    axactis.hist(
        acti_blocks_len,
        bins=0.5 + np.arange(np.max(acti_blocks_len))
    )
    axactis.set_title(
        f"CDL acti blocks lengths, $\\lambda$={reg}, thresh={threshold}"
    )
    axactis.set_xlabel("Block length")
    axactis.set_xlim([0, 10])
    if save_fig is not None:
        figactis.savefig(
            f"{save_fig}CDL_blocks_lengths_lmbd{reg}_thresh{threshold}.svg"
        )
    if plothist:
        plt.show()

    if sparsify:
        # Only keep first activation of each successive activation block
        actis_bool = np.diff(actis_bool, append=np.array([0]))
        actis_bool = np.roll(actis_bool, 1)
    # Recover index of remaining activations
    events_ind = np.where(actis_bool == 1)[0]

    if marked:
        return [np.array([[i / freq, z[i]] for i in events_ind])]
    else:
        # Convert index to time in seconds
        events_t = np.reshape(events_ind / freq, (1, len(events_ind)))
        return events_t


@memory.cache
def cdl(clean_df, reg, key, plotfig=False, save_fig=None):
    """
    Learn convolutional dictionary on ECG data. A solver can then
    estimate MHP parameters using dictionary activations as events.

    Parameters
    ----------
    clean_df: pandas DataFrame
        Signal DataFrame. index=timestamps, column ECG_NAME contain ECG values.

    reg: float
        Regularization parameter of CDL.

    plotfig: boolean, default=False
        If True, learned temporal atom, CDL reconstructed signal, activations,
        and original signal are plotted.

    save_fig: string or None, default=None.
        Path where to save figure if plotfig=true. If set to None, figure will
        not be saved.

    Returns
    -------
    cd.z_hat_: numpy array
        Activation vector of learned dictionary.
    """
    # Reshape timeseries for alphaCSC
    cd_timeseries = (
        clean_df[ECG_CH_NAME].to_numpy().reshape(1, 1, -1).astype(np.float32)
    )

    # CDL solver

    cd = alphacsc.BatchCDL(
        n_atoms=N_ATOMS,
        n_times_atom=N_TIMES_ATOMS,
        rank1=True,
        n_iter=N_ITERATIONS,
        n_jobs=1,
        reg=reg,
        sort_atoms=SORT_ATOMS,
        window=WINDOW,
        random_state=RANDOM_SEED,
        lmbd_max="fixed",
        verbose=1,
    )
    cd.fit(cd_timeseries)

    if plotfig:
        # Display learned atoms
        fig, axes = plt.subplots(
            N_ATOMS, 1, figsize=(4, 4 * N_ATOMS), squeeze=False
        )
        for k in range(N_ATOMS):
            axes[k, 0].plot(
                np.arange(len(cd.v_hat_[k])) / RESAMP_FREQ,
                cd.v_hat_[k],
                linewidth=3,
                color="blue",
            )
            axes[N_ATOMS - 1, 0].set_xlabel("Time(seconds)", size="xx-large")
            axes[0, 0].set_title("Temporal atom", size="xx-large")
            axes[0, 0].tick_params(
                axis="both", which="major", labelsize="xx-large")
            axes[0, 0].set_ylabel("mV", size="xx-large")
            if save_fig is not None:
                fig.savefig(f"{save_fig}temp_atom_lmbd{reg}_key{key}.svg")
            fig.show()

        # Plot activations parallelled with data
        # Reconstructed signal
        rec_sig = alphacsc.utils.convolution.construct_X_multi(
            cd.z_hat_, cd.D_hat_, n_channels=1
        )
        z_hat_thresh = cd.z_hat_.copy()
        z_hat_thresh[z_hat_thresh < 0.5] = 0
        rec_sig_thresh = alphacsc.utils.convolution.construct_X_multi(
            z_hat_thresh, cd.D_hat_, n_channels=1
        )
        # Plots
        # Original signal
        fig3 = go.Figure(
            go.Scattergl(
                x=clean_df["Time"],
                y=clean_df[ECG_CH_NAME],
                name=ECG_CH_NAME,
                opacity=0.4,
            )
        )
        # CDL activations
        fig3.add_trace(
            go.Scattergl(
                x=clean_df["Time"],
                y=cd.z_hat_[0][0],
                name="CDL activations",
                line_color="green",
                opacity=0.4,
            )
        )
        # CDL reconstructed signal
        fig3.add_trace(
            go.Scattergl(
                x=clean_df["Time"],
                y=rec_sig[0][0],
                name="CDL reconstructed signal",
                line_color="blue",
                opacity=0.4,
                line={"dash": "dash"},
            )
        )
        # CDL reconstructed signal from thresholded activations
        fig3.add_trace(
            go.Scattergl(
                x=clean_df["Time"],
                y=rec_sig_thresh[0][0],
                name="CDL threshold reconstructed signal",
                line_color="red",
                opacity=0.4,
                line={"dash": "dash"},
            )
        )
        fig3.show(renderer="browser")
    return cd.z_hat_


def evaluate_intensity(baseline, alpha, mean, sigma, delta, events_grid):
    L = int(1 / delta)
    TG = DiscreteKernelFiniteSupport(
        delta, n_dim=1, kernel="truncated_gaussian", lower=0, upper=1
    )

    intens = TG.intensity_eval(
        torch.tensor(baseline),
        torch.tensor(alpha),
        [torch.Tensor(mean), torch.Tensor(sigma)],
        events_grid,
        torch.linspace(0, 1, L),
    )
    return intens


@memory.cache
def fadin_fit(
    cleandf_hr,
    TIME_START,
    TIME_STOP,
    model,
    events,
    T,
    kernel_type=KERNEL_TYPE,
    kernel_length=KERNEL_LENGTH,
    lr=LR,
    delta=DELTA,
    max_iter=N_ITERATIONS_FADIN,
    random_state=RANDOM_STATE,
    init="moment_matching_max",
    figtitle="",
    figname="0",
    verbose=False,
    return_ll=False,
    **fadin_init,
):
    """
    Builds FaDIn solver and estimates its parameters on events.
    Parameters:
    -----------
    cleandf_hr: pandas DataFrame
        DataFrame with time index and 1 column: heart rate with no NaN value.
    TIME_START: float
        Beginning time of events time interval, in minutes.
    TIME_STOP: float
        End time of events time interval, in minutes.
    model: class instance
        Type of FaDIn solver. Supported values are FaDIn, MarkedFaDIn,
        JointFaDIn and JointFaDInDenoising.

    events: list
        Events on which FaDIn solver will be fitted. List of length n_dim.
        Each list element is a numpy array containing events for one dimension.
        If the solver is non-marked, events are listed as their timestamps.
        If the solver is marked, events are listed as [timestamp, mark].

    kernel_type: str
        Type of kernel for FaDIn solver. Supported values are
        'truncated_gaussian', 'raised_cosine', and 'truncated_gaussian'.

    lr: float
        Solver learning rate.

    delta: float
        Time delta for solver time discretization.

    T: int
        Length of events time interval.

    max_iter: int
        Max number of iterations of solver.

    random_state: int
        Random state for solver parameters intialization.
        Only used if `moment_matching=False`.

    moment_matching: boolean, default=`False`
        `moment_matching` parameter of solver.

    figtitle: str
        Title of kernel figure.

    figname: str
        Name of file for saving kernel figure, if SAVEFIG is not None.

    verbose: boolean, `default=False`

    return_ll: boolean, `default=False`
        If `True`, computes the log-likelihood of the solver on the events.

    **fadin_init:
    Additional parameters of solver constructor.
    """
    # Solver initialization
    if model == FaDIn:
        solver = model(
            n_dim=1,
            kernel=kernel_type,
            kernel_length=kernel_length,
            params_optim={"lr": lr},
            delta=delta,
            max_iter=max_iter,
            random_state=random_state,
            **fadin_init,
        )
    else:
        solver = model(
            n_dim=1,
            kernel=kernel_type,
            kernel_length=kernel_length,
            params_optim={"lr": lr},
            delta=delta,
            max_iter=max_iter,
            random_state=random_state,
            init=init,
            **fadin_init,
        )
    # Fit
    solver.fit(events, T)

    # Print estimated solver parameters
    estimated_baseline = solver.baseline_
    estimated_alpha = solver.alpha_
    param_kernel = solver.kernel_

    if param_kernel[0] == 0:
        abs_error = np.nan
    else:
        abs_error = np.abs(60 / param_kernel[0].item() - np.mean(cleandf_hr))
    rel_abs_error = abs_error / np.mean(cleandf_hr)

    if return_ll:
        # Compute solver intensity function on events
        if model in [UNHaP]:
            intens_bl = estimated_baseline + solver.baseline_noise_
        else:
            intens_bl = estimated_baseline
        _, events_grid = projected_grid_marked(events, delta, L * T + 1)
        intens = evaluate_intensity(
            [intens_bl],
            solver.alpha_,
            solver.kernel_[0],
            solver.kernel_[1],
            delta,
            events_grid,
        )
        # Compute log-likelihood of intensity function on events
        solver_ll = discrete_ll_loss_conv(intens, events_grid, delta, T)
    if verbose:
        print("Data time interval:", TIME_START, " - ", TIME_STOP, "minutes")
        # Print estimated parameters
        print(
            "Estimated baseline is:",
            torch.round(estimated_baseline, decimals=4).item()
        )
        if model in [UNHaP]:
            print(
                "Estimated noise baseline is:",
                torch.round(solver.baseline_noise_, decimals=4).item(),
            )
        print(
            "Estimated alpha is:",
            torch.round(estimated_alpha, decimals=4).item()
        )
        print(
            "Estimated mean of gaussian kernel:",
            np.round(param_kernel[0].item(), decimals=4)
        )
        print(
            "Estimated std of gaussian kernel:",
            np.round(param_kernel[1].item(), decimals=4)
        )
        # Print monitor QRS interval for comparison
        print(
            "Mean monitored QRS interval:",
            np.round(np.mean(60 / cleandf_hr), decimals=4),
            "s",
        )
        print(
            "std of monitored QRS interval:",
            np.round(np.std(60 / cleandf_hr), decimals=4),
            "s",
        )
        print("Heart Rate Absolute Error",
              np.round(abs_error, decimals=4),
              "beats/min")
        print(
            "Heart Rate Relative Absolute Error",
            np.round(rel_abs_error, decimals=4),
            "\n",
        )

    if not return_ll:
        return solver, abs_error, rel_abs_error
    if return_ll:
        return solver, abs_error, rel_abs_error, solver_ll
