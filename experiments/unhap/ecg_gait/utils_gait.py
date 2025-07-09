
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
from download import download


SLOTS = {
    1: [1, 1],
    2: [2, 1],
    3: [3, 1],
    4: [4, 1],
    5: [5, 1],
    6: [6, 1],
    7: [7, 1],
    8: [8, 1],
    9: [9, 1],
    10: [10, 1],
    11: [11, 1],
    12: [12, 1],
    13: [13, 1],
    14: [14, 1],
    15: [15, 1],
    16: [16, 1],
    17: [17, 1],
    18: [18, 1],
    19: [19, 1],
    20: [20, 1],
    21: [21, 1],
    22: [22, 1],
    23: [23, 1],
    24: [24, 1],
    25: [25, 1],
    26: [26, 1],
    27: [27, 1],
    28: [28, 1],
    29: [29, 1],
    30: [30, 1],
    40: [40, 1],
    41: [41, 1],
    42: [42, 1],
    43: [43, 1],
    44: [44, 1],
    45: [45, 1],
    46: [46, 1],
    47: [47, 1],
    48: [48, 1],
    49: [49, 1],
}
# DATA_HOME = Path("gait_data")
# GAIT_RECORD_ID_LIST_FNAME = DATA_HOME / "gait_record_id_list.json"
# GAIT_PARTICIPANTS_FNAME = DATA_HOME / "gait_participants.tsv"
# SLOTS_FNAME = DATA_HOME / "gait_slots.json"


# def download_gait(verbose=True):
#     gait_dir = DATA_HOME / "gait"
#     gait_dir.mkdir(parents=True, exist_ok=True)
#     gait_zip = download(
#         "http://dev.ipol.im/~truong/GaitData.zip",
#         gait_dir / "GaitData.zip",
#         replace=False,
#         verbose=verbose
#     )

#     return gait_zip


def get_gait_data(subject=1, trial=1, only_meta=False, verbose=True):
    """
    Retrieve gait data from this `dataset`_.

    Parameters
    ----------
    subject: int, defaults to 1
        Subject identifier.
        Valid subject-trial pairs can be found in this `list`_.
    trial: int, defaults to 1
        Trial number.
        Valid subject-trial pairs can be found in this `list`_.
    only_meta: bool, default to False
        If True, only returns the subject metadata
    verbose : bool, default to True
        Whether to print download status to the screen.

    Returns
    -------
    dictDATA_HOME = Path("gait_data")
# GAIT_RECORD_ID_LIST_FNAME = DATA_HOME / "gait_record_id_list.json"
# GAIT_PARTICIPANTS_FNAME = DATA_HOME / "gait_participants.tsv"
# SLOTS_FNAME = DATA_HOME / "gait_slots.json"


# def download_gait(verbose=True):
#     gait_dir = DATA_HOME / "gait"
#     gait_dir.mkdir(parents=True, exist_ok=True)
#     gait_zip = download(
#         "http://dev.ipol.im/~truong/GaitData.zip",
#         gait_dir / "GaitData.zip",
#         replace=False,
#         verbose=verbose
#     )

#     return gait_zip

        A dictionary containing metadata and data relative
        to a trial. The 'data' attribute contains time
        series for the trial, as a Pandas dataframe.


    .. _dataset: https://github.com/deepcharles/gait-data
    .. _list:
       https://github.com/deepcharles/gait-data/blob/master/code_list.json
    """
    # coerce subject and trial
    subject = int(subject)
    trial = int(trial)

    gait_zip = download_gait(verbose=verbose)

    with ZipFile(gait_zip) as zf:
        with zf.open(f"GaitData/{subject}-{trial}.json") as meta_file, \
                zf.open(f"GaitData/{subject}-{trial}.csv") as data_file:
            meta = json.load(meta_file)
            if not only_meta:
                data = pd.read_csv(data_file, sep=',', header=0)
                meta['data'] = data
            return meta


def save_errors(est, gt, key, ae_df, rae_df, method='', verbose=True,
                unit='seconds'):
    """Computes and saves Absolute Error and Relative Error between
    est and gt into DataFrames.
    Parameters:
    -----------
    est: float
        Estimated value which will be conpared to the ground truth value.
    gt: float
        Ground truth value.
    key: Any
        Key of DataFrame line in which the errors will be saved.
    ae_df: pandas.DataFrame
        DataFrame where the Absolute Error will be saved. Must be created
        before calling this function.
    rae_df: pandas.DataFrame
        DataFrame where the Relative Absolute Error will be saved.
        Must be created before calling this function.
    method: `str`
        Name of method used to compute `est`. Must be a column of dataframes
        ae_df and rae_df.
    verbose: `bool`
        If True, the Absolute Error is printed.DATA_HOME = Path("gait_data")

    Returns:
    --------
    ae: float
        Absolute Error.
    rae: float
        Relative Absolute Error.
    """
    ae = np.abs(est - gt)
    rae = ae / gt
    ae_df.loc[key, method] = ae
    rae_df.loc[key, method] = rae
    if verbose:
        print(f'{method} Absolute Error:', round(ae, 3), unit)
    return ae, rae
