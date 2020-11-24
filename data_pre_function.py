# [Author]  vv-nuke

import numpy as np
import pandas as pd


def steady_data_filter(
        data_raw: pd.DataFrame, window: int,
        filter_col = None, return_index=False,
        thre=None, mode='std', prev_zero=1e-9,
        center=False
):
    if filter_col:
        data = data_raw[filter_col]
    else:
        data = data_raw
    # threshold default 3std
    # or a list, coef. of std or fraction, depend on mode
    # prev_zero prevent threshold from zero
    if not thre:
        thre_list = 3 * data.std() + prev_zero
    elif type(thre) == list and len(thre) == len(data.columns):
        if mode == 'std':
            thre_list = data.std() * thre + prev_zero
        elif mode == 'frac':
            thre_list = abs(data) * thre + prev_zero
        else:
            return
    else:
        return

    window_mean = data.rolling(window=window, center=center).mean()
    bool_cmp = abs(data - window_mean) > thre_list
    index_drop = data.index[list(set(np.where(bool_cmp)[0]))]
    if return_index:
        return data_raw.drop(index=index_drop), index_drop
    else:
        return data_raw.drop(index=index_drop)


def on_filter(data_raw, filter_col=None, return_index=False):
    if filter_col:
        data = data_raw[filter_col]
    else:
        data = data_raw

    index_drop = data.index[list(set(np.where(data == 0)[0]))]
    if return_index:
        return data_raw.drop(index=index_drop), index_drop
    else:
        return data_raw.drop(index=index_drop)



