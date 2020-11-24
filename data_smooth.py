import numpy as np
from copy import deepcopy


def data_smooth(ts_, bd_coef=2, print_bd=False):

    def bd_cal(sm_std, coef):
        return coef * sm_std

    ts = deepcopy(ts_)
    if len(ts) < 5:
        return None
    else:
        smoothing = []
        for i in range(len(ts)):
            if i in [0, len(ts) - 1]:
                smoothing.append(ts[i])
            else:
                smoothing.append(np.mean(ts[i - 1: i + 2]))
        bd = bd_cal(np.std(smoothing), bd_coef)

    if ts[0] > smoothing[1] + bd or ts[0] < smoothing[1] - bd:
        rep = 2 * ts[1] - ts[2]
        print(f"ts[{0}]={ts[0]} replaced by {rep}")
        ts[0] = rep
        smoothing[0] = ts[0]
        smoothing[1] = np.mean(ts[:3])
        bd = bd_cal(np.std(smoothing), bd_coef)

    for i in range(2, len(ts) - 1):
        if ts[i] > smoothing[i - 1] + bd or ts[i] < smoothing[i - 1] - bd:
            rep = 2 * ts[i - 1] - ts[i - 2]
            print(f"ts[{i}]={ts[i]} replaced by {rep}")
            ts[i] = rep
            smoothing[i - 1] = np.mean(ts[i - 2: i + 1])
            smoothing[i] = np.mean(ts[i - 1: i + 2])
            smoothing[i + 1] = np.mean(ts[i: i + 3])
            bd = bd_cal(np.std(smoothing), bd_coef)

    if ts[-1] > smoothing[-2] + bd or ts[-1] < smoothing[-2] - bd:
        rep = 2 * ts[-2] - ts[-3]
        print(f"ts[{len(ts) - 1}]={ts[len(ts) - 1]} replaced by {rep}")
        ts[-1] = rep
        smoothing[-1] = ts[-1]
        smoothing[-2] = np.mean(ts[-3:])
        bd = bd_cal(np.std(smoothing), bd_coef)
    if print_bd:
        print(f"bd is {bd}")
    return ts
