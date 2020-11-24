# [Author]  vv-nuke

import datetime
import time


def round_time(t_: datetime, lvl):
    round_lvl = "round_" + lvl
    if round_lvl in globals():
        return globals()[round_lvl](t_)
    else:
        return "No support"


def round_min(t_: datetime.datetime):
    return t_ - datetime.timedelta(seconds=t_.second, microseconds=t_.microsecond)


def round_hou(t_: datetime.datetime):
    return round_min(t_) - datetime.timedelta(minutes=t_.minute)


def round_day(t_: datetime.datetime):
    return round_hou(t_) - datetime.timedelta(hours=t_.hour)


def round_10min(t_: datetime.datetime):
    t_ = round_min(t_)
    return t_ - datetime.timedelta(minutes=t_.minute % 10)


def time_check(t_, st, timed_thre):
    if t_ - st >= timed_thre:
        return True
    else:
        return False


def time_block(t_, timed_thre, sleep=120):
    while True:
        now = datetime.datetime.now()
        if now - t_ >= timed_thre:
            return now
        else:
            time.sleep(sleep)