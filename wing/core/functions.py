# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Tuple


def make_edges(vector: np.ndarray, cuts: int, print_func, unique=True, is_add_infinity=True) -> np.ndarray:
    """
    Initial edges split.
    :param vector
         vector - vectorized data array of shape (n,1) to be splitted
    :param cuts
        cuts - int number of parts to equally-sized split the X vector
    :param unique
        unique
    :param print_func
        function to outprint results
    :return edges
    :rtype np.ndarray
    """
    # here we do check of data to be qcutted.
    is_oversampled = True
    edges = None
    try:
        splits, edges = pd.qcut(vector, q=cuts, retbins=True)
    except ValueError:
        print_func("Too oversampled dataset for qcut, will be used only unique values for splitting")
        is_oversampled = True
    if is_oversampled:
        try:
            splits, edges = pd.qcut(np.unique(vector), q=cuts, retbins=True)
        except ValueError:
            print_func('Even after deleting duplicate values in X the data set got too low variance')
            print_func('Current X unique values:'+str(np.unique(vector)))
    if is_add_infinity:
        edges = add_infinity(edges[1:-1])
    if unique:
        edges = np.unique(edges)
    return edges


def generate_combs(vector: np.ndarray, k: int, k_start=1) -> List:
    """
    Generates combinations with next algo:
        C(n,1) + C(n,2) + ... + C(n,k)
    :rtype: object
    :param vector
        vector - np. array to generate combinations
    :param k
        k (int) - int value of max combinations
    :param k_start
        k (int) - int value
    :rtype List
    """
    collector = []
    for r in range(k_start, k + 1):
        variants = [el for el in combinations(vector, r)]
        collector.append(variants)
    collector = sum(collector, [])
    return collector


def add_infinity(vector: np.ndarray) -> np.ndarray:
    """
    Adds -inf and +inf bounds at 0 and -1 positions of input vector
    :param vector
        vector - array to add infs
    :return inf_vector
        vector - array with added inf
    :rtype np.ndarray
    """
    inf_vector = np.pad(vector, pad_width=(1, 1), mode='constant', constant_values=(-np.inf, np.inf))
    return inf_vector


def check_mono(vector: np.ndarray) -> bool:
    """
    This function defines does vector is monotonic
    :param vector
        vector - np.ndarray of size 1
    :return is_mono
        boolean value,which defines is the value boolean
    :rtype bool
    """
    diffs = np.diff(vector)
    mono_inc = np.all(diffs > 0)
    mono_dec = np.all(diffs < 0)
    mono_any = mono_dec | mono_inc
    is_mono = bool(mono_any)
    return is_mono


def split_by_edges(vector: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Splits input vector by edges and returns index of each value
    :param vector
        vector - array to split
    :param edges
        edges - array of edges to split
    :return bins
        bins array of len(vector) with index of each element
    :rtype np.ndarray
    """
    bins = np.digitize(vector, edges)
    assert len(bins) == len(vector)
    assert len(np.unique(bins)) <= len(edges)
    return bins


def calculate_loc_woe(vector: pd.Series, goods: int, bads: int, woe_adjustment_factor=0.5) -> np.float:
    """
    Calculates woe in bucket
    :param  vector
        Vector with keys "good" and "bad"
    :param goods
        total amount of "event" in frame
    :param bads
        total amount of "non-event" in frame
    :param woe_adjustment_factor
        WoE adjustment factor to apply if total bad or total good equals to zero
    :return local woe value
    :rtype np.float64
    """
    t_good = np.float(vector["good"]) / np.float(goods)
    t_bad = np.float(vector["bad"]) / np.float(bads)
    t_bad = woe_adjustment_factor if t_bad == 0 else t_bad
    t_good = woe_adjustment_factor if t_good == 0 else t_good
    return np.log(t_bad / t_good)


def gini_index(events: np.ndarray, non_events: np.ndarray) -> np.float:
    """
    Calculates Gini index in SAS format
    :param events
        Vector of good group sizes
    :param non_events
        Vector of non-event group sizes
    :return Gini index
    :rtype np.float64
    """
    assert len(events) > 0
    assert len(non_events) > 0
    assert (events >= 0).all()
    assert (non_events >= 0).all()
    # precalculate values
    p1 = float(2 * sum(events[i] * sum(non_events[:i]) for i in range(1, len(events))))
    p2 = float(sum(events * non_events))
    p3 = float(events.sum() * non_events.sum())
    # calculate coefficient
    coefficient = 1 - ((p1 + p2) / p3)
    index = coefficient * 100
    assert 0 < index < 100
    return index


def calc_descriptive_from_vector(bins: np.ndarray, y: np.ndarray, total_good: int, total_bad: int) -> pd.DataFrame:
    """
    Calculates IV/WoE + other descriptive data in df by grouper column
    :param bins
        array of pre-binned vector to calculate woe
    :param y
        array of target variable
    :param total_good
        int value of total good in y data
    :param total_bad
        int value of total bad in y data
    :rtype pd.DataFrame
    """
    df = pd.DataFrame(np.array([bins, y]).T, columns=["grp", "y"])
    tg_good = df.groupby("grp")["y"].sum()
    tg_all = df.groupby("grp")["y"].count()
    tg_bad = tg_all - tg_good
    woe_df = pd.concat([tg_good, tg_bad, tg_all], axis=1)
    woe_df.columns = ["good", "bad", "total"]
    woe_df["woe"] = woe_df.apply(lambda row: calculate_loc_woe(row, total_good, total_bad), axis=1)
    woe_df["local_event_rate"] = woe_df["good"] / tg_all
    return woe_df


def check_variant(bins: np.ndarray, y: np.ndarray, t_good: int, t_bad: int) -> Tuple[bool, np.float]:
    """
    Checks if privided bins are monotonic by WoE
    if woe are not monotonic, gini will be None
    :param bins:
        vector of groups after binning X
    :param y:
        target vector
    :param t_good
        Total amount of good events in frame
    :param t_bad
        Total amount of bad events in frame
    :returns is_mono,gini
    """
    wdf = calc_descriptive_from_vector(bins, y, t_good, t_bad)
    is_mono = check_mono(wdf["woe"])
    wdf = wdf.sort_values(by="local_event_rate", ascending=False)
    gini_index_value = gini_index(wdf["good"].values, wdf["bad"].values)
    return is_mono, gini_index_value


if __name__ == "__main__":
    print("Non executable module")
