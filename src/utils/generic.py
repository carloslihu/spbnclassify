import re

import numpy as np
import pandas as pd


def safe_exp(x: float | np.ndarray | pd.Series) -> float | np.ndarray:
    """Returns the exponential of x, clipping the input to avoid overflow.

    Args:
        x (float | np.ndarray | pd.Series): Input value

    Returns:
        float | np.ndarray: The exponential of x
    """
    LOG_MAX = 709  # approximately the log of the maximum float
    x = np.clip(x, -LOG_MAX, LOG_MAX)
    return np.exp(x)


def dict2html(di1: dict, di2: dict = {}) -> str:
    """Converts two dictionaries into an HTML string with the format:
        <b>key1</b>: value1
        <b>key2</b>: value2

    Args:
        di1 (dict): Dictionary with the first set of key-value pairs
        di2 (dict, optional): Dictionary with the second set of key-value pairs. Defaults to {}.

    Returns:
        str: HTML string with the key-value pairs
    """
    res = "<br/>".join([f"<b>{k:15}</b>:{v}" for k, v in di1.items()])
    if di2 != {}:
        res += "<br/><br/>"
        res += "<br/>".join([f"<b>{k:15}</b>:{v}" for k, v in di2.items()])
    return res


def extract_class_name(class_name):
    """Returns the cluster index from the cluster name

    Args:
        cluster_name (str): string with the p_<k> format

    Returns:
        str: Returns the cluster index
    """
    # regular expression to extract substring after "p_"
    p = re.compile(r"p_(.*)")
    m = p.search(class_name)
    if m:
        k = m.group(1)
    else:
        k = None
    return k


def get_boxplot_threshold(score):
    """Calculates the threshold for a boxplot.
    We assume that the data above the upper whisker is an outlier.

    Args:
        score (pd.Series): Score to have the threshold calculated

    Returns:
        float: The value of just below the upper whisker.
    """
    Q1 = score.quantile(0.25)
    Q3 = score.quantile(0.75)
    IQR = Q3 - Q1

    upper = Q3 + 1.5 * IQR
    threshold = score.loc[score < upper].max()
    # Symmetric threshold
    # lower = Q1 - 1.5*IQR
    # threshold = score.loc[score > lower].min()

    return threshold
