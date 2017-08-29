"""
False Discovery Rate Routines
"""
from __future__ import division
import numpy as np


def bhp(pv_list, alpha=0.05):
    """
    Uses the Benjamini-Hochberg procedure to control the false discovery rate of rejecting the null hypothesis.

    Parameters
    ----------
    pv_list: list,  shape: (n_pv)
        List of p values corresponding to each statistical test

    alpha: float
        Family-wise error rate

    Returns
    -------
    thresh_list: list, shape: (n_pv)
        List of booleans corresponding to rejecting the null hypothesis associated with a p value
    """

    assert alpha < 1.0
    pv_list = np.array(pv_list)

    n_comp = len(pv_list)
    pv_ix = np.argsort(pv_list)

    sig_ix = []
    for ii, ix in enumerate(pv_ix):
        if pv_list[ix] <= (alpha * (ii+1) / np.float(n_comp)):
            sig_ix.append(ix)
    thresh_ix = np.max(sig_ix)

    thresh_list = pv_list <= pv_list[thresh_ix]

    return thresh_list
