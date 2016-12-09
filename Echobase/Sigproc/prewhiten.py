"""
Functions for pre-whitening time-varying data.
Reduces the autocorrelative structure for a signal -- flattening the power spectrum
and making temporal structure more Gaussian

Created by: Ankit Khambhati

Change Log
----------
2016/11/11 - Implemented ar_one
"""

from __future__ import division
import numpy as np

from ..Common import errors


def ar_one(data):
    """
    The ar_one function fits an AR(1) model to the data and retains the residual as
    the pre-whitened data

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

    Returns
    -------
        data_white: ndarray, shape (T, N)
            Whitened signal with reduced autocorrelative structure
    """

    # Standard param checks
    errors.check_type(data, np.ndarray)
    errors.check_dims(data, 2)

    # Retrieve data attributes
    n_samp, n_chan = data.shape

    # Apply AR(1)
    data_white = np.zeros((n_samp-1, n_chan))
    for i in xrange(n_chan):
        win_x = np.vstack((data[:-1, i],
                           np.ones(n_samp-1)))
        w = np.linalg.lstsq(win_x.T, data[1:, i])[0]
        data_white[:, i] = data[1:, i] - (data[:-1, i]*w[0] + w[1])

    return data_white
