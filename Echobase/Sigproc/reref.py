"""
Functions for re-referencing the time-varying sensor data.

Created by: Ankit Khambhati

Change Log
----------
2016/11/11 - Implemented common_avg_ref
"""

from __future__ import division
import numpy as np

from ..Common import errors


def common_avg_ref(data):
    """
    The common_avg_ref function subtracts the common mode signal from the original
    signal. Suggested for removing correlated noise, broadly over a sensor array.

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

    Returns
    -------
        data_reref: ndarray, shape (T, N)
            Referenced signal with common mode removed
    """

    # Standard param checks
    errors.check_type(data, np.ndarray)
    errors.check_dims(data, 2)

    # Remove common mode signal
    data_reref = (data.T - data.mean(axis=1)).T

    return data_reref
