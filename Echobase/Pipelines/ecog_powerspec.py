"""
Function pipelines for filtering time-varying data

Created by: Ankit Khambhati

Change Log
----------
2017/10/30 - Implemented multiband function
"""

from __future__ import division
import numpy as np

from ..Common import errors
from ..Sigproc import reref, prewhiten, filters
from mtspec import mtspec


def multiband(data, fs, avgref=True):
    """
    Pipeline function for computing the power spectral density of ECoG
    using multitaper spectral estimation.

    Data --> CAR Filter --> Multitaper power spectral density

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

        fs: int
            Sampling frequency

        reref: True/False
            Re-reference data to the common average (default: True)

    Returns
    -------
        freq: ndarray, shape (F,)
            Range of frequencies over which power spectrum is being measured.

        powerspec: ndarray, shape (F, N)
            Power spectral density across F frequencies for N variates.
    """

    # Standard param checks
    errors.check_type(data, np.ndarray)
    errors.check_dims(data, 2)
    errors.check_type(fs, int)

    # Parameter set
    param = {}
    param['time_band'] = 5.
    param['n_taper'] = 9

    T, N = data.shape

    # Build pipeline
    if avgref:
        data_hat = reref.common_avg_ref(data)
    else:
        data_hat = data.copy()

    for n1 in xrange(N):
        out = mtspec(data=data[:, n1],
                     delta=1.0/fs,
                     time_bandwidth=param['time_band'],
                     number_of_tapers=param['n_taper'],
                     adaptive=True)

        if n1 == 0:
            freq = out[1]
            powerspec = np.zeros((freq.shape[0], N))
        powerspec[:, n1] = out[0]

    return freq, powerspec
