"""
Function pipelines for filtering time-varying data

Created by: Ankit Khambhati

Change Log
----------
2016/12/11 - Implemented broadband_conn
"""

from __future__ import division
import numpy as np

from ..Common import errors
from ..Sigproc import reref, prewhiten, filters
from ..Network.Functional import correlation, coherence


def broadband_conn(data, fs):
    """
    Pipeline function for computing a broadband functional network from ECoG.

    See: Khambhati, A. N. et al. (2015).
    Dynamic Network Drivers of Seizure Generation, Propagation and Termination in
    Human Neocortical Epilepsy. PLOS Computational Biology, 11(12).

    Data --> CAR Filter --> Notch Filter --> Band-pass Filter --> Cross-Correlation

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

        fs: int
            Sampling frequency
    """

    # Standard param checks
    errors.check_type(data, np.ndarray)
    errors.check_dims(data, 2)
    errors.check_type(fs, int)

    # Parameter set
    param = {}
    param['Notch_60Hz'] = {'wpass': [58.0, 62.0],
                           'wstop': [59.0, 61.0],
                           'gpass': 0.1,
                           'gstop': 60.0}
    param['HPF_5Hz'] = {'wpass': [5.0],
                        'wstop': [4.0],
                        'gpass': 0.1,
                        'gstop': 60.0}
    param['LPF_115Hz'] = {'wpass': [115.0],
                          'wstop': [120.0],
                          'gpass': 0.1,
                          'gstop': 60.0}
    param['XCorr'] = {'tau': 0.25}

    # Build pipeline
    data_hat = reref.common_avg_ref(data)
    data_hat = prewhiten.ar_one(data_hat)
    data_hat = filters.elliptic(data_hat, fs, **param['Notch_60Hz'])
    data_hat = filters.elliptic(data_hat, fs, **param['HPF_5Hz'])
    data_hat = filters.elliptic(data_hat, fs, **param['LPF_115Hz'])
    adj = correlation.xcorr_mag(data_hat, fs, **param['XCorr'])

    return adj
