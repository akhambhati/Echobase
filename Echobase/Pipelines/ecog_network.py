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


def broadband_conn(data, fs, avgref=True):
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

        reref: True/False
            Re-reference data to the common average (default: True)

    Returns
    -------
        adj: ndarray, shape (N, N)
            Adjacency matrix for N variates
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
    if avgref:
        data_hat = reref.common_avg_ref(data)
    else:
        data_hat = data.copy()
    data_hat = prewhiten.ar_one(data_hat)
    data_hat = filters.elliptic(data_hat, fs, **param['Notch_60Hz'])
    data_hat = filters.elliptic(data_hat, fs, **param['HPF_5Hz'])
    data_hat = filters.elliptic(data_hat, fs, **param['LPF_115Hz'])
    adj = correlation.xcorr_mag(data_hat, fs, **param['XCorr'])

    return adj


def multiband_conn(data, fs, avgref=True):
    """
    Pipeline function for computing a band-specific functional network from ECoG.

    See: Khambhati, A. N. et al. (2016).
    Virtual Cortical Resection Reveals Push-Pull Network Control
    Preceding Seizure Evolution. Neuron, 91(5).

    Data --> CAR Filter --> Multi-taper Coherence

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
        adj_alphatheta: ndarray, shape (N, N)
            Adjacency matrix for N variates (Alpha/Theta Band 5-15 Hz)

        adj_beta: ndarray, shape (N, N)
            Adjacency matrix for N variates (Beta Band 15-25 Hz)

        adj_lowgamma: ndarray, shape (N, N)
            Adjacency matrix for N variates (Low Gamma Band 30-40 Hz)

        adj_highgamma: ndarray, shape (N, N)
            Adjacency matrix for N variates (High Gamma Band 95-105 Hz)
    """

    # Standard param checks
    errors.check_type(data, np.ndarray)
    errors.check_dims(data, 2)
    errors.check_type(fs, int)

    # Parameter set
    param = {}
    param['time_band'] = 5.
    param['n_taper'] = 9
    param['AlphaTheta_Band'] = [5., 15.]
    param['Beta_Band'] = [15., 25.]
    param['LowGamma_Band'] = [30., 40.]
    param['HighGamma_Band'] = [95., 105.]

    # Build pipeline
    if avgref:
        data_hat = reref.common_avg_ref(data)
    else:
        data_hat = data.copy()
    adj_alphatheta = coherence.multitaper(data_hat, fs, param['time_band'], param['n_taper'], param['AlphaTheta_Band'])
    adj_beta = coherence.multitaper(data_hat, fs, param['time_band'], param['n_taper'], param['Beta_Band'])
    adj_lowgamma = coherence.multitaper(data_hat, fs, param['time_band'], param['n_taper'], param['LowGamma_Band'])
    adj_highgamma = coherence.multitaper(data_hat, fs, param['time_band'], param['n_taper'], param['HighGamma_Band'])

    return adj_alphatheta, adj_beta, adj_lowgamma, adj_highgamma
