"""
Correlation-based measure for computing functional connectivity

Created by: Ankit Khambhati

Change Log
----------
2016/03/18 - Changed XCorr and Corr to __Mag and implement Corr (nonmag)
2016/03/06 - Implemented XCorr and Corr pipes
"""

from __future__ import division
import numpy as np
import scipy.signal as spsig

from ...Common import errors


def xcorr_mag(data, fs, tau):
    """
    The xcorr_mag function implements a cross-correlation similarity function
    for computing functional connectivity -- maximum magnitude cross-correlation

    This function implements an FFT-based cross-correlation (using convolution).

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

        fs: int
            Sampling frequency

        tau: float
            The max lag limits of cross-correlation in seconds

    Returns
    -------
        adj: ndarray, shape (N, N)
            Adjacency matrix for N variates
    """

    # Standard param checks
    errors.check_type(data, np.ndarray)
    errors.check_dims(data, 2)
    errors.check_type(fs, int)
    errors.check_type(tau, float)

    # Get data attributes
    n_samp, n_chan = data.shape
    tau_samp = int(tau*fs)
    triu_ix, triu_iy = np.triu_indices(n_chan, k=1)

    # Normalize the signal
    data -= data.mean(axis=0)
    data /= data.std(axis=0)

    # Initialize adjacency matrix
    adj = np.zeros((n_chan, n_chan))
    lags = np.hstack((range(0, n_samp, 1),
                      range(-n_samp, 0, 1)))
    tau_ix = np.flatnonzero(lags <= tau_samp)

    # Use FFT to compute cross-correlation
    data_fft = np.fft.rfft(
        np.vstack((data, np.zeros_like(data))),
        axis=0)

    # Iterate over all edges
    for n1, n2 in zip(triu_ix, triu_iy):
        xc = 1 / n_samp * np.fft.irfft(
            data_fft[:, n1] * np.conj(data_fft[:, n2]))
        adj[n1, n2] = np.max(np.abs(xc[tau_ix]))
    adj += adj.T

    return adj


def xcorr(data, fs, tau):
    """
    The xcorr function implements a cross-correlation similarity function
    for computing functional connectivity.

    This function implements an FFT-based cross-correlation (using convolution).

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

        fs: int
            Sampling frequency

        tau: float
            The max lag limits of cross-correlation in seconds

    Returns
    -------
        adj: ndarray, shape (N, N)
            Adjacency matrix for N variates
    """

    # Standard param checks
    errors.check_type(data, np.ndarray)
    errors.check_dims(data, 2)
    errors.check_type(fs, int)
    errors.check_type(tau, float)

    # Get data attributes
    n_samp, n_chan = data.shape
    tau_samp = int(tau*fs)
    triu_ix, triu_iy = np.triu_indices(n_chan, k=1)

    # Normalize the signal
    data -= data.mean(axis=0)
    data /= data.std(axis=0)

    # Initialize adjacency matrix
    adj = np.zeros((n_chan, n_chan))
    lags = np.hstack((range(0, n_samp, 1),
                      range(-n_samp, 0, 1)))
    tau_ix = np.flatnonzero(lags <= tau_samp)

    # Use FFT to compute cross-correlation
    data_fft = np.fft.rfft(
        np.vstack((data, np.zeros_like(data))),
        axis=0)

    # Iterate over all edges
    for n1, n2 in zip(triu_ix, triu_iy):
        xc = 1 / n_samp * np.fft.irfft(
            data_fft[:, n1] * np.conj(data_fft[:, n2]))

        if xc[tau_ix].max() > np.abs(xc[tau_ix].min()):
            adj[n1, n2] = xc[tau_ix].max()
        else:
            adj[n1, n2] = xc[tau_ix].min()
    adj += adj.T

    return adj
