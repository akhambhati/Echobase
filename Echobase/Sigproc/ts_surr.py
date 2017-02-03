"""
Functions for computing surrogate time-series

Created by: Ankit Khambhati

Change Log
----------
2016/11/11 - Adapted AAFTsur
"""

from __future__ import division
import numpy as np

from ..Common import errors


def AAFTsur(xV):
    """
    Amplitude Adjusted Fourier Transform Surrogates

    This code is based on the following MATLAB code

    <AAFTsur.m>, v 1.0 2010/02/11 22:09:14  Kugiumtzis & Tsimpiris
    This is part of the MATS-Toolkit http://eeganalysis.web.auth.gr/

    Copyright (C) 2010 by Dimitris Kugiumtzis and Alkiviadis Tsimpiris
                           <dkugiu@gen.auth.gr>

    Reference : D. Kugiumtzis and A. Tsimpiris, "Measures of Analysis of Time
                Series (MATS): A Matlab  Toolkit for Computation of Multiple
                Measures on Time Series Data Bases", Journal of Statistical
                Software, 2010

    The original python author was: https://github.com/lneisenman/aaft

    Parameters
    ----------
        xV: ndarray, shape (N,)
            A univariate time-series with N samples

    Returns
    -------
        zM: ndarray, shape (N,)
            A univariate surrogate of the input time-series
    """

    # Standard param checks
    errors.check_type(xV, np.ndarray)
    errors.check_dims(xV, 1)

    n = len(xV)
    zM = np.empty(n)
    T = np.argsort(xV)
    oxV = np.sort(xV)
    ixV = np.argsort(T)

    # Rank order a white noise time series 'wV' to match the ranks of 'xV'
    wV = np.random.randn(n) * np.std(xV, ddof=1)    # match Matlab std
    owV = np.sort(wV)
    yV = owV[ixV].copy()

    # Fourier transform, phase randomization, inverse Fourier transform
    n2 = n//2
    tmpV = np.fft.fft(yV, 2*n2)
    magnV = np.abs(tmpV)
    fiV = np.angle(tmpV)
    rfiV = np.random.rand(n2-1) * 2 * np.pi
    nfiV = np.append([0], rfiV)
    nfiV = np.append(nfiV, fiV[n2+1])
    nfiV = np.append(nfiV, -rfiV[::-1])
    tmpV = np.append(magnV[:n2+1], magnV[n2-1:0:-1])
    tmpV = tmpV * np.exp(nfiV * 1j)
    yftV = np.real(np.fft.ifft(tmpV, n))  # Transform back to time domain

    # Rank order the 'xV' to match the ranks of the phase randomized
    # time series
    T2 = np.argsort(yftV)
    iyftV = np.argsort(T2)
    zM = oxV[iyftV]  # the AAFT surrogate of xV
    return zM
