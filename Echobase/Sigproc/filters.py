"""
Functions for filtering time-varying data

Created by: Ankit Khambhati

Change Log
----------
2016/11/11 - Implemented elliptic
"""

from __future__ import division
import numpy as np
import scipy.signal as spsig

from ..Common import errors


def elliptic(data, fs, wpass, wstop, gpass, gstop):
    """
    The elliptic function implements bandpass, lowpass, highpass filtering

    This implements zero-phase filtering to pre-process and analyze
    frequency-dependent network structure. Implements Elliptic IIR filter.

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

        fs: int
            Sampling frequency

        wpass: tuple, shape: (1,) or (1,1)
            Pass band cutoff frequency (Hz)

        wstop: tuple, shape: (1,) or (1,1)
            Stop band cutoff frequency (Hz)

        gpass: float
            Pass band maximum loss (dB)

        gstop: float
            Stop band minimum attenuation (dB)
    """

    # Standard param checks
    errors.check_type(data, np.ndarray)
    errors.check_dims(data, 2)
    errors.check_type(fs, int)
    errors.check_type(wpass, list)
    errors.check_type(wstop, list)
    errors.check_type(gpass, float)
    errors.check_type(gstop, float)
    if not len(wpass) == len(wstop):
        raise Exception('Frequency criteria mismatch for wpass and wstop')
    if not (len(wpass) < 3):
        raise Exception('Must only be 1 or 2 frequency cutoffs in wpass and wstop')

    # Design filter
    nyq = fs / 2.0
    wpass_nyq = map(lambda f: f/nyq, wpass)
    wstop_nyq = map(lambda f: f/nyq, wstop)
    coef_b, coef_a = spsig.iirdesign(wp=wpass_nyq,
                                     ws=wstop_nyq,
                                     gpass=gpass,
                                     gstop=gstop,
                                     analog=0, ftype='ellip',
                                     output='ba')

    # Perform filtering and dump into signal_packet
    data_filt = spsig.filtfilt(coef_b, coef_a, data, axis=0)

    return data_filt
