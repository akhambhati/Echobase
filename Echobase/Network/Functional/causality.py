"""
Correlation-based measure for computing functional connectivity

Created by: Jason Grosz

Change Log
----------
2016/04/24 - Implemented Granger pipe
"""

from __future__ import division
import numpy as np
from statsmodels.tsa.api import VAR

from ...Common import errors
from ...Sigproc import ts_surr


def Granger(data, lagPar, signif_value, n_perm=1000):
    """
    The Granger function adapts a Granger Causality measure from statsmodels
    by fitting an autoregressive model to the data.

    See: <insert a reference or two on this>

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

        lagPar: int
            Number of lags to use in the model fit

        signif_value: float
            The Type I error rate for the causality measure under an
            amplitude adjusted Fourier transform surrogate model

        n_perm: int
            Number of permutations over which to generate amplitude adjusted
            Fourier transform surrogate signals

    Returns
    -------
        adj_filter: ndarray, shape (N, N)
            Causal, adjacency matrix for N variates after permutation-based filtering
    """

    # Standard param checks
    errors.check_type(data, np.ndarray)
    errors.check_dims(data, 2)
    errors.check_type(lagPar, int)
    errors.check_type(signif_value, float)
    errors.check_type(n_perm, int)
    if signif_value < 1.0/n_perm:
        raise Exception('Type I error rate is too large relative to number of permutations')

    # Get data attributes
    n_samp, n_chan = data.shape

    adj=np.zeros((n_chan, n_chan))
    adj_filter=np.zeros((n_chan, n_chan))
    for ii in range(0, n_chan):
        for jj in range(0, n_chan):
            if ii == jj:
                continue

            signal1 = data[ii, :]
            signal2 = data[jj, :]
            signal = np.vstack([signal1, signal2]).T
            model = VAR(signal);
            results = model.fit(lagPar);
            granger_test = results.test_causality('y1', ['y2'], kind='f');

            granger_test_scrambled2 = np.zeros(n_perm)
            for perm_i in range(0, n_perm):
                permsignal1 = AAFTsur(signal1)
                permsignal2 = AAFTsur(signal2)

                signal=np.vstack([permsignal1, permsignal2]).T
                model = VAR(signal);
                results = model.fit(lagPar);
                granger_test_scrambled = results.test_causality('y1', ['y2'], kind='f');
                granger_test_scrambled2[perm_i] = granger_test_scrambled['statistic'];

            #compute pvalue from real granger test and surrogate model
            p_value = np.mean(granger_test_scrambled2 > granger_test['statistic'])

            adj[ii, jj] = granger_test['statistic']

            if p_value < signif_value:
                adj_filter[ii, jj] = granger_test['statistic']

    return adj_filter
