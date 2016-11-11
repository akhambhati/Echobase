"""
Functional Data Analysis Routines
"""
from __future__ import division
import numpy as np


def _curve_area(A, B):
    r1 = np.mean(A-B)
    r2 = np.mean(B-A)

    if r1 > r2:
        return r1
    else:
        return r2


def curve_test(Y, cnd_1, cnd_2, n_perm=1000):
    """
    Assess whether two curves are statistically significant based on
    permutation test over conditions and replicates.

    Parameters
    ----------
    Y: 2d array,    shape: (time x var)
        Observations matrix for each variable over time.

    cnd_1: list,    shape: (n_reps_1)
        List of replicate indices in columns of Y for condition 1

    cnd_2: list,    shape: (n_reps_2)
        List of replicate indices in columns of Y for condition 2

    n_perm: int
        Number of permutations to group
    """

    n_reps_1 = len(cnd_1)
    n_reps_2 = len(cnd_2)
    n_reps = Y.shape[1]
    assert n_reps == (n_reps_1 + n_reps_2)

    # Get true area between condition curves
    Y_1 = np.mean(Y[:, cnd_1], axis=1)
    Y_2 = np.mean(Y[:, cnd_2], axis=1)
    true_area = _curve_area(Y_1, Y_2)

    # Estimate null distribution of area between curves
    p_count = 0
    for pr in xrange(n_perm):
        rnd_reps = np.random.permutation(n_reps)
        rnd_cnd_1 = rnd_reps[:n_reps_1]
        rnd_cnd_2 = rnd_reps[n_reps_1:]

        rnd_Y_1 = np.mean(Y[:, rnd_cnd_1], axis=1)
        rnd_Y_2 = np.mean(Y[:, rnd_cnd_2], axis=1)
        rnd_area = _curve_area(rnd_Y_1, rnd_Y_2)

        if rnd_area > true_area:
            p_count += 1

    p = p_count / n_perm

    return p
