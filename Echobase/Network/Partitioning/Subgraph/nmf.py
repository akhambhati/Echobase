'''
Non-Negative Matrix Factorization for dynamic brain networks, such that:
    A ~= WH
    Constraints:
        A, W, H >= 0
        L2-Regularization on W
        L1-Sparsity on H

Implementation is based on :
    1. Jingu Kim, Yunlong He, and Haesun Park. Algorithms for Nonnegative
            Matrix and Tensor Factorizations: A Unified View Based on Block
            Coordinate Descent Framework.
            Journal of Global Optimization, 58(2), pp. 285-319, 2014.
    2. Jingu Kim and Haesun Park. Fast Nonnegative Matrix Factorization:
            An Active-set-like Method And Comparisons.
            SIAM Journal on Scientific Computing (SISC), 33(6),
            pp. 3261-3281, 2011.
Modified from: https://github.com/kimjingu/nonnegfac-python

Created by: Ankit Khambhati

Change Log
----------
2016/02/22 - Implement sparse and regularization penalties
'''

import numpy as np
import nnls
import matrix_utils as matr_util
import time

import errors
from display import my_display


def snmf_bcd(cfg_matr, alpha, beta,
             fac_subnet_init, fac_coef_init,
             max_iter, verbose=True):
    """
    Compute Sparse-NMF based on Kim and Park (2011).
    By default, enforces a sparse penalty on the coefficients matrix H
    and regularizes the sub-network, basis matrix W.

    A -> cfg_matr.T
    W -> fac_subnet.T
    H -> fac_coefs

    Parameters
    ----------
        cfg_matr: numpy.ndarray
            The network configuration matrix
            shape: [n_win x n_conn]

        alpha: float
            Regularization parameter on W

        beta: float
            Sparsity parameter on H

        fac_subnet_init: numpy.ndarray
            Initial sub-network basis matrix
            shape: [n_fac x n_conn]

        fac_coef_init: numpy.ndarray
            Initial coefficients matrix
            shape: [n_fac x n_win]

        max_iter: int
            Maximum number of optimization iterations to perform
            Typically around 100

        verbose: bool
            Print progress information to the screen

    Returns
    -------
        fac_subnet_init: numpy.ndarray
            Final sub-network basis matrix
            shape: [n_fac x n_conn]

        fac_coef_init: numpy.ndarray
            Final coefficients matrix
            shape: [n_fac x n_win]

        rel_err: numpy.ndarray
            Froebenius norm of the error matrix over iterations
            shape: [n_iter,]
    """

    # Standard param checks
    errors.check_type(cfg_matr, np.ndarray)
    errors.check_type(alpha, float)
    errors.check_type(beta, float)
    errors.check_type(fac_subnet_init, np.ndarray)
    errors.check_type(fac_coef_init, np.ndarray)
    errors.check_type(max_iter, int)
    errors.check_type(verbose, bool)

    # Check input dimensions
    if not len(cfg_matr.shape) == 2:
        raise ValueError('%r does not have two dimensions' % cfg_matr)
    n_win = cfg_matr.shape[0]
    n_conn = cfg_matr.shape[1]

    if not len(fac_subnet_init.shape) == 2:
        raise ValueError('%r does not have two dimensions' % fac_subnet_init)
    n_fac = fac_subnet_init.shape[0]
    if not fac_subnet_init.shape[1] == n_conn:
        raise ValueError('%r should have same number of connections as %r' %
                         (fac_subnet_init, cfg_matr))

    if not len(fac_coef_init.shape) == 2:
        raise ValueError('%r does not have two dimensions' % fac_coef_init)
    if not fac_coef_init.shape[0] == n_fac:
        raise ValueError('%r should specify same number of factors as %r' %
                         (fac_coef_init, fac_subnet_init))
    if not fac_coef_init.shape[1] == n_win:
        raise ValueError('%r should have same number of windows as %r' %
                         (fac_coef_init, cfg_matr))

    # Initialize matrices
    # A - [n_conn x n_win]
    # W - [n_conn x n_fac]
    # H - [n_win x n_fac]
    A = cfg_matr.T.copy()
    W = fac_subnet_init.T.copy()
    H = fac_coef_init.T.copy()

    # Regularization matrix
    # alpha_matr - [n_fac x n_fac]
    alpha_matr = np.sqrt(alpha) * np.eye(n_fac)
    alpha_zeros_matr = np.zeros((n_conn, n_fac))

    # Sparsity matrix
    # beta_matr - [1 x n_fac]
    beta_matr = np.sqrt(beta) * np.ones((1, n_fac))
    beta_zeros_matr = np.zeros((1, n_win))

    # Capture error minimization
    rel_error = np.zeros(max_iter)
    norm_A = matr_util.norm_fro(A)

    my_display('\nBeginning Non-Negative Matrix Factorization\n', verbose)
    t_iter_start = time.time()
    for ii in xrange(max_iter):
        # Use the Block-Pivot Solver
        # First solve for H
        W_beta = np.vstack((W, beta_matr))
        A_beta = np.vstack((A, beta_zeros_matr))
        Sol, info = nnls.nnlsm_blockpivot(W_beta, A_beta, init=H.T)
        H = Sol.T

        # Now, solve for W
        H_alpha = np.vstack((H, alpha_matr))
        A_alpha = np.hstack((A, alpha_zeros_matr))
        Sol, info = nnls.nnlsm_blockpivot(H_alpha, A_alpha.T, init=W.T)
        W = Sol.T

        t_iter_elapsed = time.time() - t_iter_start
        err = matr_util.norm_fro_err(A, W, H, norm_A) / norm_A
        rel_error[ii] = err

        str_header = 'Running -- '
        str_iter = 'Iteration %4d' % (ii+1)
        str_err = 'Relative Error: %0.5f' % err
        str_elapsed = 'Elapsed Time: %0.3f sec' % t_iter_elapsed
        my_display('{} {} | {} | {} \r'.format(str_header, str_iter,
                                               str_err, str_elapsed),
                   verbose)
    my_display('\nDone.\n', verbose)

    W, H, weights = matr_util.normalize_column_pair(W, H)

    return W.T, H.T, rel_error
