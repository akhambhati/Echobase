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
import nmf
import nnls
import matrix_utils
import multiprocessing as mp

from ....Common import errors, display


# Helper function for cross validation
def _cross_val(param_dict):
    # Display output
    display.my_display('Optimizing parameter set: {} \n'.format(param_dict['param_id']), True, param_dict['str_path'])

    # Derive params from dict
    n_train_win, n_train_conn = param_dict['train_cfg_matr'].shape
    n_test_win, n_test_conn = param_dict['test_cfg_matr'].shape

    # Run NMF on training set
    fac_subnet_init = np.random.uniform(low=0.0, high=1.0,
                                        size=(param_dict['rank'], n_train_conn))
    fac_coef_init = np.random.uniform(low=0.0, high=1.0,
                                      size=(param_dict['rank'], n_train_win))

    train_fac_subnet, train_fac_coef, train_err = nmf.snmf_bcd(param_dict['train_cfg_matr'],
                                                               alpha=param_dict['alpha'],
                                                               beta=param_dict['beta'],
                                                               fac_subnet_init=fac_subnet_init,
                                                               fac_coef_init=fac_coef_init,
                                                               max_iter=25, verbose=False)

    # Solve single least squares estimate for coefficients on held-out fold
    fac_coef_init = np.random.uniform(low=0.0, high=1.0,
                                      size=(param_dict['rank'], n_test_win))

    _, test_fac_coef, _ = nmf.snmf_bcd(param_dict['test_cfg_matr'],
                                       alpha=param_dict['alpha'],
                                       beta=param_dict['beta'],
                                       fac_subnet_init=train_fac_subnet,
                                       fac_coef_init=fac_coef_init,
                                       max_iter=1, verbose=False)

    # Compute error
    norm_test_cfg_matr = matrix_utils.norm_fro(param_dict['test_cfg_matr'].T)
    err = matrix_utils.norm_fro_err(param_dict['test_cfg_matr'].T,
                                    train_fac_subnet.T,
                                    test_fac_coef.T,
                                    norm_test_cfg_matr) / norm_test_cfg_matr

    return {'param_id': param_dict['param_id'],
            'error': err}


def cross_validation(cfg_matr, alpha_list, beta_list,
                     rank_list, fold, n_proc=8, str_path=None):
    """
    Pursue k-fold cross-validation of NMF over a list of
    alpha, beta, rank

    Parameters
    ----------
        cfg_matr: numpy.ndarray
            The network configuration matrix
            shape: [n_win x n_conn]

        alpha_list: list, float
            Regularization parameter on W

        beta_list: list, float
            Sparsity parameter on H

        rank_list: list, int
            Number of subgraphs to find

        fold: int
            The number of cross-validation folds to compute

        n_proc: int
            Number of parallel processes

        str_path: str
            Text file path to store progress

    Returns
    -------
        param_error: list, dict: {alpha, beta, rank, fold}
            Frobenius error over all parameter combinations
    """

    # Standard param checks
    errors.check_type(alpha_list, list)
    errors.check_type(beta_list, list)
    errors.check_type(rank_list, list)
    errors.check_type(fold, int)

    # Check input dimensions
    if not len(cfg_matr.shape) == 2:
        raise ValueError('%r does not have two dimensions' % cfg_matr)
    n_win = cfg_matr.shape[0]
    n_conn = cfg_matr.shape[1]

    # Generate fold list
    n_obs_per_fold = int(np.floor(n_win / fold))
    n_max_fold = int(np.floor(n_win / (2*np.max(rank_list))))
    if n_obs_per_fold < 2*np.max(rank_list):
        raise ValueError('Number of folds can be max {}'.format(n_max_fold))
    all_obs_ix = np.random.permutation(n_win)[:n_obs_per_fold*fold]

    # Generate parameter list
    param_list = []
    param_id = 0
    for fold_id in xrange(fold):
        test_obs_ix = all_obs_ix[fold_id*n_obs_per_fold:(fold_id+1)*n_obs_per_fold]
        train_obs_ix = np.setdiff1d(all_obs_ix, test_obs_ix)

        train_cfg_matr = cfg_matr[train_obs_ix, :]
        test_cfg_matr = cfg_matr[test_obs_ix, :]

        for alpha in alpha_list:
            for beta in beta_list:
                for rank in rank_list:

                    param_dict =  {'param_id': param_id,
                                   'alpha': alpha,
                                   'beta': beta,
                                   'rank': rank,
                                   'fold_id': fold,
                                   'train_cfg_matr': train_cfg_matr,
                                   'test_cfg_matr': test_cfg_matr,
                                   'str_path': str_path}

                    param_list.append(param_dict)

                    param_id += 1

    # Run parloop
    pool = mp.Pool(processes=n_proc,)
    pop_err = pool.map(_cross_val, param_list)

    # Reformulate into a single dictionary list
    optimization_dict = {'alpha': [],
                         'beta': [],
                         'rank': [],
                         'error': []}

    for run_err in pop_err:
        error_id = run_err['param_id']
        optimization_dict['alpha'].append(param_list[error_id]['alpha'])
        optimization_dict['beta'].append(param_list[error_id]['beta'])
        optimization_dict['rank'].append(param_list[error_id]['rank'])
        optimization_dict['error'].append(run_err['error'])

    return optimization_dict


def min_crossval_param(opt_dict):
    """
    Compute the parameter set that produces the minimum cross-validation error.

    Parameters
    ----------
        opt_dict: dict, entries: {'alpha', 'beta', 'rank', 'error'}
            Output from cross_validation function. Each dict entry is a list of length n_param

    Returns
    -------
        opt_params: dict, {'rank', 'alpha, 'beta'}
            Optimum rank, alpha, and beta based on the minimum average cross validation error
    """

    # Standard param checks
    errors.check_type(opt_dict, dict)

    # Compute average error as a function of each parameter
    error_rank = [opt_dict['error'][np.flatnonzero(opt_dict['rank']==rank)].mean()
                            for rank in np.unique(opt_dict['rank'])]
    opt_ix = np.argmin(error_rank)
    opt_rank = np.unique(opt_dict['rank'])[opt_ix]

    error_alpha = [opt_dict['error'][np.flatnonzero(opt_dict['alpha']==alpha)].mean()
                                for alpha in np.unique(opt_dict['alpha'])]
    opt_ix = np.argmin(error_alpha)
    opt_alpha = np.unique(opt_dict['alpha'])[opt_ix]

    error_beta = [opt_dict['error'][np.flatnonzero(opt_dict['beta']==beta)].mean()
                                for beta in np.unique(opt_dict['beta'])]
    opt_ix = np.argmin(error_beta)
    opt_beta = np.unique(opt_dict['beta'])[opt_ix]

    opt_params = {'rank': opt_rank,
                  'alpha': opt_alpha,
                  'beta': opt_beta}

    return opt_params
