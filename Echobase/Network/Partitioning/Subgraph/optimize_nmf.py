'''
NMF optimization for dynamic brain networks

Uses cross-validation to find the optimal parameter set for a collection of
network adjacency matrices.

Created by: Ankit Khambhati

Change Log
----------
2016/12/25 - Implemented consensus detection
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


def _cons_seeds(param_dict):
    # Display output
    display.my_display('Optimizing parameter set: {} \n'.format(param_dict['param_id']), True, param_dict['str_path'])

    # Derive params from dict
    n_win, n_conn = param_dict['cfg_matr'].shape

    # Run NMF
    fac_subnet_init = np.random.uniform(low=0.0, high=1.0,
                                        size=(param_dict['rank'], n_conn))
    fac_coef_init = np.random.uniform(low=0.0, high=1.0,
                                      size=(param_dict['rank'], n_win))

    fac_subnet, fac_coef, err = nmf.snmf_bcd(param_dict['cfg_matr'],
                                             alpha=param_dict['alpha'],
                                             beta=param_dict['beta'],
                                             fac_subnet_init=fac_subnet_init,
                                             fac_coef_init=fac_coef_init,
                                             max_iter=100, verbose=False)

    return {'param_id': param_dict['param_id'],
            'fac_subnet': fac_subnet}


def consensus_nmf(cfg_matr, opt_alpha, opt_beta,
                  opt_rank, n_seed, n_proc, str_path=None):
    """
    Consensus clustering for NMF

    Parameters
    ----------
        cfg_matr: numpy.ndarray
            The network configuration matrix
            shape: [n_win x n_conn]

        opt_alpha: float
            Regularization parameter on W

        opt_beta: float
            Sparsity parameter on H

        opt_rank: list, int
            Number of subgraphs to find

        n_seed: int
            Number of initial seeds to identify consensus subgraphs

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
    errors.check_type(cfg_matr, np.ndarray)
    errors.check_type(opt_alpha, float)
    errors.check_type(opt_beta, float)
    errors.check_type(opt_rank, int)
    errors.check_type(n_seed, int)

    # Check input dimensions
    if not len(cfg_matr.shape) == 2:
        raise ValueError('%r does not have two dimensions' % cfg_matr)
    n_win = cfg_matr.shape[0]
    n_conn = cfg_matr.shape[1]

    # Generate parameter list
    param_list = []
    param_id = 0
    for seed_id in xrange(n_seed):
        param_dict =  {'param_id': param_id,
                       'alpha': opt_alpha,
                       'beta': opt_beta,
                       'rank': opt_rank,
                       'seed_id': seed_id,
                       'cfg_matr': cfg_matr,
                       'str_path': str_path}

        param_list.append(param_dict)

        param_id += 1

    # Run parloop
    pool = mp.Pool(processes=n_proc,)
    pop_fac = pool.map(_cons_seeds, param_list)

    # Concatenate subgraphs into ensemble
    subnet_ensemble = np.zeros((opt_rank*n_seed, n_conn))
    obs_id = 0
    for fac in pop_fac:
        for subnet in fac['fac_subnet']:
            subnet_ensemble[obs_id, :] = subnet[...]
            obs_id += 1

    # NMF on the ensemble to uncover subgraphs
    subnet_ens_init = np.random.uniform(low=0.0, high=1.0, size=(opt_rank, n_conn))
    coef_ens_init = np.random.uniform(low=0.0, high=1.0, size=(opt_rank, opt_rank*n_seed))

    subnet_ens, coef_ens, err = nmf.snmf_bcd(
        subnet_ensemble,
        alpha=0.0,
        beta=0.0,
        fac_subnet_init=subnet_ens_init,
        fac_coef_init=coef_ens_init,
        max_iter=100, verbose=True)

    # Final NMF to compute coefficients
    coef_ens_init = np.random.uniform(low=0.0, high=1.0, size=(opt_rank, n_win))

    subnet_ens, coef_ens, err = nmf.snmf_bcd(
        cfg_matr,
        alpha=opt_alpha,
        beta=opt_beta,
        fac_subnet_init=subnet_ens,
        fac_coef_init=coef_ens_init,
        max_iter=100, verbose=True)

    return subnet_ens, coef_ens, err
