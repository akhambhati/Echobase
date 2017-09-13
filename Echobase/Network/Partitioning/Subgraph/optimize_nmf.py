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

def gen_random_sampling_paramset(rank_range, alpha_range, beta_range, n_param,
                                 fold_list, str_path=None):
    """
    Generate a parameter dictionary for optimizing NMF

    Parameters
    ----------
        rank_range: tuple, (2,)
            Number of subgraphs to find
            Supply lower and upper bounds on the search space (uniform sampling)

        alpha_range: tuple, (2,)
            Regularization parameter on W
            Supply lower and upper bounds on the search space (uniform sampling)

        beta_range: tuple, (2,)
            Sparsity parameter on H
            Supply lower and upper bounds on the search space (uniform sampling)

        n_param: int
            Number of random samples to pursue

        fold_list: list of lists
            Each nested list contains the observations indices that are members
            of each fold

        str_path: str
            Text file path to store progress

    Returns
    -------
        param_list: list of dict: {param_id, alpha, beta, rank, fold_id,
                                   train_ix, test_ix, str_path}
            Each list entry contains a dictionary of a single parameter set
            for NMF optimization
    """

    # Standard param checks
    errors.check_type(alpha_range, tuple)
    errors.check_type(beta_range, tuple)
    errors.check_type(rank_range, tuple)
    errors.check_type(n_param, int)
    errors.check_type(fold_list, list)

    # Generate the random samples
    rank_list = np.random.randint(low=rank_range[0], high=rank_range[1]+1, size=n_param)
    alpha_list = np.random.uniform(low=alpha_range[0], high=alpha_range[1], size=n_param)
    beta_list = np.random.uniform(low=beta_range[0], high=beta_range[1], size=n_param)

    # Generate fold list
    n_fold = len(fold_list)

    # Check the folds for size-matching and repeats
    all_fold_ix = []
    all_fold_size = []
    for fold_id in xrange(n_fold):
        all_fold_size.append(len(fold_list[fold_id]))
        for fold_ix in fold_list[fold_id]:
            all_fold_ix.append(fold_ix)

    if not len(all_fold_ix) == len(np.unique(all_fold_ix)):
        raise Exception('Folds have overlapping, non-unique observations')

    if np.sum(np.abs(np.diff(np.array(all_fold_size)))) > 0:
        print('Warning: Folds are of unequal size')

    # Generate parameter list
    param_list = []
    param_id = 0
    for fold_id in xrange(n_fold):
        test_ix = fold_list[fold_id]

        train_ix = []
        for nonfold_id in np.setdiff1d(np.arange(n_fold), fold_id):
            for fold_ix in fold_list[nonfold_id]:
                train_ix.append(fold_ix)

        for alpha, beta, rank in zip(alpha_list, beta_list, rank_list):
            param_dict =  {'param_id': param_id,
                           'alpha': alpha,
                           'beta': beta,
                           'rank': rank,
                           'fold_id': fold_id,
                           'train_ix': train_ix,
                           'test_ix': test_ix,
                           'str_path': str_path}
            param_list.append(param_dict)
            param_id += 1

    return param_list


# Helper function for cross validation
def run_xval_paramset(cfg_matr, param_dict):
    """
    Run NMF cross-validation using a single parameter dictionary generated from
    gen_random_sampling_paramset

    Parameters
    ----------
        cfg_matr: numpy.ndarray, shape:(n_win, n_conn)
            The network configuration matrix

        param_dict: dict with keys: {param_id, alpha, beta, rank, fold_id,
                                     train_ix, test_ix, str_path}
            Single entry of list of dicts returned by gen_random_sampling_paramset


    Return
    ------
        qmeas_dict: dict with keys: {param_id, err,
                                     pct_sparse_subnet, pct_sparse_coef}
            Quality measures associated with param_dict
    """

    # Check input dimensions of cfg_matr
    if not len(cfg_matr.shape) == 2:
        raise ValueError('%r does not have two dimensions' % cfg_matr)
    n_win = cfg_matr.shape[0]
    n_conn = cfg_matr.shape[1]

    # Derive params from dict
    train_cfg_matr = cfg_matr[param_dict['train_ix'], :]
    test_cfg_matr = cfg_matr[param_dict['test_ix'], :]
    n_train_win, n_train_conn = train_cfg_matr.shape
    n_test_win, n_test_conn = test_cfg_matr.shape

    # Display output
    display.my_display('Optimizing parameter set: {} \n'.format(param_dict['param_id']), True, param_dict['str_path'])

    # Run NMF on training set
    fac_subnet_init = np.random.uniform(low=0.0, high=1.0,
                                        size=(param_dict['rank'], n_train_conn))
    fac_coef_init = np.random.uniform(low=0.0, high=1.0,
                                      size=(param_dict['rank'], n_train_win))

    train_fac_subnet, train_fac_coef, train_err = nmf.snmf_bcd(train_cfg_matr,
                                                               alpha=param_dict['alpha'],
                                                               beta=param_dict['beta'],
                                                               fac_subnet_init=fac_subnet_init,
                                                               fac_coef_init=fac_coef_init,
                                                               max_iter=25, verbose=False)

    # Solve single least squares estimate for coefficients on held-out fold
    fac_coef_init = np.random.uniform(low=0.0, high=1.0,
                                      size=(param_dict['rank'], n_test_win))

    _, test_fac_coef, _ = nmf.snmf_bcd(test_cfg_matr,
                                       alpha=param_dict['alpha'],
                                       beta=param_dict['beta'],
                                       fac_subnet_init=train_fac_subnet,
                                       fac_coef_init=fac_coef_init,
                                       max_iter=1, verbose=False)

    # Compute error
    norm_test_cfg_matr = matrix_utils.norm_fro(test_cfg_matr.T)
    err = matrix_utils.norm_fro_err(test_cfg_matr.T,
                                    train_fac_subnet.T,
                                    test_fac_coef.T,
                                    norm_test_cfg_matr) / norm_test_cfg_matr

    # Compute sparsity of the subgraphs and coefficients
    pct_sparse_subnet = (train_fac_subnet==0).mean(axis=1).mean()
    pct_sparse_coef = (test_fac_coef==0).mean(axis=1).mean()

    qmeas_dict = {'param_id': param_dict['param_id'],
                  'error': err,
                  'pct_sparse_subgraph': pct_sparse_subnet,
                  'pct_sparse_coef': pct_sparse_coef}

    return qmeas_dict


def find_optimum_xval_paramset(param_list, qmeas_list, search_pct=25):
    """
    Integrate the parameter set and the quality measures to identify
    optimal parameter set based on minimum cross-validation error

    Parameters
    ----------
        param_list: list of dict with keys: {param_id, alpha, beta, rank, fold_id,
                                     train_ix, test_ix, str_path}
            List of dicts returned by gen_random_sampling_paramset

        qmeas_list: list of dict with keys: {param_id, err,
                                             pct_sparse_subnet, pct_sparse_coef}
            List of dicts aggregated over param_list runs of run_xval_paramset

    Return
    ------
        optimization_dict: dict with each quality measure and associated parameters

        opt_param_dict: dict with optimum rank, alpha, beta
    """

    # Reformulate into a single dictionary list
    optimization_dict = {'alpha': [],
                         'beta': [],
                         'rank': [],
                         'error': [],
                         'pct_sparse_subgraph': [],
                         'pct_sparse_coef': []}

    for qmeas_dict in qmeas_list:
        for param_dict in param_list:
            if qmeas_dict['param_id'] == param_dict['param_id']:
                optimization_dict['alpha'].append(param_dict['alpha'])
                optimization_dict['beta'].append(param_dict['beta'])
                optimization_dict['rank'].append(param_dict['rank'])
                optimization_dict['error'].append(qmeas_dict['error'])
                optimization_dict['pct_sparse_subgraph'].append(qmeas_dict['pct_sparse_subgraph'])
                optimization_dict['pct_sparse_coef'].append(qmeas_dict['pct_sparse_coef'])

    for key in optimization_dict.keys():
        optimization_dict[key] = np.array(optimization_dict[key])

    # Find optimum parameter set
    opt_param_dict = {}
    opt_param_dict['rank'] = int(optimization_dict['rank'][optimization_dict['error'] < np.percentile(optimization_dict['error'], search_pct)].mean().round())
    opt_param_dict['alpha'] = optimization_dict['alpha'][optimization_dict['error'] < np.percentile(optimization_dict['error'], search_pct)].mean()
    opt_param_dict['beta'] = optimization_dict['beta'][optimization_dict['error'] < np.percentile(optimization_dict['error'], search_pct)].mean()

    print('Optimal Rank: {}'.format(opt_param_dict['rank']))
    print('Optimal Alpha: {}'.format(opt_param_dict['alpha']))
    print('Optimal Beta: {}'.format(opt_param_dict['beta']))


    return optimization_dict, opt_param_dict


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
        subnet_ens: numpy.ndarray shape: [rank x n_conn]
            Consensus partitions across ensemble of subgraph estimates

        coef_ens: numpy.ndarray shape: [rank x n_win]
            Coefficients for consensus subgraphs

        err: numpy.ndarray
            Reconstruction error for consensus subgraphs and coefficients
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
