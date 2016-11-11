from __future__ import division
import sys
import time
import numpy as np
import scipy.sparse as sp

from ....Common import errors
from ....Common.display import my_display

eps = np.finfo('float').eps


def _tidyconfig(S):
    '''
    Helper function
    '''
    T = np.zeros_like(S)+-1
    for i in xrange(len(S)):
        if T[i] == -1:
            T[S == S[i]] = np.max(T) + 1
    return T


def _metanetwork(J, S):
    '''
    Helper function
    '''
    if sp.issparse(J):
        PP = sp.csr_matrix((np.ones(len(S)), (np.arange(len(S)), S)))
        M = PP.T.dot(PP.T.dot(J).T)
        M = M.tolil()
    else:
        PP = sp.csr_matrix((np.ones(len(S)), (np.arange(len(S)), S)))
        M = PP.T.dot(PP.T.dot(J).T)
    return M


def genlouvain(ml_mod_matr, limit, verbose=True):
    '''
    Implementation of the Louvain algorithm to perform
    modularity optimization and assign network nodes into communities.

    Parameters
    ----------
        ml_mod_matr: np.ndarray
            Multilayer modularity matrix
            Has shape: [n_nodes*n_layers x n_nodes*n_layers]

        limit: int
            Number of iterations before resuming to next level of optimization

    Returns
    -------
        comm_vec: np.ndarray
            Final community assignments vector
            Has shape: [n_nodes*n_layers,]

        q: float
            Optimized modularity statistic Q
    '''
    # Standard param checks
    errors.check_type(ml_mod_matr, np.ndarray)

    # Check ml_mod_matr dimensions
    if not len(ml_mod_matr.shape) == 2:
        raise ValueError('%r does not have two dimensions' % ml_mod_matr)
    if not ml_mod_matr.shape[0] == ml_mod_matr.shape[1]:
        raise ValueError('%r should be a square matrix' % ml_mod_matr)
    n_nodelayer = ml_mod_matr.shape[0]

    # Check comm_vec dimensions
    comm_vec = np.arange(n_nodelayer)
    comm_vec_old = np.zeros(n_nodelayer)

    # Aggregated community modularity matrix
    # (see reduced size approach Carchiolo et al. [2011])
    ml_mod_matr_new = ml_mod_matr.copy()

    # Total change in modularity
    dtot = eps

    # Begin greedy algorithm, loop around each pass
    while True:
        cond_a1 = np.array_equal(comm_vec_old, comm_vec)
        if (cond_a1):
            break

        # Re-initialize loop variables
        # Adapt to changes via. community -> nodes
        n_comm_node = ml_mod_matr_new.shape[0]

        # Unique community assignments --> Community IDs
        comm_vec_old = comm_vec.copy()
        comm_id = np.unique(comm_vec)
        n_comm_id = len(comm_id)
        comm_id_old = np.zeros(n_comm_id)

        # Change in modularity for pass
        dstep = 1

        my_display('\nMerging %d communities \n' % n_comm_id, verbose)

        # Search modularity local maximum
        # Loop over community nodes in first phase
        t_phaseStart = time.time()
        t_phaseEnd = t_phaseStart
        i_phaseCount = 0
        for count in range(limit):
            cond_b1 = np.array_equal(comm_id_old, comm_id)
            cond_b2 = ((dstep/dtot) <= (2*eps))
            if (cond_b1 & cond_b2):
                break

            # Re-initialize loop variables
            comm_id_old = comm_id.copy()

            # Sparse matrix [community nodes x community assignment]
            comm_lut = sp.coo_matrix((np.ones(n_comm_id),
                                      (np.arange(n_comm_id), comm_id)))
            # Sparse matrix [community assignment x community nodes]
            comm_lut = comm_lut.tocsc()

            dstep = 0

            for i in np.random.permutation(n_comm_node):
                # Community IDs of connected nodes
                u = np.unique(np.hstack(
                    (comm_id[i],
                     comm_id[ml_mod_matr_new[:, i] > 0])))

                # All possible in-community edge strengths
                dH = comm_lut[:, u].T.dot(ml_mod_matr_new[:, i])

                # Current node's community ID index
                comm_id_i = np.flatnonzero(u == comm_id[i])

                # Potentially remove edge from current community
                dH[comm_id_i] = dH[comm_id_i] - ml_mod_matr_new[i, i]
                k = np.argmax(dH)

                # Check if moving to a new community optimizes modularity more
                # than staying in current community
                if (dH[k] > dH[comm_id_i]):
                    dtot = dtot + dH[k] - dH[comm_id_i]
                    dstep = dstep + dH[k] - dH[comm_id_i]

                    # Update community lookup table with change
                    comm_lut = comm_lut.tocoo()
                    comm_lut.col[comm_lut.row == i] = u[k]
                    comm_lut = comm_lut.tocsc()

                    # Update Community ID
                    comm_id[i] = u[k]

            t_phaseEnd = time.time()
            i_phaseCount = len(np.unique(comm_id))
            my_display('   --- Found %4d communities in %.5f sec \r' %
                    (i_phaseCount, t_phaseEnd-t_phaseStart),
                    verbose)

        comm_id = _tidyconfig(comm_id)
        for i in xrange(n_comm_id):
            comm_vec[comm_vec == i] = comm_id[i]

        # When community assignment converges
        # compute quality function measurement
        # of the community assignments
        if np.array_equal(comm_vec_old, comm_vec):
            Q = np.sum(np.diag(ml_mod_matr_new))
            break

        ml_mod_matr_new = _metanetwork(ml_mod_matr, comm_vec)

    my_display('\n', verbose)

    return comm_vec, Q
