"""
Measurements of global network topology

Created by: Ankit Khambhati

Change Log
----------
2016/03/10 - Implemented DegrCentral, EvecCentral, SyncCentral pipes
"""

from __future__ import division
import numpy as np

from ...Common import errors


def synchronizability(adj):
    """
    Function for computing synchronizability given the network topology

    Parameters
    ----------
        adj: ndarray, shape (N, N)
            Undirected, symmetric adjacency matrix with N nodes

    Returns
    -------
        sync: float
            Synchronizability of the adjacency matrix (lambda_2 / lambda_max)
    """

    # Standard param checks
    errors.check_type(adj, np.ndarray)
    errors.check_dims(adj, 2)
    if not (adj == adj.T).all():
        raise Exception('Adjacency matrix is not undirected and symmetric.')

    # Get the degree vector of the adj
    deg_vec = np.sum(adj, axis=0)
    deg_matr = np.diag(deg_vec)

    # Laplacian
    lapl = deg_matr - adj

    # Compute eigenvalues and eigenvectors, ensure they are real
    eigval, eigvec = np.linalg.eig(lapl)
    eigval = np.real(eigval)

    # Sort smallest to largest eigenvalue
    eigval = np.sort(eigval)
    sync = np.abs(eigval[1] / eigval[-1])

    return sync
