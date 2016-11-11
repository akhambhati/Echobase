'''
Construct and manipulate multilayer representations of configuration vectors

Created by: Ankit Khambhati

Change Log
----------
2016/02/03 - Implement functions to construct multilayer networks
'''

import numpy as np
import scipy.sparse as sp

from ....Common import errors
from ...Transforms import configuration


def ml_modularity_matr(conn_matr, gamma, omega, null):
    """
    Find the multilayer modularity matrix of network and an assocomm_initated
    null model. Method assumes sequential linking between layers with
    homogenous weights.

    Parameters
    ----------
        conn_matr: numpy.ndarray
            Connection matrix over multiple layers
            Has shape: [n_layers x n_conns]

        gamma: float
            Intra-layer resolution parameter, typical values around 1.0

        omega: float
            Inter-layer resolution parameter, typical values around 1.0

        null: str
            Choose a null mode type: ['None', 'temporal',
                                      'connectional', 'nodal']

    Returns
    -------
        ml_mod_matr: numpy.ndarray
            Multilayer modularity matrix
            Has shape: [n_nodes*n_layers x n_nodes*n_layers]

        twomu: float
            Total edge weight in the network
    """
    # Standard param checks
    errors.check_type(conn_matr, np.ndarray)
    errors.check_type(gamma, float)
    errors.check_type(omega, float)
    errors.check_type(null, str)

    # Check conn_matr dimensions
    if not len(conn_matr.shape) == 2:
        raise ValueError('%r does not have two-dimensions' % conn_matr)
    n_layers = conn_matr.shape[0]
    n_conns = conn_matr.shape[1]
    n_nodes = int(np.floor(np.sqrt(2*n_conns))+1)

    # Check null model specomm_initfication
    valid_null_types = ['none', 'temporal', 'connectional', 'nodal']
    null = null.lower()
    if null not in valid_null_types:
        raise ValueError('%r is not on of %r' % (null, valid_null_types))

    # Initialize multilayer matrix
    B = np.zeros((n_nodes*n_layers, n_nodes*n_layers))
    twomu = 0

    if null == 'temporal':
        rnd_layer_ix = np.random.permutation(n_layers)
        conn_matr = conn_matr[rnd_layer_ix, :]

    if null == 'connectional':
        rnd_node_ix = np.random.permutation(n_nodes)
        rnd_node_iy = np.random.permutation(n_nodes)
        ix, iy = np.mgrid[0:n_nodes, 0:n_nodes]

    for ll, conn_vec in enumerate(conn_matr):
        A = configuration.convert_conn_vec_to_adj_matr(conn_vec)
        if null == 'connectional':
            A = A[rnd_node_ix[ix], rnd_node_iy[iy]]
            A = np.triu(A, k=1)
            A += A.T

        # Compute node degree
        k = np.sum(A, axis=0)
        twom = np.sum(k)        # Intra-layer average node degree
        twomu += twom           # Inter-layer accumulated node degree

        # NG Null-model
        if twom < 1e-6:
            P = np.dot(k.reshape(-1, 1), k.reshape(1, -1)) / 1.0
        else:
            P = np.dot(k.reshape(-1, 1), k.reshape(1, -1)) / twom

        # Multi-slice modularity matrix
        start_ix = ll*n_nodes
        end_ix = (ll+1)*n_nodes
        B[start_ix:end_ix, start_ix:end_ix] = A - gamma*P

    # Add inter-slice degree
    twomu += twomu + 2*omega*n_nodes*(n_layers-1)

    # Add the sequential inter-layer model
    interlayer = sp.spdiags(np.ones((2, n_nodes*n_layers)),
                            [-n_nodes, n_nodes],
                            n_nodes*n_layers, n_nodes*n_layers).toarray()
    if null == 'nodal':
        null_layer = np.random.permutation(np.diag(np.ones(n_nodes)))
        for ll in xrange(n_layers-1):
            interlayer[ll*n_nodes:(ll+1)*n_nodes,
                       (ll+1)*n_nodes:(ll+2)*n_nodes] = null_layer
        interlayer = np.triu(interlayer, k=1)
        interlayer += interlayer.T

    B = B + omega*interlayer
    B = np.triu(B, k=1)
    B += B.T
    ml_mod_matr = B

    return ml_mod_matr, twomu
