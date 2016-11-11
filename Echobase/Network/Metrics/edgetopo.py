"""
Measurements of edge importance (centrality)

Created by: Ankit Khambhati

Change Log
----------
2016/03/10 - Implemented DegrCentral, EvecCentral, SyncCentral pipes
"""

from __future__ import division
import numpy as np

from ...Common import errors
from .globaltopo import synchronizability
from ..Transforms import lesion


def edge_control(adj):
    """
    Function for computing control centrality of an edge (change in synchronizability)

    Parameters
    ----------
        adj: ndarray, shape (N, N)
            Undirected, symmetric adjacency matrix with N nodes
    """

    # Standard param checks
    errors.check_type(adj, np.ndarray)
    errors.check_dims(adj, 2)
    if not (adj == adj.T).all():
        raise Exception('Adjacency matrix is not undirected and symmetric.')

    # Get data attributes
    n_node = adj.shape[0]
    triu_ix, triu_iy = np.triu_indices(n_node, k=1)

    # Get the original synchronizability
    base_sync = synchronizability(adj)

    control_matr = np.zeros((n_node, n_node))
    for node_pair in np.array(zip(triu_ix, triu_iy)):
        adj_lesion = lesion.edge_lesion(adj, node_pair.reshape(1,-1))
        lesion_sync = synchronizability(adj_lesion)

        control_matr[node_pair[0], node_pair[1]] = (lesion_sync-base_sync) / base_sync
        control_matr[node_pair[1], node_pair[0]] = (lesion_sync-base_sync) / base_sync

    return control_matr
