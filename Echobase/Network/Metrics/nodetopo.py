"""
Measurements of nodal importance (centrality)

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


def node_strength(adj):
    """
    Function for computing sum of connection strengths from a node

    Parameters
    ----------
        adj: ndarray, shape (N, N)
            Undirected, symmetric adjacency matrix with N nodes

    Returns
    -------
        deg_vec: ndarray, shape (N,)
            Node strength vector over N variates
    """

    # Standard param checks
    errors.check_type(adj, np.ndarray)
    errors.check_dims(adj, 2)
    if not (adj == adj.T).all():
        raise Exception('Adjacency matrix is not undirected and symmetric.')

    # Get the degree vector of the adj
    deg_vec = np.sum(adj, axis=0)

    return deg_vec


def node_control(adj):
    """
    Function for computing control centrality of a node (change in synchronizability)

    Parameters
    ----------
        adj: ndarray, shape (N, N)
            Undirected, symmetric adjacency matrix with N nodes

    Returns
    -------
        control_vec: ndarray, shape (N,)
            Control centrality based on delta_sync for each of N variates
    """

    # Standard param checks
    errors.check_type(adj, np.ndarray)
    errors.check_dims(adj, 2)
    if not (adj == adj.T).all():
        raise Exception('Adjacency matrix is not undirected and symmetric.')

    # Get data attributes
    n_node = adj.shape[0]

    # Get the original synchronizability
    base_sync = synchronizability(adj)

    control_vec = np.zeros(n_node)
    for i in xrange(n_node):
        adj_lesion = lesion.node_lesion(adj, [i])
        lesion_sync = synchronizability(adj_lesion)

        control_vec[i] = (lesion_sync-base_sync) / base_sync

    return control_vec
