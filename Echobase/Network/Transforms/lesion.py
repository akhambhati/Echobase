"""
Tools to lesion the network

Created by: Ankit Khambhati

Change Log
----------
2016/11/11 - Implemented lesion
"""

from __future__ import division
import numpy as np

from ...Common import errors


def node_lesion(adj, node_list):
    """
    Function for removing full nodes from the network

    Parameters
    ----------
        adj: ndarray, shape (N, N)
            Undirected, symmetric adjacency matrix with N nodes

        node_list: list
            List of node indices to simultaneously remove from the network
    """

    # Standard param checks
    errors.check_type(adj, np.ndarray)
    errors.check_dims(adj, 2)
    if not (adj == adj.T).all():
        raise Exception('Adjacency matrix is not undirected and symmetric.')
    errors.check_type(node_list, list)

    # Remove node from network
    adj_lesion = adj.copy()
    adj_lesion = np.delete(adj_lesion, node_list, axis=0)
    adj_lesion = np.delete(adj_lesion, node_list, axis=1)

    return adj_lesion


def edge_lesion(adj, edge_list):
    """
    Function for removing full nodes from the network

    Parameters
    ----------
        adj: ndarray, shape (N, N)
            Undirected, symmetric adjacency matrix with N nodes

        edge_list: ndarray, shape (L, 2)
            Array of node-pairs (edges) to remove from the network
    """

    # Standard param checks
    errors.check_type(adj, np.ndarray)
    errors.check_dims(adj, 2)
    if not (adj == adj.T).all():
        raise Exception('Adjacency matrix is not undirected and symmetric.')
    errors.check_type(edge_list, np.ndarray)
    errors.check_dims(edge_list, 2)

    # Remove edges from network
    adj_lesion = adj.copy()
    for node_pair in edge_list:
        adj_lesion[node_pair[0], node_pair[1]] = 0
        adj_lesion[node_pair[1], node_pair[0]] = 0

    return adj_lesion
