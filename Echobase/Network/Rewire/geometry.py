"""
Network rewiring to preserve geometry

Adapted by: Ankit Khambhati
Reference: Roberts et al. (2016) NeuroImage 124:379-393.

Change Log
----------
2017/01/30 - Implemented surrogate_trend, surrogate_bin
"""

from __future__ import division
import numpy as np

from ...Common import errors
from ..Transforms import configuration


def surrogate_trend(adj, dist, mean_ord, std_ord):
    """
    Random graphs that preserve distance effect

    Parameters
    ----------
        adj: ndarray, shape (N, N)
            Undirected, symmetric adjacency matrix with N nodes

        dist: ndarray, shape (N, N)
            Undirected, symmetric distance matrix with N nodes

        mean_ord: int
            Polynomial fit order for preserving mean

        std_ord: int
            Polynommial fit order for preserving standard deviation

    Returns
    -------
        adj_wgt: ndarray, shape (N, N)
            Undirected, symmetric surrogate adjacency matrix with N nodes
            The edge distribution is preserved, but not strengths
    """

    # Standard param checks
    errors.check_type(adj, np.ndarray)
    errors.check_dims(adj, 2)
    if not (adj == adj.T).all():
        raise Exception('Adjacency matrix is not undirected and symmetric.')
    errors.check_type(dist, np.ndarray)
    errors.check_dims(dist, 2)
    if not (dist == dist.T).all():
        raise Exception('Distance matrix is not undirected and symmetric.')
    errors.check_type(mean_ord, int)
    errors.check_type(std_ord, int)

    # Preliminaries
    n_node = adj.shape[0]
    adj_wgt = np.zeros((n_node, n_node))

    cfg_vec = configuration.convert_adj_matr_to_cfg_matr(adj.reshape(1, n_node, n_node)).reshape(-1)
    dist_vec = configuration.convert_adj_matr_to_cfg_matr(dist.reshape(1, n_node, n_node)).reshape(-1)

    # Remove mean
    pfit1 = np.polyfit(dist_vec, cfg_vec, mean_ord)
    demean_cfg_vec = cfg_vec - np.polyval(pfit1, dist_vec)

    # Adjust variance
    pfit2 = np.polyfit(dist_vec, np.abs(demean_cfg_vec), std_ord)
    std_cfg_vec = demean_cfg_vec / np.polyval(pfit2, dist_vec)

    # Rewire the edges
    rnd_cfg_vec = np.random.permutation(std_cfg_vec)

    # Add geometry back
    rnd_cfg_vec = rnd_cfg_vec * np.polyval(pfit2, dist_vec)
    rnd_cfg_vec = rnd_cfg_vec + np.polyval(pfit1, dist_vec)

    # Rank re-order by using surrogate weights as a scaffold to reorder the original weights
    rank_ix_rnd = np.argsort(np.argsort(rnd_cfg_vec))
    rank_cfg_vec = np.sort(cfg_vec)[rank_ix_rnd]

    # Put into adjacency matrix
    adj_wgt = configuration.convert_conn_vec_to_adj_matr(rank_cfg_vec)

    return adj_wgt
