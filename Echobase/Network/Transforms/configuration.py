'''
Functions to construct configuration vectors

Created by: Ankit Khambhati

Change Log
----------
2016/02/05 - Shift coherence estimation to ..Pipes.spectral.Spectral
2016/02/03 - Added full coherence and power spectrum analysis
2016/02/02 - Implement class for computing multitaper coherence
'''

import numpy as np

from ...Common import errors

CONN_STR = ' <> '


def label_nodes_to_symm_conns(node_labels):
    '''
    Convert label of nodes to label of connections
    Assumes symmetric connectivity

    Parameters
    ----------
        node_labels: list(str)
            List of node labels as strings

    Returns
    -------
        conn_labels: list(str)
            List of connections as strings.
            Nomenclature: '[Node_1]<>[Node_2]' (ignore [] in string)

        node_pair_labels: list(tuples(str, str))
            List of pairwise nodes
    '''
    # Standard param checks
    errors.check_type(node_labels, list)
    errors.check_type(node_labels[0], str)

    # Compute number of nodes and connections
    n_nodes = len(node_labels)
    triu_ix, triu_iy = np.triu_indices(n_nodes, k=1)

    conn_labels = ['{}{}{}'.format(node_labels[ix], CONN_STR, node_labels[iy])
                   for ix, iy in zip(triu_ix, triu_iy)]

    node_pair_labels = [(node_labels[ix], node_labels[iy])
                        for ix, iy in zip(triu_ix, triu_iy)]

    return conn_labels, node_pair_labels


def label_symm_conns_to_nodes(conn_labels):
    '''
    Convert label of connections to label of nodes
    Assumes symmetric connectivity

    Parameters
    ----------
        conn_labels: list(str)
            List of connections as strings.
            Nomenclature: '[Node_1]<>[Node_2]' (ignore [] in string)

    Returns
    -------
        node_labels: list(str)
            List of pairwise nodes

        node_pair_labels: list(tuples(str, str))
            List of pairwise nodes
    '''
    # Standard param checks
    errors.check_type(conn_labels, list)
    errors.check_type(conn_labels[0], str)

    # Split all the connections into node pairs
    node_pair_labels = []
    node_labels = []
    for lbl in conn_labels:
        nodes = lbl.split(CONN_STR)

        node_pair_labels.append(
            tuple(nodes))

        if not nodes[0] in node_labels:
            node_labels.append(nodes[0])
        if not nodes[1] in node_labels:
            node_labels.append(nodes[1])

    return node_labels, node_pair_labels


def find_symm_conn(node_pair_labels, conn_labels):
    '''
    Return connection indices for a list of node pairs
    Assumes symmetric connectivity

    Parameters
    ----------
        node_pair_labels: list(tuples(str, str))
            List of pairwise nodes

        conn_labels: list(str)
            List of connections as strings.
            Nomenclature: '[Node_1]<>[Node_2]' (ignore [] in string)

    Returns
    -------
        conn_ix: list(int)
            List of connection indices
    '''
    # Standard param checks
    errors.check_type(node_pair_labels, list)
    errors.check_type(node_pair_labels[0], tuple)
    errors.check_type(node_pair_labels[0][1], str)
    errors.check_type(conn_labels, list)
    errors.check_type(conn_labels[0], str)

    # Iterate over node pairs
    conn_ix = []
    for pair in node_pair_labels:
        ix = None
        conn_ref_label = '{}{}{}'.format(pair[0], CONN_STR, pair[1])

        if conn_ref_label in conn_labels:
            ix = conn_labels.index(conn_ref_label)

        if ix is None:
            conn_ref_label = '{}{}{}'.format(pair[1], CONN_STR, pair[0])

        if conn_ref_label in conn_labels:
            ix = conn_labels.index(conn_ref_label)

        if ix is None:
            print('Could not locate: {}'.format(pair))
        else:
            conn_ix.append(ix)

    return conn_ix


def convert_conn_vec_to_adj_matr(conn_vec):
    '''
    Convert connections to adjacency matrix
    Assumes symmetric connectivity

    Parameters
    ----------
        conn_vec: numpy.ndarray
            Vector with shape (n_conn,) specifying unique connections

    Returns
    -------
        adj_matr: numpy.ndarray
            Symmetric matrix with shape (n_node, n_node)
    '''
    # Standard param checks
    errors.check_type(conn_vec, np.ndarray)
    if not len(conn_vec.shape) == 1:
        raise ValueError('%r has more than 1-dimension')

    # Compute number of nodes
    n_node = int(np.floor(np.sqrt(2*len(conn_vec)))+1)

    # Compute upper triangle indices (by convention)
    triu_ix, triu_iy = np.triu_indices(n_node, k=1)

    # Convert to adjacency matrix
    adj_matr = np.zeros((n_node, n_node))
    adj_matr[triu_ix, triu_iy] = conn_vec

    adj_matr += adj_matr.T

    return adj_matr


def convert_adj_matr_to_cfg_matr(adj_matr):
    '''
    Convert connections to adjacency matrix
    Assumes symmetric connectivity

    Parameters
    ----------
        adj_matr: numpy.ndarray
            Matrix with shape (n_win, n_node, n_node)

    Returns
    -------
        cfg_matr: numpy.ndarray
            Symmetric matrix with shape (n_win, n_conn)
    '''
    # Standard param checks
    errors.check_type(adj_matr, np.ndarray)
    if not len(adj_matr.shape) == 3:
        raise ValueError('%r requires 3-dimensions (n_win, n_node, n_node)')

    # Compute number of nodes
    n_node = adj_matr.shape[1]

    # Compute upper triangle indices (by convention)
    triu_ix, triu_iy = np.triu_indices(n_node, k=1)

    # Convert to configuration matrix
    cfg_matr = adj_matr[:, triu_ix, triu_iy]

    return cfg_matr
