"""
Functions to render connectivity in a circular plot

Created by: Ankit Khambhati

Change Log
----------
2017/04/09 -- Created the draw function
"""
from __future__ import print_function

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: Simplified BSD


from itertools import cycle
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as m_path
import matplotlib.patches as m_patches


def draw(conn_list, conn_pct=[90, 100], conn_cmap='YlGnBu', conn_linewidth=1.5,
         node_color=None, ax=None):
    """
    Visualize connectivity as a circular graph.

    Note: This code is based on the circle graph example by Nicolas P. Rougier
    http://www.labri.fr/perso/nrougier/coding/.

    Parameters
    ----------
        conn_list: numpy.ndarray, shape: (n_node*(n_node-1)*0.5,)
            Vector of connection or edge strengths associated with each node
            Assumes undirected, symmetric connectivity

        conn_pct: tuple, shape: (2,)
            Lower and upper percentiles for thresholding connections to plot

        conn_cmap: str
            Colormap to map connection strengths to colors

        conn_linewidth : float
            Line width to use for connections.

        node_color: list, shape: (n_node,)
            Colors corresponding to the nodes arranged around the circle
    """

    # Get number of nodes
    n_node = int(np.ceil(np.sqrt(len(conn_list)*2)))
    if node_color is not None:
        assert(n_node == len(node_color))
    else:
        node_color = [plt.cm.spectral(i / float(n_node))
                      for i in range(n_node)]

    # Layout the nodes around the unit circle
    node_angle = np.linspace(0, 2 * np.pi, n_node, endpoint=False)

    # widths correspond to the minimum angle between two nodes
    dist_mat = node_angle[None, :] - node_angle[:, None]
    dist_mat[np.diag_indices(n_node)] = 1e9
    node_width = np.min(np.abs(dist_mat))

    # get the colormap
    colormap = plt.get_cmap(conn_cmap)

    # Make figure background the same colors as axes
    fig = plt.figure(figsize=(8, 8), facecolor='white')

    # Use a polar axes
    axes = plt.subplot(111, polar=True, axisbg='white')

    # No ticks, we'll put our own
    plt.xticks([])
    plt.yticks([])

    # Set y axes limit, add additional space if requested
    plt.ylim(0, 10 + 6)

    # Remove the black axes border which may obscure the labels
    axes.spines['polar'].set_visible(False)

    # Draw lines between connected nodes, only draw the strongest connections
    lo_thr = np.percentile(conn_list, conn_pct[0])
    hi_thr = np.percentile(conn_list, conn_pct[1])

    # get the connections which we are drawing and sort by connection strength
    # this will allow us to draw the strongest connections first
    triu_ix, triu_iy = np.triu_indices(n_node, k=1)
    conn_draw_idx = np.flatnonzero((conn_list >= lo_thr) & (conn_list <= hi_thr))

    sel_conn = conn_list[conn_draw_idx]
    sel_triu = [triu_ix[conn_draw_idx], triu_iy[conn_draw_idx]]

    # now sort them
    conn_sort_idx = np.argsort(sel_conn)
    sel_conn = sel_conn[conn_sort_idx]
    sel_triu = [sel_triu[0][conn_sort_idx], sel_triu[1][conn_sort_idx]]
    n_conn = len(sel_conn)

    # We want to add some "noise" to the start and end position of the
    # edges: We modulate the noise with the number of connections of the
    # node and the connection strength, such that the strongest connections
    # are closer to the node center
    node_n_conn = np.zeros((n_node), dtype=np.int)
    for i, j in zip(sel_triu[0], sel_triu[1]):
        node_n_conn[i] += 1
        node_n_conn[j] += 1

    # initialize random number generator so plot is reproducible
    rng = np.random.mtrand.RandomState(seed=0)

    noise_max = 0.25 * node_width
    start_noise = rng.uniform(-noise_max, noise_max, n_conn)
    end_noise = rng.uniform(-noise_max, noise_max, n_conn)

    node_n_conn_seen = np.zeros_like(node_n_conn)
    for i, (start, end) in enumerate(zip(sel_triu[0], sel_triu[1])):
        node_n_conn_seen[start] += 1
        node_n_conn_seen[end] += 1

        start_noise[i] *= ((node_n_conn[start] - node_n_conn_seen[start]) /
                           float(node_n_conn[start]))
        end_noise[i] *= ((node_n_conn[end] - node_n_conn_seen[end]) /
                         float(node_n_conn[end]))

    # scale connectivity for colormap (vmin<=>0, vmax<=>1)
    conn_val_scaled = (sel_conn - sel_conn.min()) / (sel_conn.max()-sel_conn.min())

    # Finally, we draw the connections
    for pos, (i, j) in enumerate(zip(sel_triu[0], sel_triu[1])):
        # Start point
        t0, r0 = node_angle[i], 10

        # End point
        t1, r1 = node_angle[j], 10

        # Some noise in start and end point
        t0 += start_noise[pos]
        t1 += end_noise[pos]

        dist_t = np.abs(t0 - t1)
        if dist_t > np.pi:
            dist_t = 2*np.pi - dist_t
        rr = 10 * (1 - np.abs(t1-t0) / np.pi)
        if rr > 7:
            rr=7
        if rr < 3:
            rr=3

        verts = [(t0, r0), (t0, rr), (t1, rr), (t1, r1)]
        codes = [m_path.Path.MOVETO,
                 m_path.Path.CURVE4,
                 m_path.Path.CURVE4,
                 m_path.Path.LINETO]
        path = m_path.Path(verts, codes)

        color = colormap(conn_val_scaled[pos])

        # Actual line
        patch = m_patches.PathPatch(path, fill=False, edgecolor=color,
                                    linewidth=conn_linewidth, alpha=1.)
        axes.add_patch(patch)

    # Draw ring with colored nodes
    height = np.ones(n_node) * 1.0
    bars = axes.bar(node_angle, height, width=node_width, bottom=9,
                    edgecolor='white', lw=0,
                    facecolor='.9', align='center')

    for bar, color in zip(bars, node_color):
        bar.set_facecolor(color)

    return fig, axes
