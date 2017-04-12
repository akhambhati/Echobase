"""
Functions to render connectivity profiles on MNI brain surface

Created by: Ankit Khambhati

Change Log
----------
2017/04/11 -- Updated the draw function with direct triangular mesh rendering
2017/02/11 -- Created the draw function
"""

"""
TODO: GlassBrain Project
1. modularize each function and generate class structure to interact with mlab
2. instantiate all edges, nodes, separately so that the scalar values can be updated
   without redrawing the whole figure
3. optimize the surface rendering for low opacity values so that there aren't weird
   shading effects
"""


from __future__ import division

import numpy as np
from mayavi import mlab
import nibabel as nib


def draw(surf_vertices, surf_triangles, surf_scalars, surf_cmap, surf_opacity,
          node_coords, node_sizes, node_colors,
          conn_list, conn_pct, conn_cmap):
    """
    Draws a brain graph superimposed on the brain surface

    Parameters
    ----------
        surf_vertices: numpy.ndarray, shape:(n_vert, 3)
            Three-dimensional coordinates of each vertex of the surface

        surf_triangles: numpy.ndarray, shape:(n_tria, 3)
            Sets of vertex indices forming a triangle face of the surface

        surf_scalars: numpy.ndarray, shape:(n_vert,)
            Scalars representing magnitude or intensity at each vertex

        surf_cmap: str
            Colormap used to map surf_scalars to color

        surf_opacity: float,  0.0<=x<=1.0
            Opacity of the colormap

        node_coords: numpy.ndarray, shape:(n_node, 3)
            XYZ MNI coordinates of the network nodes
            Perhaps centroid location of the ROIs

        node_sizes: numpy.ndarray, shape:(n_node,)
            Radii of spheres representing nodes

        node_colors: numpy.ndarray, shape:(n_node, 4)
            RGBA values to fill spheres representing nodes

        conn_list: numpy.ndarray, shape:(n_node*(n_node-1) / 2,)
            List of connections between nodes in the network
            Assumes list represents the upper triangle of the adjacency matrix

        conn_pct: tuple, shape:(2,)
            Percentile range of connections to plot

        conn_cmap: str
            Colormap used to map connection strengths to color
    """

    ### Setup the engine and scene
    my_engine = mlab.get_engine()
    fig = mlab.figure(size=(1000, 1000), bgcolor=(1.0, 1.0, 1.0), engine=my_engine)

    ### Plot the nodes on the brain
    n_node = node_coords.shape[0]
    for n_i in xrange(n_node):

        node_source = mlab.pipeline.scalar_scatter(node_coords[n_i, 0],
                                                   node_coords[n_i, 1],
                                                   node_coords[n_i, 2],
                                                   figure=fig)

        node_surface = mlab.pipeline.glyph(node_source,
                                           scale_mode='none',
                                           scale_factor=node_sizes[n_i],
                                           mode='sphere',
                                           color=tuple(node_colors[n_i][:3]),
                                           opacity=node_colors[n_i][3],
                                           figure=fig)

    ### Plot the connections on the brain
    n_conn = conn_list.shape[0]

    # Generate connection vectors
    e_start = np.zeros((n_conn, 3))
    e_vec = np.zeros((n_conn, 3))

    triu_ix, triu_iy = np.triu_indices(n_node, k=1)

    for ii, (ix, iy) in enumerate(zip(triu_ix, triu_iy)):
        e_start[ii, :] = node_coords[ix, :]
        e_vec[ii, :] = (node_coords[iy, :] - node_coords[ix, :])

    # Threshold the connections
    lo_thr = np.percentile(conn_list, conn_pct[0])
    hi_thr = np.percentile(conn_list, conn_pct[1])
    conn_thr_idx = np.flatnonzero((conn_list >= lo_thr) & (conn_list <= hi_thr))

    # Import vectors into pipeline
    conn_source = mlab.pipeline.vector_scatter(e_start[:, 0],
                                               e_start[:, 1],
                                               e_start[:, 2],
                                               e_vec[:, 0],
                                               e_vec[:, 1],
                                               e_vec[:, 2], figure=fig)
    conn_source.mlab_source.dataset.point_data.scalars = conn_list

    #conn_thresh = mlab.pipeline.threshold(conn_source, low=lo_thr, up=hi_thr, figure=fig)

    # Change connection attributes
    conn_surface = mlab.pipeline.vectors(conn_source, colormap=conn_cmap,
                                         scale_factor=1, line_width=4, transparent=True,
                                         scale_mode='vector', figure=fig)
    conn_surface.glyph.glyph.clamping = False
    #conn_surface.actor.property.opacity = 0.1
    conn_surface.module_manager.vector_lut_manager.reverse_lut = False

    conn_surface.glyph.glyph_source.glyph_source = (\
        conn_surface.glyph.glyph_source.glyph_dict['glyph_source2d'])
    conn_surface.glyph.glyph_source.glyph_source.glyph_type='dash'

    ### Plot the colored brain regions
    surf_source = mlab.pipeline.triangular_mesh_source(surf_vertices[:, 0],
                                                       surf_vertices[:, 1],
                                                       surf_vertices[:, 2],
                                                       surf_triangles,
                                                       scalars=surf_scalars,
                                                       opacity=1.0,
                                                       figure=fig)
    surf_norms = mlab.pipeline.poly_data_normals(surf_source, figure=fig)
    surf_norms.filter.splitting = True
    surf_surf = mlab.pipeline.surface(surf_norms, figure=fig)

    surf_surf.parent.scalar_lut_manager.set(lut_mode=surf_cmap,
                                            data_range=[0, 1],
                                            use_default_range=False)
    lut = surf_surf.module_manager.scalar_lut_manager.lut.table.to_array()
    lut[:, 3] = surf_opacity
    surf_surf.module_manager.scalar_lut_manager.lut.table = lut
    surf_surf.actor.property.backface_culling = True

    return my_engine, conn_source
