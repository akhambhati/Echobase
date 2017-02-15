"""
Functions to render connectivity profiles on MNI brain surface

Created by: Ankit Khambhati

Change Log
----------
2017/02/11 -- Created the draw function
"""

from __future__ import division

import numpy as np

from mayavi import mlab


def draw(vtk_files, node_coords, conn_list=None, brain_rgba=(0.5, 0.5, 0.5, 0.2), node_rgba=(0.30, 0.69, 1.0, 1.0), node_rad=2.5, conn_thr=[0.75, 1.0], conn_cmap='YlOrRd'):
    """
    Draws a brain graph superimposed on the brain surface

    Parameters
    ----------
        vtk_files: list
            List of paths to VTK surface reconstructions for a brain

        node_coords: numpy.ndarray, shape: (n_node, 3)
            MNI coordinates for nodes associated with the brain graph

        conn_list: numpy.ndarray, shape: (n_node*(n_node-1)*0.5,)
            Vector of connection or edge strengths associated with each node
            Assumes undirected, symmetric connectivity

        brain_rgba: tuple, shape: (4,)
            RGBA color scheme for the brain surface reconstruction

        node_rgba: tuple, shape: (4,)
            RGBA color scheme for spheres representing each node

        node_rad: float
            Radius specifying sphere size associated with each node

        conn_thr: list, shape: (2,)
            Lower and upper percentiles for thresholding connections to plot

        conn_cmap: str
            Colormap to map connection strengths to colors
    """

    ### Setup the engine and scene
    my_engine = mlab.get_engine()

    # Import VTK file to pipeline
    for vtk_file in vtk_files:
        vtk_source = my_engine.open(vtk_file)

        # Render surface
        vtk_surface = mlab.pipeline.surface(vtk_source)
        vtk_surface.actor.property.specular_color = brain_rgba[:3]
        vtk_surface.actor.property.diffuse_color = brain_rgba[:3]
        vtk_surface.actor.property.ambient_color = brain_rgba[:3]
        vtk_surface.actor.property.color = brain_rgba[:3]
        vtk_surface.actor.property.opacity = brain_rgba[3]

    ### Sensor-Locations
    # Load Coordinates in MRI-space
    n_node = node_coords.shape[0]
    n_conn = np.int(n_node*(n_node-1)*0.5)

    # Import coordinates into pipeline
    crd_source = mlab.pipeline.scalar_scatter(node_coords[:, 0],
                                              node_coords[:, 1],
                                              node_coords[:, 2])
    # Render Glyphs for node points
    crd_surface = mlab.pipeline.glyph(crd_source,
                                      scale_mode='none',
                                      scale_factor=node_rad,
                                      mode='sphere',
                                      colormap='cool',
                                      color=node_rgba[:3],
                                      opacity=node_rgba[3])

    ### Connection-Locations
    if conn_list is None:
        print('No connections specified')
    else:
        assert len(conn_list) == n_conn

        # Generate all vectors
        e_start = np.zeros((n_conn, 3))
        e_vec = np.zeros((n_conn, 3))

        triu_ix, triu_iy = np.triu_indices(n_node, k=1)

        for ii, (ix, iy) in enumerate(zip(triu_ix, triu_iy)):
            e_start[ii, :] = node_coords[ix, :]
            e_vec[ii, :] = (node_coords[iy, :] - node_coords[ix, :])

        # Threshold the connections
        thresh_lower_ix = np.flatnonzero(conn_list > np.percentile(conn_list, 100*conn_thr[0]))
        thresh_upper_ix = np.flatnonzero(conn_list < np.percentile(conn_list, 100*conn_thr[1]))
        thr_ix = np.intersect1d(thresh_lower_ix, thresh_upper_ix)

        # Import vectors (connections) into pipeline
        edg_source = mlab.pipeline.vector_scatter(e_start[thr_ix, 0],
                                                e_start[thr_ix, 1],
                                                e_start[thr_ix, 2],
                                                e_vec[thr_ix, 0],
                                                e_vec[thr_ix, 1],
                                                e_vec[thr_ix, 2])
        edg_source.mlab_source.dataset.point_data.scalars = conn_list[thr_ix]

        edg_surface = mlab.pipeline.vectors(edg_source, colormap=conn_cmap, scale_factor=1,
                                            line_width=2.0, scale_mode='vector')

        edg_surface.glyph.glyph.clamping = False
        edg_surface.actor.property.opacity = 0.75
        edg_surface.module_manager.vector_lut_manager.reverse_lut = False

        edg_surface.glyph.glyph_source.glyph_source = (
        edg_surface.glyph.glyph_source.glyph_dict['glyph_source2d'])
        edg_surface.glyph.glyph_source.glyph_source.glyph_type='dash'

    # Get the associated scene
    my_scene = my_engine.scenes[0]
    my_scene.scene.background = (1.0, 1.0, 1.0)
    my_scene.scene.reset_zoom()
