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
from mayavi.api import Engine
from mayavi.modules.surface import Surface

import nibabel as nib
from nibabel.affines import apply_affine


def draw(vtk_files, coords_file, conn_list=None, brain_rgba=(0.5, 0.5, 0.5, 0.2), node_rgba=(0.30, 0.69, 1.0, 1.0), node_rad=2.5, conn_thr=[0.75, 1.0], conn_cmap='YlOrRd'):

    ### Setup the engine and scene
    my_engine = Engine()
    my_engine.start()

    # Create a new scene
    my_scene = my_engine.new_scene()

    # Set background
    my_scene.scene.background = (1.0, 1.0, 1.0)

    # Initialize the rendering
    my_scene.scene.disable_render = True

    # Import VTK file to pipeline
    for vtk_file in vtk_files:
        vtk_source = my_engine.open(vtk_file)

        # Render surface
        vtk_surface = Surface()
        my_engine.add_module(vtk_surface, obj=vtk_source)
        vtk_surface.actor.property.specular_color = brain_rgba[:3]
        vtk_surface.actor.property.diffuse_color = brain_rgba[:3]
        vtk_surface.actor.property.ambient_color = brain_rgba[:3]
        vtk_surface.actor.property.color = brain_rgba[:3]
        vtk_surface.actor.property.opacity = brain_rgba[3]

    # Reset camera
    my_scene.scene.disable_render = False
    my_scene.scene.reset_zoom()

    ### Sensor-Locations
    # Load Coordinates in MRI-space
    node_coords = np.loadtxt(coords_file, delimiter=',')
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
        return
    assert len(conn_list) == n_conn

    # Generate all vectors
    e_start = np.zeros((n_conn, 3))
    e_vec = np.zeros((n_conn, 3))

    triu_ix, triu_iy = np.triu_indices(n_node, k=1)

    for ii, (ix, iy) in enumerate(zip(triu_ix, triu_iy)):
        e_start[ii, :] = node_coords[ix, :]
        e_vec[ii, :] = 1e-3*(node_coords[iy, :] - node_coords[ix, :])

    # Import vectors (connections) into pipeline
    edg_source = mlab.pipeline.vector_scatter(e_start[:, 0],
                                              e_start[:, 1],
                                              e_start[:, 2],
                                              e_vec[:, 0],
                                              e_vec[:, 1],
                                              e_vec[:, 2])
    edg_source.mlab_source.dataset.point_data.scalars = conn_list

    edg_thresh = mlab.pipeline.threshold(edg_source,
                                         low=np.percentile(conn_list, 100*conn_thr[0]),
                                         up=np.percentile(conn_list, 100*conn_thr[1]))
    edg_thresh.auto_reset_lower = False
    edg_thresh.auto_reset_upper = False

    edg_surface = mlab.pipeline.vectors(edg_thresh, colormap=conn_cmap,
                                        line_width=3.0, scale_factor=1000,
                                        scale_mode='vector')

    edg_surface.glyph.glyph.clamping = False
    edg_surface.actor.property.opacity = 0.75
    edg_surface.module_manager.vector_lut_manager.reverse_lut = True

    edg_surface.glyph.glyph_source.glyph_source = (
        edg_surface.glyph.glyph_source.glyph_dict['glyph_source2d'])
    edg_surface.glyph.glyph_source.glyph_source.glyph_type='dash'
