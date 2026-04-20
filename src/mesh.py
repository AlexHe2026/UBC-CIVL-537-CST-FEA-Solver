# src/mesh.py
import numpy as np
import os


def generate_rect_mesh(L, h, nx, ny):
    """
    Generate a structured triangular mesh over a rectangular domain [0, L] x [-h/2, h/2].

    Each rectangular cell is split into 2 triangles. The split direction must be
    consistent across the mesh (e.g., always along the same diagonal).

    Parameters
    ----------
    L : float
        Length of the plate (x-direction).
    h : float
        Height of the plate (y-direction).
    nx : int
        Number of element divisions in x.
    ny : int
        Number of element divisions in y.

    Returns
    -------
    nodes : ndarray, shape (n_nodes, 2)
        Nodal coordinates.
    elements : ndarray, shape (n_elems, 3)
        Element connectivity (node indices per triangle).
    boundary_tags : dict
        Must contain at minimum:
        - 'fixed': list of node indices on the x=0 boundary (cantilever root)
        - 'loaded': list of node indices on the x=L boundary (cantilever tip)
    """
    raise NotImplementedError


def generate_plate_with_hole_mesh(W, H, R, n_radial, n_angular):
    """
    Generate a triangular mesh for a rectangular plate [0, W] x [0, H]
    with a circular hole of radius R centered at the origin.

    Due to symmetry, only the quarter-plate (first quadrant) needs to be meshed.
    The hole boundary and plate edges must be tagged for boundary conditions.

    Parameters
    ----------
    W : float
        Half-width of the plate (x-direction extent from center).
    H : float
        Half-height of the plate (y-direction extent from center).
    R : float
        Radius of the circular hole.
    n_radial : int
        Number of element divisions in the radial direction (from hole to edge).
    n_angular : int
        Number of element divisions in the angular direction (quarter circle).

    Returns
    -------
    nodes : ndarray, shape (n_nodes, 2)
        Nodal coordinates.
    elements : ndarray, shape (n_elems, 3)
        Element connectivity.
    boundary_tags : dict
        Must contain:
        - 'hole': list of node indices on the hole boundary
        - 'right': list of node indices on the x=W boundary (applied tension)
        - 'sym_x': list of node indices on the y=0 boundary (symmetry: v=0)
        - 'sym_y': list of node indices on the x=0 boundary (symmetry: u=0)

    Notes
    -----
    Option A (manual): Create a structured mesh in (r, theta) space for
    r in [R, outer] and theta in [0, pi/2], then map to (x, y) using
    x = r cos(theta), y = r sin(theta). The outer boundary must conform to the
    rectangular plate edges.

    Option B (Gmsh): Use the gmsh Python API to define the geometry
    (rectangle minus circle), set mesh sizes, generate the mesh, and
    extract nodes, elements, and physical groups for boundary tagging.
    See https://gmsh.info/doc/textures/gmsh_api.html for the Python API docs.
    If you use Gmsh, add 'gmsh' to your requirements.txt.

    Option C (fallback): Use load_fallback_hole_mesh() below to load the
    pre-generated mesh from data/plate_with_hole_mesh.npz. This lets you
    proceed with the rest of the project while you work on your own mesher.
    """
    raise NotImplementedError


def load_fallback_hole_mesh(filepath=None):
    """
    Load the pre-generated plate-with-hole mesh from the .npz file.

    Parameters
    ----------
    filepath : str, optional
        Path to the .npz file. Defaults to data/plate_with_hole_mesh.npz
        relative to the project root.

    Returns
    -------
    nodes : ndarray, shape (n_nodes, 2)
    elements : ndarray, shape (n_elems, 3)
    boundary_tags : dict with keys 'hole', 'right', 'sym_x', 'sym_y'

    Notes
    -----
    This is a fallback so you can work on the solver and validation plots
    without being blocked by mesh generation. Your final submission should
    include your own mesh generator (Option A or B above).
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'plate_with_hole_mesh.npz')
    data = np.load(filepath)
    nodes = data['nodes']
    elements = data['elements']
    boundary_tags = {
        'hole': data['hole'].tolist(),
        'right': data['right'].tolist(),
        'sym_x': data['sym_x'].tolist(),
        'sym_y': data['sym_y'].tolist(),
    }
    return nodes, elements, boundary_tags
