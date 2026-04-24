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

    #count number of nodes
    nx_nodes = nx + 1
    ny_nodes = ny + 1
    n_nodes = nx_nodes * ny_nodes

    #Measure elememnt length
    dx = L / nx
    dy = h / ny

    # Generate nodes starting from bottom left corner and shift the nodes down by h/2
    nodes = []
    for j_y in range(ny_nodes):
        for i_x in range(nx_nodes):
            x_coord = i_x * dx
            y_coord = (j_y * dy) - (h / 2)
            nodes.append((x_coord, y_coord))
    nodes = np.array(nodes)

    elements = []
    for row in range(ny_nodes - 1):
        for col in range(nx_nodes - 1):
            i = nx_nodes * row + col     # Bottom-Left
            j = i + 1                    # Bottom-Right = Bottom left +1
            k = i + nx_nodes             # Top-Left add one entire row node number
            l = k + 1                    # Top-Right = Top left + 1

            # Use += to cleanly unpack and append both CCW triangles at once
            # T1: Bottom-Left -> Bottom-Right -> Top-Left
            # T2: Top-Right -> Top-Left -> Bottom-Right
            elements += [[i, j, k], [l, k, j]]
            
    elements = np.array(elements)

    # Use modulo math to reliably find the left and right edges
    # example if nx_node = 5 then 0 % 5 = 0, 5 % 5 = 0 that count for left most edge
    # and the right most edge would be 4 % 5 = 4, 9 % 5 = 4 equal to node - 1
    boundary_tags = {
        "fixed" : [n for n in range(n_nodes) if n % nx_nodes == 0],
        "loaded" : [n for n in range(n_nodes) if n % nx_nodes == nx_nodes - 1]
    }

    return nodes, elements, boundary_tags


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
    # Total number of nodes in each direction is the number of divisions + 1
    N_r = n_radial + 1
    N_theta = n_angular + 1
    
    nodes = []
    
    # 1. Generate angles from 0 to pi/2 (90 degrees)
    theta_vals = np.linspace(0, np.pi / 2, N_theta)
    
    # Calculate the critical angle where the ray hits the top-right corner
    theta_corner = np.arctan2(H, W)
    
    # 2. Generate Nodes mapping from (r, theta) to Cartesian (x, y)
    for th in theta_vals:
        if th <= theta_corner:
            r_max = W / np.cos(th)  # Ray hits the right edge (x = W)
        else:
            r_max = H / np.sin(th)  # Ray hits the top edge (y = H)
            
        r_vals = np.linspace(R, r_max, N_r)
        
        for r in r_vals:
            x = r * np.cos(th)
            y = r * np.sin(th)
            nodes.append([x, y])
            
    nodes = np.array(nodes)
    
    # 3. Generate Elements (Triangles)
    elements = []
    for j in range(n_angular):
        for i in range(n_radial):
            # 1D node IDs for the 4 corners of the current quadrilateral
            n1 = i + j * N_r
            n2 = (i + 1) + j * N_r
            n3 = (i + 1) + (j + 1) * N_r
            n4 = i + (j + 1) * N_r
            
            # Split the quad into two triangles
            elements.append([n1, n2, n3])
            elements.append([n1, n3, n4])
            
    elements = np.array(elements)
    
    # 4. Generate Boundary Tags
    boundary_tags = {
        'hole': [],
        'right': [],
        'sym_x': [],
        'sym_y': []
    }
    
    # Loop over the logical grid to tag the nodes easily
    for j in range(N_theta):
        for i in range(N_r):
            node_id = i + j * N_r
            x, y = nodes[node_id]
            
            # i=0 is the inner radius (hole)
            if i == 0:
                boundary_tags['hole'].append(node_id)
            
            # j=0 is the bottom edge (theta = 0)
            if j == 0:
                boundary_tags['sym_x'].append(node_id)
                
            # j=n_angular is the left edge (theta = pi/2)
            if j == n_angular:
                boundary_tags['sym_y'].append(node_id)
                
            # i=n_radial is the outer edge, but we specifically only want 
            # the nodes that hit the right-hand wall (x = W). 
            # We use np.isclose to avoid floating point rounding errors.
            if i == n_radial and np.isclose(x, W):
                boundary_tags['right'].append(node_id)
                
    return nodes, elements, boundary_tags