# src/assembly.py
import numpy as np
from scipy.sparse import lil_matrix
from src.elements import compute_B, compute_D, compute_k


def assemble_K(nodes, elements, D, thickness):
    """
    Assemble the global stiffness matrix from element contributions.

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    elements : ndarray, shape (n_elems, 3)
    D : ndarray, shape (3, 3)
    thickness : float

    Returns
    -------
    K : scipy.sparse.csr_matrix, shape (n_dof, n_dof)
        where n_dof = 2 * n_nodes.

    Notes
    -----
    Use scipy.sparse.lil_matrix for assembly (efficient for incremental insertion),
    then convert to CSR before returning (efficient for solving).
    The DOF ordering convention is: [u0, v0, u1, v1, ...] -- interleaved.
    For each element, extract the 6 global DOF indices from the 3 node indices,
    then scatter the 6x6 element stiffness into the global matrix.
    """
    #Get the number of nodes
    n_nodes = len(nodes)
    #Each node has two DOF
    n_dof = 2 * n_nodes
    
    # Initialize the global stiffness matrix as LIL for efficient assembly
    # eg. 4 nodes have 8x8 matrix
    K = lil_matrix((n_dof, n_dof))
    
    # Loop over every element in the mesh
    for elem in elements:
        # Get the coordinates of the 3 nodes for this element
        # k -- l    Left to right node 2 and 3
        # | \\ |
        # i -- j    Left to right node 0 and 1
        # elements += [[i, j, k], [l, k, j]]
        coords = nodes[elem]
        
        # Compute the 6x6 element stiffness matrix using previous function
        k_e = compute_k(coords, D, thickness)
        
        # Told by AI to use Direct Stiffness Method instead of using LT k L.
        # Map local to global Degrees of Freedom (DOFs)
        # For an interleaved convention [u0, v0, u1, v1...], node `n` has DOFs `2n` and `2n+1`
        # like example above
        # i (DOF0, DOF1),   J (DOF 2, DOF3) k (DOF 4, DOF5)
        n1, n2, n3 = elem
        global_dofs = [
            2 * n1, 2 * n1 + 1,  # DOFs for the first node
            2 * n2, 2 * n2 + 1,  # DOFs for the second node
            2 * n3, 2 * n3 + 1   # DOFs for the third node
        ]
        
        # Scatter the element stiffness into the global matrix
        # np.ix_ creates an open meshgrid and then add the 6x6 matrix into the correct grid spots
        K[np.ix_(global_dofs, global_dofs)] += k_e
        
    # Convert to CSR format as requested for efficient equation solving later
    return K.tocsr()


def assemble_R_parabolic_shear(nodes, loaded_nodes, P, h):
    """
    Assemble the global load vector for a parabolic shear traction at the cantilever tip.

    The traction distribution along the tip edge (x = L) is:
        t_y(y) = (3P / 2h) * (1 - 4y^2/h^2)

    This must be integrated consistently using shape functions along each edge
    segment between adjacent loaded nodes (not applied as point loads).

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    loaded_nodes : list of int
        Node indices along the loaded edge (x = L), to be sorted by y-coordinate.
    P : float
        Total applied tip shear force.
    h : float
        Plate height.

    Returns
    -------
    R : ndarray, shape (n_dof,)

    Notes
    -----
    For each edge segment between two adjacent loaded nodes, use at least 2-point
    Gauss quadrature to integrate t_y(y) * N_a(y) dy and t_y(y) * N_b(y) dy,
    where N_a and N_b are the linear (1D) shape functions along the edge.
    Verify: R.sum() should equal P (global force equilibrium).
    """

    n_dof = 2 * len(nodes)
    R = np.zeros(n_dof)
    
    # 1. Sort the loaded nodes by their Y-coordinate
    # This ensures we are drawing the line segments correctly from bottom to top
    loaded_nodes = sorted(loaded_nodes, key=lambda n: nodes[n, 1])
    
    # 2. Define the traction distribution formula
    def t_y(y):
        return (3 * P / (2 * h)) * (1 - 4 * (y**2) / (h**2))
    
    # 3. Define 2-Point Gauss Quadrature rules for a 1D line [-1, 1]
    # 2 points are perfectly exact for integrating a cubic polynomial
    gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    gauss_weights = [1.0, 1.0]
    
    # 4. Loop over each segment connecting two adjacent loaded nodes
    for i in range(len(loaded_nodes) - 1):
        nA = loaded_nodes[i]
        nB = loaded_nodes[i + 1]
        
        yA = nodes[nA, 1]
        yB = nodes[nB, 1]
        
        # Calculate the length of this specific edge segment
        Le = yB - yA
        
        # The Jacobian (J) maps the real length to the natural [-1, 1] Gauss coordinate system
        J = Le / 2.0
        
        # The midpoint of the segment
        y_mid = (yA + yB) / 2.0
        
        # Initialize forces for this segment
        F_A = 0.0
        F_B = 0.0
        
        # 5. Numerical Integration (Gauss Quadrature)
        for xi, w in zip(gauss_points, gauss_weights):
            # Map the natural coordinate (xi) to the real Y coordinate
            y_eval = y_mid + J * xi
            
            # Evaluate the shear traction at this specific point
            ty_val = t_y(y_eval)
            
            # Evaluate the 1D linear shape functions at this point
            N_A = (1 - xi) / 2.0
            N_B = (1 + xi) / 2.0
            
            # Add the contribution to the forces (Force = traction * shape_function * Jacobian * weight)
            F_A += ty_val * N_A * J * w
            F_B += ty_val * N_B * J * w
            
        # 6. Scatter into the global load vector R
        # The force is purely vertical (shear), so we only add it to the Y-Degree of Freedom (2*n + 1)
        R[2 * nA + 1] += F_A
        R[2 * nB + 1] += F_B
        
    return R

    raise NotImplementedError


def assemble_R_uniform_tension(nodes, loaded_nodes, sigma_inf, thickness):
    """
    Assemble the global load vector for uniform tension applied
    to a set of boundary nodes (used for the plate-with-hole problem).

    The traction is: t_x = sigma_inf applied along the loaded edge.

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    loaded_nodes : list of int
    sigma_inf : float
    thickness : float

    Returns
    -------
    R : ndarray, shape (n_dof,)
    """
    n_dof = 2 * len(nodes)
    R = np.zeros(n_dof)
    
    # 1. Sort the loaded nodes by their Y-coordinate
    # This ensures we process the boundary edge as contiguous physical segments
    loaded_nodes = sorted(loaded_nodes, key=lambda n: nodes[n, 1])
    
    # 2. Loop over each segment connecting two adjacent loaded nodes
    for i in range(len(loaded_nodes) - 1):
        nA = loaded_nodes[i]
        nB = loaded_nodes[i + 1]
        
        yA = nodes[nA, 1]
        yB = nodes[nB, 1]
        
        # Calculate the length of this specific edge segment
        Le = abs(yB - yA)
        
        # Calculate the total force acting on this single segment
        # Force = Stress * Area = sigma_inf * (Length * thickness)
        F_total = sigma_inf * Le * thickness
        
        # For a uniform load on a linear element, the consistent nodal forces 
        # dictate that the total force is distributed equally to the two nodes
        F_node = F_total / 2.0
        
        # 3. Scatter into the global load vector R
        # Since this is tension in the X-direction (t_x), we add it to the 
        # horizontal Degrees of Freedom (2 * n)
        R[2 * nA] += F_node
        R[2 * nB] += F_node
