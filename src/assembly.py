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
    raise NotImplementedError


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
    raise NotImplementedError
