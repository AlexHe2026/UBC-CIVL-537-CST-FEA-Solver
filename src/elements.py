# src/elements.py
import numpy as np


def compute_area(coords):
    """
    Compute the signed area of a triangle.

    Parameters
    ----------
    coords : ndarray, shape (3, 2)
        Node coordinates [[x0,y0], [x1,y1], [x2,y2]].

    Returns
    -------
    float
        Signed area of the triangle (positive if nodes are counter-clockwise).
    """
    raise NotImplementedError


def compute_B(coords):
    """
    Compute the strain-displacement matrix B for a CST element.

    The CST element has constant strain throughout, so B is constant.
    B maps the 6x1 element displacement vector [u0,v0,u1,v1,u2,v2]
    to the 3x1 strain vector [eps_xx, eps_yy, gamma_xy].

    Parameters
    ----------
    coords : ndarray, shape (3, 2)
        Node coordinates.

    Returns
    -------
    ndarray, shape (3, 6)
        Strain-displacement matrix.

    Notes
    -----
    Derive B from the linear shape functions N_i(x,y) = (a_i + b_i*x + c_i*y) / (2A).
    The coefficients b_i and c_i come from cyclic permutations of the node coordinates.
    Refer to CIVL 537 Lecture Notes, Section 4.
    """
    raise NotImplementedError


def compute_D(E, nu, mode="plane_stress"):
    """
    Compute the 3x3 constitutive (material stiffness) matrix D.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    mode : str
        Either "plane_stress" or "plane_strain".

    Returns
    -------
    ndarray, shape (3, 3)
        Constitutive matrix relating stress to strain.

    Notes
    -----
    Plane stress: sigma_zz = 0, eliminate eps_zz from 3D Hooke's law.
    Plane strain: eps_zz = 0, eliminate sigma_zz from 3D Hooke's law.
    The resulting D matrices are different. Implement them based on the course notes.
    """
    raise NotImplementedError


def compute_k(coords, D, thickness):
    """
    Compute the 6x6 element stiffness matrix for a CST element.

    Parameters
    ----------
    coords : ndarray, shape (3, 2)
        Node coordinates.
    D : ndarray, shape (3, 3)
        Constitutive matrix.
    thickness : float
        Element thickness (relevant for plane stress; set to 1 for plane strain).

    Returns
    -------
    ndarray, shape (6, 6)
        Element stiffness matrix: k = t * A * B^T D B.

    Notes
    -----
    Since B is constant over the element, the integral simplifies to
    a single multiplication (no numerical quadrature needed).
    """
    raise NotImplementedError
