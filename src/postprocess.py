# src/postprocess.py
import numpy as np
from src.elements import compute_B


def compute_stresses(nodes, elements, u, D):
    """
    Recover element stresses from the displacement solution.

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    elements : ndarray, shape (n_elems, 3)
    u : ndarray, shape (n_dof,)
    D : ndarray, shape (3, 3)

    Returns
    -------
    stresses : ndarray, shape (n_elems, 3)
        Stress components [sigma_xx, sigma_yy, tau_xy] at each element centroid.
    """
    raise NotImplementedError


def compute_von_mises(stresses):
    """Von Mises stress from [sigma_xx, sigma_yy, tau_xy] per element."""
    raise NotImplementedError


def strain_energy(K, u):
    """Return 0.5 * u^T K u."""
    raise NotImplementedError
