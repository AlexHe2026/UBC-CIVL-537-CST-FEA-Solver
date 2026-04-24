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

    n_elems = len(elements)
    
    # Initialize an array to hold the 3 stress components for every element
    stresses = np.zeros((n_elems, 3))
    
    for i, elem in enumerate(elements):
        # 1. Get the local coordinates for the 3 nodes of this triangle
        coords = nodes[elem]
        
        # 2. Compute the Strain-Displacement matrix (B)
        B = compute_B(coords)
        
        # 3. Extract the local displacements for this specific triangle from the global u vector
        n1, n2, n3 = elem
        global_dofs = [
            2 * n1, 2 * n1 + 1,
            2 * n2, 2 * n2 + 1,
            2 * n3, 2 * n3 + 1
        ]
        u_e = u[global_dofs]  # This is a 6-item array of [u1, v1, u2, v2, u3, v3]
        
        # 4. Calculate Strain: epsilon = B * u_e
        # The '@' symbol performs matrix multiplication in NumPy
        strain = B @ u_e 
        
        # 5. Calculate Stress: sigma = D * epsilon (Hooke's Law)
        stress = D @ strain
        
        # 6. Store the [sigma_xx, sigma_yy, tau_xy] result in our final array
        stresses[i, :] = stress
        
    return stresses


def compute_von_mises(stresses):
    """Von Mises stress from [sigma_xx, sigma_yy, tau_xy] per element."""
    # Slice the Nx3 array into three separate 1D arrays for easy math
    sigma_xx = stresses[:, 0]
    sigma_yy = stresses[:, 1]
    tau_xy = stresses[:, 2]
    
    # Apply the 2D Plane Stress Von Mises formula
    von_mises = np.sqrt(sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3 * tau_xy**2)
    
    return von_mises


def strain_energy(K, u):
    """Return 0.5 * u^T K u."""

    # K @ u computes the internal force vector.
    # np.dot(u, ...) computes the final dot product (u^T * Forces)
    energy = 0.5 * np.dot(u, K @ u)
    
    return energy
