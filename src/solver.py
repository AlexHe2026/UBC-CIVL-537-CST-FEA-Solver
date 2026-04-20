# src/solver.py
import numpy as np
from scipy.sparse.linalg import spsolve


def apply_bc_and_solve(K, R, fixed_dofs):
    """
    Apply Dirichlet boundary conditions by elimination and solve Ku = R.

    Parameters
    ----------
    K : scipy.sparse.csr_matrix
    R : ndarray
    fixed_dofs : list or ndarray of int
        DOF indices to be fixed (set to zero displacement).

    Returns
    -------
    u : ndarray, shape (n_dof,)
        Full displacement vector (zeros at fixed DOFs).

    Notes
    -----
    Strategy: identify free DOFs as the complement of fixed_dofs,
    extract the submatrix K_ff = K[free, free] and subvector R_f = R[free],
    solve K_ff u_f = R_f using scipy.sparse.linalg.spsolve,
    then scatter u_f back into the full displacement vector.

    For the cantilever: fixed_dofs = all DOFs (both u and v) at x=0.
    For the plate with hole (quarter model):
        - Symmetry on y=0: fix v-DOFs of nodes on y=0
        - Symmetry on x=0: fix u-DOFs of nodes on x=0
    """
    raise NotImplementedError
