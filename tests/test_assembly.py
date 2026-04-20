import numpy as np
import pytest
from src.mesh import generate_rect_mesh
from src.elements import compute_D
from src.assembly import assemble_K, assemble_R_parabolic_shear


def test_K_shape():
    """K must be (2*n_nodes) x (2*n_nodes)."""
    nodes, elems, tags = generate_rect_mesh(1.0, 0.25, 4, 2)
    D = compute_D(200e9, 0.25, "plane_stress")
    K = assemble_K(nodes, elems, D, 0.01)
    n_dof = 2 * len(nodes)
    assert K.shape == (n_dof, n_dof)


def test_K_symmetric():
    """Global K must be symmetric."""
    nodes, elems, tags = generate_rect_mesh(1.0, 0.25, 4, 2)
    D = compute_D(200e9, 0.25, "plane_stress")
    K = assemble_K(nodes, elems, D, 0.01)
    assert np.allclose(K.toarray(), K.toarray().T, atol=1e-6)


def test_R_equilibrium():
    """Total vertical reaction must equal applied load P."""
    P = 6000.0
    h = 0.25
    nodes, elems, tags = generate_rect_mesh(1.0, h, 8, 4)
    R = assemble_R_parabolic_shear(nodes, tags["loaded"], P, h)
    assert abs(R.sum() - P) < 1e-6, f"R.sum()={R.sum()}, expected {P}"
