import numpy as np
import pytest
from src.elements import compute_B, compute_D, compute_k, compute_area


def test_area_known_triangle():
    """A right triangle with legs of length 1 has area 0.5."""
    coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    assert abs(compute_area(coords) - 0.5) < 1e-12


def test_area_negative_orientation():
    """Clockwise node ordering should give negative area."""
    coords = np.array([[0, 0], [0, 1], [1, 0]], dtype=float)
    assert compute_area(coords) < 0


def test_B_matrix_shape():
    """B must be 3x6 for a 3-node triangle with 2 DOFs per node."""
    coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    B = compute_B(coords)
    assert B.shape == (3, 6)


def test_B_rigid_body_motion():
    """Rigid body translation must produce zero strain."""
    coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    B = compute_B(coords)
    d_rigid = np.array([1, 2, 1, 2, 1, 2], dtype=float)
    eps = B @ d_rigid
    assert np.allclose(eps, 0, atol=1e-12)


def test_B_known_values():
    """
    Hand-verification for the reference triangle (0,0), (1,0), (0,1).
    With 2A = 1:
    B = [[-1,  0,  1,  0,  0,  0],
         [ 0, -1,  0,  0,  0,  1],
         [-1, -1,  0,  1,  1,  0]]
    """
    coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    B = compute_B(coords)
    B_expected = np.array([
        [-1,  0,  1,  0,  0,  0],
        [ 0, -1,  0,  0,  0,  1],
        [-1, -1,  0,  1,  1,  0]
    ], dtype=float)
    assert np.allclose(B, B_expected, atol=1e-12), f"B =\n{B}\nExpected:\n{B_expected}"


def test_k_symmetric():
    """Element stiffness matrix must be symmetric."""
    coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    D = compute_D(200e9, 0.25, "plane_stress")
    k = compute_k(coords, D, 0.01)
    assert np.allclose(k, k.T, atol=1e-6)


def test_k_positive_semidefinite():
    """Eigenvalues of k must be non-negative (3 zero for rigid body modes)."""
    coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    D = compute_D(200e9, 0.25, "plane_stress")
    k = compute_k(coords, D, 0.01)
    eigvals = np.linalg.eigvalsh(k)
    assert np.all(eigvals > -1e-6)


def test_D_plane_stress_vs_plane_strain():
    """Plane stress and plane strain D matrices must differ."""
    D_ps = compute_D(200e9, 0.3, "plane_stress")
    D_pe = compute_D(200e9, 0.3, "plane_strain")
    assert not np.allclose(D_ps, D_pe)


def test_D_plane_stress_symmetry():
    """D must be symmetric."""
    D = compute_D(200e9, 0.3, "plane_stress")
    assert np.allclose(D, D.T)
