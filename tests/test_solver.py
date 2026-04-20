import numpy as np
import pytest
from src.mesh import generate_rect_mesh
from src.elements import compute_D
from src.assembly import assemble_K, assemble_R_parabolic_shear
from src.solver import apply_bc_and_solve
from src.postprocess import strain_energy, compute_stresses


def test_fixed_dofs_zero():
    """Displacements at fixed DOFs must be exactly zero."""
    nodes, elems, tags = generate_rect_mesh(1.0, 0.25, 4, 2)
    D = compute_D(200e9, 0.25, "plane_stress")
    K = assemble_K(nodes, elems, D, 0.01)
    R = assemble_R_parabolic_shear(nodes, tags["loaded"], 6000.0, 0.25)
    fixed_dofs = np.array([[2*n, 2*n+1] for n in tags["fixed"]]).ravel()
    u = apply_bc_and_solve(K, R, fixed_dofs)
    assert np.allclose(u[fixed_dofs], 0.0)


def test_strain_energy_positive():
    """Strain energy must be strictly positive for a loaded structure."""
    nodes, elems, tags = generate_rect_mesh(1.0, 0.25, 4, 2)
    D = compute_D(200e9, 0.25, "plane_stress")
    K = assemble_K(nodes, elems, D, 0.01)
    R = assemble_R_parabolic_shear(nodes, tags["loaded"], 6000.0, 0.25)
    fixed_dofs = np.array([[2*n, 2*n+1] for n in tags["fixed"]]).ravel()
    u = apply_bc_and_solve(K, R, fixed_dofs)
    U = strain_energy(K, u)
    assert U > 0


def test_tip_deflection_order_of_magnitude():
    """
    Tip deflection should be within an order of magnitude of Euler-Bernoulli.
    This is a sanity check, not a precision test.
    """
    L, h, P, E, nu = 1.0, 0.25, 6000.0, 200e9, 0.25
    nodes, elems, tags = generate_rect_mesh(L, h, 8, 4)
    D = compute_D(E, nu, "plane_stress")
    K = assemble_K(nodes, elems, D, 0.01)
    R = assemble_R_parabolic_shear(nodes, tags["loaded"], P, h)
    fixed_dofs = np.array([[2*n, 2*n+1] for n in tags["fixed"]]).ravel()
    u = apply_bc_and_solve(K, R, fixed_dofs)

    I = h**3 / 12
    delta_EB = P * L**3 / (3 * E * I)
    tip_nodes = [n for n in tags["loaded"] if abs(nodes[n, 1]) < h / 8]
    tip_v = np.mean(u[2 * np.array(tip_nodes) + 1])
    assert abs(tip_v) > 0.1 * abs(delta_EB), "Tip deflection unreasonably small"
    assert abs(tip_v) < 10 * abs(delta_EB), "Tip deflection unreasonably large"


def test_patch_test():
    """
    Patch test: apply a known uniform strain state and verify that every
    element recovers exactly that strain.
    """
    L, h = 1.0, 0.5
    nodes, elems, tags = generate_rect_mesh(L, h, 2, 2)
    E, nu = 200e9, 0.25
    D = compute_D(E, nu, "plane_stress")
    thickness = 0.01

    eps_xx = 1.0e-4
    n_dof = 2 * len(nodes)

    # Exact displacements: u = eps_xx * x, v = 0
    u_exact = np.zeros(n_dof)
    for i in range(len(nodes)):
        u_exact[2*i] = eps_xx * nodes[i, 0]
        u_exact[2*i + 1] = 0.0

    # Identify boundary nodes
    tol = 1e-10
    boundary_nodes = set()
    for i in range(len(nodes)):
        x, y = nodes[i]
        if abs(x) < tol or abs(x - L) < tol or abs(y + h/2) < tol or abs(y - h/2) < tol:
            boundary_nodes.add(i)
    boundary_dofs = []
    for n in boundary_nodes:
        boundary_dofs.extend([2*n, 2*n+1])
    boundary_dofs = np.array(boundary_dofs)

    K = assemble_K(nodes, elems, D, thickness)
    R = np.zeros(n_dof)

    free_dofs = np.setdiff1d(np.arange(n_dof), boundary_dofs)

    if len(free_dofs) > 0:
        R_mod = R[free_dofs] - K[free_dofs, :][:, boundary_dofs].dot(u_exact[boundary_dofs])
        K_ff = K[free_dofs, :][:, free_dofs]
        u_free = np.linalg.solve(K_ff.toarray(), R_mod)
        u_sol = u_exact.copy()
        u_sol[free_dofs] = u_free
    else:
        u_sol = u_exact.copy()

    stresses = compute_stresses(nodes, elems, u_sol, D)
    expected_stress = D @ np.array([eps_xx, 0.0, 0.0])

    for i, sig in enumerate(stresses):
        assert np.allclose(sig, expected_stress, rtol=1e-6), \
            f"Element {i}: stress={sig}, expected={expected_stress}"
