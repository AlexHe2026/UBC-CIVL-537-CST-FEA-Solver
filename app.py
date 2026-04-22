# app.py — Full Streamlit frontend
# This file does NO math. It only calls functions from src/ and renders outputs.

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

from src.elements import compute_D
from src.mesh import generate_rect_mesh, generate_plate_with_hole_mesh
from src.assembly import assemble_K, assemble_R_parabolic_shear, assemble_R_uniform_tension
from src.solver import apply_bc_and_solve
from src.postprocess import compute_stresses, compute_von_mises, strain_energy
from src.analytics import (
    timoshenko_deflection, euler_bernoulli_deflection,
    timoshenko_sigma_xx, timoshenko_tau_xy,
    kirsch_stress_polar, kirsch_stress_cartesian,
)

st.set_page_config(page_title="CST FEA Solver", layout="wide")
st.title("2D Plane Stress / Plane Strain FEA Solver")

# ──────────────────────────────────────────────
# Sidebar: global inputs
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("Material Properties")
    mode = st.selectbox("Analysis Mode", ["Plane Stress", "Plane Strain"])
    E = st.number_input("Young's Modulus E (Pa)", value=200e9, format="%.2e")
    nu = st.number_input("Poisson's Ratio ν", value=0.25, min_value=0.0, max_value=0.4999,
                         format="%.4f")
    t = st.number_input("Thickness (m)", value=0.01, format="%.4f")

    if nu >= 0.499:
        st.warning("Near-incompressible — results may be unreliable with CST elements. "
                   "See the Locking tab.")

mode_key = "plane_stress" if mode == "Plane Stress" else "plane_strain"

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔧 Cantilever Beam", "🕳️ Plate with Hole",
                             "📊 Convergence & Locking"])


# ══════════════════════════════════════════════
# Helper: plot mesh
# ══════════════════════════════════════════════
def plot_mesh(nodes, elements, title="Mesh Preview"):
    fig = go.Figure()
    for tri in elements:
        pts = nodes[list(tri) + [tri[0]]]
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1],
                                 mode='lines', line=dict(color='steelblue', width=0.5),
                                 showlegend=False))
    fig.update_layout(yaxis_scaleanchor="x", title=title,
                      height=350, margin=dict(l=0, r=0, t=30, b=0))
    return fig


def plot_deformed_mesh(nodes, elements, u, stresses, scale, title="Deformed Mesh"):
    vm = compute_von_mises(stresses)
    u_disp = u.reshape(-1, 2)
    nodes_def = nodes + scale * u_disp

    fig = go.Figure(go.Mesh3d(
        x=nodes_def[:, 0], y=nodes_def[:, 1], z=np.zeros(len(nodes_def)),
        i=elements[:, 0], j=elements[:, 1], k=elements[:, 2],
        intensity=vm, colorscale="Jet", showscale=True,
        colorbar=dict(title="σ_vm (Pa)"),
        flatshading=True,
    ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (m)", yaxis_title="y (m)",
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(eye=dict(x=0, y=0, z=2)),
        ),
        height=500, margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


# ══════════════════════════════════════════════
# TAB 1: Cantilever Beam
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Cantilever Plate — Parabolic Tip Shear")

    col_input, col_mesh = st.columns([1, 2])
    with col_input:
        L = st.number_input("Plate Length L (m)", value=1.0, key="cant_L")
        h = st.number_input("Plate Height h (m)", value=0.25, key="cant_h")
        P = st.number_input("Tip Load P (N)", value=6000.0, key="cant_P")
        nx = st.slider("Elements in x", 2, 32, 8, key="cant_nx")
        ny = st.slider("Elements in y", 2, 16, 4, key="cant_ny")
        solve_cant = st.button("Solve Cantilever", type="primary")

    # Mesh preview
    nodes_c, elems_c, tags_c = generate_rect_mesh(L, h, nx, ny)
    with col_mesh:
        st.plotly_chart(plot_mesh(nodes_c, elems_c,
                        f"Mesh: {len(nodes_c)} nodes · {len(elems_c)} elements"),
                        use_container_width=True)

    if solve_cant:
        with st.spinner("Assembling and solving..."):
            D = compute_D(E, nu, mode_key)
            K = assemble_K(nodes_c, elems_c, D, t)
            R = assemble_R_parabolic_shear(nodes_c, tags_c["loaded"], P, h)
            fixed_dofs = np.array([[2*n, 2*n+1] for n in tags_c["fixed"]]).ravel()
            u = apply_bc_and_solve(K, R, fixed_dofs)
            stresses = compute_stresses(nodes_c, elems_c, u, D)
            U = strain_energy(K, u)

        st.success(f"Solved — Strain energy U = {U:.6e} J")

        # ── Plot 1: Deflection curve along neutral axis ──
        st.markdown("#### Deflection v(x) along neutral axis (y ≈ 0)")
        tol_y = h / (2 * ny) + 1e-10
        na_mask = np.abs(nodes_c[:, 1]) < tol_y
        na_nodes = np.where(na_mask)[0]
        x_na = nodes_c[na_nodes, 0]
        v_na = u[2 * na_nodes + 1]
        sort_idx = np.argsort(x_na)

        x_anal = np.linspace(0, L, 200)
        v_timo = timoshenko_deflection(x_anal, L, h, P, E, nu)
        v_eb = euler_bernoulli_deflection(x_anal, L, P, E, h)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=x_na[sort_idx], y=v_na[sort_idx],
                                  mode='markers+lines', name='FEA'))
        fig1.add_trace(go.Scatter(x=x_anal, y=v_timo,
                                  mode='lines', name='Timoshenko', line=dict(dash='dash')))
        fig1.add_trace(go.Scatter(x=x_anal, y=v_eb,
                                  mode='lines', name='Euler–Bernoulli', line=dict(dash='dot')))
        fig1.update_layout(xaxis_title="x (m)", yaxis_title="v (m)", height=400)
        st.plotly_chart(fig1, use_container_width=True)

        # ── Plot 2: σ_xx along bottom fiber ──
        st.markdown("#### σ_xx along bottom fiber (y = −h/2)")
        centroids = np.mean(nodes_c[elems_c], axis=1)
        bottom_tol = h / (2 * ny)
        bot_mask = np.abs(centroids[:, 1] - (-h / 2)) < bottom_tol
        bot_elems = np.where(bot_mask)[0]
        x_bot = centroids[bot_elems, 0]
        sig_xx_bot = stresses[bot_elems, 0]
        sort_b = np.argsort(x_bot)

        sig_xx_anal = timoshenko_sigma_xx(x_anal, -h/2, L, P, h)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x_bot[sort_b], y=sig_xx_bot[sort_b],
                                  mode='markers+lines', name='FEA'))
        fig2.add_trace(go.Scatter(x=x_anal, y=sig_xx_anal,
                                  mode='lines', name='Timoshenko', line=dict(dash='dash')))
        fig2.update_layout(xaxis_title="x (m)", yaxis_title="σ_xx (Pa)", height=400)
        st.plotly_chart(fig2, use_container_width=True)

        # ── Plot 3: Deformed mesh with von Mises ──
        st.markdown("#### Deformed Mesh — Von Mises Stress")
        scale = st.slider("Deformation scale", 1, 500, 100, key="cant_scale")
        st.plotly_chart(plot_deformed_mesh(nodes_c, elems_c, u, stresses, scale),
                        use_container_width=True)

        # ── Plot 4: τ_xy across beam depth at mid-span ──
        st.markdown("#### τ_xy across beam depth at mid-span (x ≈ L/2)")
        mid_tol = L / (2 * nx)
        mid_mask = np.abs(centroids[:, 0] - L/2) < mid_tol
        mid_elems = np.where(mid_mask)[0]
        y_mid = centroids[mid_elems, 1]
        tau_mid = stresses[mid_elems, 2]
        sort_m = np.argsort(y_mid)

        y_anal = np.linspace(-h/2, h/2, 200)
        tau_anal = timoshenko_tau_xy(y_anal, P, h)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=tau_mid[sort_m], y=y_mid[sort_m],
                                  mode='markers+lines', name='FEA'))
        fig4.add_trace(go.Scatter(x=tau_anal, y=y_anal,
                                  mode='lines', name='Timoshenko', line=dict(dash='dash')))
        fig4.update_layout(xaxis_title="τ_xy (Pa)", yaxis_title="y (m)", height=400)
        st.plotly_chart(fig4, use_container_width=True)

        # ── CSV export ──
        st.markdown("#### Export Results")
        disp_df = pd.DataFrame({
            'node': np.arange(len(nodes_c)),
            'x': nodes_c[:, 0], 'y': nodes_c[:, 1],
            'u': u[0::2], 'v': u[1::2]
        })
        stress_df = pd.DataFrame(stresses, columns=['sigma_xx', 'sigma_yy', 'tau_xy'])
        stress_df.insert(0, 'element', np.arange(len(stresses)))

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button("Download displacements (CSV)",
                               disp_df.to_csv(index=False), "displacements.csv", "text/csv")
        with col_d2:
            st.download_button("Download stresses (CSV)",
                               stress_df.to_csv(index=False), "stresses.csv", "text/csv")


# ══════════════════════════════════════════════
# TAB 2: Plate with Hole
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Plate with Circular Hole — Kirsch Validation")

    col_input2, col_mesh2 = st.columns([1, 2])
    with col_input2:
        W = st.number_input("Half-width W (m)", value=5.0, key="hole_W")
        H = st.number_input("Half-height H (m)", value=5.0, key="hole_H")
        R = st.number_input("Hole radius R (m)", value=1.0, key="hole_R")
        sigma_inf = st.number_input("Far-field stress σ∞ (Pa)", value=1e6, format="%.2e",
                                     key="hole_sigma")
        n_rad = st.slider("Radial divisions", 4, 24, 10, key="hole_nrad")
        n_ang = st.slider("Angular divisions", 4, 24, 12, key="hole_nang")
        solve_hole = st.button("Solve Plate with Hole", type="primary")

    # Mesh preview
    nodes_h, elems_h, tags_h = generate_plate_with_hole_mesh(W, H, R, n_rad, n_ang)
    with col_mesh2:
        st.plotly_chart(plot_mesh(nodes_h, elems_h,
                        f"Mesh: {len(nodes_h)} nodes · {len(elems_h)} elements"),
                        use_container_width=True)

    if solve_hole:
        with st.spinner("Assembling and solving..."):
            D_h = compute_D(E, nu, mode_key)
            K_h = assemble_K(nodes_h, elems_h, D_h, t)
            R_h = assemble_R_uniform_tension(nodes_h, tags_h["right"], sigma_inf, t)

            # Symmetry BCs: fix v on y=0, fix u on x=0
            fixed_dofs_h = []
            for n in tags_h["sym_x"]:
                fixed_dofs_h.append(2*n + 1)   # v = 0 on y=0
            for n in tags_h["sym_y"]:
                fixed_dofs_h.append(2*n)         # u = 0 on x=0
            fixed_dofs_h = np.array(fixed_dofs_h)

            u_h = apply_bc_and_solve(K_h, R_h, fixed_dofs_h)
            stresses_h = compute_stresses(nodes_h, elems_h, u_h, D_h)

        # ── Plot 5: σ_θθ along hole boundary ──
        st.markdown("#### σ_θθ along hole boundary (r = R)")
        centroids_h = np.mean(nodes_h[elems_h], axis=1)
        r_cent = np.sqrt(centroids_h[:, 0]**2 + centroids_h[:, 1]**2)
        theta_cent = np.arctan2(centroids_h[:, 1], centroids_h[:, 0])

        hole_tol = 1.5 * R * (1.0 / n_rad)
        near_hole = r_cent < (R + hole_tol)
        theta_near = theta_cent[near_hole]
        sig_xx_near = stresses_h[near_hole, 0]
        sig_yy_near = stresses_h[near_hole, 1]
        tau_xy_near = stresses_h[near_hole, 2]

        # Transform FEA Cartesian stress to σ_θθ
        sig_tt_fea = (sig_xx_near * np.sin(theta_near)**2
                      + sig_yy_near * np.cos(theta_near)**2
                      - 2 * tau_xy_near * np.sin(theta_near) * np.cos(theta_near))

        theta_anal = np.linspace(0, np.pi/2, 200)
        _, sig_tt_anal, _ = kirsch_stress_polar(R, theta_anal, R, sigma_inf)

        fig5 = go.Figure()
        sort5 = np.argsort(theta_near)
        fig5.add_trace(go.Scatter(x=np.degrees(theta_near[sort5]),
                                  y=sig_tt_fea[sort5] / sigma_inf,
                                  mode='markers', name='FEA'))
        fig5.add_trace(go.Scatter(x=np.degrees(theta_anal),
                                  y=sig_tt_anal / sigma_inf,
                                  mode='lines', name='Kirsch', line=dict(dash='dash')))
        fig5.update_layout(xaxis_title="θ (degrees)", yaxis_title="σ_θθ / σ∞", height=400)
        st.plotly_chart(fig5, use_container_width=True)

        st.info(f"FEA max σ_θθ / σ∞ = {np.max(sig_tt_fea) / sigma_inf:.2f}  "
                f"(Kirsch exact = 3.00)")

        # ── Plot 6: σ_xx along x-axis (θ=0) ──
        st.markdown("#### σ_xx along x-axis (y ≈ 0, showing stress decay from hole)")
        axis_tol = H / (2 * n_ang)
        xaxis_mask = (np.abs(centroids_h[:, 1]) < axis_tol) & (centroids_h[:, 0] > R * 0.8)
        r_xaxis = centroids_h[xaxis_mask, 0]
        sig_xx_xaxis = stresses_h[xaxis_mask, 0]
        sort6 = np.argsort(r_xaxis)

        r_anal = np.linspace(R, W, 200)
        sig_xx_anal, _, _ = kirsch_stress_cartesian(r_anal, 0, R, sigma_inf)

        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=r_xaxis[sort6], y=sig_xx_xaxis[sort6] / sigma_inf,
                                  mode='markers+lines', name='FEA'))
        fig6.add_trace(go.Scatter(x=r_anal, y=sig_xx_anal / sigma_inf,
                                  mode='lines', name='Kirsch', line=dict(dash='dash')))
        fig6.update_layout(xaxis_title="r (m)", yaxis_title="σ_xx / σ∞", height=400)
        st.plotly_chart(fig6, use_container_width=True)

        # ── Plot 7: Deformed mesh ──
        st.markdown("#### Deformed Mesh — Von Mises Stress")
        scale_h = st.slider("Deformation scale", 1, 1000, 200, key="hole_scale")
        st.plotly_chart(plot_deformed_mesh(nodes_h, elems_h, u_h, stresses_h, scale_h,
                                           "Deformed Plate with Hole"),
                        use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3: Convergence & Locking
# ══════════════════════════════════════════════
with tab3:
    st.subheader("h-Refinement Convergence")

    conv_col1, conv_col2 = st.columns(2)

    with conv_col1:
        st.markdown("##### Cantilever: Tip Deflection Convergence")
        if st.button("Run Cantilever Convergence", key="conv_cant"):
            mesh_sizes = [2, 4, 8, 16, 32]
            D_conv = compute_D(E, nu, mode_key)
            L_c, h_c, P_c = 1.0, 0.25, 6000.0
            tip_exact = timoshenko_deflection(L_c, L_c, h_c, P_c, E, nu)
            n_elems_list, errors = [], []

            progress = st.progress(0)
            for idx, n in enumerate(mesh_sizes):
                ny_c = max(2, n // 2)
                nodes_cv, elems_cv, tags_cv = generate_rect_mesh(L_c, h_c, n, ny_c)
                K_cv = assemble_K(nodes_cv, elems_cv, D_conv, t)
                R_cv = assemble_R_parabolic_shear(nodes_cv, tags_cv["loaded"], P_c, h_c)
                fd = np.array([[2*nd, 2*nd+1] for nd in tags_cv["fixed"]]).ravel()
                u_cv = apply_bc_and_solve(K_cv, R_cv, fd)

                tip_ns = [nd for nd in tags_cv["loaded"]
                          if abs(nodes_cv[nd, 1]) < h_c / (2 * ny_c) + 1e-10]
                tip_v = np.mean(u_cv[2 * np.array(tip_ns) + 1])
                errors.append(abs(tip_v - tip_exact) / abs(tip_exact))
                n_elems_list.append(len(elems_cv))
                progress.progress((idx + 1) / len(mesh_sizes))

            log_n = np.log10(n_elems_list)
            log_e = np.log10(errors)
            p_slope = np.polyfit(log_n, log_e, 1)[0]

            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(x=n_elems_list, y=errors,
                                          mode='markers+lines', name='FEA error'))
            fig_conv.update_layout(xaxis_title="Number of elements", yaxis_title="Relative error",
                                   xaxis_type="log", yaxis_type="log", height=400,
                                   title=f"Convergence rate p = {p_slope:.2f}")
            st.plotly_chart(fig_conv, use_container_width=True)
            st.metric("Convergence rate p", f"{p_slope:.2f}", help="Expected ≈ -1 for CST")

    with conv_col2:
        st.markdown("##### Plate with Hole: Stress Concentration Convergence")
        if st.button("Run Hole Convergence", key="conv_hole"):
            rad_sizes = [4, 6, 8, 12, 16, 20]
            W_h, H_h, R_h, sig_h = 5.0, 5.0, 1.0, 1e6
            D_hconv = compute_D(E, nu, mode_key)
            kt_list, n_elems_h_list = [], []

            progress2 = st.progress(0)
            for idx, nr in enumerate(rad_sizes):
                na = nr + 2
                nodes_hc, elems_hc, tags_hc = generate_plate_with_hole_mesh(W_h, H_h, R_h, nr, na)
                K_hc = assemble_K(nodes_hc, elems_hc, D_hconv, t)
                R_hc = assemble_R_uniform_tension(nodes_hc, tags_hc["right"], sig_h, t)
                fd_h = []
                for nd in tags_hc["sym_x"]:
                    fd_h.append(2*nd + 1)
                for nd in tags_hc["sym_y"]:
                    fd_h.append(2*nd)
                u_hc = apply_bc_and_solve(K_hc, R_hc, np.array(fd_h))
                stresses_hc = compute_stresses(nodes_hc, elems_hc, u_hc, D_hconv)

                cent_hc = np.mean(nodes_hc[elems_hc], axis=1)
                r_hc = np.sqrt(cent_hc[:, 0]**2 + cent_hc[:, 1]**2)
                th_hc = np.arctan2(cent_hc[:, 1], cent_hc[:, 0])
                near = r_hc < R_h * 1.3
                sig_tt_hc = (stresses_hc[near, 0] * np.sin(th_hc[near])**2
                             + stresses_hc[near, 1] * np.cos(th_hc[near])**2
                             - 2 * stresses_hc[near, 2] * np.sin(th_hc[near]) * np.cos(th_hc[near]))
                kt_list.append(np.max(sig_tt_hc) / sig_h)
                n_elems_h_list.append(len(elems_hc))
                progress2.progress((idx + 1) / len(rad_sizes))

            fig_kt = go.Figure()
            fig_kt.add_trace(go.Scatter(x=n_elems_h_list, y=kt_list,
                                        mode='markers+lines', name='FEA K_t'))
            fig_kt.add_hline(y=3.0, line_dash="dash", annotation_text="Kirsch exact (K_t=3)")
            fig_kt.update_layout(xaxis_title="Number of elements", yaxis_title="K_t",
                                 xaxis_type="log", height=400,
                                 title="Stress Concentration Factor Convergence")
            st.plotly_chart(fig_kt, use_container_width=True)

    # ── Locking Study ──
    st.markdown("---")
    st.subheader("Volumetric Locking Investigation")
    st.markdown("Run the cantilever in **plane strain** for increasing ν → 0.5")

    if st.button("Run Locking Study", type="primary", key="locking"):
        nu_values = [0.3, 0.4, 0.45, 0.49, 0.499, 0.4999]
        L_l, h_l, P_l = 1.0, 0.25, 6000.0
        nx_l, ny_l = 8, 4
        norm_deflections, cond_numbers = [], []

        progress3 = st.progress(0)
        for idx, nu_val in enumerate(nu_values):
            D_l = compute_D(E, nu_val, "plane_strain")
            nodes_l, elems_l, tags_l = generate_rect_mesh(L_l, h_l, nx_l, ny_l)
            K_l = assemble_K(nodes_l, elems_l, D_l, t)
            R_l = assemble_R_parabolic_shear(nodes_l, tags_l["loaded"], P_l, h_l)
            fd_l = np.array([[2*n, 2*n+1] for n in tags_l["fixed"]]).ravel()

            free_l = np.setdiff1d(np.arange(K_l.shape[0]), fd_l)
            K_free = K_l[free_l, :][:, free_l].toarray()
            try:
                cond = np.linalg.cond(K_free)
            except Exception:
                cond = np.inf
            cond_numbers.append(cond)

            u_l = apply_bc_and_solve(K_l, R_l, fd_l)
            tip_exact_l = timoshenko_deflection(L_l, L_l, h_l, P_l, E, nu_val)
            tip_ns_l = [n for n in tags_l["loaded"]
                        if abs(nodes_l[n, 1]) < h_l / (2 * ny_l) + 1e-10]
            tip_v_l = np.mean(u_l[2 * np.array(tip_ns_l) + 1])
            norm_deflections.append(tip_v_l / tip_exact_l if tip_exact_l != 0 else 0)
            progress3.progress((idx + 1) / len(nu_values))

        lock_col1, lock_col2 = st.columns(2)
        with lock_col1:
            fig_lock = go.Figure()
            fig_lock.add_trace(go.Scatter(x=nu_values, y=norm_deflections,
                                          mode='markers+lines'))
            fig_lock.add_hline(y=1.0, line_dash="dash", annotation_text="Exact")
            fig_lock.update_layout(xaxis_title="Poisson's ratio ν",
                                   yaxis_title="v_FEA / v_exact",
                                   title="Normalized Tip Deflection vs. ν (Plane Strain)",
                                   height=400)
            st.plotly_chart(fig_lock, use_container_width=True)

        with lock_col2:
            fig_cond = go.Figure()
            fig_cond.add_trace(go.Scatter(x=nu_values, y=cond_numbers,
                                          mode='markers+lines'))
            fig_cond.update_layout(xaxis_title="Poisson's ratio ν",
                                   yaxis_title="Condition number",
                                   yaxis_type="log",
                                   title="Stiffness Matrix Condition Number vs. ν",
                                   height=400)
            st.plotly_chart(fig_cond, use_container_width=True)

        st.warning("Observe: as ν → 0.5, the FEA tip deflection drops well below the "
                   "exact value (the structure appears artificially stiff). This is "
                   "volumetric locking. You must explain why in your walkthrough.")
