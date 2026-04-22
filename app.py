### app.py (use this until your modules work)
import streamlit as st
import plotly.graph_objects as go

from src.mesh import generate_rect_mesh
from src.elements import compute_D

st.set_page_config(page_title="CST FEA Solver", layout="wide")
st.title("2D Plane Stress / Plane Strain FEA Solver")

with st.sidebar:
    st.header("Problem Setup")
    mode = st.selectbox("Analysis Mode", ["Plane Stress", "Plane Strain"])
    E = st.number_input("Young's Modulus E (Pa)", value=200e9, format="%.2e")
    nu = st.number_input("Poisson's Ratio ν", value=0.25, min_value=0.0, max_value=0.499)
    t = st.number_input("Thickness (m)", value=0.01)
    L = st.number_input("Plate Length L (m)", value=1.0)
    h = st.number_input("Plate Height h (m)", value=0.25)
    P = st.number_input("Tip Load P (N)", value=6000.0)
    nx = st.slider("Elements in x", 2, 32, 8)
    ny = st.slider("Elements in y", 2, 16, 4)
    solve = st.button("Solve", type="primary")
    
st.info("Configure inputs in the sidebar and click Solve.")

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
