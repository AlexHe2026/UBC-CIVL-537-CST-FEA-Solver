# app.py
import streamlit as st
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