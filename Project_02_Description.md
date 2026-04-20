---
title: "Project 02 — Web-Based Plane Stress/Strain FEA Solver with CST Elements"
fontsize: 11pt
linestretch: 1.1
geometry: margin=0.8in
mainfont: "TeX Gyre Pagella"
sansfont: "TeX Gyre Heros"
monofont: "DejaVu Sans Mono"
colorlinks: true
linkcolor: black
urlcolor: blue
citecolor: black
---

**CIVL 537 — Computational Mechanics I | The University of British Columbia | Due: April 26, 2026**

---

## Overview

You will build, containerize, and deploy a web application that solves 2D plane stress and plane strain problems using CST (Constant Strain Triangle) elements. The app is packaged in Docker so it runs identically on any machine and can be deployed to any cloud provider.

**Primary validation:** cantilever plate under parabolically distributed tip shear load, the same benchmark from Project 01, with exact comparison to the Timoshenko elasticity solution.

**Secondary validation:** plate with a central circular hole under uniaxial tension, compared against the Kirsch analytical solution (stress concentration factor $K_t = 3$).

**Numerical investigation:** volumetric locking under near-incompressibility in plane strain.

---

## Starter Files

You are provided with a complete set of starter files. Copy them into your repo as the first commit.

+----------------------------+----------------------------------------------+-----------------------------------------------+
| File                       | What it does                                 | Your job                                      |
+============================+==============================================+===============================================+
| `src/elements.py`          | CST element functions (B, D, k matrices)     | **Implement all function bodies**             |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `src/mesh.py`              | Rectangular mesh + plate-with-hole mesh      | **Implement all function bodies**             |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `src/assembly.py`          | Global K and R assembly                      | **Implement all function bodies**             |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `src/solver.py`            | BC application and linear solve              | **Implement the function body**               |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `src/postprocess.py`       | Stress recovery, von Mises, strain energy    | **Implement all function bodies**             |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `src/analytics.py`         | Timoshenko & Kirsch reference solutions      | Provided complete — just copy it              |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `app.py`                   | Streamlit frontend with all 7 plots          | Provided complete — adjust if needed          |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `tests/test_elements.py`   | 9 unit tests for elements                    | Provided — your code must pass these          |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `tests/test_assembly.py`   | 3 unit tests for assembly                    | Provided — your code must pass these          |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `tests/test_solver.py`     | 4 unit tests including patch test            | Provided — your code must pass these          |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `data/plate_with_hole_     | Pre-generated fallback mesh for hole         | See "Fallback Mesh" below                     |
| mesh.npz`                  | problem                                      |                                               |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `Dockerfile`               | Container definition                         | Provided — copy as-is                         |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `docker-compose.yml`       | One-command local dev                        | Provided — copy as-is                         |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `.streamlit/config.toml`   | Streamlit config                             | Provided — copy as-is                         |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `requirements.txt`         | Pinned dependencies                          | Provided — add packages if needed             |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `.gitignore`,             | Git/Docker ignore patterns                    | Provided — copy as-is                         |
| `.dockerignore`            |                                              |                                               |
+----------------------------+----------------------------------------------+-----------------------------------------------+
| `AI_Usage_Summary_         | Example of what a good AI summary looks like | Reference only — do not submit this           |
| Example.md`                |                                              |                                               |
+----------------------------+----------------------------------------------+-----------------------------------------------+

**Rule:** `app.py` never does any math. It only calls functions from `src/` and renders outputs.

**Note:** The provided `app.py` imports all `src/` modules. It will crash until you implement them. Use the skeleton below for Day 1, then switch to the full `app.py` once your modules are working.

### Day 1 Skeleton 

```python
### app.py (use this until your modules work)
import streamlit as st
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
```

---

## Project Structure

```
fea-solver/
├── src/
│   ├── __init__.py
│   ├── mesh.py
│   ├── elements.py
│   ├── assembly.py
│   ├── solver.py
│   ├── postprocess.py
│   └── analytics.py
├── tests/
│   ├── test_elements.py
│   ├── test_assembly.py
│   └── test_solver.py
├── data/
│   └── plate_with_hole_mesh.npz
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .streamlit/config.toml
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Step 1 — Repository Setup

1. Create the GitHub repo with the folder structure above.
2. Copy all starter files into the repo.
3. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/), run `docker compose up`, open `http://localhost:8501`.
4. Write a skeleton `app.py` with all input widgets but no solver logic — use the starter code below.
5. Deploy to [Streamlit Community Cloud](https://share.streamlit.io): connect repo → deploy. If Streamlit Cloud is slow or down, Render.com and Railway are acceptable alternatives.

### `requirements.txt`

This file pins every dependency to an exact version. This guarantees that everyone runs the identical library versions. Without pinning, `pip install numpy` might install different versions on different machines, leading to subtle numerical differences or API breakage.

### `Dockerfile`

A Dockerfile is a recipe for building a self-contained environment (called an "image") that has everything your app needs: an operating system, Python, your libraries, and your code. When you or anyone else runs this image, they get an identical container regardless of what OS or software they have installed locally.

### `docker-compose.yml`

Docker Compose lets you start your entire application with a single command (`docker compose up`) instead of typing long `docker run` commands. This file defines one service (your FEA solver) and how to configure it.

### `.streamlit/config.toml`

Streamlit reads this file at startup. `headless = true` tells Streamlit not to try opening a browser inside the container (there is no browser in a container).

### `.gitignore`

This file tells Git which files to never track. Without it, you will accidentally commit compiled bytecode, OS metadata, editor settings, and Docker build artifacts into your repository.

### `.dockerignore`

Similar to `.gitignore`, but for Docker. When Docker builds an image, it copies your project directory into the build context. This file excludes things the container doesn't need, making builds faster.

### Starter Code: `app.py`

```python
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
```

### Tools

- Link: [Streamlit deploy on Docker — official docs](https://docs.streamlit.io/deploy)
- Link: [Docker Desktop download](https://www.docker.com/products/docker-desktop/)

**Checkpoint:** `docker compose up` opens the app. GitHub repo is public. Streamlit Cloud URL exists. This is your **first commit**.

---

## Step 2 — CST Element Module: `src/elements.py`

This is the mechanics core. **You must implement all function bodies yourself.** Open `src/elements.py`, the function signatures, docstrings, and expected input/output formats are all there.

### Unit Tests: `tests/test_elements.py`

These tests are your specification. Your implementations must pass all of them. Run tests with: `docker compose run fea-solver pytest tests/test_elements.py -v`

### Hand Verification

After your implementation passes the tests, do this exercise:

Take the reference triangle with nodes at (0,0), (1,0), (0,1). Compute the B matrix by hand starting from the shape functions, and verify that your code output matches exactly. The `test_B_known_values` test checks the same thing, but doing it by hand forces you to internalize the derivation. 

### Tools

- Link: [pytest docs](https://docs.pytest.org/)
- Do not use AI to write the implementation for you — see the [AI Use Policy](#ai-use-policy).

**Checkpoint:** All 9 tests in `test_elements.py` pass.

---

## Step 3 — Mesh Generator: `src/mesh.py`

You need two mesh generators: a structured rectangular mesh for the cantilever problem, and a mesh with a circular hole for the Kirsch problem.

**Rectangular mesh:** implement `generate_rect_mesh` yourself. It's a straightforward structured grid.

**Plate-with-hole mesh:** this one is harder. Three options:

- **Option A:** Write your own mesher (polar-to-rectangular mapping). See the docstring in `mesh.py`.
- **Option B:** Use [Gmsh](https://gmsh.info/) (industry-standard). Script it from Python, extract nodes/elements/boundary tags into the format your `assembly.py` expects. Add `gmsh` to `requirements.txt`.
- **Option C (fallback):** Call `load_fallback_hole_mesh()` from `mesh.py` to load `data/plate_with_hole_mesh.npz`. This unblocks you for the solver and validation plots while you work on your own mesher. **Your final submission should include your own mesh generator** (Option A or B), but the fallback ensures you're never stuck waiting on the mesh to make progress elsewhere.

### Mesh Preview

Add a mesh preview in `app.py` using Plotly. When the user adjusts mesh sliders, the mesh should update in real time **before** clicking Solve. Use `plotly.graph_objects.Scatter` to draw triangle edges. Check the full version of `app.py` for sample of the code.

**Checkpoint:** Both meshes render in the app. Sliders update node/element count in real time. Both rectangular and plate-with-hole meshes display correctly.

---

## Step 4 — Global Assembly: `src/assembly.py` 

Implement `assemble_K`, `assemble_R_parabolic_shear`, and `assemble_R_uniform_tension`. Signatures and docstrings are in the file.

### Common pitfalls

Assembly is where most bugs hide. Check these:

- **K is not symmetric?** Your DOF index mapping is wrong. Print the 6 global DOF indices for one element and verify by hand that they correspond to [u0, v0, u1, v1, u2, v2] for that element's three nodes.
- **R.sum() ≠ P?** Your loaded nodes are probably not sorted by y-coordinate before integration. The Gauss quadrature assumes the edge goes from $y_a$ to $y_b$ in order.
- **Singular matrix when solving?** Check that your boundary conditions eliminate enough DOFs. A 2D structure needs at least 3 constrained DOFs to prevent rigid body motion.
- **Stress results are wildly wrong but K looks okay?** Check your units. If $E$ is in $Pa$ and lengths are in meters, forces must be in Newtons and stresses come out in Pa. Mixing $mm$ and $m$ is a classic error.

### Unit Tests: `tests/test_assembly.py`

Your implementations must pass all of these tests. Run tests with: `docker compose run fea-solver pytest tests/test_assembly.py -v`

**Checkpoint:** `K` is square with `shape == (2*n_nodes, 2*n_nodes)`. `K` is symmetric. `R.sum()` equals `P`.

---

## Step 5 — Solver and Postprocessing

Implement `apply_bc_and_solve` in `src/solver.py` and the three functions in `src/postprocess.py`.

**Checkpoint:** All tests in `test_solver.py` pass, including the **patch test**. Tip deflection v(L, 0) is within 20% of Euler–Bernoulli $\delta = \frac{PL^3}{3EI}$ for a coarse mesh.

---

## Step 6 — Analytical Solutions

`src/analytics.py` is provided complete.These are just reference solutions from elasticity theory. Copy it and verify: `kirsch_stress_polar(R, π/2, R, σ)` should give $\sigma_{\theta\theta} = 3\sigma$. Make sure you understand what `timoshenko_deflection` computes and why it differs from Euler-Bernoulli.

### Checkpoint

Plot analytical curves in Streamlit, they should look correct before FEA results are overlaid. Verify that `kirsch_stress_polar(R, π/2, R, σ∞)` gives $\sigma_{\theta\theta} = 3\sigma$.

---

## Step 7 — Full App Integration

Switch from the Day 1 skeleton to the provided full `app.py`. It wires together all your modules and renders 7 validation plots plus the convergence and locking studies. **Copy this file, then adjust it as needed for your specific implementation**. Feel free to use this pieces of this file to check plots in earlier steps of the project.

If you want to customize the UI beyond what is provided here, see the Plotly and Streamlit docs:

- Link: [Plotly Mesh3d docs](https://plotly.com/python/3d-mesh/)
- Link: [Streamlit columns and layout](https://docs.streamlit.io/library/api-reference/layout)
- Link: [Streamlit tabs](https://docs.streamlit.io/library/api-reference/layout/st.tabs)

### Required plots

**Cantilever:** (1) deflection v(x) — FEA vs. Timoshenko vs. Euler-Bernoulli, (2) $\sigma_{xx}$ along bottom fiber, (3) deformed mesh with von Mises color, (4) $\tau_{xy}$ across beam depth at mid-span.

**Plate with hole:** (5) $\sigma_{\theta\theta}$ along hole boundary vs. Kirsch, (6) $\sigma_{xx}$ along x-axis showing stress decay, (7) deformed mesh with von Mises color.

**Checkpoint:** All 7 plots render correctly.

---

## Step 8 — Convergence Study and Volumetric Locking Investigation

The UI is already in `app.py`. Your job in this step is to make sure your `src/` modules are correct to produce meaningful results, and then to **analyze and interpret** what you see.

### Part A: h-Refinement convergence

Using the Convergence & Locking tab:

- Run the cantilever problem for mesh densities: nx = [2, 4, 8, 16, 32]
- For each mesh, compute the tip deflection and the relative error against the Timoshenko solution
- Plot on a **log-log** scale: number of elements vs. relative error
- Compute and display the **convergence rate** p (the slope of the log-log plot)

**Expected result:** CST elements converge at rate p ≈ 1 (first-order convergence in displacement). If you get a significantly different slope, debug your implementation before proceeding.

Repeat for the plate-with-hole problem: plot the maximum $\sigma_{\theta\theta}$ at the hole vs. mesh density, compared to the Kirsch value of $3\sigma$.

### Part B: Volumetric locking

**Task:** Run the cantilever problem in **plane strain** mode for a sequence of Poisson's ratios:

ν = [0.3, 0.4, 0.45, 0.49, 0.499, 0.4999]

For each value of ν:

1. Compute the tip deflection and normalize by the analytical (Timoshenko) value
2. Compute the condition number of the stiffness matrix (use `np.linalg.cond` on the free-DOF submatrix, or estimate with `scipy.sparse.linalg.eigsh`)
3. Examine the deformed shape — does it look physical?

**Required deliverables for the locking study:**

- A plot of **normalized tip deflection** vs. ν (you should observe the solution becoming dramatically too stiff as ν → 0.5)
- A plot of **condition number** vs. ν (expect rapid growth)
- A clear **explanation** of:
  - What volumetric locking is, physically
  - Why the CST element is particularly susceptible (hint: count the constraints imposed by incompressibility vs. the number of degrees of freedom per element)
  - What element formulations would fix the problem (name at least two approaches)

> **Note:** This investigation cannot be answered by running code alone. You need to understand the constraint ratio argument (number of incompressibility constraints vs. number of displacement DOFs) and explain it in your own words.


### Checkpoint

- Log-log convergence plot shows slope p ≈ 1 for the cantilever
- Locking plot clearly shows tip deflection collapsing toward zero as ν → 0.5 in plane strain

---

## Step 9 — Polish and Deploy

- [ ] `pytest tests/ -v` passes inside Docker: `docker compose run fea-solver pytest tests/ -v`
- [ ] CSV export works
- [ ] Input validation warnings (e.g., near-incompressible ν)
- [ ] `README.md` with: theory overview, how to run with Docker, how to run tests, screenshot
- [ ] Public URL works

### Final Checkpoint

Public URL works. `docker compose up` reproduces the exact same app locally. All tests pass. README is complete.

---

## Git Requirements

Your commit history is part of the deliverable. We will review it.

- **At least 10 meaningful commits** spread across the development period
- Each commit message describes what changed (e.g., `"Implement B-matrix for CST element"`, not `"update"`)
- Committing the entire project in 1–2 dumps does not count

### What Counts as a "Meaningful" Commit

- Adding or completing a module implementation
- Fixing a bug discovered through testing
- Adding or updating tests
- Wiring a new feature into the Streamlit app
- Refactoring or improving an existing module

### What Does NOT Count

- Committing the entire project in 1–2 large dumps
- Empty commits or commits that change only whitespace
- Commits generated by AI without your review

---

## Deliverables

| Item | Description |
|------|-------------|
| **Live URL** | Publicly accessible deployed app — must work on the due date |
| **GitHub repo** | Full project with Docker, `src/`, `tests/`, README. **10+ meaningful commits.** |
| **Demo video (3-5 min video)** | The video is a screen recording showing: solving both the cantilever and plate-with-hole problems, walking through the validation plots, convergence study with slope p, and explaining the locking investigation results. The video replaces a separate written discussion of the plots — show and explain them on camera instead. |
| **Live walkthrough (5 min)** | In-person demo + code walkthrough + Q&A. The instructor will ask you to explain functions, modify parameters, and predict outcomes. |
| **AI usage summary (1 page)** | See below. An example is provided in `AI_Usage_Summary_Example.md`. |

---

## AI Use Policy

Use AI tools freely for: debugging, learning Streamlit/Plotly/Docker/Git syntax, generating boilerplate, understanding library APIs, reviewing code after you've written a first draft.

**You must write these yourself first** (before using AI to check or improve):

- `compute_B`: the B-matrix
- `compute_D`: the constitutive matrix
- `assemble_R_parabolic_shear`: the consistent load vector
- The **locking explanation**: your own words, your own understanding

A word of caution: AI tools are very good at producing code that looks correct and passes simple tests but has subtle numerical bugs — wrong sign conventions, transposed indices, off-by-one errors in DOF mapping. These are exactly the bugs that will make your FEA results wrong in ways that are hard to diagnose. The students who do best on this project are the ones who write the core functions themselves first, understand what every line does, and then use AI to catch what they missed — not the other way around.

### AI Usage Summary (1 page)

Submit a one-page summary describing how you used AI tools. Not a log of every prompt — a reflective description covering:

1. **Which tools** you used and for what purposes
2. **Which parts** of the code were substantially AI-assisted vs. written by you
3. **One specific example** where AI gave you incorrect or misleading output and how you identified and fixed it
4. **What you learned** from the interaction that you wouldn't have learned from docs alone

See `AI_Usage_Summary_Example.md` for what a good summary looks like. This will not reduce your grade.

---

## Grading

Your project is graded in two parts: a **base score** and a **walkthrough multiplier**.

### Base Score (100 points)

| Component | Weight | Key Criteria |
|-----------|--------|-------------|
| Working solver (cantilever) | 25% | Correct results, all tests pass |
| Plate with hole | 15% | Mesh generation, K_t ≈ 3, Kirsch comparison |
| Convergence & locking | 25% | Correct convergence rate, locking demonstrated, quality of explanation |
| App quality & deployment | 15% | Clean UI, Docker works, deployed URL, CSV export |
| Code quality & Git | 10% | Clean structure, 10+ commits, tests pass, README |
| AI usage summary | 10% | Thoughtful, specific, honest |

### Walkthrough Multiplier (×0.6 to ×1.0)

The walkthrough determines what fraction of your base score you keep.

| Performance | Multiplier |
|-------------|-----------|
| Explains all code and results confidently, answers follow-ups | ×1.0 |
| Explains most code, minor hesitation on details | ×0.9 |
| High-level flow OK but struggles with implementation details | ×0.8 |
| Cannot explain key functions or predict parameter changes | ×0.7 |
| Cannot explain own code at a basic level | ×0.6 |

**Final grade = Base score × Walkthrough multiplier**

---

## When You're Stuck

You will get stuck. Everyone does. Here is a debugging strategy that works:

1. **Shrink the problem.** Use a 2×1 mesh (4 nodes, 2 elements). At this size, you can trace every number by hand — print the element stiffness, print the DOF indices, print the assembled K, print R, print u.

2. **Visualize intermediate quantities.** Plot R as arrows on the mesh — does the load vector look parabolic? Plot the sparsity pattern of K (`plt.spy(K)`) — does it have the structure you expect? Plot the deformed mesh at an absurdly large scale factor — does the deformation direction make physical sense?

3. **Check units.** If E is in Pa (N/m²) and lengths are in meters, then forces must be in Newtons, areas in m², and stresses come out in Pa. Mixing mm and m is the most common silent error.

4. **Isolate the bug.** If tip deflection is wrong, first check: does the patch test pass? If yes, assembly and BCs are likely correct and the problem is in the load vector or stress recovery. If the patch test fails, the bug is in assembly or BCs.

5. **Compare against a known solution at every step.** Don't wait until the full cantilever solve to discover something is wrong. Check `compute_area` on a known triangle. Check `compute_B` against the hand calculation. Check `K` symmetry. Check `R.sum() == P`. Each of these is a 1-minute check that can save hours.

6. **Read the error message.** Scipy's `spsolve` will tell you if the matrix is singular. NumPy will tell you about shape mismatches. These messages are precise — read them carefully before searching online.

---

## Timeline

| Days | Milestone |
|------|-----------|
| 1 | Repo setup, Docker running, skeleton app deployed |
| 2–4 | `elements.py` done, all element tests pass |
| 5–7 | Both meshes working, preview in app |
| 8–10 | Assembly done, K symmetric, R balanced |
| 11–12 | Solver + postprocessing, patch test passes |
| 13–14 | Analytics integrated, reference curves plotted |
| 15–17 | Full app wired, all 7 validation plots |
| 18–19 | Convergence + locking study |
| 19–20 | Polish, deploy, README, AI summary |
