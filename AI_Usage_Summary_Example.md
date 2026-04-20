# AI Usage Summary — Example

*This is a fictional example to show the level of detail expected. Your summary should reflect your own experience.*

---

## Tools Used

I used **GitHub Copilot** (in VS Code) throughout the project for autocomplete suggestions, and **Claude** for debugging sessions when I got stuck on specific errors. I also used **ChatGPT** once to understand how `scipy.sparse.lil_matrix` differs from `csr_matrix` and when to use each.

## What Was AI-Assisted vs. Written By Me

I wrote `compute_B`, `compute_D`, and `assemble_R_parabolic_shear` entirely by hand first, following the lecture notes. Copilot was turned off for these files. After my implementations passed the unit tests, I asked Claude to review `compute_B` and it confirmed the logic was correct but suggested I use `np.zeros((3, 6))` instead of building the matrix row by row with lists — a style improvement, not a bug fix.

For `assemble_K`, I wrote the loop structure myself but Copilot autocompleted the DOF index mapping line (`dofs = np.array([[2*n, 2*n+1] for n in tri]).ravel()`). I verified this was correct by printing the DOF indices for one element and checking by hand.

The `generate_plate_with_hole_mesh` function was the most AI-assisted. I described the geometry to Claude and asked it to help me script the Gmsh Python API, since I had never used Gmsh before. It generated a working script, but the boundary tagging was wrong — it tagged all outer edges as "right" instead of separating x=W from y=H. I had to debug the physical group assignments myself.

## One Case Where AI Was Wrong

When I asked Claude to help debug why my `R.sum()` was not equal to P, it suggested that I had the wrong Gauss quadrature weights. I spent an hour adjusting the quadrature before realizing the actual problem was that my loaded nodes were not sorted by y-coordinate. The AI confidently pointed me in the wrong direction. I found the real bug by printing the y-coordinates of adjacent loaded nodes and noticing they were not monotonically increasing.

## What I Learned

The Gmsh interaction taught me that AI is very useful for learning new library APIs quickly — I went from zero knowledge of Gmsh to a working mesh in about 2 hours, which would have taken much longer reading the docs alone. But the boundary tagging bug taught me that AI-generated code for geometry-specific logic needs careful manual verification. The AI doesn't "see" the mesh — it's guessing at the geometry from the code structure.
