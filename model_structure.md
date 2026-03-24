# GreenONet Model Structure (Current Implementation)

## Inputs & Shapes
- Trunk grid `X`: shared meshgrid over the unit square, shape `(m, m, 2)` with coordinates `(x, ξ)`.
- Coefficient samples `a, ap, b, c`: shape `(B, 2, n_lines, m)` (batch, axis={x,y}, line index, points along line).
- Output: `(B, 2, n_lines, m, m)` — for each batch and axis/line, a value at every `(x, ξ)` pair.

## Branch/Trunk
- Branch MLPs `B_a(·), B_b(·), B_c(·)`: input dim = `m` (one coefficient profile per line), hidden dim = `H`.
- Trunk MLP `T(·)`: input dim = `2` (x, ξ), hidden dim = `H` (or Fourier-embedded dim when `use_fourier=true`).
- Core (per axis/line):  
  `core = (B_a(a_line) ⊙ B_b(b_line) ⊙ B_c(c_line)) @ T(X_flat)^T` where `X_flat = X.reshape(m*m, 2)`; reshape `core` to `(B, m, m)`.

## Green Assembly
- Envelope: `E(x, ξ) = x(1−x) ξ(1−ξ)` on the shared trunk grid.
- Green primitives (piecewise, from `greens.py`):  
  `G(x, ξ) = x(1−ξ)` if `x < ξ` else `ξ(1−x)`  
  `I(x, ξ) = 0.5 x^2 (1−ξ)` if `x < ξ` else `0.5 ξ (2x − x^2 − ξ)`
- Coefficient broadcasts (all shaped `(B, 2, n_lines, m, m)`):  
  `A0 = refactor_x0(a)`, `A1 = refactor_x1(a)`, `AP1 = refactor_x1(ap)`, `B1 = refactor_x1(b)`.
- Simple path: `S = E * core + G / (A0 + eps)` (used when `AP1 + B1` is numerically zero).
- Full path (current code):
  - `invA0 = 1/(A0 + eps)`, `coeff = (AP1 + B1)/(A0*A1 + eps)`
  - `F = invA0 * G + coeff * I + coeff * E * core − coeff * bias`, with `bias = 0.5 * x^2 * ξ * (1 − x)` from the trunk grid.
- Masked output: `Y = where(AP1 + B1 ≈ 0, S, F)` keeps the original DeepONet gating behavior.

## Flux Head
- Separate coupling MLP `C(·)` available via `predict_flux_divergence(coords)`; not used in the main Green assembly.

## Notes vs Original DeepONet
- Uses a shared `(m, m, 2)` trunk grid (no per-line trunk inputs).
- Coefficients stacked by axis in the batch dims; refactors broadcast to `(m, m)` instead of diagonal/paired handling.
- Output retains full `(m, m)` grid per axis/line; trainer expands targets accordingly.
