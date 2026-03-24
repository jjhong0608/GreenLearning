# CouplingNet Overview

## Architecture
- **Purpose:** Predict axial flux divergences \(\varphi, \psi\) along x/y axial lines for variable-coefficient Poisson.
- **Inputs:**
  - Branches: sampled coefficients \(a, b, c\) and source term \(f\) along each axial line, shaped `(B, 2, n_lines, m_points)`.
  - Trunk: coordinates along each axial line, shaped `(2, n_lines, m_points, 2)` (x-lines: varying x with fixed y; y-lines: varying y with fixed x).
- **Networks:** MIONet/DeepONet style with four branch MLPs and one trunk MLP (hidden size `H`).
  - Branch\_a: \(B_a(a) \in \mathbb{R}^H\)
  - Branch\_b: \(B_b(b) \in \mathbb{R}^H\)
  - Branch\_c: \(B_c(c) \in \mathbb{R}^H\)
  - Branch\_rhs: \(B_f(f) \in \mathbb{R}^H\)
  - Trunk: \(T(x,y) \in \mathbb{R}^H\)
- **Combination:** Hadamard product of branch outputs, then inner product with trunk via batched matmul:  
  \( \text{flux}_{b,a,l,m} = \sum_h (B_a \odot B_b \odot B_c \odot B_f)_{b,a,l,h} \; T_{a,l,m,h} \)  
  yielding flux divergence for batch `b`, axis `a∈{x,y}`, line `l`, point `m`.

## Loss (Cross-Integral Equations)
For each batch and axial grid:
- Predicted fluxes: \(\varphi = \text{flux}_{x},\ \psi = \text{flux}_{y}\).
- Green kernels (precomputed from GreenONet): \(G^{(x)}, G^{(y)} \in \mathbb{R}^{n \times m \times m}\).
- Integrals (Simpson over along-line coordinate):  
$$
  U_\varphi^{(x)}(\bar{x},\bar{y}) = \int G^{(x)}(\bar{x};\xi)\,\varphi(\xi,\bar{y})\,d\xi,\quad
     U_\psi^{(y)}(\bar{x},\bar{y}) = \int G^{(y)}(\bar{y};\eta)\,\psi(\bar{x},\eta)\,d\eta.
$$
- Source integrals:  
$$
F^{(x)}(\bar{x},\bar{y}) = \int G^{(x)}(\bar{x};\xi)\,f(\xi,\bar{y})\,d\xi,\quad
     F^{(y)}(\bar{x},\bar{y}) = \int G^{(y)}(\bar{y};\eta)\,f(\bar{x},\eta)\,d\eta.
$$
- Cross equations (one per axis):  
$$
U_\varphi^{(x)} + U_\varphi^{(y)} = F^{(y)},\quad
     U_\psi^{(x)} + U_\psi^{(y)} = F^{(x)}.
$$
- Loss: mean squared residual of these equations over all lines/points:  
$$
\mathcal{L} = \|U_\varphi^{(x)} + U_\varphi^{(y)} - F^{(y)}\|_2^2 + \|U_\psi^{(x)} + U_\psi^{(y)} - F^{(x)}\|_2^2.
$$
- Metrics: relative $L^2$ (integral-based) for flux and solution reconstruction.

## Data Flow
- Dataset: loads `rhs, sol, uxx, uyy` from npz; pads `uxx/uyy`; samples along interior axial lines; samples \(a, b, c\) (and optionally \(\partial_x a, \partial_y a\)) via provided functions.
- Green kernel: computed once from the pretrained GreenONet using the same axial grid and \(\kappa, \partial \kappa\) profiles.
- Training: Adam with optional LBFGS fine-tuning; logs loss curve and saves weights to `coupling_model.safetensors`.
