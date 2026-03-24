# CouplingNet (for slides)

## Model structure
- Architecture: MIONet/DeepONet style with four branches and one trunk.
  - Branch a: MLP over sampled coefficient a on each axial line (input dim = points per line).
  - Branch b: MLP over sampled coefficient b on each axial line.
  - Branch c: MLP over sampled coefficient c on each axial line.
  - Branch f̃: MLP over normalized source f̃ (line-wise L2 normalization).
  - Trunk: MLP over (x, y) axial coordinates; combined with branch features via batched matmul.
- Output: normalized axial flux-divergences (φ̃, ψ̃) per axis, then denormalized by the line norm ‖f‖ to physical fluxes (φ, ψ).
- Shapes: inputs (coords: (2, n, m, 2); a, b, c, f̃, f raw: (B, 2, n, m)); output flux: (B, 2, n, m) with n+2 = m.

## Losses
- Consistency (x/y reconstructions agree):
  \[
  \mathcal{L}_{\text{cons}} = \frac{1}{NM} \sum_{i,j} |u^{(x),i}_{\theta,j} - u^{(y),i}_{\theta,j}|^2
  \]

## Normalization and reconstruction
- Line-wise L2 norm: ‖f‖_l = sqrt(∫_l f^2); inputs use f̃ = f / ‖f‖_l; outputs are renormalized by ‖f‖_l.
- Green kernels are precomputed from GreenONet for each axis/line and reused for integration losses and evaluation.

## Notation details (for losses)
- Indices: \(i = 1,\dots,N\) (axial lines), \(j = 1,\dots,M\) (samples per line).
- Flux fields: \(\phi_i(\xi)\) along x-lines, \(\psi_i(\eta)\) along y-lines; samples \(\phi^i_j, \psi^i_j\).
- Green kernels: \(G^{(x)}_i(x;\xi), G^{(y)}_i(y;\eta)\); convolutions with f:
  - \((G^{(x)}*f)^i_j = \int G^{(x)}_i(x_j;\xi)\,f_i(\xi)\,d\xi\), similarly \((G^{(y)}*f)^i_j\).
- Green-weighted projections (not the full solution):
  - \(v^{(x)}_{\phi,ij} = \int G^{(x)}_i(x_j;\xi)\,\phi_i(\xi)\,d\xi\), \(v^{(y)}_{\phi,ij} = \int G^{(y)}_i(y_j;\eta)\,\phi_i(\eta)\,d\eta\).
  - \(v^{(x)}_{\psi,ij}, v^{(y)}_{\psi,ij}\) defined analogously with \(\psi\).
- Consistency uses reconstructed solutions \(u^{(x)}, u^{(y)}\) from integrating fluxes with \(G^{(x)}\), \(G^{(y)}\).
