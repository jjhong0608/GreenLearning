# Local Weak-Residual Reliability Reconstruction

## Scope

Complex CouplingNet retains two directional physical sources and two Green
reconstructions,

\[
\phi+\psi=f,
\qquad
u_\phi=G_x\Phi,
\qquad
u_\psi=G_y\Psi.
\]

The optional local weak-residual reliability rule changes only the final
reported solution,

\[
u_{\mathrm{pred}}
=w_\phi u_\phi+w_\psi u_\psi.
\]

It does not modify projection, directional Green reconstruction, the training
objective, optimizer, scheduler, or checkpoint selection. The disabled default
is the exact equal mean

\[
u_{\mathrm{equal}}=\frac12(u_\phi+u_\psi).
\]

## Candidate Weak Residuals

For each valid-point P1 nodal test function \(\chi_i\), the existing axial weak
operator defines

\[
R_x(v)_i=B_x(v,\chi_i)-\langle\phi,\chi_i\rangle,
\qquad
R_y(v)_i=B_y(v,\chi_i)-\langle\psi,\chi_i\rangle.
\]

The directional bilinear forms are

\[
B_x(v,\chi_i)
=\int_\Omega
\left[
a v_x(\chi_i)_x+b_xv_x\chi_i+\frac12cv\chi_i
\right],d\mathbf x,
\]

\[
B_y(v,\chi_i)
=\int_\Omega
\left[
a v_y(\chi_i)_y+b_yv_y\chi_i+\frac12cv\chi_i
\right],d\mathbf x.
\]

Each candidate \(v\in\{u_\phi,u_\psi\}\) is tested against both directional
equations:

\[
R(v)_i=R_x(v)_i+R_y(v)_i.
\]

The implementation reuses the segment-local P1 gather/scatter operator. It
includes true segment boundary endpoints with homogeneous Dirichlet hard-zero
values, evaluates coefficients directly at physical element midpoints, splits
the reaction term as \(c/2\) per direction, and uses the transverse measures
\(h_y\) for x-elements and \(h_x\) for y-elements. It does not assemble or
solve a global matrix system.

## Local Indicator And Smoothing

Let

\[
m_i=m_{x,i}+m_{y,i}
\]

be the combined lumped nodal mass. The raw squared indicators are

\[
\eta_{\phi,\mathrm{raw},i}^2
=\frac{R(u_\phi)_i^2}{m_i+\varepsilon},
\qquad
\eta_{\psi,\mathrm{raw},i}^2
=\frac{R(u_\psi)_i^2}{m_i+\varepsilon}.
\]

Smoothing uses only the valid physical adjacency

\[
E=E_x\cup E_y.
\]

One relaxation step is

\[
\mathcal S_\rho(z)_i
=(1-\rho)z_i
+\rho\frac{1}{\deg(i)}
\sum_{j:(i,j)\in E}z_j.
\]

The default applies two steps with \(\rho=0.5\). Because the geometry edge
schema never joins disconnected segments across a hole, smoothing cannot cross
such a gap.

## Reliability Partition

For each sample, define the arithmetic-mean floor

\[
\tau
=r_{\mathrm{floor}}
\frac{\langle\eta_\phi^2\rangle+
\langle\eta_\psi^2\rangle}{2}
+\varepsilon.
\]

The signed reliability and partition weights are

\[
\theta
=\gamma
\frac{\eta_\psi^2-\eta_\phi^2}
{\eta_\phi^2+\eta_\psi^2+2\tau},
\]

\[
w_\phi=\frac12(1+\theta),
\qquad
w_\psi=1-w_\phi.
\]

Thus the candidate with the smaller local weak defect receives the larger
weight. With the default \(\gamma=0.5\), both weights remain in
\([0.25,0.75]\). With \(\gamma=0\), the method exactly recovers the equal
mean.

## Configuration

```json
"cross_axis_reconstruction": {
  "enabled": true,
  "mode": "local_weak_residual_reliability",
  "gamma": 0.5,
  "smoothing_steps": 2,
  "smoothing_relaxation": 0.5,
  "relative_floor": 0.1,
  "eps": 1e-12
}
```

This block belongs under `coupling_model`. It is accepted only by complex
CouplingNet. Geometry-only compact C2 and mismatch-detected seam C2 are not
valid production modes.

## Runtime And Artifact Contract

- Trainer: only detached `rel_sol` uses the selected final reconstruction.
  Source-only train/validation does not construct the reliability context.
- Evaluator: `rel_sol` uses the selected `u_pred`; enabled mode also reports
  `rel_sol_equal_mean` and weight statistics. `rel_flux` is unchanged.
- Artifact: `u_pred` and `u_pred_error` use the selected reconstruction. Enabled
  mode also stores the equal-mean baseline, directional/full candidate
  residuals, nodal mass, raw/smoothed indicators, floor, signed reliability,
  and both partition weights.
- Checkpoint: the feature has no trainable parameters and changes neither model
  state keys nor the Complex CouplingNet output contract.
- Reference boundary: `sol`, `target_phi`, and `target_psi` are never inputs to
  the reliability calculation. They remain evaluation-only references.

The indicator is a reference-free candidate-selection heuristic, not a
certified pointwise a posteriori error estimator. Its weights can retain axial
stripe structure even when aggregate solution metrics improve.
