# Codex Task: Extend GreenLearning from Unit-Square to Complex Geometry

## 1. Context

You are working on the `GreenLearning` codebase. The current implementation is primarily designed for a unit-square setting, where axial lines live on fixed intervals inside

\[
[0,1]\times[0,1].
\]

The goal of this task is to extend the existing unit-square implementation to support **complex 2D geometries** while preserving the current implementation style as much as possible.

This task is not a request to replace the entire codebase with a new design. Instead, refactor and extend the current code so that the complex-geometry path follows the two design documents listed below.

---

## 2. Source Documents

Before modifying code, read the following two documents carefully and treat them as the source of truth.

1. `greennet_unit_interval_normalization.md`
2. `couplingnet_complex_geometry_design.md`

If a design choice is ambiguous in the existing code, follow these two documents rather than guessing.

The two documents specify:

- how `GreenNet` should handle physical 1D connected intervals by mapping each interval to \([0,1]\);
- how transformed coefficients and right-hand sides must be scaled;
- how `CouplingNet` should represent valid points, axial segments, raw outputs, projection, reconstruction, and energy loss in complex geometry;
- that cross consistency is completely excluded in complex-geometry mode.

---

## 3. High-Level Goal

Extend the current unit-square code to support complex geometry by introducing a complex-geometry path with the following properties:

\[
\boxed{\text{GreenNet operates on normalized 1D intervals.}}
\]

\[
\boxed{\text{CouplingNet operates on flattened valid physical points and axial connected segments.}}
\]

\[
\boxed{\text{CouplingNet raw output is a segment-local unit-domain source-like quantity.}}
\]

\[
\boxed{\text{Projection and energy consistency are evaluated in physical space.}}
\]

\[
\boxed{\text{Cross consistency is completely excluded in complex-geometry mode.}}
\]

Preserve the current unit-square functionality unless a refactor is strictly necessary. Prefer adding a complex-geometry mode/path rather than destructively rewriting all existing code.

---

## 4. Files to Inspect First

Inspect the current repository structure first. In particular, look for the following files or their equivalents:

```text
src/greenonet/axial.py
src/greenonet/coupling_data.py
src/greenonet/coupling_model.py
src/greenonet/coupling_trainer.py
src/greenonet/trainer.py
src/greenonet/greens.py
cli/train.py
configs/
```

Your first step is to identify all unit-square assumptions, including but not limited to:

- hardcoded \([0,1]\times[0,1]\) grids;
- hardcoded `torch.linspace(0.0, 1.0, ...)` used as a physical grid;
- rectangular tensor assumptions such as `(B, 2, m, n)`;
- equal x-line and y-line counts;
- boundary hardcoding at \(x=0,1\) and \(y=0,1\);
- line extraction by rectangular grid index slicing;
- smooth mask projection tied to unit-square factors such as \(x(1-x)\), \(y(1-y)\), \(\sin(\pi x)\), and \(\sin(\pi y)\);
- transpose-based cross consistency that assumes a rectangular full grid.

Do not start implementation before mapping these assumptions.

---

## 5. Non-Negotiable Design Decisions

### 5.1 GreenNet unit-interval normalization

Each physical connected 1D interval must be mapped to the unit interval.

For a connected physical interval

\[
I_\ell=[s_{\ell,0},s_{\ell,1}],
\qquad
L_\ell=s_{\ell,1}-s_{\ell,0},
\]

use

\[
s=s_{\ell,0}+L_\ell t,
\qquad
t\in[0,1].
\]

Every connected interval is an independent GreenNet domain.

If an axial line intersects the geometry in disconnected components,

\[
\ell\cap\Omega
=
I_{\ell,1}\cup I_{\ell,2}\cup\cdots,
\]

then each \(I_{\ell,k}\) is processed independently.

Do **not** merge disconnected intervals.

### 5.2 GreenNet coefficient and RHS scaling

For the physical 1D operator

\[
-\frac{d}{ds}
\left(
a(s)\frac{du}{ds}
\right)
+
b(s)\frac{du}{ds}
+
c(s)u
=
f(s),
\]

the unit-interval operator must use

\[
a_{\mathrm{unit}}(t)
=
a(s_{\ell,0}+L_\ell t),
\]

\[
a'_{\mathrm{unit}}(t)
=
L_\ell a_s(s_{\ell,0}+L_\ell t),
\]

\[
b_{\mathrm{unit}}(t)
=
L_\ell b(s_{\ell,0}+L_\ell t),
\]

\[
c_{\mathrm{unit}}(t)
=
L_\ell^2 c(s_{\ell,0}+L_\ell t),
\]

\[
f_{\mathrm{unit}}(t)
=
L_\ell^2 f(s_{\ell,0}+L_\ell t).
\]

The value passed as `ap_vals` must be the unit-coordinate derivative

\[
a'_{\mathrm{unit}}=\frac{d a_{\mathrm{unit}}}{dt},
\]

not the physical derivative \(a_s\).

This is mandatory.

### 5.3 GreenNet trunk coordinate convention

GreenNet trunk coordinates must remain normalized coordinates:

\[
(t,\eta)\in[0,1]^2.
\]

GreenNet output is interpreted as

\[
G_{\mathrm{unit}}(t,\eta).
\]

If a physical Green kernel is explicitly needed, use

\[
G_{\mathrm{phys}}(s,r)
=
L_\ell
G_{\mathrm{unit}}
\left(
\frac{s-s_{\ell,0}}{L_\ell},
\frac{r-s_{\ell,0}}{L_\ell}
\right).
\]

If reconstruction is performed in unit coordinates using \(f_{\mathrm{unit}}\), do **not** multiply the kernel by \(L_\ell\) again.

Avoid double-counting length scaling.

### 5.4 CouplingNet valid-point representation

For complex geometry, do not represent fields using a full rectangular tensor such as

\[
(B,2,m,n).
\]

Instead, use flattened valid physical points:

\[
\mathcal{P}_\Omega
=
\{p_q=(x_q,y_q)\}_{q=1}^{P}.
\]

The field representation must be

\[
\mathrm{flux}\in\mathbb{R}^{B\times 2\times P}.
\]

Interpretation:

\[
\mathrm{flux}[:,0,q]=\phi(p_q),
\]

\[
\mathrm{flux}[:,1,q]=\psi(p_q).
\]

Boundary endpoints are **not** included in \(P\). They are used only as reconstruction quadrature endpoints.

### 5.5 Geometry is fixed across samples

Assume the geometry is the same for all samples.

Thus, the following metadata is batch-shared:

- valid physical point set;
- x/y segment ids;
- local coordinates;
- segment endpoints;
- segment lengths;
- reconstruction nodes;
- trapezoid weights;
- energy edge graph.

The batch dimension should represent sample variation in coefficients/source, not geometry variation.

### 5.6 Axial segment metadata

Add data structures for horizontal and vertical connected segments.

Required valid-point metadata:

```text
coords_valid: (P, 2)
x_segment_id: (P,)
y_segment_id: (P,)
x_local_t: (P,)
y_local_t: (P,)
```

Required horizontal segment metadata:

```text
x_segment_left:   (Sx,)
x_segment_right:  (Sx,)
x_segment_y:      (Sx,)
x_segment_length: (Sx,)
```

Required vertical segment metadata:

```text
y_segment_bottom: (Sy,)
y_segment_top:    (Sy,)
y_segment_x:      (Sy,)
y_segment_length: (Sy,)
```

Do not assume

\[
S_x=S_y.
\]

Horizontal and vertical segment counts may differ.

### 5.7 CouplingNet raw-output convention

CouplingNet raw output is **not** physical \(\phi,\psi\).

It is a segment-local unit-domain source-like quantity.

For a horizontal segment,

\[
\Phi_k^{\mathrm{raw}}(t)
\approx
(L_k^x)^2
\phi_{\mathrm{phys}}(x_k^-+L_k^x t,y_k).
\]

For a vertical segment,

\[
\Psi_l^{\mathrm{raw}}(t)
\approx
(L_l^y)^2
\psi_{\mathrm{phys}}(x_l,y_l^-+L_l^y t).
\]

Thus,

\[
\Phi=L_x^2\phi_{\mathrm{phys}},
\qquad
\Psi=L_y^2\psi_{\mathrm{phys}}.
\]

This convention is mandatory.

### 5.8 Function branch input

The CouplingNet function branch uses only transformed

\[
a,\quad b,\quad c.
\]

Do not add a source branch, Green response feature, source stencil lift, or \(a'\) feature to the CouplingNet function branch by default.

For a horizontal segment,

\[
a_{\mathrm{unit}}^x(t)
=
a(x_k^-+L_k^x t,y_k),
\]

\[
b_{\mathrm{unit}}^x(t)
=
L_k^x b_x(x_k^-+L_k^x t,y_k),
\]

\[
c_{\mathrm{unit}}^x(t)
=
(L_k^x)^2 c(x_k^-+L_k^x t,y_k).
\]

For a vertical segment,

\[
a_{\mathrm{unit}}^y(t)
=
a(x_l,y_l^-+L_l^y t),
\]

\[
b_{\mathrm{unit}}^y(t)
=
L_l^y b_y(x_l,y_l^-+L_l^y t),
\]

\[
c_{\mathrm{unit}}^y(t)
=
(L_l^y)^2 c(x_l,y_l^-+L_l^y t).
\]

Use fixed-size branch sampling on the segment-local unit interval:

\[
\tau_j=\frac{j}{m-1}.
\]

The branch sampling grid does not need to equal the reconstruction quadrature grid.

### 5.9 Shared horizontal/vertical networks

Horizontal and vertical branches must share the same function branch network.

\[
B_{\mathrm{func}}^x=B_{\mathrm{func}}^y.
\]

They should also share the same geometry branch and trunk design whenever possible.

Use a canonical segment representation rather than separate x/y-specific networks.

### 5.10 Geometry/transverse branch

Use the canonical notation

\[
I_\ell=[s_\ell^-,s_\ell^+],
\qquad
r=r_\ell,
\qquad
L_\ell=s_\ell^+-s_\ell^-.
\]

For horizontal segments:

\[
s=x,\qquad r=y.
\]

For vertical segments:

\[
s=y,\qquad r=x.
\]

The geometry branch feature must be

\[
g_\ell
=
[
\operatorname{PE}(r_\ell),
s_\ell^-,
s_\ell^+,
s_{\ell,\mathrm{mid}},
L_\ell,
L_\ell^2,
1/L_\ell
],
\]

where

\[
s_{\ell,\mathrm{mid}}
=
\frac{s_\ell^-+s_\ell^+}{2}.
\]

Only \(r_\ell\) is Fourier encoded:

\[
\operatorname{PE}(r_\ell)
=
[
\sin(\pi f_k r_\ell),
\cos(\pi f_k r_\ell)
]_{k=1}^{K}.
\]

The frequency set must follow the existing config schema. The default is

```text
[1, 2, 4, 8]
```

Do not include:

- axis one-hot;
- raw \(r_\ell\) scalar.

Use `product_fuser` as the default branch fusion mode:

\[
[
h_{\mathrm{func}},
h_{\mathrm{geom}},
h_{\mathrm{func}}\odot h_{\mathrm{geom}}
]
\]

followed by a learned fuser.

### 5.11 Hard symmetric projection

Use only hard symmetric projection.

Do not use balance loss.

Do not use smooth masked projection.

Projection must be applied in physical variables.

At valid point \(p_q\), let

\[
\alpha(q)=x\_segment\_id(q),
\qquad
\beta(q)=y\_segment\_id(q).
\]

Convert unit raw outputs to physical raw outputs:

\[
\phi_q^{\mathrm{raw}}
=
\frac{\Phi_q^{\mathrm{raw}}}{(L_{\alpha(q)}^x)^2},
\]

\[
\psi_q^{\mathrm{raw}}
=
\frac{\Psi_q^{\mathrm{raw}}}{(L_{\beta(q)}^y)^2}.
\]

Compute the physical residual:

\[
r_q
=
f_q-\phi_q^{\mathrm{raw}}-\psi_q^{\mathrm{raw}}.
\]

Apply symmetric projection:

\[
\phi_q^{\mathrm{proj}}
=
\phi_q^{\mathrm{raw}}+\frac12 r_q,
\]

\[
\psi_q^{\mathrm{proj}}
=
\psi_q^{\mathrm{raw}}+\frac12 r_q.
\]

Convert projected physical outputs back to unit quantities:

\[
\Phi_q^{\mathrm{proj}}
=
(L_{\alpha(q)}^x)^2
\phi_q^{\mathrm{proj}},
\]

\[
\Psi_q^{\mathrm{proj}}
=
(L_{\beta(q)}^y)^2
\psi_q^{\mathrm{proj}}.
\]

### 5.12 Segment-wise Green reconstruction

Perform Green reconstruction segment by segment.

For each segment \(I_\ell\), define reconstruction nodes

\[
\mathcal{T}_\ell
=
\{0\}
\cup
\{t_i:\text{interior valid local coordinates on }I_\ell\}
\cup
\{1\}.
\]

After sorting,

\[
0=t_{\ell,0}<t_{\ell,1}<\cdots<t_{\ell,N_\ell}=1.
\]

Endpoint values must be hard-coded to zero:

\[
\Phi_\ell(0)=\Phi_\ell(1)=0,
\]

\[
\Psi_\ell(0)=\Psi_\ell(1)=0.
\]

Do not evaluate the network at endpoints.

Use nonuniform composite trapezoid weights:

\[
w_0=\frac{t_1-t_0}{2},
\]

\[
w_i=\frac{t_{i+1}-t_{i-1}}{2},
\qquad
1\le i\le N-1,
\]

\[
w_N=\frac{t_N-t_{N-1}}{2}.
\]

Precompute segment nodes and trapezoid weights in geometry preprocessing.

GreenNet should be queried at required pairs:

\[
G_\ell(t_i,t_j)
=
\mathrm{GreenNet}(\mathrm{branch}_\ell,(t_i,t_j)).
\]

Use projected unit output only for reconstruction.

Do not use raw-output reconstruction loss.

### 5.13 Default energy consistency loss

The default consistency loss is physical valid-point face-energy consistency.

Define

\[
r(p)=u_\phi(p)-u_\psi(p).
\]

Then use

\[
\mathcal{L}_{\mathrm{energy}}
\approx
\int_\Omega
a(x,y)|\nabla r(x,y)|^2
\,dx\,dy.
\]

The discrete energy must be computed on the valid physical point graph.

For x-edges:

\[
E_x
=
\sum_{(p,p')\in\mathcal{E}_x}
a_{pp'}
\left|
\frac{r(p')-r(p)}{h_x}
\right|^2
h_xh_y.
\]

For y-edges:

\[
E_y
=
\sum_{(p,p')\in\mathcal{E}_y}
a_{pp'}
\left|
\frac{r(p')-r(p)}{h_y}
\right|^2
h_xh_y.
\]

Use current-code style area weight:

\[
h_xh_y.
\]

Use arithmetic face averaging:

\[
a_{pp'}=\frac12(a(p)+a(p')).
\]

### 5.14 Energy edge criterion

Do not use endpoint-validity alone to create energy edges.

Use the same axial connected-segment criterion.

For an x-edge between

\[
p=(x_i,y_j),
\qquad
p'=(x_{i+1},y_j),
\]

include the edge only if:

\[
p,p'\in\Omega
\quad\text{and}\quad
x\_segment\_id(p)=x\_segment\_id(p').
\]

For a y-edge between

\[
p=(x_i,y_j),
\qquad
p'=(x_i,y_{j+1}),
\]

include the edge only if:

\[
p,p'\in\Omega
\quad\text{and}\quad
y\_segment\_id(p)=y\_segment\_id(p').
\]

This prevents edges from connecting across holes, gaps, slits, or disconnected components on the same axial line.

### 5.15 Cross consistency exclusion

Cross consistency must be completely excluded in complex-geometry mode.

Do not compute it.

Do not log it.

Do not include a disabled flag.

Do not include it in metric summaries.

Do not leave a placeholder field.

In complex-geometry logs and metric dictionaries, there must be no key such as:

```text
loss_cross_consistency
cross_consistency
cross_consistency_status
```

or similar.

This is complete exclusion, not disabled logging.

---

## 6. Required Implementation Plan

### Step 1. Inspect current unit-square assumptions

Before modifying code, inspect existing files and identify where the unit-square assumptions occur.

Record the main findings in code comments, commit notes, or a short implementation note.

Look especially for:

- `torch.linspace(0.0, 1.0, ...)` being treated as a physical coordinate;
- rectangular tensor shape assumptions;
- `make_square_axial_lines`;
- unit-square boundary coordinate construction;
- smooth mask projection;
- cross consistency;
- rectangular-grid line slicing;
- transpose-based consistency logic.

### Step 2. Add complex-geometry metadata structures

Add a metadata representation for fixed complex geometry.

The metadata should include:

- valid physical coordinates;
- x/y segment ids;
- x/y local coordinates;
- horizontal segment endpoints and lengths;
- vertical segment endpoints and lengths;
- segment reconstruction nodes;
- trapezoid weights;
- valid x/y energy edge lists.

Use torch tensors where appropriate.

Avoid mixing boundary endpoints into the main valid point set \(P\).

### Step 3. Add GreenNet unit-interval helper utilities

Add or refactor utilities for:

- mapping physical interval coordinates to unit coordinates;
- sampling branch inputs on fixed unit grids;
- transforming \(a,a',b,c,f\);
- evaluating GreenNet on arbitrary \((t,\eta)\) pairs;
- avoiding double-counted Green kernel length scaling.

Keep backward compatibility with the current unit-square path where possible.

### Step 4. Add CouplingNet complex-geometry forward path

Add a complex-geometry forward path that:

- supports \(S_x\neq S_y\);
- processes x/y segments separately if necessary;
- shares function/geometry/trunk networks across x/y segments;
- uses segment-local \(t\) as trunk input;
- interprets the two raw outputs as physical quantities
  \(\phi_{\mathrm{raw}},\psi_{\mathrm{raw}}\);
- returns enough intermediate outputs for projection and reconstruction.

Do not force complex geometry into a rectangular `(B, 2, m, n)` tensor.

### Step 5. Implement hard symmetric projection

Implement projection in physical variables.

The projection path must:

1. receive physical raw \(\phi_{\mathrm{raw}},\psi_{\mathrm{raw}}\) directly;
2. compute physical residual \(f-\phi_{\mathrm{raw}}-\psi_{\mathrm{raw}}\);
3. apply the equal-half symmetric split in physical variables;
4. convert projected physical outputs to unit outputs with the axis-specific
   \(L_x^2\) and \(L_y^2\) factors;
5. use only those post-projection unit outputs for reconstruction.

Remove or bypass smooth masked projection and balance loss in complex-geometry mode.

### Step 6. Implement segment-wise Green reconstruction

Implement segment-wise reconstruction using:

- segment-local reconstruction nodes;
- hard-zero endpoint values;
- no network evaluation at endpoints;
- nonuniform trapezoid weights;
- arbitrary GreenNet pair queries;
- projected unit output only.

Do not add a raw-output reconstruction loss.

### Step 7. Implement physical valid-point energy loss

Implement valid-point graph energy loss.

Use:

- \(r=u_\phi-u_\psi\);
- x/y valid edge lists;
- same-segment edge criterion;
- area weight \(h_xh_y\);
- arithmetic face coefficient average.

The energy loss should operate on physical valid points, not on reference coordinates.

### Step 8. Remove cross consistency from complex-geometry mode

In complex-geometry mode:

- remove the computation path;
- remove metric keys;
- remove logging keys;
- remove summary keys;
- do not add disabled placeholders.

Do not include cross consistency anywhere in complex-geometry outputs.

### Step 9. Preserve unit-square behavior

Where possible, keep the existing unit-square behavior intact.

If shared code must be refactored, add tests proving that the unit-square path still works or document any unavoidable behavior change.

Prefer a mode/config switch such as:

```text
geometry_mode: unit_square | complex
```

or an equivalent design consistent with the existing config system.

---

## 7. Testing Requirements

Add tests covering the following cases.

### 7.1 GreenNet scaling tests

Test that:

- \(a_{\mathrm{unit}}\) is not multiplied by \(L\);
- \(a'_{\mathrm{unit}}=L a_s\);
- \(b_{\mathrm{unit}}=L b\);
- \(c_{\mathrm{unit}}=L^2 c\);
- \(f_{\mathrm{unit}}=L^2 f\);
- unit-coordinate reconstruction does not multiply the Green kernel by \(L\) again.

Use simple analytic functions and a non-unit length \(L\neq 1\).

### 7.2 Segment metadata tests

Test that:

- disconnected intervals are stored as separate segments;
- each valid point has the correct x/y segment id;
- local coordinates are in \([0,1]\);
- boundary endpoints are excluded from \(P\);
- reconstruction nodes include endpoints even though \(P\) does not.

### 7.3 CouplingNet raw-output convention tests

Test that:

- raw model channels are physical \(\phi_{\mathrm{raw}},\psi_{\mathrm{raw}}\);
- symmetric projection preserves the raw difference mode while enforcing balance;
- projected physical outputs convert to unit reconstruction outputs with the correct
  axis-specific \(L^2\) factors;
- unversioned legacy complex CouplingNet checkpoints are rejected as raw-unit contracts.

### 7.4 Projection tests

Test that hard symmetric projection enforces

\[
\phi_{\mathrm{proj}}+\psi_{\mathrm{proj}}=f
\]

up to numerical tolerance.

Test that:

- balance loss is not used;
- smooth masked projection is not used;
- projection is applied in physical variables.

### 7.5 Reconstruction tests

Test that:

- endpoint values are hard-zero;
- no network call is made at endpoints;
- trapezoid weights are correct for nonuniform nodes;
- trapezoid weights sum to \(1\) on \([0,1]\);
- reconstruction uses projected unit output;
- no raw-output reconstruction loss exists.

### 7.6 Energy edge tests

Construct a small synthetic geometry with disconnected intervals on the same row or column.

Test that:

- points in the same horizontal segment produce an x-edge;
- points in different horizontal segments do not produce an x-edge;
- points in the same vertical segment produce a y-edge;
- points in different vertical segments do not produce a y-edge;
- endpoint-validity alone is insufficient to create an edge.

### 7.7 Energy loss tests

Test that:

- energy loss uses physical valid points;
- x-differences divide by \(h_x\);
- y-differences divide by \(h_y\);
- area weight is \(h_xh_y\);
- face coefficient is arithmetic average.

### 7.8 Cross consistency absence tests

Test that, in complex-geometry mode:

- no `loss_cross_consistency` key exists;
- no `cross_consistency` key exists;
- no disabled status key exists;
- no placeholder field exists;
- the cross-consistency computation path is not called.

This is mandatory.

### 7.9 Unit-square regression tests

If the code already has unit-square tests, keep them passing.

If no such tests exist, add at least a minimal smoke test to confirm that the original unit-square path still works.

---

## 8. Prohibited Changes and Failure Modes

Do not make the following mistakes.

### 8.1 GreenNet failures

Do not:

- merge disconnected intervals;
- pass physical coordinates as GreenNet trunk coordinates;
- use physical \(a_s\) as `ap_vals`;
- forget \(b\), \(c\), or \(f\) scaling;
- double-count Green kernel length scaling;
- use physical quadrature weights inside unit-coordinate reconstruction unless intentionally converting the full formula.

### 8.2 CouplingNet failures

Do not:

- assume \(S_x=S_y\);
- force complex geometry into `(B,2,m,n)`;
- include boundary endpoints in \(P\);
- interpret raw output as physical \(\phi,\psi\);
- use axis one-hot in the geometry branch;
- include raw \(r_\ell\) in the geometry branch;
- use different horizontal and vertical branch networks by default;
- add source branch or Green response feature to the CouplingNet function branch by default.

### 8.3 Projection/loss failures

Do not:

- apply projection in unit variables without converting to physical variables;
- use balance loss in complex-geometry mode;
- use smooth masked projection in complex-geometry mode;
- use endpoint-validity alone for energy edges;
- use rectangular transpose-based cross consistency.

### 8.4 Logging/metric failures

Do not:

- compute cross consistency;
- log cross consistency;
- include a disabled cross-consistency flag;
- include a cross-consistency placeholder;
- include cross consistency in metric summaries.

Cross consistency must be absent in complex-geometry mode.

### 8.5 Repository-level failures

Do not:

- replace the whole repository with an unrelated implementation;
- remove working unit-square functionality unless explicitly necessary;
- silently change public config behavior without documenting it;
- leave untested complex-geometry numerical scaling paths.

---

## 9. Completion Criteria

The task is complete when all of the following are true.

### 9.1 GreenNet

- Connected intervals are mapped to unit intervals.
- Transformed \(a,a',b,c,f\) are implemented correctly.
- `ap_vals` represents \(a'_{\mathrm{unit}}\), not physical \(a_s\).
- GreenNet trunk coordinates are normalized \((t,\eta)\).
- GreenNet arbitrary pair evaluation is supported or wrapped cleanly.
- Unit reconstruction avoids double-counting \(L\).

### 9.2 CouplingNet

- Complex-geometry valid-point metadata exists.
- Field representation supports `(B,2,P)`.
- \(S_x\neq S_y\) is supported.
- Horizontal and vertical branches share networks.
- Function branch uses transformed \(a,b,c\).
- Geometry branch uses \([\operatorname{PE}(r),s^-,s^+,s_{\mathrm{mid}},L,L^2,1/L]\).
- Axis one-hot and raw \(r\) are not included.
- `product_fuser` is the default fusion mode.

### 9.3 Projection and reconstruction

- Hard symmetric projection is applied in physical variables.
- Smooth masked projection is not used.
- Balance loss is not used.
- Segment-wise Green reconstruction is implemented.
- Endpoint outputs are hard-zero.
- No endpoint network evaluation occurs.
- Reconstruction uses projected unit output only.

### 9.4 Loss and metrics

- Default complex-geometry loss is valid physical point energy consistency.
- Energy area weight is \(h_xh_y\).
- Face coefficient uses arithmetic average.
- Energy edges follow same axial connected-segment criterion.
- Cross consistency is completely absent from computation, metrics, logs, and summaries.

### 9.5 Tests

- GreenNet scaling tests pass.
- Segment metadata tests pass.
- Projection tests pass.
- Reconstruction tests pass.
- Energy edge tests pass.
- Energy loss tests pass.
- Cross consistency absence tests pass.
- Unit-square regression or smoke tests pass.

---

## 10. Implementation Note

If full optimization is too large for one pass, prioritize mathematical correctness and clear structure over speed.

A segment-wise loop is acceptable for the first implementation.

Do not introduce padding/masking or grouped segment batching unless the simpler segment-wise implementation is already correct and tested.

Document intentionally deferred optimizations in code comments or implementation notes.

---

## 11. Final Instruction

Implement the complex-geometry extension according to the two design documents.

The most important invariants are:

\[
\boxed{\text{GreenNet uses normalized unit intervals.}}
\]

\[
\boxed{\text{CouplingNet raw outputs are unit-domain source-like quantities.}}
\]

\[
\boxed{\text{Projection and energy consistency are physical-space operations.}}
\]

\[
\boxed{\text{Energy edges follow axial connected segments.}}
\]

\[
\boxed{\text{Cross consistency is completely excluded.}}
\]

Do not proceed by approximating complex geometry as a rectangular unit-square tensor. The complex-geometry path must explicitly use valid points, segment ids, segment-local coordinates, segment-wise reconstruction, and valid-edge energy loss.
