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

#### 5.6.1 Regular pentagram geometry convention

The canonical pentagram domain is a filled, hole-free, simple concave 10-gon.
It is centered at the origin, has circumradius \(R>0\), and fixes the first
vertex at \((0,R)\), so its orientation is \(\pi/2\). With
\(\varphi=(1+\sqrt{5})/2\), the alternating inner vertex radius is derived as

\[
r=\frac{R}{\varphi^2}.
\]

Neither the inner radius nor the orientation is a user option. The polygon
includes its central pentagon and is one connected Gmsh surface; do not use the
self-intersecting \(\{5/2\}\) star outline as the physical boundary.

`cli/make_pentagram_geometry.py` stores the counter-clockwise `(10, 2)`
`boundary_vertices` array in the geometry NPZ. Both Cartesian scanline
intersection and `examples/pentagram_gmsh.py` must consume those saved vertices
as their common source of truth. Valid points are strict polygon-interior points,
while segment endpoints are exact polygon intersections and may split at a
concave vertex. Reconstruction CSR arrays and edge lists remain segment-local.

### 5.7 CouplingNet raw-output convention

CouplingNet output contract v6 returns two directional reference-response fields.
The mandatory source branch uses the normalized physical profile

\[
\widetilde f=\frac{f_{\mathrm{phys}}}{A},
\qquad
A=\left(\int_0^1|f_{\mathrm{phys}}(s(t))|^2dt\right)^{1/2}.
\]

For a horizontal segment,

\[
P_k(t)=(L_k^x)^2A_k^x\widetilde P_k(t).
\]

For a vertical segment,

\[
Q_l(t)=(L_l^y)^2A_l^y\widetilde Q_l(t).
\]

The output shape remains `(B,2,P)`. The first channel is the x response \(P\),
and the second is the y response \(Q\). These are not physical source fields;
physical \(\phi,\psi\) are derived by pre-projection coordinate scaling. The
physical projection result is then pulled back to reference-response space. This
convention is mandatory, and complex CouplingNet checkpoints older than contract v6 must be
retrained.

The corresponding projected response and physical variables satisfy

\[
\Phi=L_x^2\phi_{\mathrm{phys}},
\qquad
\Psi=L_y^2\psi_{\mathrm{phys}}.
\]

### 5.8 Function branch input

The source branch is mandatory. The optional coefficient branch uses enabled
transformed channels in active order

\[
a,\quad b_{\mathrm{primary}},\quad b_{\mathrm{transverse}},\quad c.
\]

Do not add Green response features, source stencil lift, or an \(a'\) channel to
the complex CouplingNet branches. Green reconstruction keeps its separate
primary-operator coefficient contract.

For a horizontal segment,

\[
a_{\mathrm{unit}}^x(t)
=
a(x_k^-+L_k^x t,y_k),
\]

\[
b_{\mathrm{primary}}^x(t)
=
L_k^x b_x(x_k^-+L_k^x t,y_k),
\]

\[
b_{\mathrm{transverse}}^x(t)
=
L_k^x b_y(x_k^-+L_k^x t,y_k),
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
b_{\mathrm{primary}}^y(t)
=
L_l^y b_y(x_l,y_l^-+L_l^y t),
\]

\[
b_{\mathrm{transverse}}^y(t)
=
L_l^y b_x(x_l,y_l^-+L_l^y t),
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

The separate fixed-line transverse branch Fourier-encodes the globally normalized
coordinate \(\widehat r_\ell\):

\[
\operatorname{PE}(\widehat r_\ell)
=
[
\sin(\pi f_k \widehat r_\ell),
\cos(\pi f_k \widehat r_\ell)
]_{k=1}^{K}.
\]

The frequency set must follow the existing config schema. The default is

```text
[1, 2, 4, 8]
```

Do not include:

- axis one-hot;
- raw \(r_\ell\) scalar.

Output contract v6 also requires one shared pointwise transverse trunk with input

\[
\left[
t_\perp,
\log\frac{L_\perp}{L_{\mathrm{ref}}},
\log\frac{L_\parallel}{L_\perp},
\frac{4L_\parallel^2L_\perp^2}
{(L_\parallel^2+L_\perp^2)^2}
\right].
\]

For x/\(\Phi\), use \((L_\parallel,L_\perp,t_\perp)=(L_x,L_y,t_y)\).
For y/\(\Psi\), swap the axes. Require all lengths and the global reference
extent to be positive. Fuse the primary and transverse trunk embeddings using the
configured `product` or `product_fuser` mode.

Use `product_fuser` as the default branch fusion mode:

\[
[
h_{\mathrm{func}},
h_{\mathrm{geom}},
h_{\mathrm{func}}\odot h_{\mathrm{geom}}
]
\]

followed by a learned fuser.

### 5.11 Physical symmetric balance projection

Complex output contract v6 requires `balance_projection.enabled=true` and
`mode="physical_symmetric"`. Do not use balance loss, smooth masked projection,
response-space orthogonal projection, or response-preconditioned projection.

At valid point \(p_q\), define

\[
\sigma_x=(L_{\alpha(q)}^x)^2,
\qquad
\sigma_y=(L_{\beta(q)}^y)^2.
\]

Given base raw reference responses \(P_{0,q},Q_{0,q}\), first map them to
physical directional-source proposals:

\[
p_{0,q}=\frac{P_{0,q}}{\sigma_x},
\qquad
q_{0,q}=\frac{Q_{0,q}}{\sigma_y}.
\]

The optional `coupling_model.pre_projection_fusion` block modifies only the
physical difference before projection. Define

\[
d_{\mathrm{base},q}=p_{0,q}-q_{0,q},
\qquad
A_q=
\sqrt{\frac{(A^x_{\alpha(q)})^2+(A^y_{\beta(q)})^2}{2}},
\]

\[
A_{\mathrm{safe},q}=\max(A_q,\varepsilon),
\qquad
z_q=
\left[
\frac{d_{\mathrm{base},q}}{A_{\mathrm{safe},q}},
\frac{f_q}{A_{\mathrm{safe},q}}
\right].
\]

One pointwise nonlinear MLP produces a normalized residual:

\[
r_{\theta,q}=\operatorname{MLP}_{\theta}(z_q),
\qquad
d_{\mathrm{fused},q}
=
d_{\mathrm{base},q}+A_{\mathrm{safe},q}r_{\theta,q}.
\]

Construct the temporary physical pair directly from the physical source and
the fused difference:

\[
p_q=\frac{f_q+d_{\mathrm{fused},q}}{2},
\qquad
q_q=\frac{f_q-d_{\mathrm{fused},q}}{2}.
\]

Thus the optional block presents an exactly balanced pair to the physical
symmetric projection. It has one `2 -> hidden_dim -> ... -> 1` MLP, a fixed
identity skip through \(d_{\mathrm{base}}\), and no learned linear branch,
learned gate, combination mode, or direct coordinate/geometry/line-length
feature. The final MLP layer weight and bias are zero-initialized, so the
enabled path initially produces the same projected result as the disabled
path. If \(A_q=0\), force the physical residual to zero. The option must default
to disabled, is complex-only, uses no reference target, and does not change
output contract v6. Its config accepts only `enabled`, `hidden_dim`, `depth`,
and `eps`; retired split-fuser configs and enabled checkpoints are rejected.

Preserve the resulting physical difference \(d_q=p_q-q_q\) while imposing exact source
balance by the symmetric physical projection

\[
d_q=p_q-q_q,
\]

\[
\phi_q=\frac{f_q+d_q}{2},
\qquad
\psi_q=\frac{f_q-d_q}{2}.
\]

This projection is performed in physical source space and must enforce

\[
\phi_q+\psi_q=f_q,
\qquad
\phi_q-\psi_q=p_q-q_q.
\]

After projection, pull the physical fields back to the unit-interval response
variables consumed by Green reconstruction:

\[
\Phi_q=\sigma_x\phi_q,
\qquad
\Psi_q=\sigma_y\psi_q.
\]

Green reconstruction consumes \(\Phi,\Psi\) directly. Do not multiply the
projected responses by another axis-specific \(L^2\) factor.

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

Use projected response output only for reconstruction, with no additional
post-projection \(L^2\) scaling.

Do not use raw-output reconstruction loss.

### 5.13 Full-domain canonical energy consistency loss

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

Sum every same-segment x/y edge exactly once. Do not classify edges by line-length
jump or normalize regular and transition subsets separately.

Complete the valid-point bulk graph with both hard-zero endpoint edges of every
connected x/y segment. If \(p_\gamma\) is the nearest represented interior point
to endpoint \(\gamma\), \(d_\gamma\) is their physical distance, and
\(h_{\perp,\gamma}\) is the transverse grid measure, add

\[
E_{\partial}
=
\sum_\gamma
a(p_\gamma)r(p_\gamma)^2
\frac{h_{\perp,\gamma}}{d_\gamma}.
\]

The production objective is

\[
\mathcal L_{\mathrm{energy}}
=E_x+E_y+E_{\partial}.
\]

Report this as `loss_energy_consistency`, with bulk and boundary x/y
decompositions. Do not emit transition-specific metrics.

In complex mode, `sol` and optional `phi/psi` targets are evaluation-only. They
must not affect gradients, loss, scheduler, early stopping, or checkpoint
selection. Reject `best_rel_sol_checkpoint.enabled=true`.
`best_energy_checkpoint` selects `complex_coupling_model_best_energy.safetensors`
using validation `loss_energy_consistency`.

### 5.14 Optional relative split consistency

The derivative energy cannot detect a constant split mismatch. Complex v6 may
therefore replace the raw canonical-energy split objective with a source-normalized
energy-plus-value objective. For every sample \(b\), define

\[
r_b=u_{\phi,b}-u_{\psi,b},
\qquad
M_b=h_xh_y\sum_{p\in\Omega_h}r_b(p)^2,
\qquad
F_b=h_xh_y\sum_{p\in\Omega_h}f_b(p)^2,
\]

and

\[
D_{\mathrm{ref}}
=
\max(x_{\max}-x_{\min},y_{\max}-y_{\min}).
\]

Then

\[
\mathcal L_{\mathrm{split},b}
=
\frac{
E_{\mathrm{balanced},b}
+
\mu D_{\mathrm{ref}}^{-2}M_b
}{
F_b+\varepsilon
}.
\]

This option is controlled by
`coupling_training.relative_split_consistency`. It is disabled by the dataclass
default. If enabled, its `weight` scales the complete sample-mean relative split
objective. It uses only `rhs`, geometry, coefficients, and predictions; it must
not read `sol` or target directional fields.

### 5.15 Optional directional weak operator closure

Complex v6 may additionally require the common prediction

\[
u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi)
\]

to satisfy both directional weak equations. On every connected x-segment and
y-segment, use the geometry reconstruction CSR nodes, including both true
boundary endpoints. Use P1 trial/test values with homogeneous endpoint values.
For physical nodal test functions \(v_i\), assemble

\[
R_{x,i}
=
B_x(u_{\mathrm{pred}},v_i)-\langle\phi,v_i\rangle,
\qquad
B_x(u,v)
=
\int
\left(
a u_xv_x+b_xu_xv+\frac12cuv
\right)\,dx,
\]

\[
R_{y,i}
=
B_y(u_{\mathrm{pred}},v_i)-\langle\psi,v_i\rangle,
\qquad
B_y(u,v)
=
\int
\left(
a u_yv_y+b_yu_yv+\frac12cuv
\right)\,dy.
\]

Use physical element lengths \(L(t_{j+1}-t_j)\), transverse measure \(h_y\) on
x-segments and \(h_x\) on y-segments, P1 diffusion stiffness, nonsymmetric
convection, and consistent reaction mass matrices. Evaluate `a`, directional
`b`, and `c` directly at physical element midpoints instead of interpolating
branch samples.

With lumped directional nodal masses \(m_x,m_y\), define

\[
\mathcal L_{\mathrm{weak},b}
=
\frac{
\frac12
\left[
\sum_i\frac{R_{x,b,i}^2}{m_{x,i}+\varepsilon}
+
\sum_i\frac{R_{y,b,i}^2}{m_{y,i}+\varepsilon}
\right]
}{
F_b+\varepsilon
}.
\]

This option is controlled by `coupling_training.weak_operator_closure` and is
disabled by the dataclass default. Assemble the residual with differentiable
element gather/scatter operations; do not introduce a sparse-matrix dependency.
When enabled, add `weight * mean(L_weak,b)` to the selected split objective.
`best_physics_checkpoint` selects
`complex_coupling_model_best_physics.safetensors` using the total validation
reference-free objective. The energy and physics checkpoints are independent.

### 5.16 General connected-segment boundary energy

Complex v6 must complete the valid-point bulk edge energy with the physical P1
edge from both hard-zero endpoints of every connected segment to the nearest
represented interior node. For residual \(r=u_\phi-u_\psi\), add

\[
a_i r_i^2\frac{h_\perp}{d_i},
\]

where \(d_i\) is the physical endpoint distance, \(h_\perp=h_y\) for x-segments,
and \(h_\perp=h_x\) for y-segments. Use the nearest valid point's one-sided
diffusion coefficient. Segments with no represented interior node add no anchor.

The former `coupling_training.admissibility_gluing`, global self-trace loss, and
transition-only cross-axis carrier are retired. Old configs containing that key
must fail fast. The canonical energy uses no 2D mesh, matrix solve, `sol`, or
target directional source.

### 5.17 Energy edge criterion

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

### 5.18 Cross consistency exclusion

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

Geometry-specific provenance may be stored as optional metadata. For the
regular pentagram this includes `domain_type`, `outer_radius`, derived
`inner_radius`, `center`, `orientation_angle`, `fill_rule`, `has_hole`, and
`boundary_vertices`. FEniCSx summary generation must preserve these values, but
they do not alter the required tensor schema used by existing geometries.

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
- uses one shared four-input pointwise transverse length-context trunk;
- interprets the two raw outputs as directional responses \(P,Q\);
- returns enough intermediate outputs for projection and reconstruction.

Do not force complex geometry into a rectangular `(B, 2, m, n)` tensor.

### Step 5. Implement physical symmetric balance projection

Implement the pre-scale, physical projection, and post-projection pull-back path.

The projection path must:

1. receive raw reference responses \(P,Q\);
2. use positive \(L_x^2,L_y^2\) response scales;
3. compute physical proposals \(p=P/L_x^2\), \(q=Q/L_y^2\);
4. preserve \(p-q\) while enforcing exact \(\phi+\psi=f\);
5. pull back with \(\Phi=L_x^2\phi\), \(\Psi=L_y^2\psi\);
6. pass projected responses directly to reconstruction.

Remove or bypass smooth masked projection and balance loss in complex-geometry mode.

### Step 6. Implement segment-wise Green reconstruction

Implement segment-wise reconstruction using:

- segment-local reconstruction nodes;
- hard-zero endpoint values;
- no network evaluation at endpoints;
- nonuniform trapezoid weights;
- arbitrary GreenNet pair queries;
- projected response output only, with no extra \(L^2\) multiplication.

Do not add a raw-output reconstruction loss.

### Step 7. Implement full-domain canonical physical energy loss

Implement valid-point graph energy loss.

Use:

- \(r=u_\phi-u_\psi\);
- x/y valid edge lists;
- same-segment edge criterion;
- area weight \(h_xh_y\);
- arithmetic face coefficient average;
- both hard-zero endpoint edges of every connected segment;
- no regular/transition edge partition or geometry-dependent reweighting.

The energy loss should operate on physical valid points, not on reference coordinates.
Use validation `loss_energy_consistency` for the optional best-energy checkpoint.

### Step 8. Add optional value and weak consistency objectives

Implement `relative_split_consistency` as the per-sample source-normalized sum
of canonical bulk-plus-boundary energy and domain-scaled split mass. Implement
`weak_operator_closure` using shared `u_pred`, connected-segment P1 nodal test
functions, direct midpoint coefficient evaluation, hard-zero endpoints, and
differentiable element gather/scatter. Save the optional best-physics checkpoint
from the total validation reference-free objective. Reference `sol/phi/psi`
values must remain detached evaluation metrics.

### Step 9. Remove cross consistency from complex-geometry mode

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

- raw model channels are responses \(P,Q\);
- output scaling is \(L_x^2A_x\) and \(L_y^2A_y\);
- the shared transverse trunk receives the four length-context features with x/y role swap;
- disabled pre-projection fusion leaves the v6 state/output path unchanged;
- enabled pre-projection fusion has one two-input nonlinear MLP, a fixed
  identity skip, and a zero-initialized final layer;
- enabled pre-projection fusion constructs an exactly balanced physical pair
  from `rhs` and the fused difference;
- zero source amplitude gives an exact zero fusion correction;
- unversioned and version 5-or-older complex CouplingNet checkpoints are rejected by
  output contract v6.

### 7.4 Projection tests

Test that physical symmetric projection enforces

\[
\phi+\psi=f
\]

up to numerical tolerance.

Test that:

- balance loss is not used;
- smooth masked projection is not used;
- pre-projection scaling produces \(p_0=P_0/L_x^2\) and \(q_0=Q_0/L_y^2\);
- optional fusion uses only normalized physical difference/source inputs and
  produces the difference presented to projection;
- optional fusion has no learned linear branch, learned gate, or direct
  geometry/length input;
- the fused physical difference is preserved by projection;
- post-projection pull-back produces \(\Phi=L_x^2\phi\), \(\Psi=L_y^2\psi\);
- very short positive segment lengths remain finite in float64.

### 7.5 Reconstruction tests

Test that:

- endpoint values are hard-zero;
- no network call is made at endpoints;
- trapezoid weights are correct for nonuniform nodes;
- trapezoid weights sum to \(1\) on \([0,1]\);
- reconstruction uses projected response output without another \(L^2\) factor;
- no raw-output reconstruction loss exists.

### 7.6 Energy edge tests

Construct a small synthetic geometry with disconnected intervals on the same row or column.

Test that:

- points in the same horizontal segment produce an x-edge;
- points in different horizontal segments do not produce an x-edge;
- points in the same vertical segment produce a y-edge;
- points in different vertical segments do not produce a y-edge;
- length-square log jumps classify transition edges correctly;
- regular and transition groups are independently normalized;
- an empty group falls back to the unweighted energy;
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
- interpret raw response output as physical \(\phi,\psi\);
- use axis one-hot in the geometry branch;
- include raw \(r_\ell\) in the geometry branch;
- use different horizontal and vertical branch networks by default;
- omit the mandatory normalized physical source branch;
- omit the four-feature pointwise cross-axis length context;
- add a Green response feature or source stencil lift to the complex branch path.

### 8.3 Projection/loss failures

Do not:

- apply complex symmetric, response-preconditioned, or geometry-weighted projection;
- multiply projected responses by another axis-specific \(L^2\) factor;
- use balance loss in complex-geometry mode;
- use smooth masked projection in complex-geometry mode;
- use endpoint-validity alone for energy edges;
- select complex checkpoints with reference `rel_sol`;
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
- Geometry branch uses \([s^-,s^+,s_{\mathrm{mid}},L,L^2,1/L]\) and a separate fixed-line transverse PE branch.
- Shared pointwise transverse trunk receives cross-axis coordinate and length context.
- Axis one-hot and raw \(r\) are not included.
- `product_fuser` is the default fusion mode.

### 9.3 Projection and reconstruction

- Physical symmetric projection is the only complex v6 mode.
- Raw reference responses are divided by length squares before projection.
- Projected physical fields are multiplied by length squares after projection.
- Smooth masked projection is not used.
- Balance loss is not used.
- Segment-wise Green reconstruction is implemented.
- Endpoint outputs are hard-zero.
- No endpoint network evaluation occurs.
- Reconstruction uses projected responses directly without another \(L^2\) factor.

### 9.4 Loss and metrics

- The base complex-geometry split loss is full-domain canonical valid-point energy consistency.
- Optional relative split consistency adds source-normalized value consistency.
- Optional directional weak closure uses common `u_pred` and full `a,bx/by,c/2`.
- Energy area weight is \(h_xh_y\).
- Face coefficient uses arithmetic average.
- Energy edges follow same axial connected-segment criterion.
- Canonical bulk and boundary x/y components are reported separately.
- Best-energy and best-physics checkpoints use independent reference-free criteria.
- Reference `sol/phi/psi` never enters a loss or checkpoint criterion.
- Cross consistency is completely absent from computation, metrics, logs, and summaries.

### 9.5 Tests

- GreenNet scaling tests pass.
- Segment metadata tests pass.
- Projection tests pass.
- Reconstruction tests pass.
- Energy edge tests pass.
- Energy loss tests pass.
- Relative split consistency tests pass.
- Directional weak closure matrix, endpoint, and manufactured-residual tests pass.
- Independent best-energy and best-physics checkpoint tests pass.
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
