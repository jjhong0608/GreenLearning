# CouplingNet Design for Complex Geometry

## Segment-Local Reference Output, Hard Symmetric Projection, and Physical Energy Consistency

---

## 1. Purpose and Scope

This document specifies the `CouplingNet`-side design for complex geometry.

The `GreenNet` unit-interval normalization has already been documented separately. Therefore, this document does **not** restate the full `GreenNet` design. Instead, it focuses on how `CouplingNet` should be defined and coupled with the already normalized `GreenNet` framework in a complex two-dimensional geometry.

The purpose of this document is to fix the following design decisions:

1. how the complex 2D domain is represented;
2. how axial segments are defined;
3. how `CouplingNet` raw outputs are interpreted;
4. how function-branch inputs are constructed;
5. how the geometry/transverse branch is constructed;
6. how hard symmetric projection is applied;
7. how segment-wise Green reconstruction is performed;
8. how the default physical energy consistency loss is computed;
9. how energy edges are selected;
10. how validation and visualization should be interpreted;
11. how cross consistency is treated.

The primary objective is to avoid future confusion about coordinate convention, tensor shape, projection location, and loss evaluation.

---

## 2. Global Assumptions

### 2.1 Fixed geometry across samples

All samples are assumed to share the same complex geometry \\(\Omega\\).

Thus, the following objects are batch-shared:

\\[
\Omega,
\qquad
\mathcal{P}_\Omega,
\qquad
\{I_k^x\}_{k=1}^{S_x},
\qquad
\{I_l^y\}_{l=1}^{S_y},
\\]

together with all segment endpoints, valid-point indices, local coordinates, and segment-local reconstruction nodes.

The batch dimension is used for variation in source/coefficient samples, not for variation in geometry.

\\[
\boxed{
\text{The geometry metadata is shared across all samples in a batch.}
}
\\]

### 2.2 No global 2D reference mapping

The full 2D domain \\(\Omega\\) is **not** mapped to a global reference domain.

However, each one-dimensional axial segment is parameterized by a segment-local reference coordinate:

\\[
t\in[0,1].
\\]

Thus, the design uses:

\\[
\boxed{
\text{No global 2D reference mapping}
}
\\]

but

\\[
\boxed{
\text{Segment-local 1D reference coordinates}
}
\\]

for axial branch/trunk/reconstruction operations.

---

## 3. Geometry and Valid Point Representation

### 3.1 Bounding box grid

Let \\(\Omega\subset\mathbb{R}^2\\) be the complex physical domain. We assume that a bounding box

\\[
B=[x_{\min},x_{\max}]\times[y_{\min},y_{\max}]
\\]

is given such that

\\[
\Omega\subset B.
\\]

A Cartesian grid is defined on \\(B\\). Among the grid points, those that lie inside \\(\Omega\\) are collected as valid interior physical points:

\\[
\mathcal{P}_\Omega
=
\{p_q=(x_q,y_q)\}_{q=1}^{P}.
\\]

### 3.2 Flattened valid-point field representation

In the unit-square setting, a field can be stored as a rectangular tensor such as

\\[
(B,2,m,n).
\\]

This is not appropriate for a complex geometry because the interior point set is generally not rectangular.

Therefore, the complex-geometry `CouplingNet` field representation is

\\[
\boxed{
\mathrm{flux}\in\mathbb{R}^{B\times 2\times P}.
}
\\]

The two components are interpreted as

\\[
\mathrm{flux}[:,0,q]=\phi(p_q),
\\]

\\[
\mathrm{flux}[:,1,q]=\psi(p_q).
\\]

Here \\(B\\) is the batch size, and \\(P\\) is the number of valid interior physical points.

### 3.3 Boundary points are not included in \\(P\\)

Boundary points are not included in the valid point set \\(P\\).

\\[
\boxed{
P=\text{number of interior valid trunk/intersection points}.
}
\\]

Boundary endpoints are used only as segment-wise quadrature nodes for Green reconstruction. They are not included in the main flattened field tensor.

---

## 4. Axial Segment Decomposition

### 4.1 Connected intervals

Each horizontal or vertical axial line may intersect \\(\Omega\\) in one or more connected intervals.

If an axial line intersects \\(\Omega\\) in multiple disconnected intervals, each connected interval is treated as a separate segment.

### 4.2 Horizontal segments

A horizontal segment is written as

\\[
I_k^x=[x_k^-,x_k^+],
\qquad
y=y_k,
\\]

with length

\\[
L_k^x=x_k^+-x_k^-.
\\]

The segment-local coordinate is

\\[
t=\frac{x-x_k^-}{L_k^x},
\qquad
x=x_k^-+L_k^x t,
\qquad
t\in[0,1].
\\]

### 4.3 Vertical segments

A vertical segment is written as

\\[
I_l^y=[y_l^-,y_l^+],
\qquad
x=x_l,
\\]

with length

\\[
L_l^y=y_l^+-y_l^-.
\\]

The segment-local coordinate is

\\[
t=\frac{y-y_l^-}{L_l^y},
\qquad
y=y_l^-+L_l^y t,
\qquad
t\in[0,1].
\\]

### 4.4 Valid point metadata

Each valid point \\(p_q=(x_q,y_q)\\) belongs to one horizontal segment and one vertical segment.

The following metadata must be stored:

| Tensor | Shape | Meaning |
|---|---:|---|
| `coords_valid` | `(P, 2)` | Interior valid physical points |
| `x_segment_id` | `(P,)` | Horizontal segment id for each valid point |
| `y_segment_id` | `(P,)` | Vertical segment id for each valid point |
| `x_local_t` | `(P,)` | Local coordinate in the horizontal segment |
| `y_local_t` | `(P,)` | Local coordinate in the vertical segment |

For a point \\(p_q=(x_q,y_q)\\), if

\\[
\alpha(q)=x\_segment\_id(q),
\qquad
\beta(q)=y\_segment\_id(q),
\\]

then

\\[
t_q^x
=
\frac{x_q-x_{\alpha(q)}^-}{L_{\alpha(q)}^x},
\\]

\\[
t_q^y
=
\frac{y_q-y_{\beta(q)}^-}{L_{\beta(q)}^y}.
\\]

---

## 5. CouplingNet Raw-Output Convention

### 5.1 Raw outputs are physical directional fields

`CouplingNet` keeps two axis-conditioned outputs, but both are interpreted directly
in physical source space:

\\[
\phi_k^{\mathrm{raw}}(t),
\qquad
\psi_l^{\mathrm{raw}}(t).
\\]

The horizontal path predicts \\(\phi_{\mathrm{raw}}\\), and the vertical path predicts
\\(\psi_{\mathrm{raw}}\\). No \\(L^2\\) conversion is applied before physical balance
projection. The axis-specific unit quantities are created only after projection for
Green reconstruction:

\\[
\Phi^{\mathrm{proj}}=L_x^2\phi^{\mathrm{proj}},
\qquad
\Psi^{\mathrm{proj}}=L_y^2\psi^{\mathrm{proj}}.
\\]

### 5.2 `axis_1d_trunk=true`

The default architecture uses

```text
axis_1d_trunk = true
```

In this complex-geometry design, the 1D trunk input is the segment-local coordinate

\\[
t\in[0,1].
\\]

Thus,

\\[
\boxed{
\text{Trunk input} = t.
}
\\]

The trunk does not receive the physical axial coordinate directly in this design.

---

## 6. Function Branch Input

### 6.1 Function branch channels

The source branch is mandatory. It receives the segment-local physical source
profile normalized by

\\[
A=\left(\int_0^1 |f_{\mathrm{phys}}(s(t))|^2\,dt\right)^{1/2},
\\]

and the model output is multiplied by \\(A\\) to restore physical source amplitude.
The optional coefficient branch uses enabled channels from
\\(a,b_{\mathrm{primary}},b_{\mathrm{transverse}},c\\). The Green response feature,
source stencil lift, and \\(a'\\) coefficient channel remain excluded from the complex
CouplingNet branch. Coefficients are transformed into the segment-local
reference-domain operator convention.

### 6.2 Horizontal transformed coefficients

For a horizontal segment

\\[
I_k^x=[x_k^-,x_k^+],
\qquad
L_k^x=x_k^+-x_k^-,
\qquad
y=y_k,
\\]

the physical coordinate is

\\[
x=x_k^-+L_k^x t.
\\]

The transformed branch inputs are

\\[
a_{\mathrm{unit}}^x(t)
=
a(x_k^-+L_k^x t,y_k),
\\]

\\[
b_{\mathrm{unit}}^x(t)
=
L_k^x b_x(x_k^-+L_k^x t,y_k),
\\]

\\[
c_{\mathrm{unit}}^x(t)
=
(L_k^x)^2 c(x_k^-+L_k^x t,y_k).
\\]

### 6.3 Vertical transformed coefficients

For a vertical segment

\\[
I_l^y=[y_l^-,y_l^+],
\qquad
L_l^y=y_l^+-y_l^-,
\qquad
x=x_l,
\\]

the physical coordinate is

\\[
y=y_l^-+L_l^y t.
\\]

The transformed branch inputs are

\\[
a_{\mathrm{unit}}^y(t)
=
a(x_l,y_l^-+L_l^y t),
\\]

\\[
b_{\mathrm{unit}}^y(t)
=
L_l^y b_y(x_l,y_l^-+L_l^y t),
\\]

\\[
c_{\mathrm{unit}}^y(t)
=
(L_l^y)^2 c(x_l,y_l^-+L_l^y t).
\\]

### 6.4 Branch sampling grid

The function branch uses a fixed-size uniform sampling grid on the segment-local reference interval:

\\[
\tau_j=\frac{j}{m-1},
\qquad
j=0,\dots,m-1.
\\]

This grid is used only to construct fixed-size branch inputs.

It does not need to coincide with the Green reconstruction quadrature nodes.

### 6.5 Axis-wise current-code style

The implementation should preserve the current-code style of axis-wise processing as much as possible.

Conceptually,

\\[
\text{axis}=0
\quad\text{corresponds to horizontal/x-segments},
\\]

\\[
\text{axis}=1
\quad\text{corresponds to vertical/y-segments}.
\\]

However, in complex geometry, the number of horizontal and vertical segments may differ:

\\[
S_x\neq S_y.
\\]

Therefore, storage may use separate tensors:

\\[
a_x,b_x,c_x\in\mathbb{R}^{B\times S_x\times m},
\\]

\\[
a_y,b_y,c_y\in\mathbb{R}^{B\times S_y\times m}.
\\]

The model should still process the two axes in an axis-wise manner internally.

### 6.6 Shared horizontal/vertical network

The horizontal and vertical branches use the same networks.

\\[
\boxed{
B_{\mathrm{func}}^x=B_{\mathrm{func}}^y.
}
\\]

Horizontal and vertical segments are interpreted through the same canonical segment representation.

---

## 7. Geometry / Transverse Branch

### 7.1 Canonical notation

For both horizontal and vertical segments, introduce a canonical segment notation:

\\[
I_\ell=[s_\ell^-,s_\ell^+],
\qquad
r=r_\ell,
\qquad
L_\ell=s_\ell^+-s_\ell^-.
\\]

For a horizontal segment,

\\[
s=x,
\qquad
r=y.
\\]

For a vertical segment,

\\[
s=y,
\qquad
r=x.
\\]

### 7.2 Geometry feature vector

The geometry/transverse branch feature is

\\[
\boxed{
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
].
}
\\]

where

\\[
s_{\ell,\mathrm{mid}}
=
\frac{s_\ell^-+s_\ell^+}{2}.
\\]

### 7.3 Positional encoding of \\(r_\ell\\)

Only the transverse coordinate \\(r_\ell\\) is Fourier-encoded.

\\[
\operatorname{PE}(r_\ell)
=
[
\sin(\pi f_1 r_\ell),
\cos(\pi f_1 r_\ell),
\dots,
\sin(\pi f_K r_\ell),
\cos(\pi f_K r_\ell)
].
\\]

The frequency set \\(\{f_k\}_{k=1}^{K}\\) follows the existing config schema.

The default value is

\\[
f_k=[1,2,4,8].
\\]

### 7.4 Excluded geometry features

The following are not included in the default geometry branch:

\\[
\boxed{
\text{No axis one-hot.}
}
\\]

\\[
\boxed{
\text{No raw }r_\ell\text{ scalar.}
}
\\]

Raw \\(r_\ell\\) may be added later as an ablation, but it is not part of the default design.

### 7.5 Geometry/function branch fusion

The function branch embedding and geometry branch embedding are fused using `product_fuser` by default.

Let

\\[
h_\ell^{\mathrm{func}}
=
B_{\mathrm{func}}(F_\ell),
\\]

\\[
h_\ell^{\mathrm{geom}}
=
B_{\mathrm{geom}}(g_\ell).
\\]

The fused branch feature is

\\[
h_\ell
=
F_{\mathrm{fuser}}
\left(
[
h_\ell^{\mathrm{func}},
h_\ell^{\mathrm{geom}},
h_\ell^{\mathrm{func}}\odot h_\ell^{\mathrm{geom}}
]
\right).
\\]

Thus,

\\[
\boxed{
\texttt{product\_fuser is the default branch fusion mode.}
}
\\]

---

## 8. Physical Balance Projection

### 8.1 Projection policy

Hard symmetric projection is the default. Response-preconditioned projection is
available only as an opt-in complex-geometry ablation.

\\[
\boxed{
\text{Symmetric by default; response-preconditioned as opt-in.}
}
\\]

No balance loss is used.

No smooth masked projection is used.

No reference solution or reference split field is used by either projection.

### 8.2 Physical raw output

At a valid point \\(p_q=(x_q,y_q)\\), let

\\[
\alpha(q)=x\_segment\_id(q),
\qquad
\beta(q)=y\_segment\_id(q).
\\]

The model directly returns the physical raw outputs

\\[
\phi_q^{\mathrm{raw}},
\qquad
\psi_q^{\mathrm{raw}}.
\\]

Segment lengths remain available to the geometry branch, but they do not rescale
these raw fields before projection.

### 8.3 Physical residual

The physical residual is

\\[
r_q
=
f_q-\phi_q^{\mathrm{raw}}-\psi_q^{\mathrm{raw}}.
\\]

### 8.4 Symmetric projection in physical variables

The projected physical quantities are

\\[
\phi_q^{\mathrm{proj}}
=
\phi_q^{\mathrm{raw}}+\frac12 r_q,
\\]

\\[
\psi_q^{\mathrm{proj}}
=
\psi_q^{\mathrm{raw}}+\frac12 r_q.
\\]

### 8.5 Response-preconditioned projection

For the optional RPS ablation, define

\[
\sigma_x=(L_{alpha(q)}^x)^2,
\qquad
\sigma_y=(L_{eta(q)}^y)^2,
\]

\[
d_0=\frac{\sigma_y-\sigma_x}{\sigma_x+\sigma_y}f_q,
\qquad
\kappa=\frac{4\sigma_x\sigma_y}{(\sigma_x+\sigma_y)^2},
\]

and

\[
d_{\mathrm{RPS}}
=d_0+\kappa
(\phi_q^{\mathrm{raw}}-\psi_q^{\mathrm{raw}}).
\]

The projected physical quantities are

\[
\phi_q^{\mathrm{proj}}=\frac12(f_q+d_{\mathrm{RPS}}),
\qquad
\psi_q^{\mathrm{proj}}=\frac12(f_q-d_{\mathrm{RPS}}).
\]

Equal segment lengths give \(d_0=0\) and \(\kappa=1\), exactly recovering the
symmetric projection. RPS does not use reference solution or split targets.

### 8.6 Physical projected output to unit projected output

For Green reconstruction, the projected physical quantities are converted back to unit quantities:

\\[
\Phi_q^{\mathrm{proj}}
=
(L_{\alpha(q)}^x)^2
\phi_q^{\mathrm{proj}},
\\]

\\[
\Psi_q^{\mathrm{proj}}
=
(L_{\beta(q)}^y)^2
\psi_q^{\mathrm{proj}}.
\\]

### 8.7 Smooth masked projection is excluded

Smooth masked projection is not used in the complex-geometry design.

The current smooth masked projection is tied to unit-square boundary expressions such as

\\[
x(1-x),
\qquad
y(1-y),
\qquad
\sin(\pi x),
\qquad
\sin(\pi y),
\\]

which are not appropriate for general complex geometry.

---

## 9. Segment-Wise Green Reconstruction

### 9.1 Segment-wise loop

Green reconstruction is performed segment by segment.

This is the default strategy because different segments may contain different numbers of valid points.

\\[
\boxed{
\text{Segment-wise loop is the default reconstruction strategy.}
}
\\]

### 9.2 Reconstruction node set

For each segment \\(I_\ell\\), define its reconstruction node set as

\\[
\mathcal{T}_\ell
=
\{0\}
\cup
\{t_i:\text{interior valid points on }I_\ell\}
\cup
\{1\}.
\\]

After sorting,

\\[
0=t_{\ell,0}<t_{\ell,1}<\cdots<t_{\ell,N_\ell}=1.
\\]

The nodes \\(0\\) and \\(1\\) are boundary endpoints.

### 9.3 Endpoint handling

Endpoint values are hard-coded as zero.

\\[
\Phi_\ell(0)=\Phi_\ell(1)=0,
\\]

\\[
\Psi_\ell(0)=\Psi_\ell(1)=0.
\\]

The network is not evaluated at endpoints.

\\[
\boxed{
\text{No network evaluation is performed at segment endpoints.}
}
\\]

### 9.4 Nonuniform composite trapezoid rule

For nodes

\\[
0=t_0<t_1<\cdots<t_N=1,
\\]

the nonuniform composite trapezoid weights are

\\[
w_0=\frac{t_1-t_0}{2},
\\]

\\[
w_i=\frac{t_{i+1}-t_{i-1}}{2},
\qquad
1\le i\le N-1,
\\]

\\[
w_N=\frac{t_N-t_{N-1}}{2}.
\\]

Segment-wise nodes and weights are precomputed and stored during geometry preprocessing.

\\[
\boxed{
\text{Segment nodes and trapezoid weights are precomputed.}
}
\\]

### 9.5 GreenNet pair query

For each segment, GreenNet is queried at the required pairs:

\\[
G_\ell(t_i,t_j)
=
\mathrm{GreenNet}(\mathrm{branch}_\ell,(t_i,t_j)).
\\]

Uniform-grid Green kernel interpolation is not the default strategy.

### 9.6 Reconstruction formula

For a segment-local unit source-like quantity \\(F_\ell(t_j)\\), the reconstruction is

\\[
u_\ell(t_i)
\approx
\sum_{j=0}^{N_\ell}
w_{\ell,j}
G_\ell(t_i,t_j)
F_\ell(t_j).
\\]

The source-like values at endpoints are zero, so endpoint contributions vanish.

### 9.7 Projected unit output only

Green reconstruction uses only the projected unit output.

Raw physical output reconstruction loss is not used.

\\[
\boxed{
\text{Reconstruction uses projected unit output only.}
}
\\]

---

## 10. Default Energy Consistency Loss

### 10.1 Residual

At each valid physical point \\(p\\), define

\\[
r(p)=u_\phi(p)-u_\psi(p).
\\]

The default consistency loss is based on the physical energy of this residual.

### 10.2 Physical energy

The target continuous form is

\\[
\mathcal{L}_{\mathrm{energy}}
\approx
\int_\Omega
a(x,y)|\nabla r(x,y)|^2
\,dx\,dy.
\\]

This follows the current-code energy-consistency philosophy: penalize the face-based physical energy of the represented-solution residual.

### 10.3 Discrete edge energy

Let \\(\mathcal{E}_x\\) be the set of valid x-direction edges and \\(\mathcal{E}_y\\) be the set of valid y-direction edges.

For an x-edge \\((p,p')\\),

\\[
D_x r(p,p')
=
\frac{r(p')-r(p)}{h_x}.
\\]

For a y-edge \\((p,p')\\),

\\[
D_y r(p,p')
=
\frac{r(p')-r(p)}{h_y}.
\\]

The energy is

\\[
E_x
=
\sum_{(p,p')\in\mathcal{E}_x}
a_{pp'}
\left|
D_x r(p,p')
\right|^2
h_xh_y,
\\]

\\[
E_y
=
\sum_{(p,p')\in\mathcal{E}_y}
a_{pp'}
\left|
D_y r(p,p')
\right|^2
h_xh_y.
\\]

Thus,

\\[
\mathcal{L}_{\mathrm{energy}}
=
E_x+E_y.
\\]

### 10.4 Area weight

The area weight follows the current-code structure:

\\[
\boxed{
\text{Area weight}=h_xh_y.
}
\\]

### 10.5 Face coefficient

The face coefficient uses arithmetic averaging:

\\[
a_{pp'}
=
\frac12(a(p)+a(p')).
\\]

---

## 11. Energy Edge Criterion

### 11.1 Same-segment rule

Energy edges are selected by the same axial connected-segment criterion.

The rule is stronger than simply checking whether both endpoints are valid.

### 11.2 x-direction edge

Let

\\[
p=(x_i,y_j),
\qquad
p'=(x_{i+1},y_j).
\\]

Then the x-edge \\((p,p')\\) is used if and only if:

1. both points are valid interior points;
2. both points belong to the same horizontal connected segment.

That is,

\\[
p,p'\in\Omega
\quad\text{and}\quad
x\_segment\_id(p)=x\_segment\_id(p').
\\]

Then

\\[
\boxed{
(p,p')\in\mathcal{E}_x.
}
\\]

If

\\[
x\_segment\_id(p)\neq x\_segment\_id(p'),
\\]

then the x-edge is excluded.

### 11.3 y-direction edge

Let

\\[
p=(x_i,y_j),
\qquad
p'=(x_i,y_{j+1}).
\\]

Then the y-edge \\((p,p')\\) is used if and only if:

1. both points are valid interior points;
2. both points belong to the same vertical connected segment.

That is,

\\[
p,p'\in\Omega
\quad\text{and}\quad
y\_segment\_id(p)=y\_segment\_id(p').
\\]

Then

\\[
\boxed{
(p,p')\in\mathcal{E}_y.
}
\\]

If

\\[
y\_segment\_id(p)\neq y\_segment\_id(p'),
\\]

then the y-edge is excluded.

### 11.4 Interpretation

The energy graph follows the axial connected-segment structure of the domain.

\\[
\boxed{
\text{Energy edges follow the axial connected-segment graph.}
}
\\]

This prevents the energy loss from connecting points across holes, slits, or disconnected components of the same axial line.

---

## 12. Cross Consistency Policy

Cross consistency is completely excluded from the complex-geometry `CouplingNet` design.

This means:

- no cross-consistency loss is computed;
- no cross-consistency metric is computed;
- no cross-consistency logging entry is produced;
- no disabled flag is logged;
- no cross-consistency summary field is produced.

\\[
\boxed{
\text{Cross consistency is not part of the complex-geometry CouplingNet design.}
}
\\]

Implementation should avoid including any `cross_consistency` field in complex-geometry logs or metric summaries.

---

## 13. Validation and Visualization Policy

### 13.1 Validation metrics

Validation metrics should follow the current `CouplingTrainer` metric philosophy as much as possible, except that cross consistency is completely removed.

The main validation quantities should include:

- total loss;
- energy consistency loss;
- relative solution metric, if available;
- relative flux metric, if target flux is available.

No cross-consistency metric or logging entry is produced.

### 13.2 Visualization output

Raw physical output is not the default visualization output.

\\[
\boxed{
\text{Raw physical output is archived for audit, not used as the default figure.}
}
\\]

The default visualization should use projected physical quantities:

\\[
\phi_{\mathrm{proj,phys}},
\qquad
\psi_{\mathrm{proj,phys}}.
\\]

The reconstructed solutions should be stored and visualized at physical valid points:

\\[
u_\phi(p_q),
\qquad
u_\psi(p_q),
\qquad
p_q\in\Omega.
\\]

---

## 14. Implementation Checklist

### Geometry and points

- [ ] Geometry is shared across all samples.
- [ ] `coords_valid` contains only interior valid points.
- [ ] Boundary endpoints are not included in \(P\).
- [ ] Each valid point has `x_segment_id` and `y_segment_id`.
- [ ] Each valid point has `x_local_t` and `y_local_t`.

### Segment metadata

- [ ] Horizontal segments store \(x^-\), \(x^+\), \(y\), and \(L^x\).
- [ ] Vertical segments store \(y^-\), \(y^+\), \(x\), and \(L^y\).
- [ ] Disconnected intervals are stored as separate segments.

### Function branches

- [ ] Mandatory source branch uses normalized physical source profiles.
- [ ] Source amplitude is restored on the two physical raw outputs.
- [ ] Coefficient branch follows active \([a,b_{\mathrm{primary}},b_{\mathrm{transverse}},c]\) channels.
- [ ] x/y source and coefficient paths use shared networks.
- [ ] Axis-wise processing is preserved internally.

### Geometry branch

- [ ] Geometry feature is \([s^-,s^+,s_{\mathrm{mid}},L,L^2,1/L]\).
- [ ] Fixed-line transverse Fourier features use a separate shared branch.
- [ ] Raw \(r\) is not included.
- [ ] Axis one-hot is not included.
- [ ] Fourier frequencies follow config.
- [ ] `product_fuser` is used by default.

### Projection

- [ ] Model output is physical \((\phi_{\mathrm{raw}},\psi_{\mathrm{raw}})\).
- [ ] Hard symmetric projection is applied in physical variables.
- [ ] Projected physical output is converted to unit output only after projection.
- [ ] Balance loss is not used.
- [ ] Smooth masked projection is not used.

### Reconstruction

- [ ] Segment-wise loop is used.
- [ ] Segment reconstruction nodes include \(0\), interior valid local coordinates, and \(1\).
- [ ] Endpoint source-like values are hard-coded to zero.
- [ ] No network evaluation is performed at endpoints.
- [ ] Trapezoid weights are precomputed.
- [ ] GreenNet is queried at required \((t_i,t_j)\) pairs.
- [ ] Reconstruction uses projected unit output only.

### Energy loss

- [ ] Residual is \(r=u_\phi-u_\psi\).
- [ ] Energy is computed at valid physical points.
- [ ] Area weight is \(h_xh_y\).
- [ ] Face coefficient uses arithmetic average.
- [ ] x-edges require same `x_segment_id`.
- [ ] y-edges require same `y_segment_id`.

### Cross consistency

- [ ] No cross-consistency loss exists.
- [ ] No cross-consistency metric exists.
- [ ] No cross-consistency logging entry exists.
- [ ] No disabled flag is logged.
- [ ] No cross-consistency summary field is produced.

### Visualization

- [ ] Default visualization uses projected physical \(\phi,\psi\).
- [ ] Raw physical output is archived but is not the default visualization target.
- [ ] Reconstructed solutions are stored at physical valid points.

---

## 15. Fixed Design Decisions Summary

\\[
\boxed{
\text{Geometry is fixed across samples.}
}
\\]

\\[
\boxed{
\text{2D fields use flattened valid-point tensors }(B,2,P).
}
\\]

\\[
\boxed{
\text{Boundary endpoints are excluded from }P.
}
\\]

\\[
\boxed{
\text{CouplingNet raw output is a pair of physical directional fields.}
}
\\]

\\[
\boxed{
\Phi_{\mathrm{proj}}=L_x^2\phi_{\mathrm{proj}},
\qquad
\Psi_{\mathrm{proj}}=L_y^2\psi_{\mathrm{proj}}.
}
\\]

\\[
\boxed{
\texttt{axis\_1d\_trunk=true}
}
\\]

\\[
\boxed{
\text{Function branch uses transformed }a,b,c.
}
\\]

\\[
\boxed{
\text{Geometry branch uses }
[
\operatorname{PE}(r),
s^-,
s^+,
s_{\mathrm{mid}},
L,
L^2,
1/L
].
}
\\]

\\[
\boxed{
\text{No axis one-hot and no raw }r.
}
\\]

\\[
\boxed{
\text{Horizontal and vertical branches share the same networks.}
}
\\]

\\[
\boxed{
\texttt{product\_fuser is the default fusion mode.}
}
\\]

\\[
\boxed{
\text{Hard symmetric projection only.}
}
\\]

\\[
\boxed{
\text{No smooth masked projection and no balance loss.}
}
\\]

\\[
\boxed{
\text{Segment-wise Green reconstruction uses projected unit output only.}
}
\\]

\\[
\boxed{
\text{Endpoint values are hard-coded to zero.}
}
\\]

\\[
\boxed{
\text{Default loss is valid-physical-point face-energy consistency.}
}
\\]

\\[
\boxed{
\text{Energy area weight is }h_xh_y.
}
\\]

\\[
\boxed{
\text{Energy edges follow same axial connected-segment criterion.}
}
\\]

\\[
\boxed{
\text{Cross consistency is completely excluded.}
}
\\]

---

## 16. Out-of-Scope Items

The following items are outside the scope of this document:

1. GreenNet unit-interval normalization details;
2. geometry extraction from boundary representation;
3. connected segment detection algorithm;
4. class-level implementation details;
5. segment-wise loop optimization;
6. short-segment filtering or clipping;
7. cross consistency reintroduction;
8. raw \(r_\ell\) ablation;
9. Fourier frequency ablation;
10. alternative projection methods;
11. alternative area weights for energy loss.

---

## 17. Final Statement

The complex-geometry `CouplingNet` design is based on the following principle:

\\[
\boxed{
\text{CouplingNet predicts physical raw fields on segment-local coordinates,}
}
\\]

while

\\[
\boxed{
\text{projection and energy consistency are evaluated in physical space.}
}
\\]

The raw network output is a pair of physical directional source fields. Physical
symmetric projection, or the opt-in response-preconditioned alternative, enforces
balance at valid interior points. Only then does segment-wise Green reconstruction
create and use axis-scaled projected unit quantities on segment-local reference nodes.

Cross consistency is not part of the design and must not appear in loss computation, metrics, logs, or summaries.
