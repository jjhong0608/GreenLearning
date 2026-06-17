# GreenNet Unit-Interval Normalization for Complex Geometry

## 1. Background and Motivation

The purpose of this document is to fix the design of `GreenNet` for complex geometry.

The current `GreenNet` is designed to learn one-dimensional Green functions along axial lines. In the unit-square setting, every axial line segment naturally has the same one-dimensional domain:

\[
[0,1].
\]

Therefore, the existing implementation can use a fixed trunk grid

\[
(x,\xi)\in [0,1]\times[0,1],
\]

and a fixed-size branch input vector sampled on the same one-dimensional interval.

In a complex geometry, this is no longer true. An axial line may intersect the physical domain in a segment whose endpoints and length vary from line to line. Therefore, the physical one-dimensional domain for each Green function may be

\[
I_\ell = [s_{\ell,0}, s_{\ell,1}],
\]

where the endpoints depend on the connected intersection component.

The design decision documented here is:

\[
\boxed{
\text{Every connected 1D interval is mapped to the unit interval } [0,1].
}
\]

`GreenNet` will always learn a Green function on the normalized coordinate domain

\[
(t,\eta)\in[0,1]\times[0,1].
\]

The physical interval is handled through coordinate transformation and operator scaling.

---

## 2. Current Role of GreenNet

`GreenNet` is not a direct neural solver for the full two-dimensional domain. Its role is to learn a one-dimensional Green function for each axial one-dimensional problem.

For each 1D interval, `GreenNet` receives:

1. branch input:
   - sampled coefficient functions along the one-dimensional interval;
   - in the current implementation, these include diffusion, derivative of diffusion, convection, and reaction terms;
2. trunk input:
   - a coordinate pair representing the Green kernel evaluation point and source point;
   - in the normalized formulation, this coordinate is \((t,\eta)\in[0,1]^2\).

The output is interpreted as a normalized one-dimensional Green kernel:

\[
G_{\mathrm{unit}}(t,\eta).
\]

The Green reconstruction relation on the normalized interval is

\[
v(t)
=
\int_0^1
G_{\mathrm{unit}}(t,\eta)
f_{\mathrm{unit}}(\eta)
\,d\eta.
\]

The current `GreenNet` learning objective compares the reconstructed solution against the reference 1D solution. Therefore, the consistency of the formulation depends on using the correctly transformed coefficients and right-hand side.

---

## 3. 1D Domain Components in Complex Geometry

Let \(\Omega\subset\mathbb{R}^2\) be a complex two-dimensional domain.

For an axial line \(\ell\), the intersection

\[
\ell \cap \Omega
\]

may consist of one or more connected components.

If the intersection is disconnected, for example

\[
\ell \cap \Omega
=
I_{\ell,1}\cup I_{\ell,2}\cup\cdots\cup I_{\ell,K},
\]

then each connected component is treated as a separate one-dimensional GreenNet domain.

That is,

\[
I_{\ell,k}
=
[s_{\ell,k,0},s_{\ell,k,1}]
\]

is treated independently from the other components, even if they lie on the same axial line.

The rule is:

\[
\boxed{
\text{One connected interval component } = \text{ one independent 1D GreenNet domain.}
}
\]

Disconnected components must not be merged into a single longer interval.

---

## 4. Physical Interval to Unit Interval Mapping

Consider one connected physical interval

\[
I_\ell=[s_{\ell,0},s_{\ell,1}].
\]

Define its length as

\[
L_\ell=s_{\ell,1}-s_{\ell,0}.
\]

The interval must satisfy

\[
L_\ell>0.
\]

The affine map from the unit interval to the physical interval is

\[
s=s_{\ell,0}+L_\ell t,
\qquad
t\in[0,1].
\]

The inverse map is

\[
t=\frac{s-s_{\ell,0}}{L_\ell}.
\]

For a physical solution \(u(s)\), define the normalized solution

\[
v(t)=u(s_{\ell,0}+L_\ell t).
\]

The same transformation applies to the source coordinate. If \(r\in I_\ell\) is the physical source coordinate, then

\[
r=s_{\ell,0}+L_\ell \eta,
\qquad
\eta\in[0,1].
\]

Thus the physical Green kernel coordinate pair \((s,r)\) corresponds to the normalized pair

\[
(t,\eta)
=
\left(
\frac{s-s_{\ell,0}}{L_\ell},
\frac{r-s_{\ell,0}}{L_\ell}
\right).
\]

---

## 5. Transformation of the 1D Differential Operator

Assume that the physical one-dimensional operator on \(I_\ell\) is

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
\qquad
s\in[s_{\ell,0},s_{\ell,1}],
\]

with homogeneous Dirichlet boundary condition

\[
u(s_{\ell,0})=u(s_{\ell,1})=0.
\]

Using

\[
s=s_{\ell,0}+L_\ell t,
\qquad
v(t)=u(s_{\ell,0}+L_\ell t),
\]

we have

\[
\frac{du}{ds}
=
\frac{1}{L_\ell}\frac{dv}{dt},
\]

and

\[
\frac{d}{ds}
=
\frac{1}{L_\ell}\frac{d}{dt}.
\]

Substituting into the physical operator gives

\[
-\frac{1}{L_\ell^2}
\frac{d}{dt}
\left(
a(s_{\ell,0}+L_\ell t)
\frac{dv}{dt}
\right)
+
\frac{1}{L_\ell}
b(s_{\ell,0}+L_\ell t)
\frac{dv}{dt}
+
c(s_{\ell,0}+L_\ell t)v
=
f(s_{\ell,0}+L_\ell t).
\]

Multiplying the equation by \(L_\ell^2\), we obtain the normalized equation

\[
-\frac{d}{dt}
\left(
a_{\mathrm{unit}}(t)
\frac{dv}{dt}
\right)
+
b_{\mathrm{unit}}(t)
\frac{dv}{dt}
+
c_{\mathrm{unit}}(t)v
=
f_{\mathrm{unit}}(t),
\qquad
t\in[0,1].
\]

This is the form that must be passed to `GreenNet`.

---

## 6. Transformation Rules for Coefficients and RHS

The transformed coefficients are defined as follows.

### 6.1 Diffusion coefficient

\[
a_{\mathrm{unit}}(t)
=
a(s_{\ell,0}+L_\ell t).
\]

The diffusion coefficient itself is sampled by composition with the affine map. No additional multiplicative scaling is applied to \(a\).

---

### 6.2 Derivative of the diffusion coefficient

The derivative coefficient used by `GreenNet` must be the derivative with respect to the normalized coordinate \(t\), not the physical coordinate \(s\).

If

\[
a_s(s)=\frac{da}{ds}(s),
\]

then

\[
a'_{\mathrm{unit}}(t)
=
\frac{d}{dt}a(s_{\ell,0}+L_\ell t)
=
L_\ell a_s(s_{\ell,0}+L_\ell t).
\]

Thus,

\[
\boxed{
a'_{\mathrm{unit}} = L_\ell a_s.
}
\]

This is a critical point. Passing the physical derivative \(a_s\) directly as `ap_vals` would be inconsistent with the normalized operator.

---

### 6.3 Convection coefficient

The transformed convection coefficient is

\[
b_{\mathrm{unit}}(t)
=
L_\ell b(s_{\ell,0}+L_\ell t).
\]

Thus,

\[
\boxed{
b_{\mathrm{unit}} = L_\ell b_{\mathrm{phys}}.
}
\]

---

### 6.4 Reaction coefficient

The transformed reaction coefficient is

\[
c_{\mathrm{unit}}(t)
=
L_\ell^2 c(s_{\ell,0}+L_\ell t).
\]

Thus,

\[
\boxed{
c_{\mathrm{unit}} = L_\ell^2 c_{\mathrm{phys}}.
}
\]

---

### 6.5 Right-hand side

The transformed right-hand side is

\[
f_{\mathrm{unit}}(t)
=
L_\ell^2 f(s_{\ell,0}+L_\ell t).
\]

Thus,

\[
\boxed{
f_{\mathrm{unit}} = L_\ell^2 f_{\mathrm{phys}}.
}
\]

---

### 6.6 Summary table

| Quantity | Physical definition | Unit-interval quantity |
|---|---:|---:|
| Coordinate | \(s\in[s_{\ell,0},s_{\ell,1}]\) | \(t=(s-s_{\ell,0})/L_\ell\) |
| Solution | \(u(s)\) | \(v(t)=u(s_{\ell,0}+L_\ell t)\) |
| Diffusion | \(a(s)\) | \(a_{\mathrm{unit}}(t)=a(s_{\ell,0}+L_\ell t)\) |
| Diffusion derivative | \(a_s(s)\) | \(a'_{\mathrm{unit}}(t)=L_\ell a_s(s_{\ell,0}+L_\ell t)\) |
| Convection | \(b(s)\) | \(b_{\mathrm{unit}}(t)=L_\ell b(s_{\ell,0}+L_\ell t)\) |
| Reaction | \(c(s)\) | \(c_{\mathrm{unit}}(t)=L_\ell^2 c(s_{\ell,0}+L_\ell t)\) |
| RHS | \(f(s)\) | \(f_{\mathrm{unit}}(t)=L_\ell^2 f(s_{\ell,0}+L_\ell t)\) |

---

## 7. Green Function Scaling Rule

Let \(G_{\mathrm{unit}}(t,\eta)\) be the Green function of the normalized unit-interval problem:

\[
v(t)
=
\int_0^1
G_{\mathrm{unit}}(t,\eta)
f_{\mathrm{unit}}(\eta)
\,d\eta.
\]

Let \(G_{\mathrm{phys}}(s,r)\) be the Green function of the physical interval problem:

\[
u(s)
=
\int_{s_{\ell,0}}^{s_{\ell,1}}
G_{\mathrm{phys}}(s,r)
f(r)
\,dr.
\]

Using

\[
s=s_{\ell,0}+L_\ell t,
\qquad
r=s_{\ell,0}+L_\ell \eta,
\qquad
dr=L_\ell d\eta,
\]

and

\[
f_{\mathrm{unit}}(\eta)=L_\ell^2 f(s_{\ell,0}+L_\ell\eta),
\]

we obtain

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

Thus,

\[
\boxed{
G_{\mathrm{phys}} = L_\ell G_{\mathrm{unit}}.
}
\]

This scaling applies when converting the learned unit Green kernel back to the physical coordinate representation.

---

## 8. Unit Reconstruction vs Physical Reconstruction

There are two equivalent ways to reconstruct the solution, provided all scaling rules are applied correctly.

### 8.1 Unit-coordinate reconstruction

Use the normalized Green kernel and normalized RHS:

\[
v(t_i)
\approx
\sum_j
w_j
G_{\mathrm{unit}}(t_i,\eta_j)
f_{\mathrm{unit}}(\eta_j).
\]

Then recover the physical solution by

\[
u(s_i)=v(t_i),
\qquad
s_i=s_{\ell,0}+L_\ell t_i.
\]

This approach keeps all GreenNet computation in normalized coordinates.

---

### 8.2 Physical-coordinate reconstruction

First convert the Green kernel:

\[
G_{\mathrm{phys}}(s_i,r_j)
=
L_\ell G_{\mathrm{unit}}(t_i,\eta_j).
\]

Then reconstruct using the physical RHS and physical quadrature:

\[
u(s_i)
\approx
\sum_j
\tilde w_j
G_{\mathrm{phys}}(s_i,r_j)
f(r_j),
\]

where the physical quadrature weights satisfy

\[
\tilde w_j = L_\ell w_j.
\]

Both approaches are equivalent.

The unit-coordinate reconstruction is usually simpler for `GreenNet`, because it avoids repeatedly converting the kernel to physical coordinates.

---

## 9. Branch Network Sampling Rule

The branch network requires a fixed-size input vector. Therefore, branch sampling is performed on the normalized unit interval, not on globally shared physical coordinates.

Let the branch input size be \(m\). Define fixed unit sampling points

\[
t_j=\frac{j}{m-1},
\qquad
j=0,\dots,m-1.
\]

For each physical interval \(I_\ell=[s_{\ell,0},s_{\ell,1}]\), the corresponding physical sampling points are

\[
s_{\ell,j}=s_{\ell,0}+L_\ell t_j.
\]

The sampled branch inputs are constructed from the transformed coefficients:

\[
\left[
a_{\mathrm{unit}}(t_0),
a_{\mathrm{unit}}(t_1),
\dots,
a_{\mathrm{unit}}(t_{m-1})
\right],
\]

\[
\left[
a'_{\mathrm{unit}}(t_0),
a'_{\mathrm{unit}}(t_1),
\dots,
a'_{\mathrm{unit}}(t_{m-1})
\right],
\]

\[
\left[
b_{\mathrm{unit}}(t_0),
b_{\mathrm{unit}}(t_1),
\dots,
b_{\mathrm{unit}}(t_{m-1})
\right],
\]

\[
\left[
c_{\mathrm{unit}}(t_0),
c_{\mathrm{unit}}(t_1),
\dots,
c_{\mathrm{unit}}(t_{m-1})
\right].
\]

The branch sampling rule is:

\[
\boxed{
\text{Branch points are identical in normalized coordinate, not in physical coordinate.}
}
\]

This is acceptable because the branch network only requires a fixed input dimension. It does not require all physical intervals to share identical physical sampling locations.

---

## 10. Trunk Coordinates and Green Kernel Evaluation

The trunk coordinates are always normalized coordinates.

The trunk grid is

\[
(t,\eta)\in[0,1]\times[0,1].
\]

Therefore, `GreenNet` learns

\[
G_{\mathrm{unit}}(t,\eta),
\]

not directly

\[
G_{\mathrm{phys}}(s,r).
\]

If a physical Green kernel is needed, the conversion rule is

\[
G_{\mathrm{phys}}(s,r)
=
L_\ell
G_{\mathrm{unit}}(t,\eta).
\]

where

\[
t=\frac{s-s_{\ell,0}}{L_\ell},
\qquad
\eta=\frac{r-s_{\ell,0}}{L_\ell}.
\]

The trunk grid should remain fixed across all connected interval components. This preserves the current `GreenNet` architecture, because the trunk input size and structure do not depend on the physical length or location of the interval.

---

## 11. Interpretation of Disconnected Intersections

In complex geometry, an axial line may intersect the domain in multiple disconnected intervals.

For example,

\[
\ell\cap\Omega
=
[s_1^-,s_1^+]\cup[s_2^-,s_2^+].
\]

These two intervals are treated as two independent one-dimensional domains:

\[
I_1=[s_1^-,s_1^+],
\qquad
I_2=[s_2^-,s_2^+].
\]

Each interval has its own:

- left endpoint;
- right endpoint;
- length;
- coordinate map;
- transformed coefficients;
- transformed RHS;
- unit Green function evaluation.

They must not be treated as one domain with an internal gap.

The design rule is:

\[
\boxed{
\text{Disconnected intervals are independent GreenNet domains.}
}
\]

---

## 12. Boundary Condition Interpretation

For each connected interval

\[
I_\ell=[s_{\ell,0},s_{\ell,1}],
\]

the normalized problem is assumed to have homogeneous Dirichlet boundary conditions:

\[
v(0)=0,
\qquad
v(1)=0.
\]

This corresponds to

\[
u(s_{\ell,0})=0,
\qquad
u(s_{\ell,1})=0.
\]

This assumption is consistent with treating each connected interval as an independent 1D Green function domain.

If nonzero Dirichlet, Neumann, Robin, or interface-type boundary conditions are introduced later, they must be documented separately. They are outside the scope of this GreenNet normalization document.

---

## 13. Implementation-Level Checklist

The following checklist must be satisfied when implementing the normalized GreenNet formulation.

### 13.1 Interval definition

- [ ] Each connected interval component is represented as \(I_\ell=[s_{\ell,0},s_{\ell,1}]\).
- [ ] The interval length is computed as \(L_\ell=s_{\ell,1}-s_{\ell,0}\).
- [ ] The length satisfies \(L_\ell>0\).
- [ ] Disconnected components are stored as separate intervals.

---

### 13.2 Coordinate mapping

- [ ] The normalized coordinate grid is defined by \(t_j=j/(m-1)\).
- [ ] Physical sampling points are computed as \(s_{\ell,j}=s_{\ell,0}+L_\ell t_j\).
- [ ] Trunk coordinates remain \((t,\eta)\in[0,1]^2\).
- [ ] Physical coordinates are not directly passed as GreenNet trunk coordinates.

---

### 13.3 Coefficient transformation

- [ ] \(a_{\mathrm{unit}}(t)=a(s_{\ell,0}+L_\ell t)\).
- [ ] \(a'_{\mathrm{unit}}(t)=L_\ell a_s(s_{\ell,0}+L_\ell t)\).
- [ ] \(b_{\mathrm{unit}}(t)=L_\ell b(s_{\ell,0}+L_\ell t)\).
- [ ] \(c_{\mathrm{unit}}(t)=L_\ell^2 c(s_{\ell,0}+L_\ell t)\).
- [ ] The value passed as `ap_vals` is the unit-coordinate derivative, not the physical derivative.

---

### 13.4 RHS transformation

- [ ] The physical RHS is sampled at \(s_{\ell,j}=s_{\ell,0}+L_\ell t_j\).
- [ ] The unit RHS is computed as \(f_{\mathrm{unit}}=L_\ell^2 f_{\mathrm{phys}}\).
- [ ] Green reconstruction in unit coordinates uses \(f_{\mathrm{unit}}\).

---

### 13.5 Green kernel interpretation

- [ ] `GreenNet` output is interpreted as \(G_{\mathrm{unit}}(t,\eta)\).
- [ ] If physical Green kernel is required, use \(G_{\mathrm{phys}}=L_\ell G_{\mathrm{unit}}\).
- [ ] If unit-coordinate reconstruction is used, do not additionally multiply the kernel by \(L_\ell\).
- [ ] Avoid double-counting the length scaling.

---

### 13.6 Branch input

- [ ] Branch input dimension is fixed for all intervals.
- [ ] Branch sampling points are fixed in normalized coordinates.
- [ ] Physical intervals of different lengths are allowed.
- [ ] Physical sample points do not need to coincide across intervals.
- [ ] Branch inputs must contain transformed coefficients, not unscaled physical coefficients.

---

### 13.7 Numerical integration

- [ ] Unit-coordinate quadrature uses weights on \([0,1]\).
- [ ] Physical-coordinate quadrature uses weights scaled by \(L_\ell\).
- [ ] Unit reconstruction and physical reconstruction are not mixed without applying the correct scaling.
- [ ] Simpson or trapezoid rules should be interpreted in the coordinate system in which the integral is evaluated.

---

## 14. Summary of Fixed Design Decisions

The following design decisions are fixed for `GreenNet`.

1. `GreenNet` operates on normalized unit intervals.
2. Every connected 1D interval from a complex geometry is mapped to \([0,1]\).
3. Disconnected intersections are treated as independent 1D GreenNet domains.
4. Branch network sampling is performed at fixed normalized coordinates.
5. Branch input points do not need to coincide in physical space.
6. Trunk coordinates are normalized coordinates \((t,\eta)\).
7. The learned Green kernel is interpreted as \(G_{\mathrm{unit}}\).
8. Coefficients and RHS must be transformed according to the coordinate scaling.
9. The derivative coefficient `ap_vals` must represent \(d a_{\mathrm{unit}}/dt\), not \(d a/ds\).
10. Physical Green kernel recovery requires the scaling \(G_{\mathrm{phys}}=L_\ell G_{\mathrm{unit}}\).
11. Unit-coordinate reconstruction uses \(f_{\mathrm{unit}}\).
12. Physical-coordinate reconstruction uses \(G_{\mathrm{phys}}\), \(f_{\mathrm{phys}}\), and physical quadrature weights.

---

## 15. Common Failure Modes

The following mistakes would make the formulation mathematically inconsistent.

### 15.1 Using physical derivative as `ap_vals`

Incorrect:

\[
\texttt{ap\_vals}=a_s(s).
\]

Correct:

\[
\texttt{ap\_vals}=a'_{\mathrm{unit}}(t)=L_\ell a_s(s_{\ell,0}+L_\ell t).
\]

---

### 15.2 Forgetting convection scaling

Incorrect:

\[
b_{\mathrm{unit}}(t)=b(s_{\ell,0}+L_\ell t).
\]

Correct:

\[
b_{\mathrm{unit}}(t)=L_\ell b(s_{\ell,0}+L_\ell t).
\]

---

### 15.3 Forgetting reaction scaling

Incorrect:

\[
c_{\mathrm{unit}}(t)=c(s_{\ell,0}+L_\ell t).
\]

Correct:

\[
c_{\mathrm{unit}}(t)=L_\ell^2 c(s_{\ell,0}+L_\ell t).
\]

---

### 15.4 Forgetting RHS scaling

Incorrect:

\[
f_{\mathrm{unit}}(t)=f(s_{\ell,0}+L_\ell t).
\]

Correct:

\[
f_{\mathrm{unit}}(t)=L_\ell^2 f(s_{\ell,0}+L_\ell t).
\]

---

### 15.5 Double-counting Green kernel scaling

If the reconstruction is performed in unit coordinates,

\[
v(t)=\int_0^1G_{\mathrm{unit}}(t,\eta)f_{\mathrm{unit}}(\eta)\,d\eta,
\]

then do not multiply \(G_{\mathrm{unit}}\) by \(L_\ell\).

The factor \(L_\ell\) is used only when converting to the physical Green kernel:

\[
G_{\mathrm{phys}}=L_\ell G_{\mathrm{unit}}.
\]

---

### 15.6 Merging disconnected intervals

Incorrect:

\[
[s_1^-,s_1^+]\cup[s_2^-,s_2^+]
\quad\text{treated as one interval}.
\]

Correct:

\[
[s_1^-,s_1^+]
\quad\text{and}\quad
[s_2^-,s_2^+]
\]

are separate GreenNet domains.

---

## 16. Out-of-Scope Items

This document intentionally does not specify the following:

1. how to detect connected interval components from a complex geometry;
2. how to represent the full two-dimensional complex domain;
3. how to construct masks, meshes, or boundary point clouds;
4. how `CouplingNet` should combine segment-wise GreenNet outputs;
5. how to impose global 2D consistency across disconnected intervals;
6. how to handle nonzero Dirichlet, Neumann, Robin, or interface conditions;
7. how to train `GreenNet` and `CouplingNet` jointly in the complex geometry setting;
8. how to perform adaptive or nonuniform branch sampling.

These topics should be documented separately.

---

## 17. Final Statement

The normalized `GreenNet` formulation is fixed as follows:

\[
\boxed{
\text{All physical 1D connected intervals are mapped to } [0,1].
}
\]

\[
\boxed{
\text{GreenNet learns } G_{\mathrm{unit}}(t,\eta) \text{ on } [0,1]^2.
}
\]

\[
\boxed{
\text{Branch inputs are sampled at fixed normalized coordinates.}
}
\]

\[
\boxed{
\text{Operator scaling must be applied to } a', b, c, \text{ and } f.
}
\]

\[
\boxed{
G_{\mathrm{phys}}(s,r)
=
L_\ell
G_{\mathrm{unit}}
\left(
\frac{s-s_{\ell,0}}{L_\ell},
\frac{r-s_{\ell,0}}{L_\ell}
\right).
}
\]

This design keeps the `GreenNet` architecture independent of the physical length and location of each one-dimensional interval while preserving the mathematical equivalence between the physical interval problem and the normalized unit-interval problem.
