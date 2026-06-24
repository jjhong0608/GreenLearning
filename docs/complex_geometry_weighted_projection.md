# Complex Geometry Weighted Projection

## Purpose

This note describes a possible `geometry_weighted` balance projection for
complex-geometry CouplingNet.

The goal is not to impose

\[
\phi = 0,\qquad \psi = 0
\]

on a curved boundary. That condition is generally false for complex domains.
For example, in a disk with

\[
u(x,y)=\sin(4\pi(x^2+y^2)),\qquad r\leq \frac12,
\]

the boundary value \(u=0\) holds at \(r=\frac12\), but

\[
\phi=-u_{xx}=8\pi,\qquad \psi=-u_{yy}=8\pi
\]

on the boundary. Therefore the unit-square smooth-mask interpretation cannot be
ported as a rule that drives \(\phi\) or \(\psi\) to zero on complex-domain
boundaries.

The purpose of `geometry_weighted` is narrower:

1. preserve the pointwise balance constraint

   \[
   \phi+\psi=f,
   \]

2. avoid adding an extra boundary loss,
3. replace the fixed symmetric split by a deterministic geometry-aware source
   partition,
4. keep the remaining difference mode as the only free CouplingNet gauge.

This makes `geometry_weighted` a structural projection, not a loss term and not
an exact curved-boundary enforcement mechanism.

## Current Symmetric Projection

The current complex path uses hard symmetric projection in physical variables.
Given raw model outputs

\[
\phi_0,\qquad \psi_0
\]

and source \(f\), define

\[
r=f-\phi_0-\psi_0.
\]

The symmetric projection is

\[
\phi=\phi_0+\frac12 r,
\]

\[
\psi=\psi_0+\frac12 r.
\]

Equivalently, with

\[
d=\phi_0-\psi_0,
\]

it can be written as

\[
\phi=\frac12 f+\frac12 d,
\]

\[
\psi=\frac12 f-\frac12 d.
\]

This always enforces

\[
\phi+\psi=f.
\]

Its limitation is that the source partition \(\frac12,\frac12\) is the same at
every valid point, regardless of local chord geometry.

## Geometry-Weighted Parameterization

The proposed geometry-weighted form keeps the same balance identity but replaces
the constant source partition by point-dependent weights:

\[
\phi=w_\phi f+\beta d,
\]

\[
\psi=w_\psi f-\beta d,
\]

with

\[
w_\phi+w_\psi=1.
\]

Then balance is exact by construction:

\[
\phi+\psi
=
(w_\phi+w_\psi)f+\beta d-\beta d
=f.
\]

The model still predicts the difference mode \(d\), while the geometry controls
how the source component \(f\) is split between \(\phi\) and \(\psi\).

This is the same structural idea as smooth-mask projection, but the weights are
not boundary-zero masks. They are fixed source-partition weights derived from
complex-geometry metadata.

## Chord Length Features

For each valid point \(p\), the complex geometry metadata already gives:

- \(L_x(p)\): length of the horizontal x-segment containing \(p\),
- \(L_y(p)\): length of the vertical y-segment containing \(p\).

These are available through the segment ids:

\[
L_x(p)=\text{x\_segment\_length}[\text{x\_segment\_id}(p)],
\]

\[
L_y(p)=\text{y\_segment\_length}[\text{y\_segment\_id}(p)].
\]

The simplest chord-length score is

\[
s_x = L_x^2,\qquad s_y = L_y^2.
\]

Then a length-weighted source partition is

\[
w_\phi = \frac{s_x}{s_x+s_y+\epsilon},
\]

\[
w_\psi = \frac{s_y}{s_x+s_y+\epsilon}.
\]

This choice says that the x-direction component receives more of the source when
the x-chord is relatively long, and the y-direction component receives more when
the y-chord is relatively long.

An inverse-stiffness alternative is also possible:

\[
s_x = \frac{1}{L_x^2+\epsilon},\qquad
s_y = \frac{1}{L_y^2+\epsilon},
\]

\[
w_\phi = \frac{s_x}{s_x+s_y+\epsilon},
\qquad
w_\psi = \frac{s_y}{s_x+s_y+\epsilon}.
\]

This gives more source to the shorter chord direction. It is motivated by the
fact that a one-dimensional Dirichlet operator on a short interval has stronger
stiffness. However, it is not guaranteed to match true Cartesian second
derivative components on curved boundaries.

For a first implementation, the safer default is the direct length-squared rule

\[
s_x=L_x^2,\qquad s_y=L_y^2,
\]

because it is bounded, smooth for non-degenerate segments, and reduces to the
symmetric split when \(L_x=L_y\).

## Difference-Mode Scaling

The factor \(\beta\) controls how much of the raw model difference mode

\[
d=\phi_0-\psi_0
\]

is preserved.

To recover symmetric projection when \(w_\phi=w_\psi=\frac12\), we need

\[
\beta=\frac12.
\]

A simple geometry-aware bounded choice is

\[
\beta = 2w_\phi w_\psi.
\]

This satisfies

\[
0\leq \beta \leq \frac12,
\]

and

\[
\beta=\frac12
\quad\text{when}\quad
w_\phi=w_\psi=\frac12.
\]

It damps the difference mode when the geometry strongly prefers one source
partition over the other. This mirrors the role of the smooth-mask difference
factor, but without claiming that either \(\phi\) or \(\psi\) vanishes at the
boundary.

The projected fields become

\[
\phi=w_\phi f+2w_\phi w_\psi(\phi_0-\psi_0),
\]

\[
\psi=w_\psi f-2w_\phi w_\psi(\phi_0-\psi_0).
\]

Again,

\[
\phi+\psi=f
\]

holds exactly.

## Behavior In A Disk

For a disk centered at the origin, near the center of the domain,

\[
L_x\approx L_y,
\]

so

\[
w_\phi\approx w_\psi\approx \frac12.
\]

The projection is close to symmetric.

Near a point where the horizontal chord is short and the vertical chord is long,
the direct length-squared rule gives

\[
w_\phi < w_\psi.
\]

This does not mean that \(\phi\) is forced to zero. It only means that the
geometry-weighted source partition gives more of the source component to the
y-direction part at that point. The model difference mode can still shift mass
between \(\phi\) and \(\psi\), controlled by \(\beta\).

The radial example above is an important caution: a true solution can have
nonzero and even equal \(\phi,\psi\) values on a circular boundary. Therefore
`geometry_weighted` should be treated as a structural inductive bias, not as an
exact boundary formula.

## Relation To Unit-Square Smooth Mask

In the unit square, smooth-mask projection used coordinate masks such as

\[
m_\phi=y(1-y),\qquad m_\psi=x(1-x)
\]

or

\[
m_\phi=\sin(\pi y),\qquad m_\psi=\sin(\pi x).
\]

Those masks are meaningful because the unit-square boundary is axis-aligned.
For example, on \(y=0,1\), the x-direction is tangential to the boundary, so the
zero Dirichlet trace implies tangential derivative constraints. That special
axis-aligned structure is not available on a circular or general curved
boundary.

The `geometry_weighted` mode should therefore not reuse

\[
m_\phi=h(y_{\mathrm{local}}),\qquad m_\psi=h(x_{\mathrm{local}})
\]

as boundary-zero masks. That would again imply a zero boundary behavior for
\(\phi\) or \(\psi\), which is not valid in general complex geometry.

Instead, `geometry_weighted` uses chord lengths only to define a fixed source
partition while leaving \(\phi,\psi\) nonzero whenever the balance and learned
difference mode require it.

## Proposed Projection Modes

A useful comparison set would be:

1. `symmetric`

   \[
   w_\phi=w_\psi=\frac12,\qquad \beta=\frac12.
   \]

2. `geometry_weighted`

   \[
   w_\phi=\frac{L_x^2}{L_x^2+L_y^2+\epsilon},
   \qquad
   w_\psi=\frac{L_y^2}{L_x^2+L_y^2+\epsilon},
   \]

   \[
   \beta=2w_\phi w_\psi.
   \]

3. `inverse_geometry_weighted`

   \[
   w_\phi=
   \frac{(L_x^2+\epsilon)^{-1}}
        {(L_x^2+\epsilon)^{-1}+(L_y^2+\epsilon)^{-1}+\epsilon},
   \]

   \[
   w_\psi=1-w_\phi,
   \qquad
   \beta=2w_\phi w_\psi.
   \]

4. `learned_geometry_weighted`

   \[
   w_\phi=\sigma(g_\theta(\text{geometry features})),
   \qquad
   w_\psi=1-w_\phi.
   \]

   This still enforces \(\phi+\psi=f\) exactly, but learns the source partition
   from geometry features. It has more freedom and should be considered only
   after fixed modes are tested.

## Implementation Surface

If implemented, the projection should remain in physical variables:

1. convert raw unit outputs to physical \(\phi_0,\psi_0\),
2. compute \(L_x,L_y\) at each valid point,
3. compute \(w_\phi,w_\psi,\beta\),
4. apply the geometry-weighted projection,
5. convert projected physical \(\phi,\psi\) back to projected unit outputs for
   Green reconstruction.

The natural implementation location is

```text
src/greenonet/complex_projection.py
```

with trainer, evaluator, and artifact paths dispatching through a
config-aware projection helper.

## Limitations

`geometry_weighted` is not an exact curved-boundary condition. A rigorous
boundary formula for \(\phi=-u_{xx}\) and \(\psi=-u_{yy}\) on a curved boundary
would involve boundary normal/tangent geometry and, in general, mixed derivative
information such as \(u_{xy}\). The current axial decomposition does not model
that term directly.

Therefore the expected benefit of `geometry_weighted` is structural bias, not a
mathematical guarantee:

- it enforces \(\phi+\psi=f\) exactly,
- it avoids false zero-boundary assumptions for \(\phi,\psi\),
- it uses available complex-geometry metadata,
- it may improve over the global \(1/2,1/2\) split when chord anisotropy matters,
- it still needs empirical comparison against symmetric projection.

