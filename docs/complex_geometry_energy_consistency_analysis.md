# Complex Geometry Energy-Consistency Analysis

## 1. Purpose and Setting

This note gives a continuous-domain energy-consistency analysis for the complex
geometry GreenNet/CouplingNet framework.  The purpose is to isolate a structural
mechanism: under explicit assumptions, the energy disagreement between two
directional Green reconstructions controls the directional split-error
variables.

This is not a proof for a particular discrete implementation or a finite
valid-point graph.  It is a continuous structural theorem for a complex physical
domain

\[
\Omega\subset \mathbb{R}^2
\]

with homogeneous Dirichlet boundary condition.  The energy norm is

\[
\|v\|_a^2
:=
\int_\Omega a|\nabla v|^2.
\]

The energy-consistency loss considered here is

\[
\mathcal{E}_{\mathrm{split}}
:=
\|u_\phi-u_\psi\|_a^2
=
\int_\Omega a|\nabla(u_\phi-u_\psi)|^2,
\]

where \(u_\phi\) and \(u_\psi\) are the two represented solutions obtained by
applying connected-interval axial Green reconstructions to the two projected
directional source components.

The theorem is useful because CouplingNet does not output the solution directly.
It predicts a balanced directional source split.  The Green reconstructions then
map this split back to represented solutions.  Energy consistency asks whether
the two represented solutions describe the same physical field.

## 2. Directional Split Operators

The full elliptic operator is

\[
Lu
=
-\nabla\cdot(a\nabla u)
+b\cdot\nabla u
+cu,
\qquad
b=(b_x,b_y).
\]

The complex geometry framework uses a symmetric 1/2 reaction split.  The
directional operators are

\[
L_xu
=
-\partial_x(a\partial_xu)
+b_x\partial_xu
+\frac12cu,
\]

\[
L_yu
=
-\partial_y(a\partial_yu)
+b_y\partial_yu
+\frac12cu.
\]

Thus

\[
L=L_x+L_y.
\]

Let \(u_*\in H_0^1(\Omega)\) be the reference solution satisfying

\[
Lu_*=f.
\]

The exact direction-split source fields induced by \(u_*\) are

\[
\phi_*:=L_xu_*,
\qquad
\psi_*:=L_yu_*.
\]

By construction,

\[
\phi_*+\psi_*
=
(L_x+L_y)u_*
=
Lu_*
=
f.
\]

The model predicts a split \((\phi,\psi)\).  After physical balance projection,
the predicted split is assumed to satisfy

\[
\phi+\psi=f.
\]

Equivalently, the split errors satisfy

\[
(\phi-\phi_*)+(\psi-\psi_*)=0.
\]

This identity is the algebraic source of the energy-consistency estimate.

## 3. Connected-Interval Green Reconstructions

In a complex domain, an axial line may intersect the domain in more than one
connected component.  For almost every horizontal slice, write

\[
\Omega\cap\{y=\bar y\}
=
\bigcup_k I_k^x(\bar y),
\]

where the \(I_k^x(\bar y)\) are disjoint connected intervals.  Similarly, for
almost every vertical slice,

\[
\Omega\cap\{x=\bar x\}
=
\bigcup_\ell I_\ell^y(\bar x).
\]

The x-direction Green reconstruction \(G_x\) is assembled by applying an
independent one-dimensional Dirichlet Green inverse on each horizontal connected
interval \(I_k^x(\bar y)\).  The y-direction Green reconstruction \(G_y\) is
assembled analogously on each vertical connected interval \(I_\ell^y(\bar x)\).

Disconnected intervals are not merged.  If a line intersects the domain in two
separate components, these components represent two independent one-dimensional
boundary value problems.  This prevents the reconstruction from transporting
information through the outside of the physical domain.

Formally, for a source field \(\mu\), \(G_x[\mu]\) is the function obtained by
solving

\[
L_x w=\mu
\]

on each horizontal connected interval, with homogeneous endpoint conditions on
that interval, and then assembling the intervalwise solutions over all slices.
The definition of \(G_y[\nu]\) is the corresponding vertical construction.

The following theorem treats \(G_x\) and \(G_y\) as connected-interval Green
reconstruction operators satisfying the weak inverse assumptions stated below.

## 4. Assumptions

The analysis is conditional on explicit structural assumptions.  These
assumptions are intentionally stated rather than hidden inside the notation.

**Assumption 1: Domain and slice structure.**  
\(\Omega\subset\mathbb{R}^2\) is a bounded Lipschitz domain.  For almost every
horizontal and vertical slice, the intersection with \(\Omega\) admits a
connected-interval decomposition as in Section 3.  The endpoints of these
connected intervals are interpreted as Dirichlet endpoints for the corresponding
one-dimensional Green problems.

**Assumption 2: Coefficient bounds.**  
There exist \(a_0,a_1>0\) such that

\[
0<a_0\le a(x,y)\le a_1<\infty
\qquad
\text{for a.e. }(x,y)\in\Omega.
\]

Moreover,

\[
b\in L^\infty(\Omega)^2,
\qquad
c\in L^\infty(\Omega).
\]

**Assumption 3: Full bilinear coercivity.**  
Define

\[
B(u,v)
:=
\int_\Omega a\nabla u\cdot\nabla v
+
\int_\Omega (b\cdot\nabla u)v
+
\int_\Omega cuv.
\]

There exists \(\gamma_a>0\) such that

\[
B(v,v)\ge \gamma_a\|v\|_a^2
\qquad
\forall v\in H_0^1(\Omega).
\]

This condition is automatic in pure diffusion with positive diffusion and
homogeneous Dirichlet boundary condition.  With convection or sign-changing
reaction, it is an assumption on the lower-order terms relative to diffusion.

**Assumption: Linearity of the Green reconstructions.**  
The connected-interval Green reconstructions are linear in the source:

\[
G_x[\alpha\mu_1+\beta\mu_2]
=
\alpha G_x[\mu_1]+\beta G_x[\mu_2],
\qquad
G_y[\alpha\nu_1+\beta\nu_2]
=
\alpha G_y[\nu_1]+\beta G_y[\nu_2].
\]

**Assumption 4: Connected-interval weak inverse identities.**  
For admissible source terms \(\mu\) and \(\nu\), the connected-interval Green
reconstructions satisfy

\[
L_xG_x[\mu]=\mu,
\qquad
L_yG_y[\nu]=\nu
\]

in the weak sense.  Equivalently, for the directional bilinear forms below,

\[
B_x(G_x[\mu],v)=\langle\mu,v\rangle,
\qquad
B_y(G_y[\nu],v)=\langle\nu,v\rangle
\]

for all admissible test functions \(v\).

**Assumption 5: Projection consistency.**  
The projected predicted split satisfies

\[
\phi+\psi=f=\phi_*+\psi_*.
\]

**Assumption 6: Exact reference reconstruction.**  
For the exact Green theorem, the connected-interval reconstructions recover the
reference solution from the exact split:

\[
G_x[\phi_*]=u_*,
\qquad
G_y[\psi_*]=u_*.
\]

The imperfect reconstruction theorem in Section 7 relaxes this assumption.

**Assumption 7: Full-domain admissibility.**  
The represented and reference solutions belong to the full Dirichlet energy
space:

\[
u_\phi,u_\psi,u_*\in H_0^1(\Omega).
\]

Here \(u_*\in H_0^1(\Omega)\) is the full-domain weak reference solution.
By contrast, \(u_\phi=G_x[\phi]\) and \(u_\psi=G_y[\psi]\) are assembled from
connected-interval reconstructions.  Their membership in \(H_0^1(\Omega)\) is
therefore an admissibility assumption on the reconstructed fields, not a
consequence of interval endpoint conditions alone.

Equivalently, the theorem can be read as assuming that the connected-interval
operators are well-defined on admissible source classes

\[
G_x:\mathcal{S}_x\to H_0^1(\Omega),
\qquad
G_y:\mathcal{S}_y\to H_0^1(\Omega).
\]

The spaces \(\mathcal{S}_x\) and \(\mathcal{S}_y\) encode the source,
coefficient, slice, and transverse regularity required for the assembled
slice-wise reconstructions to be full-domain Sobolev functions.  The issue is
discussed separately in Section 8.

## 5. Exact Green Reconstruction Theorem

Define the represented solutions

\[
u_\phi:=G_x[\phi],
\qquad
u_\psi:=G_y[\psi].
\]

Let

\[
\mathcal{E}_{\mathrm{split}}
=
\|u_\phi-u_\psi\|_a^2.
\]

Define the split-error variables

\[
q_x:=2G_x[\phi-\phi_*],
\qquad
q_y:=2G_y[\psi-\psi_*],
\]

and define the average component

\[
q_c:=\frac{q_x+q_y}{2}.
\]

**Theorem 1: Energy-consistency control under exact connected-interval Green reconstruction.**  
Suppose Assumptions 1-7 hold.  Then

\[
\|q_c\|_a
\le
C_E\sqrt{\mathcal{E}_{\mathrm{split}}},
\]

and

\[
\|q_x\|_a,\ \|q_y\|_a
\le
(1+C_E)\sqrt{\mathcal{E}_{\mathrm{split}}}.
\]

A conservative admissible constant is

\[
C_E=\frac{M}{\gamma_a},
\qquad
M=
1+\frac{C_P(\Omega)\|b\|_{L^\infty}}{a_0},
\]

where \(C_P(\Omega)\) is the Poincare constant of \(\Omega\).

Consequently,

\[
\mathcal{E}_{\mathrm{split}}\to0
\quad\Longrightarrow\quad
q_c,q_x,q_y\to0
\quad
\text{in the energy norm}.
\]

Because the variables are assumed to lie in \(H_0^1(\Omega)\), energy-norm
convergence also controls the \(H_0^1\)-seminorm, and Poincare's inequality
controls the \(L^2\) component.

## 6. Proof of Theorem 1

Under exact reference reconstruction,

\[
q_x
=
2(G_x[\phi]-G_x[\phi_*])
=
2(u_\phi-u_*),
\]

and

\[
q_y
=
2(G_y[\psi]-G_y[\psi_*])
=
2(u_\psi-u_*).
\]

Let

\[
r:=u_\phi-u_\psi.
\]

Then

\[
q_x-q_y
=
2(u_\phi-u_\psi)
=
2r.
\]

Since \(q_c=(q_x+q_y)/2\), we have the average-difference decomposition

\[
q_x=q_c+r,
\qquad
q_y=q_c-r.
\]

By the connected-interval weak inverse identities,

\[
L_xq_x=2(\phi-\phi_*),
\qquad
L_yq_y=2(\psi-\psi_*).
\]

Adding these identities gives

\[
L_xq_x+L_yq_y
=
2\{(\phi-\phi_*)+(\psi-\psi_*)\}.
\]

Projection consistency makes the right-hand side zero.  Therefore

\[
L_xq_x+L_yq_y=0.
\]

Substituting \(q_x=q_c+r\) and \(q_y=q_c-r\) yields

\[
L_x(q_c+r)+L_y(q_c-r)=0.
\]

Equivalently,

\[
(L_x+L_y)q_c+(L_x-L_y)r=0.
\]

Since \(L=L_x+L_y\),

\[
Lq_c=-(L_x-L_y)r.
\]

This is the structural identity.

Now define the directional bilinear forms

\[
B_x(w,v)
:=
\int_\Omega a\partial_xw\,\partial_xv
+
\int_\Omega b_x\partial_xw\,v
+
\frac12\int_\Omega cwv,
\]

\[
B_y(w,v)
:=
\int_\Omega a\partial_yw\,\partial_yv
+
\int_\Omega b_y\partial_yw\,v
+
\frac12\int_\Omega cwv.
\]

Then

\[
B(w,v)=B_x(w,v)+B_y(w,v).
\]

The weak form of the structural identity is

\[
B(q_c,v)
=
-\{B_x(r,v)-B_y(r,v)\}
\qquad
\forall v\in H_0^1(\Omega).
\]

Taking \(v=q_c\) gives

\[
B(q_c,q_c)
=
-\{B_x(r,q_c)-B_y(r,q_c)\}.
\]

By coercivity,

\[
\gamma_a\|q_c\|_a^2
\le
B(q_c,q_c).
\]

Therefore

\[
\gamma_a\|q_c\|_a^2
\le
|B_x(r,q_c)-B_y(r,q_c)|.
\]

It remains to bound the directional difference.  The reaction terms cancel:

\[
B_x(r,v)-B_y(r,v)
=
\int_\Omega a(\partial_xr\,\partial_xv-\partial_yr\,\partial_yv)
+
\int_\Omega (b_x\partial_xr-b_y\partial_yr)v.
\]

For the diffusion part, set

\[
P_r=(\partial_xr,-\partial_yr),
\qquad
P_v=(\partial_xv,\partial_yv).
\]

Then \(|P_r|=|\nabla r|\), \(|P_v|=|\nabla v|\), and

\[
\left|
\int_\Omega a(\partial_xr\,\partial_xv-\partial_yr\,\partial_yv)
\right|
\le
\|r\|_a\|v\|_a.
\]

For the advection part, let \(\widetilde b=(b_x,-b_y)\).  Then
\(|\widetilde b|=|b|\), and

\[
\left|
\int_\Omega (b_x\partial_xr-b_y\partial_yr)v
\right|
\le
\|b\|_{L^\infty}\|\nabla r\|_{L^2}\|v\|_{L^2}.
\]

Using \(a\ge a_0\) and Poincare's inequality on \(\Omega\),

\[
\|\nabla r\|_{L^2}
\le
\frac{1}{\sqrt{a_0}}\|r\|_a,
\]

\[
\|v\|_{L^2}
\le
C_P(\Omega)\|\nabla v\|_{L^2}
\le
\frac{C_P(\Omega)}{\sqrt{a_0}}\|v\|_a.
\]

Thus

\[
\left|
\int_\Omega (b_x\partial_xr-b_y\partial_yr)v
\right|
\le
\frac{C_P(\Omega)\|b\|_{L^\infty}}{a_0}
\|r\|_a\|v\|_a.
\]

Combining the two estimates gives

\[
|B_x(r,v)-B_y(r,v)|
\le
M\|r\|_a\|v\|_a,
\]

where

\[
M=
1+\frac{C_P(\Omega)\|b\|_{L^\infty}}{a_0}.
\]

Applying this with \(v=q_c\),

\[
\gamma_a\|q_c\|_a^2
\le
M\|r\|_a\|q_c\|_a.
\]

If \(\|q_c\|_a=0\), the estimate is immediate.  Otherwise, divide by
\(\|q_c\|_a\):

\[
\|q_c\|_a
\le
\frac{M}{\gamma_a}\|r\|_a.
\]

Since

\[
\|r\|_a^2
=
\mathcal{E}_{\mathrm{split}},
\]

we obtain

\[
\|q_c\|_a
\le
C_E\sqrt{\mathcal{E}_{\mathrm{split}}}.
\]

Finally,

\[
q_x=q_c+r,
\qquad
q_y=q_c-r.
\]

The triangle inequality gives

\[
\|q_x\|_a
\le
\|q_c\|_a+\|r\|_a
\le
(1+C_E)\sqrt{\mathcal{E}_{\mathrm{split}}},
\]

and the same argument gives

\[
\|q_y\|_a
\le
(1+C_E)\sqrt{\mathcal{E}_{\mathrm{split}}}.
\]

This proves Theorem 1.

**Corollary: final solution error bound.**  
Under the assumptions of Theorem 1, define

\[
u_{\mathrm{pred}}
:=
\frac12(u_\phi+u_\psi).
\]

Then

\[
u_{\mathrm{pred}}-u_*
=
\frac12q_c,
\]

and therefore

\[
\|u_{\mathrm{pred}}-u_*\|_a
\le
\frac{C_E}{2}\sqrt{\mathcal{E}_{\mathrm{split}}}.
\]

Moreover, since

\[
u_\phi-u_*=\frac12q_x,
\qquad
u_\psi-u_*=\frac12q_y,
\]

the represented solution errors satisfy

\[
\|u_\phi-u_*\|_a,\ \|u_\psi-u_*\|_a
\le
\frac{1+C_E}{2}\sqrt{\mathcal{E}_{\mathrm{split}}}.
\]

Thus, in the exact connected-interval Green reconstruction setting, the energy
loss directly bounds the final solution energy error and the two represented
solution energy errors.

## 7. Imperfect Green Reconstruction Perturbation

The exact theorem assumes that the connected-interval Green reconstructions
recover the reference solution from the exact split.  Learned or approximate
Green operators may violate this.  We therefore include a perturbation theorem.

Assume

\[
G_x[\phi_*]=u_*+\varepsilon_x,
\qquad
G_y[\psi_*]=u_*+\varepsilon_y.
\]

Here \(\varepsilon_x,\varepsilon_y\in H_0^1(\Omega)\) represent reference
reconstruction errors.

For the perturbation theorem, the approximate reconstructions are assumed to be
linear in the source, or at least to satisfy

\[
G_x[\phi-\phi_*]=G_x[\phi]-G_x[\phi_*],
\qquad
G_y[\psi-\psi_*]=G_y[\psi]-G_y[\psi_*].
\]

Keep the same definitions

\[
q_x:=2G_x[\phi-\phi_*],
\qquad
q_y:=2G_y[\psi-\psi_*],
\qquad
q_c:=\frac{q_x+q_y}{2}.
\]

Then

\[
q_x
=
2(u_\phi-u_*-\varepsilon_x),
\]

and

\[
q_y
=
2(u_\psi-u_*-\varepsilon_y).
\]

Define

\[
\delta\varepsilon:=\varepsilon_x-\varepsilon_y,
\qquad
s:=(u_\phi-u_\psi)-\delta\varepsilon.
\]

Then

\[
q_x-q_y=2s,
\]

so

\[
q_x=q_c+s,
\qquad
q_y=q_c-s.
\]

**Theorem 2: Perturbed energy-consistency control.**  
Assume the same structural hypotheses as Theorem 1, except replace exact
reference reconstruction with the perturbed reconstruction identities above.  If
the weak inverse identities remain valid for the error variables \(q_x,q_y\),
then

\[
\|q_c\|_a
\le
C_E
\left(
\sqrt{\mathcal{E}_{\mathrm{split}}}
+
\|\varepsilon_x-\varepsilon_y\|_a
\right),
\]

and

\[
\|q_x\|_a,\ \|q_y\|_a
\le
(1+C_E)
\left(
\sqrt{\mathcal{E}_{\mathrm{split}}}
+
\|\varepsilon_x-\varepsilon_y\|_a
\right).
\]

**Proof.**  
The projection consistency argument still gives

\[
L_xq_x+L_yq_y=0.
\]

Substitute

\[
q_x=q_c+s,
\qquad
q_y=q_c-s.
\]

Then

\[
L_x(q_c+s)+L_y(q_c-s)=0,
\]

and hence

\[
Lq_c=-(L_x-L_y)s.
\]

The same coercivity and boundedness argument used in Theorem 1 gives

\[
\|q_c\|_a
\le
C_E\|s\|_a.
\]

But

\[
\|s\|_a
=
\|(u_\phi-u_\psi)-(\varepsilon_x-\varepsilon_y)\|_a
\le
\|u_\phi-u_\psi\|_a+\|\varepsilon_x-\varepsilon_y\|_a.
\]

Since

\[
\|u_\phi-u_\psi\|_a
=
\sqrt{\mathcal{E}_{\mathrm{split}}},
\]

we obtain the stated estimate for \(q_c\).  The estimates for \(q_x\) and
\(q_y\) follow from

\[
q_x=q_c+s,
\qquad
q_y=q_c-s.
\]

This proves Theorem 2.

**Corollary: perturbed final solution error bound.**  
Under the assumptions of Theorem 2, define

\[
u_{\mathrm{pred}}
:=
\frac12(u_\phi+u_\psi).
\]

Using

\[
q_c
=
u_\phi+u_\psi-2u_*-\varepsilon_x-\varepsilon_y,
\]

we have

\[
u_{\mathrm{pred}}-u_*
=
\frac12q_c
+
\frac12(\varepsilon_x+\varepsilon_y).
\]

Therefore,

\[
\|u_{\mathrm{pred}}-u_*\|_a
\le
\frac{C_E}{2}
\left(
\sqrt{\mathcal{E}_{\mathrm{split}}}
+
\|\varepsilon_x-\varepsilon_y\|_a
\right)
+
\frac12\|\varepsilon_x+\varepsilon_y\|_a.
\]

The first perturbation term \(\varepsilon_x-\varepsilon_y\) measures mismatch
between the two exact-split Green reconstruction errors.  The second term
\(\varepsilon_x+\varepsilon_y\) contains the common Green bias and cannot be
removed by energy consistency alone.

The perturbation theorem has an important interpretation.  Energy consistency
penalizes disagreement between the two represented solutions, and the theorem
also sees the branch mismatch error \(\varepsilon_x-\varepsilon_y\).  However,
it does not directly control a common Green reconstruction bias.  If

\[
\varepsilon_x=\varepsilon_y=\varepsilon,
\]

then

\[
\varepsilon_x-\varepsilon_y=0.
\]

The mismatch term disappears, but the represented solution errors still contain
the individual Green reconstruction bias:

\[
2(u_\phi-u_*)=q_x+2\varepsilon_x,
\qquad
2(u_\psi-u_*)=q_y+2\varepsilon_y.
\]

Thus, a common Green bias can be invisible to energy consistency while still
affecting the actual solution error.

If a learned Green operator also fails the weak inverse identity, additional
inverse residuals enter the structural identity.  These residuals would appear
as extra dual-energy error terms.  The present perturbation theorem isolates the
reference reconstruction error but does not claim to cover every possible
learned-operator residual.

## 8. Admissibility Discussion

The assumption

\[
u_\phi,u_\psi,u_*\in H_0^1(\Omega)
\]

is essential for the energy method.  It ensures that the full-domain energy norm
and weak bilinear forms are meaningful.

The three memberships in this assumption have different mathematical origins.
The reference solution \(u_*\in H_0^1(\Omega)\) is the weak solution of the
full elliptic problem.  Under the coefficient bounds and coercivity assumptions
above, this is the standard energy setting for the reference PDE.

The represented solutions \(u_\phi=G_x[\phi]\) and \(u_\psi=G_y[\psi]\) are
different.  They are not obtained by directly solving the full two-dimensional
PDE.  Instead, they are assembled from one-dimensional connected-interval Green
reconstructions.  Their full-domain admissibility is therefore a condition on
the reconstructed fields.

Endpoint zero values are necessary but not sufficient.  Connected-interval
Green reconstruction enforces zero endpoint values on each one-dimensional
interval.  For example, on a horizontal connected interval \(I_k^x(y)\), the
reconstructed x-direction function vanishes at the two interval endpoints.
This is the correct slice-wise Dirichlet compatibility condition, but
\(H_0^1(\Omega)\) requires more:

\[
u\in H^1(\Omega),
\qquad
\operatorname{Tr}_{\partial\Omega}u=0.
\]

The trace condition is a full-domain Sobolev statement.  Without first knowing
that the assembled field lies in \(H^1(\Omega)\), interval endpoint values alone
do not prove membership in \(H_0^1(\Omega)\).

The missing issue is transverse regularity.  The operator \(G_x\) is elliptic in
the x-direction on each horizontal connected interval, so it naturally controls
the axial derivative of \(u_\phi\).  It does not automatically control
\(\partial_yu_\phi\).  As \(y\) varies, the interval endpoints, interval length,
coefficients, source profile, and one-dimensional solution may all vary.  If
this dependence is too rough, the assembled function can fail to have
\(\partial_yu_\phi\in L^2(\Omega)\).  The y-direction reconstruction has the
analogous issue: \(G_y\) controls the vertical axial derivative of \(u_\psi\),
but \(\partial_xu_\psi\in L^2(\Omega)\) is an additional transverse
regularity requirement.

Moving interval geometry is part of this obstruction.  In a complex domain, the
connected interval endpoints and lengths can change with the transverse
coordinate, and the number of connected components can change at exceptional
slices.  Near a narrowing portion of the domain, interval lengths may degenerate
to zero.  Even for smooth domain boundaries, this can require weighted
integrability estimates to ensure that the assembled reconstructions remain in
the full energy space.

A possible sufficient-condition route would be to assume that slice endpoints
vary regularly in the transverse variable, that the coefficients and source
profiles have compatible transverse regularity, and that the corresponding
parameter-dependent one-dimensional Green solutions satisfy transverse
derivative estimates.  Under such a separate regularity theorem, the maps

\[
G_x:\mathcal{S}_x\to H_0^1(\Omega),
\qquad
G_y:\mathcal{S}_y\to H_0^1(\Omega)
\]

could be justified for appropriate admissible source spaces
\(\mathcal{S}_x,\mathcal{S}_y\).  This document does not prove that theorem.
It instead keeps full-domain admissibility as a structural assumption.

This is the conservative choice for an energy-consistency theorem.  The theorem
explains what energy consistency controls once the represented fields are
admissible full-domain energy functions; it does not claim that connected
interval endpoint conditions automatically produce such fields.

In numerical complex-mode training, the energy loss is evaluated on valid
same-segment edges.  That discrete construction should be understood as a
finite-dimensional analogue of the continuous energy principle, not as a direct
proof that the continuous \(H_0^1(\Omega)\) assumptions automatically hold.

## 9. Why \(L^2\)-Consistency Is Insufficient

The same proof cannot be obtained from an \(L^2\)-consistency loss alone.  If
one used

\[
\mathcal{L}_0
:=
\|u_\phi-u_\psi\|_{L^2(\Omega)}^2
=
\|r\|_{L^2(\Omega)}^2,
\]

then the loss would control only the amplitude of \(r\), not its derivatives.

The structural identity requires bounding

\[
B_x(r,v)-B_y(r,v).
\]

This expression contains derivative terms:

\[
\int_\Omega a(\partial_xr\,\partial_xv-\partial_yr\,\partial_yv)
+
\int_\Omega (b_x\partial_xr-b_y\partial_yr)v.
\]

Thus the energy method requires control of \(\nabla r\), naturally provided by
\(\|r\|_a\), but not by \(\|r\|_{L^2}\).

The obstruction is local and does not depend on the global domain being a
square.  On any small rectangle contained in \(\Omega\), one can construct
oscillatory functions whose \(L^2\)-norm stays bounded while their gradients
grow.  Such functions make the derivative terms in \(B_x-B_y\) large while an
\(L^2\)-only loss remains small or bounded.  Therefore, \(L^2\)-consistency
alone cannot support the same theorem without additional regularity, spectral,
or smoothing assumptions.

For example, choose \(r_n\in C_c^\infty(Q)\) oscillatory in \(x\) on a small
rectangle \(Q\Subset\Omega\), and take \(v_n=r_n\).  Then
\(\|r_n\|_{L^2}\) can remain bounded while the diffusion part of
\(B_x(r_n,v_n)-B_y(r_n,v_n)\) grows like the squared frequency.

## 10. Interpretation for the Framework

The energy-consistency theorem should be read as a conditional structural
statement:

\[
\text{small } \mathcal{E}_{\mathrm{split}}
\quad
\Rightarrow
\quad
\text{small split-error variables}
\]

provided the connected-interval reconstructions are admissible weak inverses and
the projected split is balanced.

The theorem explains why the energy loss is more than a heuristic agreement
loss.  It directly controls the difference component \(u_\phi-u_\psi\), and the
projection constraint converts this control into an estimate for the average
split-error component.  The average-difference decomposition then controls the
two directional split-error variables.

At the same time, the theorem does not claim that energy consistency alone
solves every error source in the learned model.  It does not by itself prove
full solution accuracy when the Green reconstructions have a common bias.  It
does not remove inverse residuals from imperfect learned Green operators.  It
also does not prove that intervalwise reconstructions automatically lie in
\(H_0^1(\Omega)\).

The final solution prediction is

\[
u_{\mathrm{pred}}
=
\frac12(u_\phi+u_\psi).
\]

Energy consistency supports this average by encouraging the two represented
solutions to agree in the PDE energy geometry.  The theorem therefore justifies
the consistency part of the framework under explicit continuous assumptions,
while leaving Green accuracy, source-split approximation, and admissibility as
separate requirements.
