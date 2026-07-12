# Complex Geometry에서의 GreenNet/CouplingNet Framework Technical Report

## Abstract

이 문서는 복잡한 2차원 domain에서 elliptic partial differential equation의 solution을
재구성하기 위한 GreenNet/CouplingNet framework를 정리한다. 목적은 학회 발표자료를 직접
작성하는 것이 아니라, 발표자료를 구성할 때 사용할 수 있는 수학적, 알고리즘적 근거를 제공하는
것이다.

Framework의 핵심은 2차원 문제를 axial direction에 따른 1차원 Green operator와
source-decomposition 문제로 나누는 데 있다. GreenNet은 complex geometry와 axial line의
교차에서 생기는 connected interval 위의 1차원 Green operator를 학습한다. CouplingNet은
source, coefficient, interval geometry, transverse position을 이용해 forcing term을 두 방향의
split field로 분해한다. Projection은 이 split field가 physical balance를 만족하도록 만들고,
Green reconstruction은 split field를 solution contribution으로 되돌린다.

이 framework는 복잡한 domain 전체를 하나의 reference domain으로 mapping하지 않는다. 대신
각 axial connected interval을 독립적인 1차원 boundary value problem으로 해석하고, 그 interval을
unit interval로 pull-back한다. 이 관점은 circle, annulus, disconnected intersection을 포함하는
general complex geometry를 같은 수학적 구조 안에서 다룰 수 있게 한다.

## 1. Problem Setting

Let \(\Omega\subset\mathbb{R}^2\) be a bounded complex domain. The target problem is
an elliptic equation of the form

\[
-\partial_x(a\,\partial_x u)
-\partial_y(a\,\partial_y u)
+b_x\,\partial_xu
+b_y\,\partial_yu
+c\,u
=f
\qquad \text{in }\Omega,
\]

with homogeneous Dirichlet boundary condition

\[
u=0
\qquad \text{on }\partial\Omega.
\]

The quantity \(u\) is the solution to be reconstructed. The right-hand side \(f\)
is the source. The scalar coefficient \(a\) controls diffusion, the vector
\((b_x,b_y)\) controls convection, and \(c\) controls reaction. The diffusion term
governs smoothing and energy geometry, convection introduces directional transport,
and reaction modifies local amplitude through damping or amplification.

The central difficulty is that \(\Omega\) is not assumed to be a rectangle or a
simple product domain. A horizontal or vertical line can intersect \(\Omega\) in
short intervals, long intervals, or multiple disconnected intervals. Therefore a
method that relies on a single global rectangular coordinate structure does not
represent the natural geometry of the problem. The framework instead treats each
connected axial intersection as the primitive one-dimensional domain on which an
axial Green operator is defined.

## 2. Axial Source Decomposition

The elliptic operator is decomposed into x-direction and y-direction source
components. Define

\[
\phi
=
-\partial_x(a\,\partial_xu)
+b_x\,\partial_xu
+\frac12 c\,u,
\]

\[
\psi
=
-\partial_y(a\,\partial_yu)
+b_y\,\partial_yu
+\frac12 c\,u.
\]

Then the original PDE can be written as the balance relation

\[
\phi+\psi=f.
\]

This decomposition is not merely a notational device. It is the interface between
CouplingNet and GreenNet. CouplingNet learns a source-conditioned decomposition of
\(f\) into \(\phi\) and \(\psi\). GreenNet then interprets each component as the
right-hand side of an axial one-dimensional problem and reconstructs a represented
solution contribution.

Let \(\mathcal{G}_x\) denote the x-direction axial Green reconstruction operator
and \(\mathcal{G}_y\) denote the y-direction axial Green reconstruction operator.
After a projected split field is obtained, the two represented solutions are

\[
u_\phi = \mathcal{G}_x[\phi],
\qquad
u_\psi = \mathcal{G}_y[\psi].
\]

The final prediction is the symmetric represented solution

\[
u_{\mathrm{pred}}
=
\frac12\left(u_\phi+u_\psi\right).
\]

Thus GreenNet and CouplingNet have complementary roles. GreenNet supplies the
axial solution operators; CouplingNet supplies the source decomposition needed to
use those operators for the two-dimensional PDE.

## 3. Complex Geometry and Axial Connected Intervals

For a horizontal line \(y=\rho\), the intersection with \(\Omega\) may be empty,
one interval, or multiple disconnected intervals:

\[
\Omega\cap\{(x,\rho):x\in\mathbb{R}\}
=
\bigcup_{k} I_k^x.
\]

Likewise, for a vertical line \(x=\rho\),

\[
\Omega\cap\{(\rho,y):y\in\mathbb{R}\}
=
\bigcup_{\ell} I_\ell^y.
\]

Each connected component is treated as an independent one-dimensional boundary
value problem. This is essential. If two separated components on the same axial
line are merged into a single longer interval, the resulting one-dimensional
problem would permit information to pass through a region outside the physical
domain. That would define the wrong Green operator.

For a connected interval

\[
I=[s_0,s_1],
\qquad
L=s_1-s_0>0,
\]

the endpoints \(s_0\) and \(s_1\) act as the one-dimensional Dirichlet boundary.
The interval length \(L\), the midpoint, and the transverse location of the
line are all part of the local geometry. In a circle these quantities vary
smoothly across axial lines. In an annulus, an axial line may intersect the domain
in two separated intervals because of the inner hole. In more general domains,
the same principle applies: the connected interval, not the full line, is the
unit of the axial Green problem.

This representation separates the geometry into two roles. Along the interval,
the local coordinate describes movement in the primary axial direction. Across
parallel intervals, the transverse coordinate describes where that interval sits
inside the two-dimensional domain.

## 4. GreenNet: Axial Green Operator

GreenNet learns the one-dimensional Green operator associated with each connected
axial interval. Consider a physical interval

\[
I=[s_0,s_1],
\qquad
L=s_1-s_0.
\]

The physical coordinate \(s\) is mapped to a normalized coordinate
\(t\in[0,1]\) by

\[
s=s_0+Lt.
\]

For a physical solution \(u(s)\), define the normalized solution

\[
v(t)=u(s_0+Lt).
\]

The source coordinate is transformed in the same way:

\[
r=s_0+L\eta,
\qquad
\eta\in[0,1].
\]

The physical one-dimensional operator along the interval has the form

\[
-\frac{d}{ds}
\left(
a_{\mathrm{phys}}(s)\frac{du}{ds}
\right)
+b_{\mathrm{phys}}(s)\frac{du}{ds}
+c_{\mathrm{phys}}(s)u(s)
=
f_{\mathrm{phys}}(s).
\]

Under the pull-back \(s=s_0+Lt\), the equivalent unit-interval problem is written
using scaled coefficients and scaled source:

\[
a_{\mathrm{unit}}(t)
=
a_{\mathrm{phys}}(s_0+Lt),
\]

\[
a'_{\mathrm{unit}}(t)
=
L\,a'_{\mathrm{phys}}(s_0+Lt),
\]

\[
b_{\mathrm{unit}}(t)
=
L\,b_{\mathrm{phys}}(s_0+Lt),
\]

\[
c_{\mathrm{unit}}(t)
=
L^2\,c_{\mathrm{phys}}(s_0+Lt),
\]

\[
f_{\mathrm{unit}}(t)
=
L^2\,f_{\mathrm{phys}}(s_0+Lt).
\]

With these definitions, the normalized Green kernel \(G_{\mathrm{unit}}\) satisfies
the reconstruction relation

\[
v(t)
=
\int_0^1
G_{\mathrm{unit}}(t,\eta)
f_{\mathrm{unit}}(\eta)
\,d\eta.
\]

No additional factor of \(L\) is multiplied into this unit-coordinate
reconstruction. The interval length has already entered through the transformed
convection, reaction, source, and diffusion-derivative terms. Multiplying the
kernel by another length factor inside this normalized reconstruction would double
count the coordinate transformation.

GreenNet has two conceptual inputs. The branch represents the one-dimensional
differential operator by encoding the coefficient profiles along the interval.
The trunk represents the coordinate pair \((t,\eta)\), where \(t\) is the
solution evaluation coordinate and \(\eta\) is the source coordinate. The output is
interpreted as a normalized Green response:

\[
(t,\eta)\mapsto G_{\mathrm{unit}}(t,\eta).
\]

More precisely, GreenNet represents the normalized kernel as a combination of a
coefficient-dependent learned correction and an analytic Green structure. The
coefficient branch first encodes the unit-interval operator profiles:

\[
z_a=\mathcal{B}_a[a_{\mathrm{unit}}],
\qquad
z_{a'}=\mathcal{B}_{a'}[a'_{\mathrm{unit}}],
\]

\[
z_b=\mathcal{B}_b[b_{\mathrm{unit}}],
\qquad
z_c=\mathcal{B}_c[c_{\mathrm{unit}}].
\]

These representations are fused into an operator representation

\[
z_{\mathrm{op}}
=
\mathcal{F}
\left(
z_a,z_{a'},z_b,z_c
\right).
\]

The trunk encodes the relation between the evaluation coordinate and the source
coordinate:

\[
z_{\mathrm{tr}}(t,\eta)=\mathcal{T}(t,\eta).
\]

The neural part of the Green kernel is then a learned correction

\[
R_\theta(t,\eta)
=
z_{\mathrm{op}}^\top z_{\mathrm{tr}}(t,\eta)+\beta.
\]

This correction is not used alone as the Green kernel. It is inserted into a
structured expression that already contains the boundary-compatible analytic
response of the unit interval. Define the base Green response

\[
G_0(t,\eta)
=
\begin{cases}
t(1-\eta), & t<\eta,\\
\eta(1-t), & t\ge \eta,
\end{cases}
\]

and the associated antiderivative-like term

\[
J_0(t,\eta)
=
\begin{cases}
\frac12 t^2(1-\eta), & t<\eta,\\
\frac12 \eta(2t-t^2-\eta), & t\ge \eta.
\end{cases}
\]

The evaluation-side coefficient factors are

\[
A(t)
=
\frac{1}{a_{\mathrm{unit}}(t)},
\]

\[
B(t)
=
\frac{
a'_{\mathrm{unit}}(t)+b_{\mathrm{unit}}(t)
}{
a_{\mathrm{unit}}(t)^2
}.
\]

Finally, define the boundary-compatible envelope

\[
E(t,\eta)=t\eta(1-\eta),
\qquad
M(t)=1-t.
\]

The GreenNet kernel used for reconstruction is

\[
G_\theta(t,\eta)
=
E(t,\eta)M(t)R_\theta(t,\eta)
+
B(t)
\left(
J_0(t,\eta)-\frac12E(t,\eta)
\right)
+
A(t)G_0(t,\eta).
\]

This expression is best interpreted through the distributional structure of a
Green function. The purpose of the analytic part is not merely to provide a
smooth baseline. It fixes the singular behavior that must occur at
\(t=\eta\).

First, the base kernel \(G_0\) has the derivative jump

\[
\partial_tG_0(t,\eta)
=
\begin{cases}
1-\eta, & t<\eta,\\
-\eta, & t>\eta,
\end{cases}
\]

so, with the convention
\[
[q]_{t=\eta}=q(\eta^+)-q(\eta^-),
\]

one obtains

\[
\left[\partial_tG_0\right]_{t=\eta}
=
-1.
\]

Therefore, in the sense of distributions,

\[
\partial_t^2G_0(t,\eta)
=
-\delta(t-\eta),
\qquad
-\partial_t^2G_0(t,\eta)
=
\delta(t-\eta).
\]

This is the essential Green-function singularity. It is the mechanism by which a
source concentrated at \(\eta\) appears when the differential operator is applied
to the kernel.

For variable diffusion, the factor

\[
A(t)=\frac{1}{a_{\mathrm{unit}}(t)}
\]

is inserted so that the conservative diffusion operator has the correct
\(\delta\)-normalization. To see this, temporarily ignore the reaction term and
write the first two parts of the unit-interval operator as

\[
\mathcal{L}_{0}u
=
-\partial_t
\left(
a_{\mathrm{unit}}(t)\partial_tu
\right)
+
b_{\mathrm{unit}}(t)\partial_tu.
\]

Applying this operator to \(A(t)G_0(t,\eta)\) begins with

\[
a_{\mathrm{unit}}(t)\partial_t
\left(
A(t)G_0(t,\eta)
\right)
=
a_{\mathrm{unit}}(t)A'(t)G_0(t,\eta)
+
\partial_tG_0(t,\eta),
\]

because \(a_{\mathrm{unit}}(t)A(t)=1\). The derivative of
\(\partial_tG_0\) produces the desired Dirac term:

\[
-\partial_t\partial_tG_0(t,\eta)
=
\delta(t-\eta).
\]

Thus \(A(t)G_0(t,\eta)\) is the analytic structure that enforces the
Dirac-\(\delta\) property of the Green kernel under variable diffusion. It is not
only a diffusion-scaled base response; it is the term that supplies the necessary
jump singularity.

However, the same operator application produces an additional non-Dirac term
involving \(\partial_tG_0\). Collecting the terms proportional to
\(\partial_tG_0\) gives

\[
-a_{\mathrm{unit}}(t)A'(t)\partial_tG_0(t,\eta)
+
b_{\mathrm{unit}}(t)A(t)\partial_tG_0(t,\eta).
\]

The coefficient multiplying \(\partial_tG_0\) is

\[
-a_{\mathrm{unit}}A'
+
b_{\mathrm{unit}}A
=
\frac{
a'_{\mathrm{unit}}+b_{\mathrm{unit}}
}{
a_{\mathrm{unit}}
}
=
a_{\mathrm{unit}}B(t).
\]

Therefore \(A(t)G_0(t,\eta)\) creates the desired Dirac-\(\delta\) contribution,
but it also leaves a Heaviside-type contribution of the form

\[
a_{\mathrm{unit}}(t)B(t)\partial_tG_0(t,\eta).
\]

This contribution is not another source singularity. Since
\(\partial_tG_0(t,\eta)\) is piecewise constant with different values on the two
sides of \(t=\eta\), it has the character of a step term.

The role of the second analytic component is to cancel this step contribution.
Define

\[
S(t,\eta)
=
J_0(t,\eta)-\frac12E(t,\eta).
\]

The functions \(S\) and \(\partial_tS\) remain continuous at \(t=\eta\), so this
term does not introduce a new Dirac singularity. Its second derivative is

\[
\partial_t^2S(t,\eta)
=
\partial_t^2J_0(t,\eta)
=
\partial_tG_0(t,\eta),
\]

because \(E(t,\eta)\) is linear in \(t\). Hence, when the leading diffusion part
of \(\mathcal{L}_0\) is applied to \(B(t)S(t,\eta)\), the leading contribution is

\[
-a_{\mathrm{unit}}(t)B(t)\partial_t^2S(t,\eta)
=
-a_{\mathrm{unit}}(t)B(t)\partial_tG_0(t,\eta).
\]

This cancels the Heaviside-type contribution generated by \(A(t)G_0(t,\eta)\).
In this sense,

\[
B(t)
\left(
J_0(t,\eta)-\frac12E(t,\eta)
\right)
\]

is an analytic compensation term. Its purpose is to remove the step contribution
that appears while enforcing the Dirac-\(\delta\) property.

After the Dirac jump and the leading Heaviside cancellation have been built into
the analytic structure, remaining terms can still arise from coefficient
derivatives, convection interactions, and reaction. These terms are regular in
comparison with the singular distributional structure. The neural correction

\[
E(t,\eta)M(t)R_\theta(t,\eta)
\]

learns this remaining smooth correction while the envelope keeps the correction
compatible with the endpoint boundary structure. The reaction profile
\(c_{\mathrm{unit}}\) does not directly create the Dirac jump or the Heaviside
cancellation factor; it enters through the operator branch and affects the
learned correction \(R_\theta\).

The learning target is not best understood as direct pointwise supervision of a
known Green kernel. The more important objective is source-to-solution
reconstruction:

\[
f_{\mathrm{unit}}
\quad\longmapsto\quad
\int_0^1
G_\theta(\cdot,\eta)
f_{\mathrm{unit}}(\eta)
\,d\eta.
\]

Equivalently, for a particular evaluation coordinate,

\[
v_\theta(t)
=
\int_0^1
G_\theta(t,\eta)
f_{\mathrm{unit}}(\eta)
\,d\eta.
\]

In other words, GreenNet is trained so that the learned kernel acts as a valid
solution operator for the normalized one-dimensional boundary value problem.

### Forward 방식의 Supervised Dataset 구성

GreenNet 학습에서 supervised target은 exact Green kernel 자체가 아니다. Forward
construction은 먼저 smooth target solution을 생성하고, 그 solution에 unit-interval
differential operator를 적용해 source를 계산한다. 따라서 학습 pair는 Green
kernel label이 아니라 source-to-solution relation

\[
\tilde f_{\mathrm{unit}}^{(q)}
\quad\longmapsto\quad
\tilde v^{(q)}
\]

으로 구성된다.

각 connected axial interval

\[
I=[s_0,s_1],
\qquad
s=s_0+Lt,
\qquad
t\in[0,1]
\]

에서 coefficient profiles

\[
a_{\mathrm{unit}},\quad
a'_{\mathrm{unit}},\quad
b_{\mathrm{unit}},\quad
c_{\mathrm{unit}}
\]

는 interval geometry와 physical coefficient field에 의해 고정된다. 같은 interval에서
여러 supervised sample을 만들 때 달라지는 것은 local operator가 아니라 generated
target solution과 그에 대응하는 source이다.

Smooth target solution은 unit interval 위의 Gaussian Process sample로 생성된다:

\[
w^{(q)}(t)\sim\mathcal{GP}(0,k_\ell),
\]

\[
k_\ell(t,t')
=
\sigma^2
\exp
\left(
-\frac{|t-t'|^2}{2\ell^2}
\right).
\]

\(\ell\)은 generated target solution의 smoothness scale을 조절한다. 큰
\(\ell\)은 interval을 따라 천천히 변하는 sample을 만들고, 작은 \(\ell\)은 더
급격한 variation을 허용한다.

Green problem은 homogeneous Dirichlet boundary condition을 사용하므로, sampled
function을 그대로 target으로 쓰지 않는다. Endpoint value를 잇는 linear
interpolant를 제거해 boundary-compatible target을 만든다:

\[
\ell_w(t)
=
(1-t)w^{(q)}(0)+tw^{(q)}(1),
\]

\[
v^{(q)}(t)
=
w^{(q)}(t)-\ell_w(t).
\]

그러면

\[
v^{(q)}(0)=0,
\qquad
v^{(q)}(1)=0.
\]

Source는 독립적으로 sampling하지 않는다. Source는 generated target solution에
unit-interval operator를 적용해 manufactured된다. Operator를

\[
\mathcal{L}_{\mathrm{unit}}v
=
-\frac{d}{dt}
\left(
a_{\mathrm{unit}}(t)\frac{dv}{dt}
\right)
+
b_{\mathrm{unit}}(t)\frac{dv}{dt}
+
c_{\mathrm{unit}}(t)v(t).
\]

로 정의하면, generated target \(v^{(q)}\)에 대응하는 source는

\[
f_{\mathrm{unit}}^{(q)}(t)
=
-a'_{\mathrm{unit}}(t)\frac{dv^{(q)}}{dt}
-
a_{\mathrm{unit}}(t)\frac{d^2v^{(q)}}{dt^2}
+
b_{\mathrm{unit}}(t)\frac{dv^{(q)}}{dt}
+
c_{\mathrm{unit}}(t)v^{(q)}(t).
\]

이다. 따라서 모든 generated pair는 construction에 의해 unit-interval PDE relation을
만족한다:

\[
\mathcal{L}_{\mathrm{unit}}v^{(q)}
=
f_{\mathrm{unit}}^{(q)}.
\]

이 pair는 target solution magnitude로 normalize된다:

\[
\gamma^{(q)}
=
\max
\left(
\left(
\int_0^1 |v^{(q)}(t)|^2\,dt
\right)^{1/2},
\varepsilon
\right),
\]

\[
\tilde v^{(q)}(t)
=
\frac{v^{(q)}(t)}{\gamma^{(q)}},
\qquad
\tilde f_{\mathrm{unit}}^{(q)}(t)
=
\frac{f_{\mathrm{unit}}^{(q)}(t)}{\gamma^{(q)}}.
\]

\(\mathcal{L}_{\mathrm{unit}}\)은 \(v\)에 대해 linear이므로, 이 normalization은
PDE consistency를 보존한다:

\[
\mathcal{L}_{\mathrm{unit}}\tilde v^{(q)}
=
\tilde f_{\mathrm{unit}}^{(q)}.
\]

각 generated pair에 대해 GreenNet은 다음 reconstruction을 만든다:

\[
\tilde v_\theta^{(q)}(t)
=
\int_0^1
G_\theta(t,\eta)
\tilde f_{\mathrm{unit}}^{(q)}(\eta)
\,d\eta.
\]

Supervised objective는 이 reconstruction이 normalized target과 일치하도록 만드는
것이다:

\[
\tilde v_\theta^{(q)}(t)
\approx
\tilde v^{(q)}(t).
\]

개념적으로 GreenNet training objective는

\[
\mathcal{J}_{\mathrm{Green}}
=
\mathbb{E}_{I,q}
\left[
\int_0^1
\left|
\tilde v_\theta^{(I,q)}(t)
-
\tilde v^{(I,q)}(t)
\right|^2
\,dt
\right].
\]

으로 볼 수 있다. 이 construction에서 GreenNet은 supervised reconstruction pair로
학습한다. Label은 generated target solution이고, source는 그 target에 알려진
unit-interval operator를 적용한 결과이다. 이 report는 이 forward construction만
다룬다.

## 5. CouplingNet: Source-Conditioned Split Field Model

CouplingNet learns the direction-split source fields \(\phi\) and \(\psi\). The
model must be source-conditioned because the correct decomposition depends on the
right-hand side \(f\). Even with the same geometry and the same coefficients, a
different source can require a different distribution of forcing between the
x-direction and y-direction axial reconstructions.

The source branch encodes the physical forcing profile on each connected interval.
After endpoint handling and interpolation on the unit coordinate, its amplitude is

\[
A
=
\left(
\int_0^1
|f_{\mathrm{phys}}(s_0+Lt)|^2\,dt
\right)^{1/2}.
\]

The source branch receives \(f_{\mathrm{phys}}/A\), and each directional model
output is multiplied by \(A\). Consequently, CouplingNet returns raw physical split
fields rather than raw unit Green sources. Segment length remains explicit in the
geometry and coefficient context; the \(L^2\) source conversion is deferred until
after physical projection.

The coefficient branch encodes the local differential operator. Diffusion,
convection, and reaction determine how forcing should be split and how it will
propagate through the axial Green reconstruction. The coefficient branch answers
the question: under this local operator, what kind of source decomposition is
appropriate?

For convection, the relevant operator context is not only the convection along
the primary axial direction.  In a complex domain, the same physical point also
lies on a transverse axial interval whose boundary locations and length may
change from point to point.  CouplingNet therefore separates the convection
context into
\[
b_{\mathrm{primary}}
\qquad\text{and}\qquad
b_{\mathrm{transverse}}.
\]

For the \(\phi\)-path, the primary direction is horizontal and the transverse
direction is vertical, so the convection information is interpreted as

\[
\phi\text{-path}:
\qquad
\left[
L_x b_x,\,
L_x b_y
\right].
\]

For the \(\psi\)-path, the primary direction is vertical and the transverse
direction is horizontal, so the corresponding interpretation is

\[
\psi\text{-path}:
\qquad
\left[
L_y b_y,\,
L_y b_x
\right].
\]

The same primary interval length scales both convection components in the
corresponding path.  This convention gives CouplingNet both the axial transport
coefficient that belongs to the one-dimensional split operator and the
orthogonal transport context that can affect how the source should be divided
near transverse boundaries.

This coefficient-branch convention is separate from Green reconstruction.  The
axial Green operator for a connected interval uses the primary one-dimensional
operator: diffusion, its primary axial derivative, primary convection, and
reaction.  The transverse convection context is used for source-split
prediction, not as an additional coefficient in the one-dimensional Green
reconstruction operator.

The geometry branch provides information about the connected interval as a
physical object. Its mathematical role is to tell the network where the interval
lies, how long it is, and what scale conversion relates local coordinate to
physical coordinate. Without this information, two intervals with similar
normalized source and coefficient profiles could become indistinguishable even
though they occupy different positions or lengths in the physical domain.

The transverse branch describes the position of an interval among parallel
intervals. For a horizontal interval, the transverse coordinate is the vertical
position of the line; for a vertical interval, it is the horizontal position of
the line. This information matters because the two-dimensional coefficient field,
boundary geometry, and source structure vary across the domain. The transverse
branch allows CouplingNet to distinguish axial intervals that have similar
internal coordinate profiles but live in different parts of \(\Omega\). This is
global transverse information: it describes where an entire axial interval is
placed relative to other parallel intervals.

The axial local trunk describes pointwise variation along the primary axial
direction. Let

\[
t_{\parallel}\in[0,1]
\]

denote the normalized coordinate inside the connected interval on which the
direction-split source is being predicted. For the \(\phi\)-path, the primary
direction is the horizontal axial direction, so
\[
\phi\text{-path}: \qquad t_{\parallel}=t_x.
\]

For the \(\psi\)-path, the primary direction is the vertical axial direction, so

\[
\psi\text{-path}: \qquad t_{\parallel}=t_y.
\]

The axial local trunk therefore answers the question: where is the current point
inside the one-dimensional interval that carries the split source being
predicted?

This is different from the transverse branch. The transverse branch can encode a
global transverse placement, which can be denoted abstractly as

\[
r_{\mathrm{global}}.
\]

However, \(r_{\mathrm{global}}\) is an interval-level descriptor. It does not
tell the model where the current point lies inside the transverse axial interval
passing through that same point. In complex geometry, that transverse interval
can have a different length and different boundary endpoints from point to point.
Therefore, global transverse placement alone cannot communicate the local
transverse boundary context needed by directional-split sources.

The pointwise transverse trunk supplies this missing information. Let

\[
t_{\perp}\in[0,1]
\]

be the normalized coordinate of the same physical point inside the transverse
axial interval. Then

\[
\phi\text{-path}: \qquad t_{\perp}=t_y,
\]

because the transverse interval for the horizontal split is vertical, while

\[
\psi\text{-path}: \qquad t_{\perp}=t_x,
\]

because the transverse interval for the vertical split is horizontal. The
pointwise transverse trunk injects pointwise transverse boundary context that
cannot be represented by the interval-level transverse branch.

Conceptually, CouplingNet uses

\[
\text{primary axial variation}
\quad+\quad
\text{pointwise transverse boundary context}
\]

to predict the local value of the directional-split source. The pointwise
transverse trunk does not replace the axial local trunk. It complements it:
the axial local trunk models variation along the split direction, while the
pointwise transverse trunk communicates how the orthogonal axial boundary
structure constrains the same physical point.

Together with the source, coefficient, geometry, and transverse branches, these
trunk components produce raw physical split fields. These fields may not yet satisfy
the balance relation exactly, so projection supplies the final physical correction.

### Learnable Rational Activation

The nonlinear transformations inside the GreenNet and CouplingNet branch/trunk
networks use a learnable rational activation. This activation is not a fixed
elementary function such as a fixed hyperbolic tangent or a fixed rectifier.
Instead, for a scalar pre-activation \(x\), it has the form

\[
\sigma_{\mathrm{rat}}(x)
=
\frac{P_\alpha(x)}{1+|Q_\beta(x)|}.
\]

Here \(P_\alpha\) and \(Q_\beta\) are polynomials. The coefficients
\(\alpha\) and \(\beta\) are trainable parameters, so the shape of the
nonlinearity itself can change during optimization. The initialization used for
the rational activation is

\[
P_{\alpha_0}(x)
=
1.1915x^3
+
1.5957x^2
+
0.5x
+
0.0218,
\]

\[
Q_{\beta_0}(x)
=
2.383x.
\]

These expressions should be interpreted only as the initial activation shape.
They do not define a fixed activation throughout training. During learning,
\(\alpha\) and \(\beta\) are updated together with the other neural network
parameters.

The denominator structure

\[
1+|Q_\beta(x)|
\]

prevents the denominator from becoming zero. This gives the activation the
expressive numerator of a polynomial nonlinearity while retaining a controlled
rational scaling through the denominator.

In GreenNet, this activation is used inside the operator branch and the trunk
that form the learned correction \(R_\theta(t,\eta)\). The analytic Green
wrapping supplies the Dirac-\(\delta\) structure and the Heaviside cancellation
structure; the learnable rational activation helps represent the remaining
smooth correction on top of that analytic structure.

In CouplingNet, the same type of activation provides nonlinear representation
inside the source, coefficient, geometry, and transverse branches, as well as
inside the axial local trunk and pointwise transverse trunk. Its role is to help
model the source-conditioned mapping from forcing and operator context to
directional-split source fields.

The rational activation therefore does not replace the analytic Green wrapping,
the physical balance projection, or the Green reconstruction. It is a learnable
representation nonlinearity used inside the branch/trunk networks that feed
those larger mathematical structures.

## 6. Projection: Enforcing PDE Balance

The split fields must satisfy

\[
\phi+\psi=f.
\]

A neural prediction need not satisfy this relation exactly. Projection enforces
the balance in physical variables. Let \(\phi_{\mathrm{raw}}\) and
\(\psi_{\mathrm{raw}}\) be the model's two raw physical split fields. Define the residual

\[
r
=
f-\phi_{\mathrm{raw}}-\psi_{\mathrm{raw}}.
\]

The symmetric balance projection assigns half of this residual to each component:

\[
\phi_{\mathrm{proj}}
=
\phi_{\mathrm{raw}}+\frac12 r,
\]

\[
\psi_{\mathrm{proj}}
=
\psi_{\mathrm{raw}}+\frac12 r.
\]

Then

\[
\phi_{\mathrm{proj}}+\psi_{\mathrm{proj}}
=
f.
\]

This projection has a simple interpretation. CouplingNet is allowed to learn the
difference mode between the two directions, while projection fixes the sum mode
required by the PDE. The split is therefore constrained to be physically
consistent before Green reconstruction is applied.

The projection is applied directly in physical source variables. The equation
\(\phi+\psi=f\) is a statement about the physical PDE on \(\Omega\), and the
equal-half correction preserves the network's raw difference mode. Only after this
projection are the Green reconstruction inputs formed as
\(\Phi_{\mathrm{unit}}=L_x^2\phi_{\mathrm{proj}}\) and
\(\Psi_{\mathrm{unit}}=L_y^2\psi_{\mathrm{proj}}\).

## 7. Green Reconstruction and Final Solution

After projection, each split field is passed through its axial Green
reconstruction operator. The x-direction component gives

\[
u_\phi
=
\mathcal{G}_x[\phi_{\mathrm{proj}}],
\]

and the y-direction component gives

\[
u_\psi
=
\mathcal{G}_y[\psi_{\mathrm{proj}}].
\]

The final solution prediction is

\[
u_{\mathrm{pred}}
=
\frac12
\left(
u_\phi+u_\psi
\right).
\]

The two represented solutions play two roles. First, their average provides the
solution prediction. Second, their difference measures whether the learned
source decomposition is internally consistent. In an ideal decomposition, the
x-direction and y-direction reconstructions describe the same physical solution:

\[
u_\phi \approx u_\psi \approx u.
\]

Thus the framework does not ask CouplingNet to output \(u\) directly. Instead,
CouplingNet outputs a physically balanced source decomposition, and GreenNet maps
that decomposition into solution space.

## 8. Training Objective and Evaluation Interpretation

GreenNet training is best understood through source-to-solution reconstruction.
For a normalized interval source \(f_{\mathrm{unit}}\), the learned kernel should
produce the corresponding normalized solution:

\[
v_{\mathrm{pred}}(t)
=
\int_0^1
G_{\mathrm{unit}}(t,\eta)
f_{\mathrm{unit}}(\eta)
\,d\eta.
\]

The reconstruction error compares \(v_{\mathrm{pred}}\) with the reference
one-dimensional solution \(v\). This objective encourages the learned kernel to
behave as an operator, not merely as a table of values.

CouplingNet training is interpreted through the solution consistency induced by
the projected split fields. Once \(\phi_{\mathrm{proj}}\) and
\(\psi_{\mathrm{proj}}\) are obtained, the two represented solutions
\(u_\phi\) and \(u_\psi\) should agree. A natural energy interpretation is

\[
\mathcal{E}_{\mathrm{split}}
=
\int_{\Omega}
a\,\left|\nabla(u_\phi-u_\psi)\right|^2
\,dx.
\]

This expression penalizes disagreement between the two axial reconstructions in
the diffusion-weighted energy geometry of the PDE. It is stronger than comparing
only pointwise values because it also measures gradient-level inconsistency.

The final solution error is interpreted by comparing

\[
u_{\mathrm{pred}}
=
\frac12(u_\phi+u_\psi)
\]

with the reference solution \(u\). When direction-split target fields are
available, the projected \(\phi\) and \(\psi\) can also be compared against those
operator components:

\[
\phi
=
-\partial_x(a\,\partial_xu)
+b_x\,\partial_xu
+\frac12cu,
\]

\[
\psi
=
-\partial_y(a\,\partial_yu)
+b_y\,\partial_yu
+\frac12cu.
\]

These comparisons have different meanings. Solution error measures whether the
final reconstructed solution is accurate. Split-field error measures whether the
learned decomposition matches a particular direction-wise operator split.
Energy consistency measures whether the two Green reconstructions agree as
solutions of the same physical PDE.

### Energy-Norm Error Bound Proposition

**Setup.**  Energy consistency is interpreted through the diffusion-weighted
energy norm

\[
\|v\|_a^2
=
\int_\Omega a|\nabla v|^2,
\qquad
\mathcal{E}_{\mathrm{split}}
=
\|u_\phi-u_\psi\|_a^2.
\]

The directional split operators use the same reaction convention as the
framework:

\[
L_xu
=
-\partial_x(a\partial_xu)
+b_x\partial_xu
+\frac12cu,
\qquad
L_yu
=
-\partial_y(a\partial_yu)
+b_y\partial_yu
+\frac12cu.
\]

After projection, the split fields satisfy the physical balance

\[
\phi+\psi=f.
\]

The two represented solutions are reconstructed from the split sources by
connected-interval Green operators:

\[
u_\phi=G_x[\phi],
\qquad
u_\psi=G_y[\psi].
\]

**Proposition (exact energy-loss error bound).**  Assume exact
connected-interval Green reconstruction for the reference split, source-linearity
of the reconstructions, weak inverse identities, projected balance
\(\phi+\psi=f\), and full-domain admissibility

\[
u_\phi,u_\psi,u_*\in H_0^1(\Omega).
\]

Then the final prediction

\[
u_{\mathrm{pred}}
=
\frac12(u_\phi+u_\psi)
\]

satisfies the energy-error bound

\[
\|u_{\mathrm{pred}}-u_*\|_a
\le
\frac{C_E}{2}\sqrt{\mathcal{E}_{\mathrm{split}}}.
\]

Moreover, the two represented solutions satisfy

\[
\|u_\phi-u_*\|_a,\ \|u_\psi-u_*\|_a
\le
\frac{1+C_E}{2}\sqrt{\mathcal{E}_{\mathrm{split}}}.
\]

The reason is that, in the exact case, the split-error variable \(q_c\) satisfies

\[
u_{\mathrm{pred}}-u_*=\frac12q_c,
\]

and the energy-consistency estimate gives

\[
\|q_c\|_a\le C_E\sqrt{\mathcal{E}_{\mathrm{split}}}.
\]

Thus the energy loss is not merely an agreement signal.  Under the stated
structural assumptions, it directly bounds the final solution error in the
energy norm.

**Perturbed Green reconstruction.**  If the Green reconstructions of the exact
split are imperfect,

\[
G_x[\phi_*]=u_*+\varepsilon_x,
\qquad
G_y[\psi_*]=u_*+\varepsilon_y,
\]

then the final prediction satisfies the perturbed bound

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

The term \(\varepsilon_x-\varepsilon_y\) measures directional reconstruction
mismatch, while \(\varepsilon_x+\varepsilon_y\) contains the common Green bias.
A common Green bias can remain invisible to \(\mathcal{E}_{\mathrm{split}}\), so learned
GreenNet error must be accounted for separately.

**Interpretation.**  This proposition is a conditional error-bound statement.
It explains why energy consistency is stronger than an \(L^2\)-only agreement:
the \(L^2\) norm controls amplitude, while \(\mathcal{E}_{\mathrm{split}}\) controls the
derivative-level error that enters the elliptic energy argument.

The admissibility condition is essential in complex geometry.  Zero endpoint
values on connected intervals provide the correct one-dimensional Dirichlet
compatibility, but endpoint zero alone does not prove
\(u_\phi,u_\psi\in H_0^1(\Omega)\).  Full-domain admissibility also requires
transverse Sobolev regularity of the assembled slice-wise reconstructions.

**Limitations.**  The exact error bound is not an unconditional statement about
all learned models.  It assumes exact Green reconstruction, source-linearity,
weak inverse identities, projected balance, and full-domain admissibility.  In
the imperfect case, the perturbation terms above are part of the bound.  If a
learned reconstruction fails to behave as a weak inverse, additional inverse
residuals are outside what energy agreement alone can remove.

## 9. End-to-End Algorithm

The following mathematical algorithm summarizes the framework.

**Algorithm 1: Complex geometry GreenNet/CouplingNet solution reconstruction**

1. Define the complex physical domain \(\Omega\), the coefficient fields
   \(a\), \((b_x,b_y)\), \(c\), and the source \(f\).
2. Intersect \(\Omega\) with horizontal and vertical axial lines.
3. Decompose each intersection into connected intervals and treat each connected
   interval as an independent one-dimensional boundary value problem.
4. For every connected interval \(I=[s_0,s_1]\), pull back the physical
   coordinate to \(t\in[0,1]\) by \(s=s_0+Lt\).
5. Transform the one-dimensional coefficients and source into unit-coordinate
   quantities using the length-scaled operator relations.
6. Train or evaluate GreenNet so that
   \[
   v(t)=\int_0^1G_{\mathrm{unit}}(t,\eta)
   f_{\mathrm{unit}}(\eta)\,d\eta
   \]
   reconstructs the interval solution.
7. Use CouplingNet to predict raw direction-split source components from source,
   coefficient, geometry, transverse position, and local coordinate information.
8. Apply symmetric balance projection so that
   \[
   \phi_{\mathrm{proj}}+\psi_{\mathrm{proj}}=f.
   \]
9. Reconstruct represented solutions:
   \[
   u_\phi=\mathcal{G}_x[\phi_{\mathrm{proj}}],
   \qquad
   u_\psi=\mathcal{G}_y[\psi_{\mathrm{proj}}].
   \]
10. Form the final solution prediction:
    \[
    u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi).
    \]
11. Evaluate solution accuracy, split-field accuracy when available, and
    energy consistency between \(u_\phi\) and \(u_\psi\).

This algorithm emphasizes that GreenNet and CouplingNet do not solve the same
subproblem. GreenNet learns axial solution operators. CouplingNet learns how the
two-dimensional forcing should be split so that those axial operators reconstruct
a consistent solution.

## 10. Presentation Preparation Notes

The following statements are useful as high-level messages for a conference
presentation.

1. **The framework solves a two-dimensional PDE through axial Green operators.**
   The problem is not treated as a direct black-box mapping from \(f\) to \(u\).
   Instead, the solution is reconstructed through one-dimensional Green
   operators along connected axial intervals.

2. **Complex geometry is handled by connected interval pull-back.**
   Each axial intersection component is mapped to \([0,1]\). This avoids forcing
   a complex domain into an inappropriate rectangular representation.

3. **GreenNet learns normalized one-dimensional Green kernels.**
   The kernel is evaluated on \((t,\eta)\in[0,1]^2\), while physical length
   effects enter through coefficient and source scaling.

4. **CouplingNet learns a source-conditioned split of the forcing.**
   The model does not output the solution directly. It predicts the direction
   split fields that will become Green reconstruction sources.

5. **Projection enforces physical balance.**
   The relation \(\phi+\psi=f\) is imposed directly on the raw physical split
   prediction, before axis-specific \(L^2\) conversion for Green reconstruction.

6. **The final solution is reconstructed, not directly regressed.**
   The prediction
   \[
   u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi)
   \]
   is obtained after applying Green reconstruction to the projected split fields.

7. **Energy consistency links the two axial reconstructions.**
   The disagreement \(u_\phi-u_\psi\) is measured in a diffusion-weighted energy
   sense, reflecting the elliptic structure of the PDE.

8. **Disconnected intervals are separate Green domains.**
   This is essential for annular or multiply connected geometries, where a
   single axial line can intersect the domain in multiple disjoint pieces.

These messages form a coherent presentation narrative: complex geometry produces
connected axial intervals; GreenNet learns the interval-wise solution operators;
CouplingNet learns the balanced source decomposition; projection enforces the PDE
constraint; Green reconstruction produces the final solution.
