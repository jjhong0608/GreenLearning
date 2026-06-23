# GreenNet and CouplingNet: A Structured Axial Green Function Model

This document is a continuous English narrative for a conference presentation. It is not a
slide-by-slide script. The goal is to explain the modeling idea: how GreenNet approximates
one-dimensional Green functions with an analytic form and a neural correction, and how
CouplingNet couples those axial Green responses into a two-dimensional solution
reconstruction.

The central message is that the method is not a generic black-box neural operator. It is a
structured model built around the Green function representation of elliptic PDEs. GreenNet
represents line-wise Green kernels with a structured analytic-neural form. CouplingNet learns
how to decompose a two-dimensional source into axial components that can be propagated
through those Green kernels.

## Motivation

We are interested in solving two-dimensional elliptic boundary value problems on the unit
square. The source term is \(f(x,y)\), and the solution \(u(x,y)\) satisfies a homogeneous
Dirichlet boundary condition,

$$
u(x,y)=0, \qquad (x,y)\in \partial [0,1]^2.
$$

The operator we have in mind is a variable-coefficient diffusion-convection-reaction
operator,

$$
\mathcal{L}u
=
-\nabla\cdot(a\nabla u)
+\mathbf{b}\cdot\nabla u
+cu
=f,
\qquad
\mathbf{b}=(b_x,b_y).
$$

Here \(a(x,y)\) is the diffusion coefficient, \(\mathbf{b}(x,y)\) is the convection field,
and \(c(x,y)\) is the reaction coefficient. We assume the diffusion is positive,
\(a(x,y)>0\), and the reaction is nonnegative, \(c(x,y)\ge 0\), so the underlying elliptic
problem has the expected coercive structure. Written componentwise, the convection term is

$$
\mathbf{b}\cdot\nabla u
=
b_x\partial_x u + b_y\partial_y u.
$$

This explicit operator is the starting point of the model. GreenNet approximates
one-dimensional Green functions induced by axial slices of this operator, and CouplingNet
learns how to combine those axial responses into a two-dimensional solution.

The classical Green function viewpoint is attractive because it turns the solution process
into an integral transform. If a Green kernel is known, the solution can be written as an
integral of the kernel against the source. In one dimension this has the simple form

$$
u(x)=\int_0^1 G(x,\xi) f(\xi)\,d\xi.
$$

For a full two-dimensional problem, however, the Green function depends on both a
two-dimensional evaluation point and a two-dimensional source point. It is therefore a
high-dimensional object. The main idea of our approach is to avoid learning this full
object directly. Instead, we use an axial decomposition: we learn one-dimensional Green
functions along x-lines and y-lines, and then couple the resulting axial responses.

## Axial Green Function Viewpoint

The coefficients are defined on the two-dimensional domain, but each coordinate line sees
a one-dimensional slice of them. If we fix \(y=y_0\), the x-line sees coefficients such as
$$
a(x,y_0), \qquad b_x(x,y_0), \qquad c(x,y_0).
$$

If we fix \(x=x_0\), the y-line sees

$$
a(x_0,y), \qquad b_y(x_0,y), \qquad c(x_0,y).
$$

Thus, even for a single fixed two-dimensional problem, the induced one-dimensional
operators vary from line to line. The corresponding Green functions also vary from line to
line. This is the key modeling reason for using a coefficient-aware Green network.

In this presentation, the problem families should be understood as a progression in
difficulty. The simplest case is Poisson, where diffusion is constant and there is no
convection or reaction. The next cases introduce variable diffusion. More difficult cases
include reaction, and the most complete family includes diffusion, directional convection,
and reaction simultaneously. This progression is useful because it separates the questions:
can the model learn a basic Green representation, can it adapt to spatially varying
diffusion, and can it still reconstruct solutions when additional operator terms are
present?

## GreenNet: Approximating a One-Dimensional Green Function

GreenNet is the first part of the method. Its role is to approximate a one-dimensional
Green function for each axial line. The model does not simply take coordinates and output a
kernel value through an unconstrained neural network. Instead, it uses a structured
approximation that combines an analytic Green form with a learned correction.

The inputs to GreenNet are the coefficient profiles along an axial line and the coordinate
pair \((x,\xi)\), where \(x\) is the evaluation coordinate and \(\xi\) is the source
coordinate. The output is the learned one-dimensional Green kernel value
\(\widehat{G}(x,\xi)\). Architecturally, this can be viewed as a MIONet-style construction:
one part of the network represents the line operator through its coefficient profiles, while
another part represents the coordinate pair. These representations are combined inside the
structured analytic formula described below.

For an axial line, consider a one-dimensional operator of the form

$$
-\frac{d}{dx}\left(a(x)\frac{du}{dx}\right) + b(x)\frac{du}{dx} + c(x)u = f(x),
$$

with homogeneous Dirichlet boundary conditions. Its Green function must satisfy two basic
requirements. First, it must produce a point-source response,

$$
L_x G(x,\xi)=\delta(x-\xi),
$$

where \(L_x\) is the one-dimensional line operator. Second, it must satisfy the endpoint
boundary condition,

$$
G(0,\xi)=G(1,\xi)=0.
$$

These two requirements are difficult for a fully unconstrained neural network to learn from
data alone. The delta property introduces a singular response at \(x=\xi\), while the
boundary condition imposes an exact structural constraint at the endpoints. GreenNet
therefore starts from an analytic Green form rather than asking the neural network to
learn the entire kernel directly.

The analytic base is built from the Poisson Green kernel \(G_0(x,\xi)\), which already has
the correct one-dimensional Dirichlet boundary behavior and the correct qualitative
point-source structure. GreenNet then modifies this base form to account for variable
diffusion and convection. The approximation is

$$
\widehat{G}(x,\xi)
=
A(x)G_0(x,\xi)
+
B(x)\left(I_0(x,\xi)-\frac12 x\xi(1-\xi)\right)
+
x\xi(1-\xi)(1-x)N_\theta(x,\xi;\text{line coefficients}).
$$

The first term, \(A(x)G_0(x,\xi)\), carries the dominant Dirichlet Green structure. The
second term uses an associated integrated Green-type term \(I_0(x,\xi)\). This term is
included to handle the lower-regularity structure that appears when the differential
operator acts on the analytic ansatz. In particular, applying a variable-coefficient
operator to Green-like functions creates not only Dirac-delta behavior but also
Heaviside-type terms. The coefficients

$$
A(x)=\frac{1}{a(x)}, \qquad
B(x)=\frac{a'(x)+b(x)}{a(x)^2}
$$

are chosen so that these singular and discontinuous contributions are structurally matched
or removed at the analytic level. In other words, \(A(x)\) and \(B(x)\) are not arbitrary
learned scalings. They encode how the local diffusion \(a(x)\), its derivative \(a'(x)\),
and the line convection \(b(x)\) should enter the Green function approximation.

The final term is the neural correction. It is multiplied by the envelope

$$
x\xi(1-\xi)(1-x).
$$

This polynomial envelope is important. It vanishes at the endpoints and prevents the neural
correction from destroying the boundary behavior supplied by the analytic Green form. The
neural network \(N_\theta\) therefore has a focused role: it learns the smoother,
coefficient-dependent correction that remains after the main singular structure and
boundary structure have already been built into the formula.

This design explains why the model is not merely a black-box kernel regressor. The analytic
terms encode the Dirac-delta response, boundary-zero behavior, and the leading effects of
variable diffusion and convection. The neural term supplies flexibility for what remains.
The network receives information about the line coefficients and the coordinate pair
\((x,\xi)\), but its output is inserted into a structured formula rather than used as the
entire Green kernel.

The training objective follows the Green representation itself. GreenNet is not trained by
directly matching \(\widehat{G}(x,\xi)\) to an exact Green kernel. Instead, the predicted
kernel is integrated against sampled source functions,

$$
\widehat{u}(x)
=
\int_0^1 \widehat{G}(x,\xi)f(\xi)\,d\xi,
$$

and the loss is imposed on the reconstructed solution:

$$
\mathcal{L}_{G}
=
\mathbb{E}_{(f,u)}
\int_0^1
\left|
u(x)-\widehat{u}(x)
\right|^2\,dx.
$$

This is an important distinction. The model output is a Green kernel, but the supervision is
through the source-to-solution map induced by that kernel. When an exact Green kernel is
available, it can be used as a diagnostic to inspect kernel quality, but it is not the direct
training target.

The presentation should emphasize this division of labor:

- The Poisson Green kernel provides the base point-source and boundary structure.
- The integrated Green-type term and the coefficients \(A(x)\), \(B(x)\) handle singular
  and Heaviside-type terms induced by the variable-coefficient operator.
- The polynomial envelope protects the boundary behavior of the neural correction.
- The neural network learns the remaining smooth coefficient-dependent correction.

This is the main modeling idea behind GreenNet.

## From Axial Green Kernels to a Two-Dimensional Solution

Once we have line-wise Green kernels, we can propagate one-dimensional source components
along x-lines and y-lines. However, the original problem is still two-dimensional. A source
\(f(x,y)\) must be represented in a way that is consistent across both axial directions.

This motivates the introduction of two scalar source-decomposition components,

$$
\phi(x,y), \qquad \psi(x,y),
$$

which are not arbitrary auxiliary variables. They are directional source components induced
by the PDE operator. For a reference solution \(u\), we define

$$
\phi(x,y)
=
-\partial_x(a\partial_x u)
+
b_x\partial_x u
+
\frac12 cu,
$$

and

$$
\psi(x,y)
=
-\partial_y(a\partial_y u)
+
b_y\partial_y u
+
\frac12 cu.
$$

The reaction term is split evenly between the two directions. With this definition,

$$
\phi(x,y)+\psi(x,y)
=
-\nabla\cdot(a\nabla u)
+
\mathbf{b}\cdot\nabla u
+
cu
=
f(x,y).
$$

Thus, \(\phi\) is the source-like component associated with the x-direction axial
reconstruction, and \(\psi\) is the source-like component associated with the y-direction
axial reconstruction. The two components should satisfy two structural conditions. The first
is source balance:

$$
\phi+\psi=f.
$$

The second is consistency of the two represented solutions. Using the learned line-wise Green
kernels, the components define

$$
u_x = G_x[\phi], \qquad u_y = G_y[\psi],
$$

where the subscripts label the axial reconstruction direction, not partial derivatives. The
two represented solutions should agree with each other,

$$
u_x \approx u_y,
$$

and, when the source split and Green kernels are accurate, they should also agree with the
true two-dimensional solution. CouplingNet is the model that predicts these directional
components.

## CouplingNet: A MIONet-Style Coupling Model

CouplingNet is designed as a multi-input operator network. Its role is not to approximate a
Green function. Its role is to predict the axial source decomposition that makes the
GreenNet responses useful for two-dimensional reconstruction.

The inputs naturally separate into several groups.

The first group is the source information. This tells the model what forcing term must be
represented. The second group is the local operator information, including diffusion,
directional convection, and reaction coefficients. The third group is transverse coordinate
context, which tells the model which axial line is being considered. The final group is the
axial coordinate itself, which describes variation along the line.

This gives the following conceptual structure:

```text
source branch
coefficient branch
transverse-coordinate branch
        |
        v
multi-input branch representation
        |
        v
axial trunk representation
        |
        v
raw components phi_0 and psi_0
```

The branch representation uses a product-style fusion. This means that source information,
coefficient information, and transverse coordinate information interact multiplicatively.
The purpose is to encode the intuition that the source response depends jointly on the
forcing, the local operator, and the line location.

The trunk is shared and one-dimensional. For the \(\phi\) path, the trunk receives the
x-coordinate. For the \(\psi\) path, it receives the y-coordinate. This matches the axial
nature of the output:

$$
\phi \text{ varies along x-lines}, \qquad
\psi \text{ varies along y-lines}.
$$

The transverse coordinate is not fed as a raw scalar. Instead, it is encoded through
boundary-aware sine and cosine features,

$$
Enc_k(t)=
[\sin(n\pi t),\cos(n\pi t)]_{n=1}^{k}.
$$

This encoding gives the branch information about the line location while reflecting the
bounded domain structure. It also separates the along-line coordinate, handled by the trunk,
from the transverse coordinate, handled by the branch.

The main consistency objective compares the two represented solutions \(u_x\) and \(u_y\) in
the diffusion energy norm:

$$
\mathcal{L}_E
=
\int_\Omega a|\nabla(u_x-u_y)|^2.
$$

This is stronger than only matching \(u_x\) and \(u_y\) pointwise in an \(L^2\) sense. The
elliptic operator is controlled naturally through weak and energy quantities, and the
gradient-level discrepancy \(\nabla(u_x-u_y)\) is directly relevant to the directional
operator split. An \(L^2\)-only consistency term can make the represented solutions close in
amplitude while still leaving high-frequency or derivative-level disagreement uncontrolled.

## Why Projection Is Needed

The raw network outputs are denoted by

$$
\phi_0, \qquad \psi_0.
$$

If these raw outputs were used directly, there would be no guarantee that they satisfy the
source balance relation

$$
\phi+\psi=f.
$$

This balance relation is central because \(\phi\) and \(\psi\) are meant to decompose the
same source term into axial components. However, balance is not the only reason for the
projection. The projection is also designed to be compatible with the boundary behavior of
fiberwise Green reconstructions.

The x-direction Green reconstruction \(G_x[\phi]\) naturally satisfies the endpoint
conditions along each x-fiber:

$$
G_x[\phi](0,y)=G_x[\phi](1,y)=0.
$$

But this fiberwise construction does not automatically guarantee the transverse boundary
conditions at \(y=0\) and \(y=1\). Similarly, \(G_y[\psi]\) naturally satisfies

$$
G_y[\psi](x,0)=G_y[\psi](x,1)=0,
$$

but it does not automatically enforce the transverse boundary conditions at \(x=0\) and
\(x=1\). Therefore, the source split should be structured not only to satisfy
\(\phi+\psi=f\), but also to avoid introducing incompatible transverse-boundary behavior in
the represented solutions.

CouplingNet therefore applies a projection that turns the raw outputs into
balance-preserving and boundary-aware components.

The current baseline uses sine smooth masks:

$$
m_\phi(y)=\sin(\pi y), \qquad
m_\psi(x)=\sin(\pi x).
$$

These masks are zero at the relevant transverse boundaries and positive in the interior.
The mask \(m_\phi(y)\) vanishes at \(y=0\) and \(y=1\), which controls the transverse
boundary behavior of the x-direction component. The mask \(m_\psi(x)\) vanishes at
\(x=0\) and \(x=1\), which controls the transverse boundary behavior of the y-direction
component. They encode the idea that the raw difference mode should be damped near the
transverse boundary while still remaining flexible in the interior.

Define

$$
w_\phi=\frac{m_\phi}{m_\phi+m_\psi}, \qquad
w_\psi=\frac{m_\psi}{m_\phi+m_\psi},
$$

and

$$
\alpha=\frac{m_\phi m_\psi}{m_\phi+m_\psi}.
$$

The projected components are then

$$
\phi = w_\phi f + \alpha(\phi_0-\psi_0),
$$

$$
\psi = w_\psi f - \alpha(\phi_0-\psi_0).
$$

This formula has an immediate and important consequence:

$$
\phi+\psi = (w_\phi+w_\psi)f = f.
$$

The raw difference mode \(\phi_0-\psi_0\) appears with opposite signs in \(\phi\) and
\(\psi\), so it cancels in the sum. The projection therefore preserves the source balance
exactly while retaining a learnable degree of freedom for how the source is split between
the two axial directions.

This is the key structural role of the projection. It is not just a numerical trick. It
builds the source-decomposition constraint into the model and uses sine masks to make that
constraint compatible with transverse boundary behavior. In this sense, the projection
addresses two constraints at once: the algebraic balance \(\phi+\psi=f\) and the boundary
compatibility required by fiberwise Green reconstructions.

## What the Neural Networks Learn

The two models learn different objects.

GreenNet represents the full one-dimensional Green kernel through a structured analytic form
plus a neural correction. The analytic terms encode known behavior, while the neural
component learns the remaining coefficient-dependent part. The model output is therefore
the complete structured kernel, not only the correction term.

CouplingNet learns a decomposition of the source into axial components. Its branches
combine source information, operator information, and transverse coordinate context. Its
trunk represents one-dimensional variation along the axial direction. Its projection turns
raw neural outputs into balance-preserving components.

Together, the models implement the following pipeline:

```text
   2D coefficients and source
        |
        v
line-wise Green kernel approximation by GreenNet
        |
        v
source decomposition by CouplingNet
        |
        v
axial Green integration
        |
        v
consistent 2D solution reconstruction
```

This makes the overall method structure-aware at two levels. GreenNet is structured around
the analytic form of one-dimensional Green functions. CouplingNet is structured around the
balance relation and the axial decomposition of the source.

## Evidence to Show in the Presentation

The figures should support the model explanation rather than highlight code-level training
details.

For GreenNet, the most important evidence is the shape of the learned Green function. A
heatmap of \(G(x,\xi)\) shows the global kernel structure on a selected line. A fixed-source
slice \(G(x,\xi_0)\) shows boundary behavior and diagonal response more clearly. A
schematic of the analytic base plus neural correction would also be useful because it
directly explains how the model is constructed.

For CouplingNet, the most important evidence is whether the predicted source decomposition
leads to a good two-dimensional reconstruction. Useful figures include the source, the
reference solution, the predicted solution, signed solution error, the predicted
\(\phi\) and \(\psi\) components, and the balance residual

$$
f-\phi-\psi.
$$

The balance residual is especially important because it directly visualizes whether the
decomposition behaves as intended. Signed error figures are preferred over absolute error
figures because they reveal systematic bias and spatial structure.

A compact architecture schematic should show the MIONet-style input separation:

```text
source information
operator coefficients
transverse coordinate encoding
        |
        v
branch representation
        |
        v
shared 1D trunk
        |
        v
raw phi_0, psi_0
        |
        v
smooth-mask projection
        |
        v
balanced phi, psi
```

This schematic would communicate the core CouplingNet idea more effectively than listing
training hyperparameters.

## Experimental Variants Outside the Main Narrative

Several experimental variants exist, but they are not part of the main baseline narrative.
The presentation should not spend much time on them unless the goal is an ablation study.
Examples include alternative input-side lifting mechanisms, optional Green-response branch
features, alternative trunk encodings, projection-off losses, and additional boundary-loss
variants.

For the main presentation, the clean story is:

- GreenNet represents one-dimensional Green functions using an analytic form plus a neural
  correction, and trains them through source-to-solution reconstruction.
- CouplingNet predicts source-decomposition components using a MIONet-style architecture.
- A smooth-mask projection turns raw components into balance-preserving components.
- Axial Green integration converts those components into a two-dimensional solution.

## Main Takeaways

The first takeaway is that GreenNet is not a generic coordinate network for a Green kernel.
It is a structured approximation of a one-dimensional Green function. The known analytic
terms provide the dominant Green behavior, the neural component supplies the remaining
coefficient-dependent flexibility, and the training signal comes from reconstructing
solutions through the Green representation.

The second takeaway is that CouplingNet is not simply a post-processing network. It solves
the coupling problem created by the axial decomposition. It predicts how the source should
be split into \(\phi\) and \(\psi\), so that x-direction and y-direction Green responses can
produce a consistent two-dimensional solution.

The third takeaway is that projection is central to the method. The projection guarantees
\(\phi+\psi=f\), while the sine masks encode transverse boundary behavior. This gives the
model both flexibility and structure.

The final takeaway is that the method combines analytic PDE structure with neural
approximation. The analytic terms make the Green function approximation physically
meaningful, and the neural components provide adaptability to variable coefficients and
two-dimensional coupling.

## Compact Presentation Message

GreenNet approximates a one-dimensional Green function by combining an analytic Green form
with a neural correction, then trains the resulting kernel through the reconstruction
identity \(u(x)=\int G(x,\xi)f(\xi)\,d\xi\).

The analytic part captures the dominant boundary-aware Green structure, while the neural
part supplies the coefficient-dependent residual behavior inside the complete kernel model.

CouplingNet predicts a source decomposition through a MIONet-style architecture with
separate source, coefficient, transverse-coordinate, and axial-coordinate representations.

The projection turns raw neural outputs into balance-preserving components satisfying
\(\phi+\psi=f\).

The sine masks encode transverse boundary behavior and control how the raw difference mode
contributes near the boundary.

In one sentence: GreenNet represents axial Green kernels with a structured analytic-neural
form, and CouplingNet couples those axial kernels through a balance-preserving source
decomposition to reconstruct the two-dimensional solution.
