# WCCM-ECCOMAS 2026 Presentation Outline

## Talk Metadata

- **Conference:** WCCM-ECCOMAS 2026
- **Minisymposium:** MS165 - Methods and Applications of Model Order Reduction
- **Talk title:** Hybrid Green's Function Learning With Axial Reduction for Multi-Dimensional Elliptic Problems
- **Time budget:** 15 minutes including Q&A
- **Target talk time:** 11-12 minutes, leaving 3-4 minutes for questions
- **Audience framing:** computational mechanics and model order reduction researchers

## Talk Goal

The talk should present the method as a hybrid model reduction framework for
multi-dimensional elliptic problems.  The core message is:

> We reduce a multi-dimensional elliptic problem to normalized one-dimensional
> Green operator learning along axial intervals, and learn the source coupling
> needed to reconstruct a consistent multi-dimensional solution.

The talk should not be framed primarily as a complex-geometry construction.  The
geometry message should be limited to this: non-square or non-unit axial
intervals are pulled back to the unit interval, and this pull-back induces
scaling of the operator coefficients and the source used by GreenNet.

## Narrative Flow

1. **Start from MOR motivation and axial reduction.**  For one fixed
   heterogeneous elliptic operator, the target map is the source-to-solution
   operator \(\mathcal S_{a,\mathbf b,c}: f\mapsto u\).  Representing this map
   globally is expensive and geometrically rigid, so the method represents it
   through one-dimensional axial Green reconstructions plus learned coupling.
2. **Show the graphic abstract before details.**  A figure-first roadmap should
   show the computation as a two-dimensional forcing \(f(x,y)\) on a general
   domain \(\rightarrow\) axial interval intersections \(\rightarrow\) a
   GreenNet KaTeX math card for kernel integration
   \(\rightarrow\) directional split coupling through \(G_x,G_y\) into a 2D
   solution \(u\).
3. **Normalize every axial interval and keep physical scale.**  A physical
   interval of length \(L\) is mapped to \([0,1]\), and \(L\) enters the normalized
   operator coefficient profiles and source profile through pull-back scaling.
4. **Explain GreenNet I: normalized axial kernels.**  GreenNet works with a
   local unit-interval Green kernel whose integral action defines a
   source-to-solution map, not with a global 2D Green function.
5. **Explain GreenNet II: analytic hybrid structure.**  The normalized kernel is
   not a black-box neural kernel.  Variable-coefficient Green kernels are rarely
   available in closed form, so the slide should say: do not learn the Green
   singularity from scratch.  The analytic component supplies the delta-induced
   jump, flux-jump behavior, and boundary structure before the neural correction
   learns the smooth residual.
6. **Explain GreenNet III: reconstruction supervision.**  GreenNet is supervised
   through one-dimensional source-to-solution reconstruction, not through direct
   pointwise labels for the exact Green kernel.  This should also set up the
   contrast with CouplingNet, which is trained without reference-solution or
   split labels.
7. **Validate GreenNet before introducing CouplingNet.**  The GreenNet evidence
   slide should appear immediately after source-to-solution supervision.  It
   uses reaction-free convection-diffusion Disk_CD artifacts, a fixed
   \(\eta=0.75\) slice, a compact axial-line context tag
   \((y\)-directed interval at \(x=-0.25\), \(L=0.866)\), and a three-state
   reveal: kernel heatmaps plus signed error and diagnostics \(\rightarrow\)
   enlarged fixed-\(\eta\) slice \(\rightarrow\) takeaway.  The slide title and
   placement make it clear that this is GreenNet kernel validation, so no
   visible `Kernel-level evidence` label is needed.
8. **Explain CouplingNet I: directional source split.**  CouplingNet learns how
   the full source should be split into horizontal and vertical line-wise
   flux-divergence/source components so axial Green operators can reconstruct
   the multi-dimensional solution.
9. **Explain CouplingNet II: branch nets and trunk nets.**  CouplingNet uses
   branch nets for source, coefficient profiles, and line-geometry structure,
   while trunk nets provide pointwise axial and transverse coordinates for split
   prediction.  The coefficient field is fixed for the problem, but its axial
   profiles vary from line to line, so the same operator-learning model is
   reused across lines.
10. **Explain CouplingNet III: projection and reconstruction.**  Projection
   imposes \(\phi+\psi=f\), and the projected split fields are passed through
   Green reconstructions to build \(u_{\mathrm{pred}}\).
11. **State the error-bound message.**  The split energy
   \(\mathcal{E}_{\mathrm{split}}\) is not only a heuristic agreement loss; under
   structural assumptions, it bounds the final solution energy error.
12. **Show CouplingNet solver-level evidence.**  After the CouplingNet and
    energy-bound slides, show full solution reconstruction and split
    consistency through the quantile-selected CouplingNet evidence matrix.
13. **Close with contributions.**  The contribution is a hybrid axial Green
    reduction plus source-conditioned coupling, supported by an energy-norm
    error-bound interpretation.

## Slide-Level Outline

| # | Slide title | Main message | Key equation / visual | Speaker focus | Time |
|---|---|---|---|---|---|
| 1 | Hybrid Green's Function Learning With Axial Reduction | The method combines axial Green operators and learned source coupling for elliptic PDEs. | Title, authors, minisymposium name, one-sentence thesis. | Say the whole talk in one sentence: reduce the operator, learn the coupling. | 0:30 |
| 2 | Axial Reduction of Elliptic Operators | The reduction is from one global source-to-solution map under a fixed heterogeneous operator to directional 1D operators and axial Green reconstructions plus learned coupling. | Homogeneous Dirichlet 2D PDE, \(\mathcal S_{a,\mathbf b,c}:f\mapsto u\), \(L_x,L_y\) directional operators, and anchor sentence: GreenNet supplies line-wise Green inverses; CouplingNet learns the source split that turns them into a 2D elliptic solver. | Explain what is hard, what is reduced, and what the directional split is without over-framing the method as traditional MOR. | 1:10 |
| 3 | From a 2D Elliptic Problem to Coupled Axial Green Solves | A figure-first roadmap shows the computation from 2D forcing on a general domain to axial intervals, kernel-integration line-wise inverses, directional source coupling, and 2D solution reconstruction. | Deck-native graphic abstract: \(f(x,y)\) on \(\Omega\) \(\rightarrow\) axial intervals \(\rightarrow\) KaTeX GreenNet card \(v(t)=\int_0^1G_\theta(t,\eta)\rho(\eta)d\eta\) for a generic line-source profile \(\rho\) \(\rightarrow\) KaTeX CouplingNet card \(f\mapsto(\phi,\psi)\), split paths through \(G_x,G_y\), and 2D solution \(u\), with \(\phi+\psi=f\) in the final stage. | Orient the audience visually; do not explain analytic Green structure, branch/trunk internals, or energy bounds. | 0:45 |
| 4 | Unit-Interval Pull-Back and Operator Scaling | Physical intervals are normalized, but their length remains in the transformed operator coefficient profiles and source profile. | Generic physical 1D operator on \(s\in[s_0,s_1]\), interval visual with \(L=s_1-s_0\), \(s=s_0+Lt\), scaling equations for \(a,a',b_\parallel,c,f\), and the expanded normalized unit equation. | Keep geometry lightweight and make scaling the technical bridge to GreenNet. | 1:00 |
| 5 | GreenNet I: Normalized Axial Green Operator | GreenNet learns a local unit-interval Green kernel, not a global 2D Green function. | Contrast strip: previous step normalized the operator; this step learns its Green inverse. \(G_{\mathrm{unit}}(t,\eta)\) as the kernel integral operator, \(\mathcal G_{\mathrm{unit}}[f_{\mathrm{unit}}](t)=\int_0^1G_{\mathrm{unit}}(t,\eta)f_{\mathrm{unit}}(\eta)\,d\eta\), and an operator branch/trunk schematic. | Define "Green operator" as the integral source-to-solution map on a normalized interval. | 0:55 |
| 6 | GreenNet II: Analytic Green Structure and Learned Correction | Do not learn the Green singularity from scratch; the analytic component supplies the delta-induced jump, flux-jump behavior, and boundary structure before learning. | Three Auto-Animate states: role thesis, Dirac delta/Heaviside identities, and learned smooth correction. Key equations include \(G_\theta=E M R_\theta+B(J_0-\frac12E)+A G_0\), \(G_0\) base kernel, \(J_0\) antiderivative identity, \(S\) cancellation identity, and \(E/M\) envelope factors; \(A/B\) coefficient-factor details move to Backup A. | Explain the three roles: Dirac delta structure, Heaviside cancellation, learned correction. | 1:05 |
| 7 | GreenNet III: Source-to-Solution Supervision | GreenNet is supervised by one-dimensional source-to-solution reconstruction pairs, not by direct exact-kernel labels; CouplingNet later uses no reference-solution or split labels. | \(w\sim\mathcal{GP}(0,k_\ell)\), endpoint-interpolant removal for \(v\), \(f_{\mathrm{unit}}=\mathcal L_{\mathrm{unit}}v\), \(v_\theta=\int G_\theta f_{\mathrm{unit}}\), and \(\mathcal J_{\mathrm{Green}}\sim\mathbb E[\int |v_\theta-v|^2]\). | State that training checks whether the learned Green operator maps sources generated from target solutions back to those target solutions, then contrast this with CouplingNet's unsupervised consistency training. | 0:55 |
| 8 | Numerical Evidence I: GreenNet Kernel Approximation | GreenNet evidence validates the line-wise kernel before CouplingNet is introduced. | Reaction-free convection-diffusion Disk_CD problem strip, compact axial-line context tag, separated assets for reference kernel, learned kernel, signed error heatmap, fixed-\(\eta=0.75\) slice, and a slide-native diagnostic card. Three states: heatmaps/diagnostics, enlarged fixed-\(\eta\) slice, takeaway. | Use this as the transition from GreenNet to CouplingNet: once the line-wise inverse is validated, the remaining problem is source coupling. | 0:55 |
| 9 | CouplingNet I: Directional Source Split | CouplingNet learns horizontal and vertical line-wise flux-divergence/source components rather than predicting the solution directly. | \(f\rightarrow(\phi,\psi)\), \(\phi=L_xu\), \(\psi=L_yu\), compact \(L_x,L_y\) definitions, and \(\phi+\psi=f\). | State that CouplingNet learns line-wise flux-divergence/source components that couple the axial Green reconstructions. | 0:50 |
| 10 | CouplingNet II: Branches and Local Context for Split Prediction | The split predictor combines branch-net features for profiles and line geometry with trunk-net features for pointwise coordinates, reusing the same model across line-varying axial coefficient profiles. | Branch nets: source profile, coefficient profiles, line-geometry structure. Trunk nets: \(t_{\parallel}\) and \(t_{\perp}\). | Explain that the coefficient field is fixed for the problem, but its axial profiles vary from line to line; the same operator-learning model is reused across lines. | 1:00 |
| 11 | CouplingNet III: Projection and Green Reconstruction | A raw directional split is projected in physical variables, then two axial Green reconstructions are averaged. | Two-stage pipeline: \((\phi_{\mathrm{raw}},\psi_{\mathrm{raw}})\rightarrow r=f-(\phi_{\mathrm{raw}}+\psi_{\mathrm{raw}})\rightarrow(\phi,\psi),\ \phi+\psi=f\rightarrow(u_\phi,u_\psi)\rightarrow u_{\mathrm{pred}}\). | Connect the two networks into one solver pipeline while making clear that balance is imposed in physical split variables before Green reconstruction. | 0:55 |
| 12 | Energy-Norm Error Bound Proposition | The split-energy loss bounds final solution error under structural assumptions. | Diffusion-weighted norm with \(a(x)\) as diffusion coefficient, \(u_*\) reference solution, \(C_E\) stability constant, and \(\|u_{\mathrm{pred}}-u_*\|_a\le\frac{C_E}{2}\sqrt{\mathcal{E}_{\mathrm{split}}}\). | Define \(a(x)\), \(u_*\), and \(C_E\) before the bound. Keep assumptions as a final footer, not the first visual focus. | 1:05 |
| 13 | Numerical Evidence II: CouplingNet Solution Reconstruction | Solver-level evidence: quantile-selected samples show that the learned directional source split supports 2D CDR solution reconstruction across the observed relative-error range, not only on a single favorable case. | Disk_CDR convection-diffusion-reaction problem strip; 5-by-4 separated field matrix from `coupling_cdr_evidence_rel_sol_quantiles_*` assets: columns selected by relative solution error (`min`, `q25`, `q50`, `q75`, `max`), rows for source, reference, prediction, and signed error, plus a separate metric card. | Use the quantile columns to avoid a best-case-only visual; explain the rows once, then point to the signed-error row and relative-error metric card. | 1:05 |
| 14 | Takeaway: Coupled Axial Green Solvers | A 2D elliptic problem is solved through axial Green inversions and a learned, balance-preserving source decomposition. | Four taller blocks: axial Green kernels, analytic structure, unsupervised source split, and energy bound. Wide closing banner: GreenNet supplies line-wise Green inverses; CouplingNet learns the source split that turns them into a 2D elliptic solver. | End with the GreenNet/CouplingNet role split and state that reference solutions are used for evaluation, not CouplingNet training. | 0:35 |
| 15 | Backup / Q&A Menu | Keep concrete Q&A support slides available without changing the 14-slide main talk. | Ready backups: Dirac/Heaviside derivation, imperfect Green perturbation, and connected-interval pull-back. Deferred backups: figure-dependent GreenNet/CouplingNet evidence. | Use only if asked; do not include this in the target 11-12 minute timing. | Backup |

This outline has 14 main slides plus one backup/Q&A prompt slide.  It fits an
11-12 minute talk if Slide 2 stays compact and the graphic abstract is used as a
brief visual roadmap rather than a detailed explanation slide.
The main compression choice is still the evidence block: show one focused
GreenNet kernel-structure evidence slide and one focused CouplingNet evidence
slide rather than a large metric table.

The backup section is organized as a menu plus three ready non-figure support
slides.  Backup A explains the operator-action view of the analytic Green
wrapping: the Dirac jump, the induced Heaviside leftover, the analytic
compensation, and the \(A/B\) coefficient factors moved out of the main slide.
Backup B explains how imperfect Green reconstructions add directional mismatch
and common-bias perturbation channels to the energy-bound story.  Backup C uses
a non-square domain slice visual to explain connected-interval pull-back and
why connected intervals are not merged across outside-domain gaps.  Extra
GreenNet/CouplingNet numerical evidence remains deferred until final figures
are selected.

## Core Equations to Show

Use only the equations needed to carry the narrative.  Avoid derivations except
for the pull-back scaling and the final error-bound statement.

### Directional split operators

\[
\begin{aligned}
-\nabla\cdot(a\nabla u)+\mathbf b\cdot\nabla u+cu &= f
&&\text{in }\Omega,\\
u &= 0
&&\text{on }\partial\Omega,
\end{aligned}
\qquad
\mathbf b=(b_x,b_y).
\]

\[
L_xu
=
-\partial_x(a\partial_xu)
+
b_x\partial_xu
+
\frac12cu,
\qquad
L_yu
=
-\partial_y(a\partial_yu)
+
b_y\partial_yu
+
\frac12cu.
\]

\[
L_xu+L_yu=f.
\]

This is the physical-domain directional split.  Unit-interval normalization
comes after this split, on each axial interval.

### Pull-back and operator scaling

\[
s\in[s_0,s_1].
\]

\[
\mathcal L_{\mathrm{phys}}u
=
-\frac{d}{ds}
\left(
a_{\mathrm{phys}}(s)\frac{du}{ds}
\right)
+
b_{\parallel,\mathrm{phys}}(s)\frac{du}{ds}
+
c_{\mathrm{phys}}(s)u
=
f_{\mathrm{phys}}(s).
\]

This is the generic axial physical operator after the directional split.

\[
s=s_0+Lt,\qquad L=s_1-s_0,\qquad v(t)=u(s_0+Lt).
\]

\[
a_{\mathrm{unit}}=a_{\mathrm{phys}},
\quad
a'_{\mathrm{unit}}=L a'_{\mathrm{phys}},
\quad
b_{\parallel,\mathrm{unit}}=L b_{\parallel,\mathrm{phys}},
\quad
c_{\mathrm{unit}}=L^2c_{\mathrm{phys}},
\quad
f_{\mathrm{unit}}=L^2f_{\mathrm{phys}}.
\]

\[
-\frac{d}{dt}
\left(
a_{\mathrm{unit}}(t)\frac{dv}{dt}
\right)
+
b_{\parallel,\mathrm{unit}}(t)\frac{dv}{dt}
+
c_{\mathrm{unit}}(t)v(t)
=
f_{\mathrm{unit}}(t),
\qquad
t\in[0,1].
\]

Here \(b_\parallel\) is the convection component along the axial interval:
\(b_\parallel=b_x\) for an \(x\)-directed interval and
\(b_\parallel=b_y\) for a \(y\)-directed interval.

### Green operator action / reconstruction

\[
\mathcal G_{\mathrm{unit}}[f_{\mathrm{unit}}](t)
=
\int_0^1
G_{\mathrm{unit}}(t,\eta)f_{\mathrm{unit}}(\eta)
\,d\eta.
\]

\[
v=\mathcal G_{\mathrm{unit}}[f_{\mathrm{unit}}].
\]

### GreenNet analytic structure

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

Use this equation to show the hybrid nature of GreenNet:
\(A(t)G_0(t,\eta)\) carries the Dirac-\(\delta\) jump structure,
\(B(t)(J_0-\frac12E)\) cancels the leading Heaviside contribution, and
\(E(t,\eta)M(t)R_\theta(t,\eta)\) learns the remaining smooth correction.  Do
not derive the full distributional proof on the main slide.

\[
G_0(t,\eta)
=
\begin{cases}
t(1-\eta), & t<\eta,\\
\eta(1-t), & t\ge \eta,
\end{cases}
\qquad
\partial_t^2G_0(t,\eta)=-\delta(t-\eta).
\]

\[
\partial_tJ_0(t,\eta)=G_0(t,\eta),
\qquad
S(t,\eta)=J_0(t,\eta)-\frac12E(t,\eta),
\qquad
\partial_t^2S(t,\eta)=\partial_tG_0(t,\eta).
\]

\[
A(t)=\frac{1}{a_{\mathrm{unit}}(t)},
\qquad
B(t)=
\frac{
a'_{\mathrm{unit}}(t)+b_{\parallel,\mathrm{unit}}(t)
}{
a_{\mathrm{unit}}(t)^2
},
\qquad
E(t,\eta)=t\eta(1-\eta),
\qquad
M(t)=1-t.
\]

The full piecewise formula for \(J_0\) and the full Dirac/Heaviside derivation
belong in backup or the technical report, not in the main slide.

### GreenNet source-to-solution supervision

\[
w(t)\sim\mathcal{GP}(0,k_\ell),
\qquad
v(t)=w(t)-\bigl((1-t)w(0)+t\,w(1)\bigr),
\qquad
v(0)=v(1)=0.
\]

\[
f_{\mathrm{unit}}=\mathcal L_{\mathrm{unit}}v.
\]

\[
v_\theta(t)
=
\int_0^1
G_\theta(t,\eta)f_{\mathrm{unit}}(\eta)\,d\eta.
\]

\[
\mathcal J_{\mathrm{Green}}
\sim
\mathbb E
\left[
\int_0^1
|v_\theta(t)-v(t)|^2\,dt
\right].
\]

Use this slide to say that GreenNet is trained by source-to-solution
reconstruction from GP-generated target solutions and the sources generated from
those target solutions, not by direct pointwise Green-kernel
labels.

### Source split and final prediction

\[
r=f-(\phi_{\mathrm{raw}}+\psi_{\mathrm{raw}}),
\qquad
\phi=\phi_{\mathrm{raw}}+\frac12r,
\qquad
\psi=\psi_{\mathrm{raw}}+\frac12r,
\qquad
\phi+\psi=f.
\]

\[
u_\phi=G_x[\phi],
\qquad
u_\psi=G_y[\psi],
\qquad
u_{\mathrm{pred}}
=
\frac12(u_\phi+u_\psi).
\]

Use this slide to say that the balance projection is applied in physical split
variables before the balanced split fields are passed to the axial Green
reconstructions.

### Energy-norm error bound

\[
\|v\|_a^2
=
\int_\Omega a(x)|\nabla v(x)|^2\,dx,
\qquad
a(x)>0.
\]

Here \(a(x)\) is the diffusion coefficient.

\[
\mathcal{E}_{\mathrm{split}}
=
\|u_\phi-u_\psi\|_a^2.
\]

\[
\mathcal L u_*=f,
\qquad
u_*|_{\partial\Omega}=0.
\]

\[
\|u_{\mathrm{pred}}-u_*\|_a
\le
\frac{C_E}{2}\sqrt{\mathcal{E}_{\mathrm{split}}}.
\]

Use \(C_E\) as the stability constant of the fixed elliptic operator.  If
there is time, mention the learned-Green perturbation message verbally:
imperfect Green reconstruction adds mismatch and common-bias terms, so GreenNet
accuracy remains part of the full error story.

Use this slide as a conditional structural proposition.  Exact or controlled
Green reconstruction and \(H_0^1(\Omega)\)-admissible represented solutions are
required, but these assumptions should appear as a final footer rather than the
opening visual focus.  The main slide should not contain the proof variables or
full perturbation bound.

### Numerical evidence slots

Numerical Evidence I now appears immediately after GreenNet supervision and
uses separated GreenNet kernel-structure evidence assets rather than one
composite image.  The slide uses the reaction-free convection-diffusion Disk_CD
artifact, places the reference Green kernel, learned Green kernel, signed error
heatmap, and fixed-\(\eta=0.75\) slice as separate figures, and keeps a separate
diagnostic card for the relative errors and diagonal-band diagnostics.  A
compact line-context tag identifies the physical axial line as a \(y\)-directed
interval at \(x=-0.25\) with \(L=0.866\), without exposing the artifact interval
id.  The visible `Kernel-level evidence` box is intentionally removed; the slide
title, subtitle, placement, and speaker notes identify it as GreenNet validation.
Numerical Evidence II uses separated CouplingNet field assets rather than
artifact-titled composite plots.  It now uses `checkpoints/Disk_CDR/coupling/artifacts`
and the `coupling_cdr_evidence_rel_sol_quantiles` basename for convection-diffusion-reaction
solver-level evidence.  Its 5-by-4 matrix has
relative-solution-error quantile columns (`min`, `q25`, `q50`, `q75`, `max`) and
source/reference/prediction/signed-error rows; row and column labels plus the
metric table are slide-native elements, so each image panel stays free of
titles, axes, colorbars, and sample ids.

## What to De-emphasize

- Do not center the talk on complex geometry generation.
- Do not explain axial line construction algorithms.
- Do not discuss disconnected interval bookkeeping unless asked.
- Do not include code, configuration, file schema, checkpoint, or dataset
  generation details.
- Do not present the full proof of the energy theorem.
- Do not derive the full Dirac/Heaviside distributional proof on the main
  GreenNet analytic slide.
- Do not turn the new CouplingNet context slide into an implementation diagram.
- Do not finalize numerical evidence claims before the actual GreenNet and
  CouplingNet result figures are selected.
- Do not use the old notation for the split energy; use
  \(\mathcal{E}_{\mathrm{split}}\) consistently.

## Expected Takeaway

The audience should leave with four points:

1. **Axial reduction:** multi-dimensional elliptic learning is reduced to
   normalized one-dimensional Green operator learning.
2. **Analytic-neural GreenNet:** GreenNet embeds the analytic singular and
   cancellation structure of the one-dimensional Green function, while learning
   the remaining smooth correction.
3. **Hybrid coupling:** CouplingNet learns a directional source split using
   source, operator, axial, and transverse context.
4. **Energy-bound interpretation:** the split consistency energy is tied to a
   final solution error bound under explicit structural assumptions.

One thesis sentence for the final slide:

> A 2D elliptic problem is solved through axial Green inversions and a learned,
> balance-preserving source decomposition.

Final closing line:

> GreenNet supplies line-wise Green inverses; CouplingNet learns the source split
> that turns them into a 2D elliptic solver.
