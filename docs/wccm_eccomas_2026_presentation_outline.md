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

1. **Start from MOR motivation and axial reduction.**  Even under a fixed
   heterogeneous elliptic operator, directly learning the global
   source-to-solution map is expensive and geometrically rigid, so the method
   represents that map through one-dimensional axial Green reconstructions plus
   learned coupling.
2. **Normalize every axial interval and keep physical scale.**  A physical
   interval of length \(L\) is mapped to \([0,1]\), and \(L\) enters the normalized
   operator coefficient profiles and source profile through pull-back scaling.
3. **Explain GreenNet I: normalized axial operators.**  GreenNet works on a
   normalized one-dimensional operator on the unit interval, not on a global 2D
   Green function.
4. **Explain GreenNet II: analytic hybrid structure.**  The normalized kernel is
   not a black-box neural kernel.  It combines analytic Green-function terms for
   the Dirac jump and Heaviside cancellation with a learned smooth correction.
5. **Explain GreenNet III: reconstruction supervision.**  GreenNet is supervised
   through source-to-solution reconstruction, not through direct pointwise labels
   for the exact Green kernel.
6. **Explain CouplingNet I: source-conditioned split.**  CouplingNet learns how
   the full source should be split into directional source components so axial
   Green operators can reconstruct the multi-dimensional solution.
7. **Explain CouplingNet II: split prediction context.**  CouplingNet uses source
   information, local operator information, axial coordinate context, and
   transverse boundary context to predict the split source.
8. **Explain CouplingNet III: projection and reconstruction.**  Projection
   imposes \(\phi+\psi=f\), and the projected split fields are passed through
   Green reconstructions to build \(u_{\mathrm{pred}}\).
9. **State the error-bound message.**  The split energy
   \(\mathcal{E}_{\mathrm{split}}\) is not only a heuristic agreement loss; under
   structural assumptions, it bounds the final solution energy error.
10. **Separate numerical evidence by model role.**  GreenNet evidence should show
    interval source-to-solution reconstruction quality, while CouplingNet
    evidence should show full solution reconstruction and split consistency.
11. **Close with contributions.**  The contribution is a hybrid axial Green
    reduction plus source-conditioned coupling, supported by an energy-norm
    error-bound interpretation.

## Slide-Level Outline

| # | Slide title | Main message | Key equation / visual | Speaker focus | Time |
|---|---|---|---|---|---|
| 1 | Hybrid Green's Function Learning With Axial Reduction | The method combines axial Green operators and learned source coupling for elliptic PDEs. | Title, authors, minisymposium name, one-sentence thesis. | Say the whole talk in one sentence: reduce the operator, learn the coupling. | 0:30 |
| 2 | MOR Motivation: Axial Reduction of Elliptic Operators | The model reduction is from one global source-to-solution map under a fixed heterogeneous operator to directional 1D operators and axial Green reconstructions plus learned coupling. | 2D PDE, \(L_x,L_y\) directional operators, and schematic: fixed 2D operator \(\rightarrow\) \(L_x,L_y\) \(\rightarrow\) axial Green operators + coupling. | Merge motivation and reduction: what is hard, what is reduced, and what the directional split is. | 1:10 |
| 3 | Unit-Interval Pull-Back and Operator Scaling | Physical intervals are normalized, but their length remains in the transformed operator coefficient profiles and source profile. | Generic physical 1D operator, \(s=s_0+Lt\), and scaling equations for \(a,a',b_\parallel,c,f\). | Keep geometry lightweight and make scaling the technical bridge to GreenNet. | 1:10 |
| 4 | GreenNet I: Normalized Axial Green Operator | GreenNet works on a normalized one-dimensional Green operator, not on a global 2D Green function. | \(\mathcal G_{\mathrm{unit}}[f_{\mathrm{unit}}](t)=\int_0^1G_{\mathrm{unit}}(t,\eta)f_{\mathrm{unit}}(\eta)\,d\eta\) and an operator branch/trunk schematic. | Define "Green operator" as the integral source-to-solution map on a normalized interval. | 0:55 |
| 5 | GreenNet II: Analytic Green Structure and Learned Correction | The kernel is a hybrid of analytic singular/cancellation structure and a learned smooth correction. | \(G_\theta=E M R_\theta+B(J_0-\frac12E)+A G_0\), \(G_0\) base kernel, \(J_0\) antiderivative identity, and \(A/B\) coefficient factors. | Explain the three roles: Dirac jump, Heaviside cancellation, learned correction. | 1:10 |
| 6 | GreenNet III: Source-to-Solution Supervision | GreenNet is supervised by reconstruction, not by direct exact-kernel labels. | \(v_\theta(t)=\int_0^1G_\theta(t,\eta)f_{\mathrm{unit}}(\eta)\,d\eta\). | State that the loss checks whether the learned kernel maps sources to solutions. | 0:55 |
| 7 | CouplingNet I: Source-Conditioned Directional Split | CouplingNet learns how the multi-dimensional source should be split between directions. | \(\phi+\psi=f\), source \(\rightarrow\) \((\phi,\psi)\) diagram. | State that the multi-dimensional coupling is learned through the split. | 0:50 |
| 8 | CouplingNet II: Branches and Local Context for Split Prediction | The split predictor uses source, operator, axial coordinate, and transverse boundary context. | Conceptual branch/trunk context diagram without implementation detail. | Explain why the split depends on both source and local geometric/operator context. | 1:00 |
| 9 | CouplingNet III: Projection and Green Reconstruction | Projection enforces balance; Green reconstruction produces two represented solutions. | \(u_\phi=G_x[\phi]\), \(u_\psi=G_y[\psi]\), \(u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi)\). | Connect the two networks into one solver pipeline. | 0:55 |
| 10 | Energy-Norm Error Bound Proposition | The split energy bounds final solution error under structural assumptions. | \(\mathcal{E}_{\mathrm{split}}=\|u_\phi-u_\psi\|_a^2\), \(\|u_{\mathrm{pred}}-u_*\|_a\le\frac{C_E}{2}\sqrt{\mathcal{E}_{\mathrm{split}}}\). | Make the theoretical claim clearly, without a full proof. Mention perturbation terms for learned Green errors. | 1:10 |
| 11 | Numerical Evidence I: GreenNet Reconstruction Quality | The learned normalized Green operator reconstructs interval solutions from sources. | Source-to-solution curve comparison or Green kernel heatmap with reconstruction error. | Show GreenNet as an axial inverse before moving to full solution evidence. | 0:55 |
| 12 | Numerical Evidence II: CouplingNet Solution Reconstruction | The learned source split leads to a consistent multi-dimensional reconstruction. | Reference, prediction, signed error, and \(u_\phi-u_\psi\) or energy diagnostic. | Show final solution quality and tie split mismatch back to the energy proposition. | 1:05 |
| 13 | Contributions and Takeaway | The framework gives an axial Green reduction, analytic-neural GreenNet, learned source coupling, and energy-bound interpretation. | Four contribution bullets and final thesis. | End with a compact message before Q&A. | 0:35 |
| 14 | Backup / Q&A Prompt | Keep extra derivation or split-consistency plots available if asked. | Optional Dirac/Heaviside derivation sketch, extra GreenNet/CouplingNet evidence, or geometry detail. | Do not include this in the target 11-12 minute timing. | Backup |

This outline has 13 main slides plus one backup/Q&A prompt slide.  It fits an
11-12 minute talk if the merged motivation and pull-back slides remain compact.
The main compression choice is still the evidence block: show one focused
GreenNet evidence slide and one focused CouplingNet evidence slide rather than a
large metric table.

## Core Equations to Show

Use only the equations needed to carry the narrative.  Avoid derivations except
for the pull-back scaling and the final error-bound statement.

### Directional split operators

\[
-\nabla\cdot(a\nabla u)+\mathbf b\cdot\nabla u+cu=f,
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
s=s_0+Lt,\qquad v(t)=u(s_0+Lt).
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
\(B(t)(J_0-\frac12E)\) cancels the leading Heaviside-type contribution, and
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

### Source split and final prediction

\[
\phi+\psi=f,
\qquad
u_{\mathrm{pred}}
=
\frac12(u_\phi+u_\psi).
\]

### Energy-norm error bound

\[
\mathcal{E}_{\mathrm{split}}
=
\|u_\phi-u_\psi\|_a^2,
\qquad
\|u_{\mathrm{pred}}-u_*\|_a
\le
\frac{C_E}{2}\sqrt{\mathcal{E}_{\mathrm{split}}}.
\]

If there is time, mention the learned-Green perturbation message verbally:
imperfect Green reconstruction adds mismatch and common-bias terms, so GreenNet
accuracy remains part of the full error story.

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
- Do not use the old notation for the split energy; use
  \(\mathcal{E}_{\mathrm{split}}\) consistently.

## Expected Takeaway

The audience should leave with four points:

1. **Axial reduction:** multi-dimensional elliptic learning is reduced to
   normalized one-dimensional Green operator learning.
2. **Analytic-neural GreenNet:** GreenNet embeds the analytic singular and
   cancellation structure of the one-dimensional Green function, while learning
   the remaining smooth correction.
3. **Hybrid coupling:** CouplingNet learns a source-conditioned split using
   source, operator, axial, and transverse context.
4. **Energy-bound interpretation:** the split consistency energy is tied to a
   final solution error bound under explicit structural assumptions.

One closing sentence for the final slide:

> Hybrid Green's function learning provides a reduced elliptic solver by learning
> one-dimensional Green operators and the source coupling that makes their
> reconstructions energy-consistent.
