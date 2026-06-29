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
intervals are pulled back to the unit interval, and this pull-back induces the
coefficient and source scaling used by GreenNet.

## Narrative Flow

1. **Start from model reduction.**  Directly learning a full multi-dimensional
   elliptic solution operator is expensive and geometrically rigid.  The method
   seeks a reduced representation that preserves the Green-operator structure.
2. **Introduce axial reduction.**  The multi-dimensional problem is reduced to
   families of one-dimensional axial Green reconstructions plus a learned
   coupling mechanism.
3. **Normalize every axial interval.**  A physical interval of length \(L\) is
   mapped to \([0,1]\).  The interval length does not disappear; it enters through
   coefficient and source scaling.
4. **Explain GreenNet I: normalized axial operators.**  GreenNet first receives
   a normalized one-dimensional operator on the unit interval, including the
   coefficient and source scaling induced by the pull-back.
5. **Explain GreenNet II: analytic hybrid structure.**  The normalized kernel is
   not a black-box neural kernel.  It combines analytic Green-function terms for
   the Dirac jump and Heaviside cancellation with a learned smooth correction.
6. **Explain GreenNet III: reconstruction supervision.**  GreenNet is supervised
   through source-to-solution reconstruction, not through direct pointwise labels
   for the exact Green kernel.
7. **Explain CouplingNet.**  CouplingNet learns how the full source should be
   split into directional source components so that axial Green operators can
   reconstruct the multi-dimensional solution.
8. **Enforce physical balance.**  Projection imposes \(\phi+\psi=f\), making the
   learned split compatible with the original PDE.
9. **State the error-bound message.**  The split energy
   \(\mathcal{E}_{\mathrm{split}}\) is not only a heuristic agreement loss; under
   structural assumptions, it bounds the final solution energy error.
10. **Show evidence.**  Use prediction/error fields and concise split-consistency
    diagnostics to demonstrate that the method reconstructs the target solution
    and keeps the two axial reconstructions consistent.
11. **Close with contributions.**  The contribution is a hybrid axial Green
   reduction plus source-conditioned coupling, supported by an energy-norm error
   bound interpretation.

## Slide-Level Outline

| # | Slide title | Main message | Key equation / visual | Speaker focus | Time |
|---|---|---|---|---|---|
| 1 | Hybrid Green's Function Learning With Axial Reduction | The method combines axial Green operators and learned source coupling for elliptic PDEs. | Title, authors, minisymposium name, one-sentence thesis. | Say the whole talk in one sentence: reduce the operator, learn the coupling. | 0:30 |
| 2 | Why This Is a Model Reduction Problem | A full multi-dimensional elliptic operator is expensive to learn directly. | PDE statement and a schematic "full operator -> reduced axial operators". | Frame the problem for MOR: reduce dimensional complexity while preserving operator structure. | 1:00 |
| 3 | Axial Reduction: What Is Reduced? | The solution operator is represented through one-dimensional axial Green reconstructions. | Diagram: 2D domain with axial intervals, but no construction details. | Emphasize that the reduced objects are 1D Green operators, not a global 2D black-box map. | 1:00 |
| 4 | Unit-Interval Pull-Back | Non-unit physical intervals are normalized before Green learning. | \(s=s_0+Lt\), physical interval \(\rightarrow\) unit interval graphic. | Keep geometry lightweight: non-square domains only motivate interval normalization. | 1:00 |
| 5 | Scaling Induced by Pull-Back | Interval length enters through transformed coefficients and source. | Scaling equations for \(a,a',b,c,f\). | This is the technical bridge from physical intervals to a shared normalized learning problem. | 1:20 |
| 6 | GreenNet I: Normalized Axial Green Operator | GreenNet works on a normalized one-dimensional operator, not on a global 2D Green function. | Operator branch/trunk schematic on \([0,1]\). | Keep this conceptual: one normalized axial Green problem per interval. | 0:55 |
| 7 | GreenNet II: Analytic Green Structure and Learned Correction | The kernel is a hybrid of analytic singular/cancellation structure and a learned smooth correction. | \(G_\theta(t,\eta)=E M R_\theta+B(J_0-\frac12E)+A G_0\). | Explain the three roles: Dirac jump, Heaviside cancellation, learned correction. | 1:10 |
| 8 | GreenNet III: Source-to-Solution Supervision | GreenNet is supervised by reconstruction, not by direct exact-kernel labels. | \(v_\theta(t)=\int_0^1G_\theta(t,\eta)f_{\mathrm{unit}}(\eta)\,d\eta\). | State that the loss checks whether the learned kernel maps sources to solutions. | 0:55 |
| 9 | CouplingNet I: Source-Conditioned Directional Split | CouplingNet learns how the multi-dimensional source should be split between directions. | \(\phi+\psi=f\), source \(\rightarrow\) \((\phi,\psi)\) diagram. | State that the multi-dimensional coupling is learned through the split. | 0:55 |
| 10 | CouplingNet II: Projection and Green Reconstruction | Projection enforces balance; Green reconstruction produces two represented solutions. | \(u_\phi=G_x[\phi]\), \(u_\psi=G_y[\psi]\), \(u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi)\). | Connect the two networks into one solver pipeline. | 0:55 |
| 11 | Energy-Norm Error Bound Proposition | The split energy bounds final solution error under structural assumptions. | \(\mathcal{E}_{\mathrm{split}}=\|u_\phi-u_\psi\|_a^2\), \(\|u_{\mathrm{pred}}-u_*\|_a\le\frac{C_E}{2}\sqrt{\mathcal{E}_{\mathrm{split}}}\). | Make the theoretical claim clearly, without a full proof. Mention perturbation terms for learned Green errors. | 1:15 |
| 12 | Numerical Evidence: Solution Reconstruction | The final prediction matches the reference solution on the test domain. | Prediction, reference, signed error, and possibly \(u_\phi-u_\psi\). | Focus on solution quality and one split-consistency diagnostic. | 1:10 |
| 13 | Contributions and Takeaway | The framework gives an axial Green reduction, analytic-neural GreenNet, learned source coupling, and energy-bound interpretation. | Four contribution bullets and final thesis. | End with a compact message before Q&A. | 0:35 |
| 14 | Backup / Q&A Prompt | Keep extra derivation or split-consistency plots available if asked. | Optional Dirac/Heaviside derivation sketch or extra numerical panel. | Do not include this in the target 11-12 minute timing. | Backup |

This outline has 13 main slides plus one backup/Q&A prompt slide.  It fits an
11-12 minute talk only if each technical slide makes one point.  The main
compression choice is the numerical evidence block: keep one main evidence slide
and move detailed split-consistency plots to backup if time is tight.

## Core Equations to Show

Use only the equations needed to carry the narrative.  Avoid derivations except
for the pull-back scaling and the final error-bound statement.

### Pull-back to the unit interval

\[
s=s_0+Lt.
\]

### Coefficient and source scaling

\[
a_{\mathrm{unit}}=a_{\mathrm{phys}},
\quad
a'_{\mathrm{unit}}=L a'_{\mathrm{phys}},
\quad
b_{\mathrm{unit}}=L b_{\mathrm{phys}},
\quad
c_{\mathrm{unit}}=L^2c_{\mathrm{phys}},
\quad
f_{\mathrm{unit}}=L^2f_{\mathrm{phys}}.
\]

### Green reconstruction

\[
v(t)
=
\int_0^1
G_{\mathrm{unit}}(t,\eta)f_{\mathrm{unit}}(\eta)
\,d\eta.
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
not derive the full distributional proof on the main slide; present the role
decomposition only.

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
- Do not claim that energy consistency removes every learned-model error source.
- Do not use the old notation for the split energy; use
  \(\mathcal{E}_{\mathrm{split}}\) consistently.

## Expected Takeaway

The audience should leave with four points:

1. **Axial reduction:** multi-dimensional elliptic learning is reduced to
   normalized one-dimensional Green operator learning.
2. **Analytic-neural GreenNet:** GreenNet embeds the analytic singular and
   cancellation structure of the one-dimensional Green function, while learning
   the remaining smooth correction.
3. **Hybrid coupling:** CouplingNet learns the source split that lets axial Green
   operators reconstruct a multi-dimensional solution.
4. **Energy-bound interpretation:** the split consistency energy is tied to a
   final solution error bound under explicit structural assumptions.

One closing sentence for the final slide:

> Hybrid Green's function learning provides a reduced elliptic solver by learning
> one-dimensional Green operators and the source coupling that makes their
> reconstructions energy-consistent.
