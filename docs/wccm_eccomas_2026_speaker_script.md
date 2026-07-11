# WCCM-ECCOMAS 2026 Speaker Script

## Metadata

- **Conference:** WCCM-ECCOMAS 2026
- **Minisymposium:** MS165 - Methods and Applications of Model Order Reduction
- **Talk title:** Hybrid Green's Function Learning With Axial Reduction for Multi-Dimensional Elliptic Problems
- **Presenter:** Junhong Jo, National Institute for Mathematical Sciences
- **Target duration:** 14-15 minutes for the spoken script
- **Language:** English
- **Status:** Editable draft before inserting the final version into Quarto speaker notes

## How to Use This Script

This document is the editable speaker-script draft for the current Quarto/Reveal.js deck.  It is not yet copied into the slide notes.  After revision, the final version can be inserted into the `::: {.notes}` blocks in the Quarto source.

Each slide section contains the target time, reveal cues, a full spoken script, a compression option, and a transition sentence.  The compression option is the sentence to use when the live timing becomes tight.

Canonical thesis sentence:

> GreenNet supplies line-wise Green inverses; CouplingNet learns the directional source split that turns them into a 2D elliptic solver.

## Timing Plan

| Slide | Title | Target time |
|---:|---|---:|
| 1 | Hybrid Green's Function Learning With Axial Reduction | 0:35 |
| 2 | Axial Reduction of Elliptic Operators | 1:15 |
| 3 | From a 2D Elliptic Problem to Coupled Axial Green Solves | 0:55 |
| 4 | Unit-Interval Pull-Back and Operator Scaling | 1:05 |
| 5 | GreenNet I: Normalized Axial Green Operator | 0:55 |
| 6 | GreenNet II: Analytic Green Structure and Learned Correction | 1:20 |
| 7 | GreenNet III: Source-to-Solution Supervision | 1:00 |
| 8 | Numerical Evidence I: GreenNet Kernel Approximation | 1:05 |
| 9 | CouplingNet I: Directional Source Split | 0:55 |
| 10 | CouplingNet II: Branches and Local Context for Split Prediction | 1:00 |
| 11 | CouplingNet III: Projection and Green Reconstruction | 1:00 |
| 12 | Energy-Norm Error Bound Proposition | 1:15 |
| 13 | Numerical Evidence II: CouplingNet Solution Reconstruction | 1:20 |
| 14 | Takeaway: Coupled Axial Green Solvers | 0:50 |
| **Total** | **Main talk only** | **14:30** |

Backup slides are not included in the timed talk.  They are Q&A support material only.

## Slide 1 - Hybrid Green's Function Learning With Axial Reduction

**Target time:** 0:35

**Reveal / click cues:** Static slide.  No animation.

**Speaker script:**

Good morning.  My name is Junhong Jo from the National Institute for Mathematical Sciences.  This is joint work with Taeyoung Ha at NIMS and Chang-Ock Lee at KAIST.

In this talk, I will describe a hybrid Green's-function learning framework for multi-dimensional elliptic problems.  The main idea is simple: instead of learning a full two-dimensional solver as one black-box map, we use line-wise Green inverses and learn how the source should be split and coupled across directions.

The short version is this: GreenNet supplies line-wise Green inverses, and CouplingNet learns the directional source split that turns them into a two-dimensional elliptic solver.

**Compression option:** GreenNet gives line-wise Green inverses, and CouplingNet learns the source split needed to assemble a two-dimensional solution.

**Transition to next slide:** I will first explain what is reduced in this axial viewpoint.

## Slide 2 - Axial Reduction of Elliptic Operators

**Target time:** 1:15

**Reveal / click cues:** Reveal the direct source-to-solution map first, then the directional split, then the GreenNet/CouplingNet anchor sentence.

**Speaker script:**

We start from a fixed heterogeneous elliptic operator.  The coefficient fields define the problem, and the sample variation comes from the source field.  So the basic learning target is the source-to-solution map, written here as \( \mathcal S_{a,\mathbf b,c}: f \mapsto u \).

Learning this map directly in two dimensions is expensive because it is high-dimensional and sensitive to the geometry.  The axial viewpoint replaces this single global representation by directional one-dimensional operators.  The operator is split into an \(x\)-direction part and a \(y\)-direction part, each with half of the reaction term, so that \(L_x u + L_y u = f\).

This is the point where the two networks enter.  GreenNet learns the line-wise Green inverses associated with these directional operators.  CouplingNet learns the directional source split, so that those line-wise inverses can be combined into a two-dimensional solution.

This is not meant as a purely geometric decomposition.  It is a structured decomposition of the elliptic operator and its source-to-solution map.

**Compression option:** The fixed operator maps \(f\) to \(u\); we replace one global representation by directional one-dimensional operators plus learned coupling.

**Transition to next slide:** The next slide shows the whole computation as a visual roadmap.

## Slide 3 - From a 2D Elliptic Problem to Coupled Axial Green Solves

**Target time:** 0:55

**Reveal / click cues:** Reveal the graphic abstract from left to right: 2D problem, axial intervals, GreenNet line-wise inverse, CouplingNet coupling, solution.

**Speaker script:**

This slide is the graphic abstract of the method.

On the left, we have a two-dimensional source field on a domain.  We then look at axial intersections of the domain.  Along each valid interval, the problem is represented as a one-dimensional line problem.

The GreenNet block learns the line-wise inverse.  Given a line-source profile, it reconstructs a line-wise solution profile through a Green kernel integral.  However, these line-wise solves do not by themselves determine how the original two-dimensional source should be distributed between directions.

That missing coupling is the role of CouplingNet.  It predicts the directional source components, which are then passed through the \(x\)- and \(y\)-direction Green reconstructions.  The final output is a two-dimensional solution field.

So the full pipeline is: slice, solve line-wise, learn the coupling, and reconstruct the field.

**Compression option:** The picture summarizes the solver: axial intervals go to line-wise Green inverses, and CouplingNet supplies the source coupling needed for a 2D field.

**Transition to next slide:** To make line-wise Green learning reusable, each physical interval is normalized to a unit interval.

## Slide 4 - Unit-Interval Pull-Back and Operator Scaling

**Target time:** 1:05

**Reveal / click cues:** Show the physical 1D operator, then the interval pull-back, then the scaling rules and normalized equation.

**Speaker script:**

Once we restrict the problem to an axial interval, that interval is generally not the unit interval.  It may have different length depending on where the line intersects the domain.

So we pull back the physical coordinate \(s\) to a normalized coordinate \(t\) in \([0,1]\), using \(s=s_0+Lt\), where \(L=s_1-s_0\).  The normalized solution is \(v(t)=u(s_0+Lt)\).

The important point is that the interval is normalized, but the physical length is not discarded.  It enters the unit-interval operator through the scaling of the coefficients and the source.  The diffusion coefficient itself is evaluated along the interval, the derivative of diffusion scales by \(L\), convection scales by \(L\), and reaction and source scale by \(L^2\).

This gives a shared unit-interval representation while still retaining the physical length and operator information of each line segment.

**Compression option:** The pull-back maps every physical interval to \([0,1]\), but the length \(L\) remains inside the scaled operator.

**Transition to next slide:** Now that each line problem is normalized, we can define the Green operator that GreenNet learns.

## Slide 5 - GreenNet I: Normalized Axial Green Operator

**Target time:** 0:55

**Reveal / click cues:** Reveal the contrast strip, then the source profile, kernel integral operator, and output profile.

**Speaker script:**

The previous step normalized the operator.  This step learns its Green inverse.

By Green operator, I mean the source-to-solution map induced by a Green kernel integral.  On the unit interval, the operator maps a source profile \(f_{\mathrm{unit}}\) to a solution profile \(v\) by integrating the source against \(G_{\mathrm{unit}}(t,\eta)\).

This is not a global two-dimensional Green function.  It is a local one-dimensional Green kernel associated with a normalized axial line problem.  The coefficient profiles along that line define the local one-dimensional operator, and the kernel coordinates \((t,\eta)\) define where the Green kernel is evaluated.

So GreenNet learns a family of normalized axial Green kernels, conditioned by the local operator profiles.

**Compression option:** A Green operator is the integral map from a line source to a line solution; GreenNet learns this normalized one-dimensional kernel.

**Transition to next slide:** The next question is how to learn this kernel without forcing the neural network to learn the Green singularity from scratch.

## Slide 6 - GreenNet II: Analytic Green Structure and Learned Correction

**Target time:** 1:20

**Reveal / click cues:** Use the three GreenNet analytic states: role thesis, analytic identities, learned correction envelope.

**Speaker script:**

This is the hybrid part of GreenNet.  For variable-coefficient operators, the exact Green kernel is usually not available in closed form.  But the Green kernel still has structural features: a source-point singularity, a jump condition, and homogeneous boundary behavior.

The idea is not to ask the neural network to learn all of that from scratch.  Instead, the final learned kernel is built from three components.

The \(A(t)G_0(t,\eta)\) term provides the Dirac-delta jump structure.  This is the singular Green-function behavior that creates the source response.

The \(B(t)(J_0-\frac12E)\) term compensates the Heaviside-type contribution generated by that jump construction.  Here \(J_0\) is an antiderivative of \(G_0\), so it creates the derivative structure needed for cancellation.

Finally, \(E(t,\eta)M(t)R_\theta(t,\eta)\) is the learned smooth correction.  The envelope factors make this learned correction compatible with the boundary structure.

So the neural network is not responsible for the singular part of the Green function.  It learns the remaining smooth residual after the analytic structure has been built in.

**Compression option:** The analytic terms encode the singular jump and cancellation structure; the neural network learns only the smooth residual.

**Transition to next slide:** With this kernel form fixed, we still need a supervised signal for GreenNet.

## Slide 7 - GreenNet III: Source-to-Solution Supervision

**Target time:** 1:00

**Reveal / click cues:** Reveal GP target generation, boundary-compatible target, source generation, reconstruction loss.

**Speaker script:**

GreenNet is supervised, but not by direct pointwise labels for the exact Green kernel.  Instead, it is trained through source-to-solution reconstruction.

We first sample a smooth target profile from a Gaussian process.  Then we subtract the endpoint interpolant so that the resulting target solution satisfies homogeneous Dirichlet boundary conditions on the unit interval.

The source is not sampled independently.  It is generated by applying the unit operator to this target solution.  Therefore, by construction, this source and target solution form a consistent one-dimensional boundary value problem.

GreenNet then produces \(v_\theta\) by integrating the learned kernel against the generated source.  The training loss checks whether \(v_\theta\) reconstructs the target solution \(v\).

So GreenNet learns the kernel through its action as a source-to-solution operator, not through direct exact-kernel supervision.

**Compression option:** GreenNet is trained by asking whether the learned kernel maps sources generated from target solutions back to those target solutions.

**Transition to next slide:** Before introducing CouplingNet, I will show that this line-wise Green kernel is actually being captured.

## Slide 8 - Numerical Evidence I: GreenNet Kernel Approximation

**Target time:** 1:05

**Reveal / click cues:** State 1: heatmaps and diagnostics.  State 2: fixed-\(\eta\) slice.  State 3: takeaway.

**Speaker script:**

This is kernel-level evidence for GreenNet, before introducing the coupling model.

The example is a reaction-free convection-diffusion problem on a disk.  We use this case because reference axial Green kernels are available for validation.

The first two heatmaps compare the reference kernel and the learned kernel on a representative axial interval.  The third heatmap is the signed error.  What I want to highlight is that the learned kernel captures the singular Green structure around the diagonal.

Now, the fixed-\(\eta\) slice gives a more direct view.  The reference and learned curves are almost overlapping, including near the singular point.  The reference curve is drawn on top so that it remains visible even when the learned curve is very close.

The signed error is also informative.  It is not concentrated only near the singular diagonal.  This supports the purpose of the analytic wrapping: the singular structure is handled well, and the remaining error is not dominated by the singular line.

**Compression option:** The learned kernel follows the singular Green structure, and the signed error is not concentrated near the singular diagonal.

**Transition to next slide:** Once the line-wise inverse is validated, the remaining question is how to split the two-dimensional source across directions.

## Slide 9 - CouplingNet I: Directional Source Split

**Target time:** 0:55

**Reveal / click cues:** Reveal why GreenNet alone is insufficient, then \(\phi,\psi\), then the balance relation.

**Speaker script:**

GreenNet gives line-wise inverses, but the original problem is still two-dimensional.  We need to decide what source should be sent to the horizontal line problems and what source should be sent to the vertical line problems.

This is the role of CouplingNet.  It predicts a directional source split.  Conceptually, \(\phi\) is the \(x\)-direction flux-divergence or source component, and \(\psi\) is the \(y\)-direction component.  With the reaction term split evenly, the exact components would satisfy \(\phi+\psi=f\).

The important point is that CouplingNet is not predicting the solution directly.  It predicts the source components that make the Green reconstructions possible.

Also, CouplingNet is trained without reference-solution labels or split labels.  The reference solution is used for evaluation, not for training this split model.

**Compression option:** CouplingNet learns \(\phi\) and \(\psi\), the directional source components that couple the line-wise Green inverses.

**Transition to next slide:** To predict this split, the model needs both line-level profiles and pointwise coordinate context.

## Slide 10 - CouplingNet II: Branches and Local Context for Split Prediction

**Target time:** 1:00

**Reveal / click cues:** Reveal the central predictor, branch nets, trunk nets, then the combined map.

**Speaker script:**

This slide shows the information used by CouplingNet.

The branch nets process profile-level information: the source profile, coefficient profiles, and line-geometry structure.  The coefficient field is fixed for a given problem, but its axial profiles vary from line to line, so the model still needs to see those local profiles.

The trunk nets process pointwise coordinate information.  The primary axial coordinate \(t_{\parallel}\) tells the model where the point lies along the line being predicted.  The transverse coordinate \(t_{\perp}\) gives local information about the orthogonal line passing through the same physical point.

This separation is useful because the split value depends both on profile-level operator information and on pointwise location, including transverse boundary context.

Together, these branch and trunk features produce the directional source split.

**Compression option:** Branch nets encode profiles and line geometry; trunk nets encode pointwise axial and transverse coordinates.

**Transition to next slide:** After CouplingNet predicts a raw split, we impose physical balance and reconstruct the solution.

## Slide 11 - CouplingNet III: Projection and Green Reconstruction

**Target time:** 1:00

**Reveal / click cues:** Reveal raw split, residual projection, balanced split, Green reconstructions, and final average.

**Speaker script:**

CouplingNet first proposes a raw directional split.  But this raw split is not guaranteed to satisfy the physical balance \(\phi+\psi=f\).

So we project the raw split in physical split variables.  We compute the residual between the true source and the sum of the raw components, and then distribute that residual symmetrically to \(\phi\) and \(\psi\).  After this projection, the balance condition \(\phi+\psi=f\) is enforced.

Then the two balanced components are passed through the axial Green reconstructions.  The \(x\)-direction source gives \(u_\phi\), and the \(y\)-direction source gives \(u_\psi\).

The final prediction is the average of these two represented solutions:
\[
u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi).
\]

So the full solver is: predict a split, impose physical balance, reconstruct with Green operators, and average.

**Compression option:** Projection enforces \(\phi+\psi=f\), then \(G_x\) and \(G_y\) reconstruct two represented solutions whose average is the prediction.

**Transition to next slide:** This split consistency also has a useful energy-norm interpretation.

## Slide 12 - Energy-Norm Error Bound Proposition

**Target time:** 1:15

**Reveal / click cues:** Reveal energy norm, split energy, reference solution, error bound, assumptions footer.

**Speaker script:**

The energy-norm result explains why the agreement between \(u_\phi\) and \(u_\psi\) is meaningful.

The norm here is diffusion-weighted.  The coefficient \(a(x)\) is the diffusion coefficient, and the norm measures the gradient of the error in the PDE energy scale.

The split-energy loss is defined as the energy norm of the difference between the two represented solutions:
\[
\mathcal E_{\mathrm{split}}=\|u_\phi-u_\psi\|_a^2.
\]

Under structural assumptions, this quantity is not only a diagnostic.  It bounds the final prediction error:
\[
\|u_{\mathrm{pred}}-u_*\|_a
\le
\frac{C_E}{2}\sqrt{\mathcal E_{\mathrm{split}}}.
\]

Here \(u_*\) is the reference solution of the full elliptic problem, and \(C_E\) is a stability constant of the fixed elliptic operator.

This is a conditional statement.  It assumes exact or controlled Green reconstruction and admissible represented solutions.  So I am not claiming an unconditional theorem for every learned model.  But it gives a structural reason why split consistency is connected to final solution error.

**Compression option:** Under controlled Green reconstruction, small split energy bounds the final energy error.

**Transition to next slide:** Now I will show solver-level evidence for the coupled model.

## Slide 13 - Numerical Evidence II: CouplingNet Solution Reconstruction

**Target time:** 1:20

**Reveal / click cues:** Source row and metric card first, then reference row, prediction row, signed-error row, and takeaway.

**Speaker script:**

This slide shows the full CouplingNet solver evidence on a convection-diffusion-reaction disk problem.

The columns are selected by relative solution error: minimum, first quartile, median, third quartile, and maximum among the selected test cases.  The goal is to avoid showing only one favorable example.

The first row is the source.  The second row is the reference solution.  The third row is the predicted solution.  The fourth row is the signed error.

The source and solution scales are per sample, so each column remains visually readable.  The signed-error row uses a shared zero-centered scale, so the error magnitudes can still be compared across the selected samples.

On the right, the metric card shows the relative solution error and the split-energy loss for each column.  In this set, the relative solution error ranges from about 2.3 percent to about 7.0 percent.

The main message is that the learned directional source split supports two-dimensional reconstruction across the observed error range, not only for a single best-case sample.

**Compression option:** Across relative-error quantiles, the predicted solutions track the references, and the signed-error row shows where the coupled solver still deviates.

**Transition to next slide:** I will close by summarizing the four contributions.

## Slide 14 - Takeaway: Coupled Axial Green Solvers

**Target time:** 0:50

**Reveal / click cues:** Reveal thesis, then the four contribution blocks, then the closing banner.

**Speaker script:**

The takeaway is that a two-dimensional elliptic problem can be solved through axial Green inversions and a learned, balance-preserving source decomposition.

There are four main pieces.

First, axial Green kernels provide line-wise inverse operators on normalized intervals.

Second, GreenNet uses analytic structure so that the singular behavior of the Green function is built in before the neural correction.

Third, CouplingNet learns the directional source split without reference-solution or split labels.  Reference solutions are used for evaluation, not for training CouplingNet.

Fourth, split-energy consistency can be connected to final solution error under explicit structural assumptions.

So the final message is: GreenNet supplies line-wise Green inverses, and CouplingNet learns the source split that turns them into a two-dimensional elliptic solver.

Thank you.

**Compression option:** GreenNet learns line-wise Green inverses; CouplingNet learns the balance-preserving source split; together they form a two-dimensional elliptic solver.

**Transition to next slide:** Stop here for the timed talk.  Use the backup menu only if a question calls for it.

## Backup / Q&A Prompt Snippets

These prompts are not part of the 14-15 minute timed script.

### Backup A - Dirac/Heaviside Derivation Sketch

**Use when asked:** Why is the analytic GreenNet wrapping necessary?

**Short response:** The analytic part separates the distributional behavior from the smooth residual.  \(G_0\) supplies the Dirac-delta jump.  Applying the variable-coefficient operator to that jump also creates a Heaviside-type contribution, and the \(J_0-\frac12E\) term is designed to cancel that contribution.  The neural network then learns the remaining smooth correction.

### Backup B - Imperfect Green Reconstruction Perturbation

**Use when asked:** What happens if GreenNet is not exact?

**Short response:** The energy bound becomes a perturbation statement.  Directional mismatch, represented by \(\varepsilon_x-\varepsilon_y\), affects the split-consistency term.  A common Green bias, represented by \(\varepsilon_x+\varepsilon_y\), can remain invisible to energy consistency.  So the bound is meaningful under exact or controlled Green reconstruction, and GreenNet accuracy remains important.

### Backup C - Connected-Interval Pull-Back Detail

**Use when asked:** How are non-square or complex domains handled?

**Short response:** Each axial line intersects the domain in one or more connected intervals.  Each connected interval is treated as an independent one-dimensional Dirichlet problem and pulled back to the unit interval.  Disconnected intervals are not merged, because merging would create artificial information flow through regions outside the domain.
