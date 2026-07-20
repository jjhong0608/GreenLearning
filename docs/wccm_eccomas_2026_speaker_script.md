# WCCM-ECCOMAS 2026 Speaker Script

## Metadata

- **Conference:** WCCM-ECCOMAS 2026
- **Minisymposium:** MS165 - Methods and Applications of Model Order Reduction
- **Talk title:** Hybrid Green's Function Learning With Axial Reduction for Multi-Dimensional Elliptic Problems
- **Presenter:** Junhong Jo, National Institute for Mathematical Sciences
- **Target duration:** Approximately 13 minutes for the spoken script
- **Language:** English
- **Status:** Compressed 13-minute canonical script synchronized with the Quarto speaker notes

## How to Use This Script

This document is the canonical editable speaker script for the current Quarto/Reveal.js deck.  Its compressed main-talk script and Q&A prompts are synchronized with the `::: {.notes}` blocks in the Quarto source.  The timing plan reserves a short delivery buffer so that clicks and pauses fit within approximately 13 minutes.  Future wording changes should be applied to both locations.

Each slide section contains the target time, reveal cues, a full spoken script, a compression option, and a transition sentence.  The compression option is the sentence to use when the live timing becomes tight.

Terminology convention: reserve **field** for vector-valued quantities, such as the convection vector field \(\mathbf b\).  Refer to \(f\) as the source or forcing, \(u\) as the solution, \(a\) and \(c\) as coefficients, and \(\phi,\psi\) as directional source components.

Canonical thesis sentence:

> GreenNet supplies line-wise Green inverses; CouplingNet learns the directional source split that turns them into a 2D elliptic solver.

## Timing Plan

| Slide | Title | Target time |
|---:|---|---:|
| 1 | Hybrid Green's Function Learning With Axial Reduction | 0:40 |
| 2 | Axial Reduction of Elliptic Operators | 1:05 |
| 3 | From a 2D Elliptic Problem to Coupled Axial Green Solves | 0:50 |
| 4 | Unit-Interval Pull-Back and Operator Scaling | 0:55 |
| 5 | GreenNet I: Normalized Axial Green Operator | 0:45 |
| 6 | GreenNet II: Analytic Green Structure and Learned Correction | 1:10 |
| 7 | GreenNet III: Source-to-Solution Supervision | 0:50 |
| 8 | Numerical Evidence I: GreenNet Kernel Approximation | 0:55 |
| 9 | CouplingNet I: Directional Source Split | 0:50 |
| 10 | CouplingNet II: Branches and Local Context for Split Prediction | 0:50 |
| 11 | CouplingNet III: Projection and Green Reconstruction | 0:50 |
| 12 | Energy-Norm Error Bound Proposition | 1:05 |
| 13 | Numerical Evidence II: CouplingNet Solution Reconstruction | 1:15 |
| 14 | Takeaway: Coupled Axial Green Solvers | 0:40 |
| **Total** | **Main talk, before delivery buffer** | **12:40** |

Backup slides are not included in the timed talk.  They are Q&A support material only.

## Slide 1 - Hybrid Green's Function Learning With Axial Reduction

**Target time:** 0:40

**Reveal / click cues:** Static slide.  No animation.

**Speaker script:**

Good afternoon.  I am Junhong Jo from the National Institute for Mathematical Sciences.  This is joint work with Taeyoung Ha at NIMS and Chang-Ock Lee at KAIST.

We solve multi-dimensional elliptic problems with two complementary components: GreenNet supplies line-wise Green inverses, and CouplingNet learns the directional source split that turns them into a two-dimensional solver.  I will explain the construction and its numerical evidence.

**Compression option:** GreenNet provides line-wise inverses, while CouplingNet learns how to couple them.

**Transition to next slide:** Let me begin with the axial operator viewpoint.

## Slide 2 - Axial Reduction of Elliptic Operators

**Target time:** 0:55

**Reveal / click cues:** Reveal the direct source-to-solution map first, then the directional split, then the GreenNet/CouplingNet anchor sentence.

**Speaker script:**

In our setting, the coefficient functions \(a\), \(\mathbf b\), and \(c\) are prescribed, while the source \(f\) varies across samples.  Our learning target is therefore the source-to-solution map \(f\mapsto u\).

Representing this map directly over the full two-dimensional domain is high-dimensional and sensitive to geometry.  Instead, we split the operator into \(L_x\) and \(L_y\), with half of the reaction term in each direction, so that \(L_xu+L_yu=f\).

GreenNet learns the directional line-wise inverses.  CouplingNet learns how the source is divided between them.  This is an operator and source decomposition, not merely a geometric slicing.

The homogeneous boundary condition is enforced within each directional line problem.

**Compression option:** We replace one global source-to-solution map with directional inverses and learned coupling.

**Transition to next slide:** This diagram summarizes the resulting solver.

## Slide 3 - From a 2D Elliptic Problem to Coupled Axial Green Solves

**Target time:** 0:50

**Reveal / click cues:** Reveal the graphic abstract from left to right: 2D problem, axial intervals, GreenNet line-wise inverse, CouplingNet coupling, solution.

**Speaker script:**

Here is the complete pipeline.  A two-dimensional source is restricted to valid horizontal and vertical intersections.  Each connected interval in both directions becomes an independent one-dimensional line problem.  GreenNet maps its line source to a line solution through a Green kernel integral.

Those independent inverses still need a directional source allocation.  CouplingNet predicts \(\phi\) and \(\psi\), routes them through the \(x\)- and \(y\)-direction Green reconstructions, and recovers a consistent two-dimensional solution.

**Compression option:** GreenNet solves along lines, and CouplingNet supplies the directional source allocation.

**Transition to next slide:** We first normalize every physical interval.

## Slide 4 - Unit-Interval Pull-Back and Operator Scaling

**Target time:** 0:55

**Reveal / click cues:** Show the physical 1D operator, then the interval pull-back, then the scaling rules and normalized equation.

**Speaker script:**

Axial intervals have different endpoints and physical lengths.  We pull each coordinate \(s\) back to \(t\in[0,1]\) by \(s=s_0+Lt\), and write the normalized solution as \(v(t)=u(s_0+Lt)\).

Normalization does not discard the length.  It re-enters through the transformed operator: diffusion and convection scale with \(L\), while reaction and source scale with \(L^2\).  The bottom equation is the same physical line problem on \([0,1]\), with its operator and boundary locations retained.

**Compression option:** Every interval becomes \([0,1]\), with its length retained through operator scaling.

**Transition to next slide:** On this normalized interval, GreenNet learns the inverse.

## Slide 5 - GreenNet I: Normalized Axial Green Operator

**Target time:** 0:45

**Reveal / click cues:** Reveal the contrast strip, then the source profile, kernel integral operator, and output profile.

**Speaker script:**

A Green operator maps a line source to a line solution by integrating against a kernel.  GreenNet learns this action on the unit interval.

It is not a global two-dimensional Green function.  The axial coefficient profiles specify the local one-dimensional operator, while \(t\) and \(\eta\) are the evaluation and source coordinates.  GreenNet therefore learns a family of normalized axial kernels.

The same GreenNet model can then be reused across the different axial intervals.

**Compression option:** GreenNet learns the unit-interval kernel integral for each axial operator.

**Transition to next slide:** We now build its singular structure into the model.

## Slide 6 - GreenNet II: Analytic Green Structure and Learned Correction

**Target time:** 1:10

**Reveal / click cues:** Use the three GreenNet analytic states: role thesis, analytic identities, learned correction envelope.

**Speaker script:**

For variable coefficients, the exact kernel is rarely available, but its singularity, derivative jump, and homogeneous boundary behavior are known.  We therefore build these features in analytically.

The \(A(t)G_0\) term supplies the Dirac-delta jump.  The variable-coefficient operator also produces a Heaviside-type contribution, cancelled by \(B(t)(J_0-\frac12E)\), where \(J_0\) is an antiderivative of \(G_0\).

The network learns only the remaining smooth correction \(R_\theta\).  Its envelope \(E(t,\eta)M(t)\) makes that correction compatible with the endpoint conditions.  Reaction effects also enter through this learned residual.  The nonsmooth Green behavior is therefore supplied analytically, while learning is reserved for the smoother residual.

**Compression option:** Analytic terms handle the jump and cancellation; the network learns the boundary-compatible smooth residual.

**Transition to next slide:** The kernel is trained through its source-to-solution action.

## Slide 7 - GreenNet III: Source-to-Solution Supervision

**Target time:** 0:50

**Reveal / click cues:** Reveal GP target generation, boundary-compatible target, source generation, reconstruction loss.

**Speaker script:**

GreenNet is supervised through reconstruction, not exact-kernel labels.  We sample a smooth Gaussian-process profile and remove its endpoint interpolant to obtain a Dirichlet target \(v\).

Applying the unit operator to \(v\) generates a consistent source.  Source and target are normalized by the same factor.  GreenNet integrates that source against \(G_\theta\), and the loss compares the reconstruction \(v_\theta\) with \(v\).  The kernel is therefore learned through its operator action.

By construction, every generated pair satisfies the same one-dimensional boundary-value problem, even though no exact kernel values are used as labels.

**Compression option:** Consistent source-solution pairs supervise the learned Green operator action.

**Transition to next slide:** We can now test whether the kernel structure was learned.

## Slide 8 - Numerical Evidence I: GreenNet Kernel Approximation

**Target time:** 1:05

**Reveal / click cues:** State 1: heatmaps and diagnostics.  State 2: fixed-\(\eta\) slice.  State 3: takeaway.

**Speaker script:**

This reaction-free convection-diffusion problem is used because its reference axial kernels are available.  The displayed example is one representative vertical interval in the disk.

First, the reference and learned heatmaps show that GreenNet captures the singular diagonal structure.  Next, the fixed-\(\eta\) slice shows that the two curves nearly overlap, including near the singular point.

Finally, the signed-error heatmap and diagnostics show that error is not concentrated only near the diagonal.  This indicates that the analytic wrapping has handled the singular structure as intended.

**Compression option:** The learned kernel captures the singular structure without diagonal-dominated error.

**Transition to next slide:** We now need to couple these line-wise inverses.

## Slide 9 - CouplingNet I: Directional Source Split

**Target time:** 0:50

**Reveal / click cues:** Reveal why GreenNet alone is insufficient, then \(\phi,\psi\), then the balance relation.

**Speaker script:**

GreenNet provides the directional inverses, but it does not determine how the two-dimensional source is split into the \(x\)- and \(y\)-directional source components.

CouplingNet predicts \(\phi\) and \(\psi\), the horizontal and vertical flux-divergence or source components.  With the reaction term divided equally, their physical balance is \(\phi+\psi=f\).  CouplingNet predicts this split rather than the solution itself.

Training uses neither reference-solution nor split labels.  It is driven by balance, Green reconstruction, and split consistency; reference solutions are used only for evaluation.

**Compression option:** CouplingNet learns the balanced directional source components, not the solution.

**Transition to next slide:** Its inputs combine line profiles with pointwise coordinates.

## Slide 10 - CouplingNet II: Branches and Local Context for Split Prediction

**Target time:** 0:50

**Reveal / click cues:** Reveal the central predictor, branch nets, trunk nets, then the combined map.

**Speaker script:**

CouplingNet separates profile-level and pointwise information.

The branch nets encode the source, axial coefficient profiles, and line geometry.  Although the coefficients are fixed for the problem, their profiles differ across lines.

The trunk nets encode the pointwise positions within the primary and transverse axial intervals.  Combining both representations at each point produces the directional split.

**Compression option:** Branches encode line profiles; trunks encode axial and transverse coordinates.

**Transition to next slide:** The predicted split is then balanced and reconstructed.

## Slide 11 - CouplingNet III: Projection and Green Reconstruction

**Target time:** 0:50

**Reveal / click cues:** Reveal raw split, residual projection, balanced split, Green reconstructions, and final average.

**Speaker script:**

GreenNet evaluates each axial inverse on the normalized unit interval.  CouplingNet, however, must enforce the directional source balance in the original physical coordinates.  The raw source components are therefore restored to the physical source scale before projection.

The raw split need not satisfy \(\phi+\psi=f\).  We form
\[
r=f-(\phi_{\mathrm{raw}}+\psi_{\mathrm{raw}})
\]
and add half of \(r\) to each component.  This gives a balanced split satisfying \(\phi+\psi=f\).

The balanced components are then pulled back along their respective axial intervals and passed through \(G_x\) and \(G_y\), producing \(u_\phi\) and \(u_\psi\).  Their average,
\[
\frac12(u_\phi+u_\psi),
\]
is the final prediction.

The projection preserves the original source exactly while changing only its directional decomposition.

**Compression option:** GreenNet solves on normalized intervals, while projection enforces the source balance in physical coordinates before reconstruction.

**Transition to next slide:** Their agreement yields an energy-error interpretation.

## Slide 12 - Energy-Norm Error Bound Proposition

**Target time:** 1:05

**Reveal / click cues:** Introduce the unsupervised loss, then reveal the energy norm, split energy, reference solution, error bound, and assumptions footer.

**Speaker script:**

CouplingNet is trained without reference-solution or directional-split labels.  We therefore need a loss that can be computed from the two directional reconstructions alone.

Since \(u_\phi\) and \(u_\psi\) are intended to represent the same physical solution, we measure their disagreement in the diffusion-weighted energy norm.  Here, \(a\) is the diffusion coefficient, and this norm captures gradient-level differences in the PDE scale.

This gives the unsupervised split-energy loss,
\[
\mathcal E_{\mathrm{split}}
=
\|u_\phi-u_\psi\|_a^2.
\]

For the theoretical analysis, let \(u_*\) denote the exact reference solution.  It does not enter the CouplingNet training loss.

Why is this loss relevant to solution accuracy?  Under the stated assumptions, it bounds the final energy error through the inequality shown here.  The constant \(C_E\) is a stability constant of the elliptic operator.

This result is conditional on \(H_0^1(\Omega)\) admissibility and exact or controlled Green reconstruction.  It provides a structural justification for the unsupervised loss, rather than an unconditional guarantee.

**Compression option:** CouplingNet minimizes the energy disagreement between its two reconstructions.  Under controlled Green reconstruction, this unsupervised loss bounds the final energy error.

**Transition to next slide:** We now evaluate the complete coupled solver.

## Slide 13 - Numerical Evidence II: CouplingNet Solution Reconstruction

**Target time:** 1:15

**Reveal / click cues:** Source row and metric card first, then reference row, prediction row, signed-error row, and takeaway.

**Speaker script:**

This is the complete solver on a convection-diffusion-reaction disk problem.  The five columns represent relative-error quantiles, from the minimum to the maximum, rather than one favorable sample.

As the reference, prediction, and signed-error rows appear, the predictions follow the reference solutions across this range.  Reference and prediction share a solution scale within each sample, while signed errors use one common zero-centered scale across all five columns.

The metric card reports both relative solution error and split-energy loss.  Relative errors range from about 2.3 to 7 percent.  These examples show that the learned source split supports reconstruction across the observed test-error distribution.

**Compression option:** Quantile-selected cases show reconstruction across the observed error range.

**Transition to next slide:** I will close with the four main contributions.

## Slide 14 - Takeaway: Coupled Axial Green Solvers

**Target time:** 0:40

**Reveal / click cues:** Reveal thesis, then the four contribution blocks, then the closing banner.

**Speaker script:**

The method has four pieces.  Axial Green kernels provide normalized line-wise inverses.  Analytic wrapping builds in the singular Green structure.  CouplingNet learns a directional source split without solution or split labels, rather than through direct supervision.  Split-energy consistency connects that decomposition to final error under explicit assumptions.

GreenNet supplies the line-wise inverses, and CouplingNet learns the balance-preserving source split that turns them into a two-dimensional elliptic solver.

Thank you.

**Compression option:** GreenNet learns the inverses; CouplingNet learns their balance-preserving coupling.

**Transition to next slide:** Stop here for the timed talk.  Use the backup menu only if a question calls for it.

## Backup / Q&A Prompt Snippets

These prompts are not part of the approximately 13-minute timed script.

### Backup A - Dirac/Heaviside Derivation Sketch

**Use when asked:** Why is the analytic GreenNet wrapping necessary?

**Short response:** The analytic part separates the distributional behavior from the smooth residual.  \(G_0\) supplies the Dirac-delta jump.  Applying the variable-coefficient operator to that jump also creates a Heaviside-type contribution, and the \(J_0-\frac12E\) term is designed to cancel that contribution.  The neural network then learns the remaining smooth correction.

### Backup B - Imperfect Green Reconstruction Perturbation

**Use when asked:** What happens if GreenNet is not exact?

**Short response:** The energy bound becomes a perturbation statement.  Directional mismatch, represented by \(\varepsilon_x-\varepsilon_y\), affects the split-consistency term.  A common Green bias, represented by \(\varepsilon_x+\varepsilon_y\), can remain invisible to energy consistency.  So the bound is meaningful under exact or controlled Green reconstruction, and GreenNet accuracy remains important.

### Backup C - Connected-Interval Pull-Back Detail

**Use when asked:** How are non-square or complex domains handled?

**Short response:** Each axial line intersects the domain in one or more connected intervals.  Each connected interval is treated as an independent one-dimensional Dirichlet problem and pulled back to the unit interval.  Disconnected intervals are not merged, because merging would create artificial information flow through regions outside the domain.
