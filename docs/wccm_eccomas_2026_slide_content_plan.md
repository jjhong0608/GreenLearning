# WCCM-ECCOMAS 2026 Slide Content Plan

## Talk Metadata

- **Conference:** WCCM-ECCOMAS 2026
- **Minisymposium:** MS165 - Methods and Applications of Model Order Reduction
- **Talk title:** Hybrid Green's Function Learning With Axial Reduction for Multi-Dimensional Elliptic Problems
- **Target talk time:** 11-12 minutes, leaving 3-4 minutes for Q&A
- **Audience:** computational mechanics and model order reduction researchers

## How to Use This Document

This document is not a slide deck.  It is a slide-by-slide content blueprint for building the deck.  Each slide section gives the exact title/subtitle direction, the message that must survive editing, and the material that can be removed if the slide becomes too dense.

- **Must include** means the content should appear on the slide body or in the main visual.
- **Optional / can omit** means the content can move to speaker notes, backup, or Q&A if the slide is too crowded.
- Main slides should not include implementation details, file formats, software commands, or data-generation procedures.
- Complex geometry should motivate unit-interval pull-back and scaling; it should not become the central story of the talk.
- The main talk should be understandable without the backup slide.

## Slide 1 - Hybrid Green's Function Learning With Axial Reduction

**Title:** Hybrid Green's Function Learning With Axial Reduction

**Subtitle:** A reduced elliptic solver built from one-dimensional Green operators and learned source coupling

**Main claim:** The method reduces multi-dimensional elliptic operator learning to normalized axial Green learning plus a learned coupling mechanism.

**Must include:**

- Talk title, conference, minisymposium, presenter, and affiliation.
- One-sentence thesis: axial Green reduction plus source-conditioned coupling.
- The phrase "elliptic problems" or "elliptic PDEs" so the technical scope is immediate.

**Optional / can omit:**

- Detailed affiliation blocks if they crowd the title slide.
- Method diagram; save it for Slide 2 unless a compact visual is available.
- Long motivation text.

**Suggested visual:** Minimal title slide with a quiet schematic: a two-dimensional domain feeding into axial one-dimensional operators and then a reconstructed solution.

**Equations / notation:** None required.

**Slide text draft:**

- Reduce a multi-dimensional elliptic problem into normalized axial Green problems.
- Learn the source coupling that makes the axial reconstructions consistent.
- Combine analytic Green structure with neural correction.

**Speaker emphasis:** Open with the main thesis in one sentence.  The audience should know immediately that this is a model reduction approach, not a geometry-generation talk and not a black-box neural solver.

## Slide 2 - MOR Motivation: Axial Reduction of Elliptic Operators

**Title:** MOR Motivation: Axial Reduction of Elliptic Operators

**Subtitle:** Replace a global source-to-solution map with structured one-dimensional Green reconstructions

**Main claim:** For a fixed heterogeneous elliptic operator, the reduction changes the source-to-solution representation from one global map to axial Green operators plus learned coupling.

**Must include:**

- A compact elliptic PDE statement:
  \[
  -\nabla\cdot(a\nabla u)+\mathbf b\cdot\nabla u+cu=f,
  \qquad
  \mathbf b=(b_x,b_y).
  \]
- The full-operator learning challenge: mapping a high-dimensional source field to \(u\) directly under a fixed heterogeneous elliptic operator is high-dimensional and geometry-sensitive.
- The fixed-coefficient framing: for each coefficient problem, the operator is fixed; the sample variation comes from the source field.
- The physical directional split:
  \[
  L_xu
  =
  -\partial_x(a\partial_x u)
  +
  b_x\partial_xu
  +
  \frac12cu,
  \]
  \[
  L_yu
  =
  -\partial_y(a\partial_y u)
  +
  b_y\partial_yu
  +
  \frac12cu,
  \]
  \[
  L_xu+L_yu=f.
  \]
- The reduced view:
  \[
  \text{fixed 2D elliptic operator}
  \rightarrow
  \text{directional 1D operators }L_x,L_y
  \rightarrow
  \text{axial Green operators}+\text{learned coupling}.
  \]
- The MOR framing: reduce operator complexity while preserving PDE structure.

**Optional / can omit:**

- Long related-work positioning.
- Detailed connected-interval construction.
- Implementation or data-layout detail.

**Suggested visual:** A three-stage reduction diagram: "fixed 2D elliptic operator" \(\rightarrow\) "\(L_x,L_y\) directional 1D operators" \(\rightarrow\) "axial Green reconstructions + learned coupling."  Keep the direct global source-to-solution map as a small contrast callout.

**Equations / notation:**

\[
-\nabla\cdot(a\nabla u)+\mathbf b\cdot\nabla u+cu=f,
\qquad
\mathbf b=(b_x,b_y).
\]

\[
\text{fixed 2D elliptic operator}
\rightarrow
\text{directional 1D operators }L_x,L_y
\rightarrow
\text{axial Green reconstructions}+\text{source coupling}.
\]

\[
L_xu
=
-\partial_x(a\partial_x u)
+
b_x\partial_xu
+
\frac12cu,
\qquad
L_yu
=
-\partial_y(a\partial_y u)
+
b_y\partial_yu
+
\frac12cu,
\]

\[
L_xu+L_yu=f.
\]

**Slide text draft:**

- Direct source-to-solution maps are high-dimensional and geometry-sensitive.
- For each coefficient problem, the operator is fixed; the sample variation comes from the source field.
- The convection coefficient is a vector field, \(\mathbf b=(b_x,b_y)\).
- Axial reduction starts by viewing the fixed 2D operator through directional 1D operators.
- Reduced objects: one-dimensional axial Green operators.
- Missing ingredient: learned coupling of the multi-dimensional source.

**Speaker emphasis:** This slide must answer both "why model reduction?" and "what is reduced?"  Show the physical \(L_x,L_y\) split here, but leave unit-interval normalization and scaling for Slide 3.

## Slide 3 - Unit-Interval Pull-Back and Operator Scaling

**Title:** Unit-Interval Pull-Back and Operator Scaling

**Subtitle:** Physical length is absorbed into normalized operator coefficient profiles and source profile

**Main claim:** Non-unit physical intervals are mapped to \([0,1]\), but their length remains in the transformed one-dimensional operator.

**Must include:**

- The generic physical one-dimensional operator after the directional split:
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
- The pull-back map:
  \[
  s=s_0+Lt,\qquad v(t)=u(s_0+Lt).
  \]
- \(t\in[0,1]\) as the normalized coordinate.
- The full scaling rule:
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
- The sentence: "The interval is normalized, but the physical length is not discarded."
- The clarification that \(b_\parallel\) is the primary convection component:
  \(b_\parallel=b_x\) on an \(x\)-directed interval and
  \(b_\parallel=b_y\) on a \(y\)-directed interval.

**Optional / can omit:**

- Chain-rule derivation.
- Full \(L_x,L_y\) split repetition; that belongs on Slide 2.
- Geometry generation.
- Interval bookkeeping or disconnected interval discussion.

**Suggested visual:** A physical 1D operator on \([s_0,s_1]\) mapped to a unit-interval operator on \([0,1]\), with a small callout listing the scaling factors.

**Equations / notation:**

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

\[
s=s_0+Lt,\qquad v(t)=u(s_0+Lt),\qquad t\in[0,1].
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
b_\parallel=b_x\ \text{on an \(x\)-directed interval},
\qquad
b_\parallel=b_y\ \text{on a \(y\)-directed interval}.
\]

**Slide text draft:**

- Slide 2 defines the physical directional split.
- Slide 3 normalizes each resulting physical 1D operator.
- Pull each physical axial interval back to \([0,1]\).
- The interval length reappears in the normalized operator coefficient profiles and source profile.
- The vector convection field contributes through the primary axial component.
- This transformed operator is the one GreenNet learns.

**Speaker emphasis:** This is the technical bridge from non-square/non-unit geometry to a shared normalized Green learning problem.  Do not repeat the full \(L_x,L_y\) split or derive the chain rule; show the physical 1D operator, the pull-back map, and the scaling rule.

## Slide 4 - GreenNet I: Normalized Axial Green Operator

**Title:** GreenNet I: Normalized Axial Green Operator

**Subtitle:** Learn one-dimensional Green responses, not a global two-dimensional kernel

**Main claim:** GreenNet represents the normalized axial Green operator: a source-to-solution map induced by integrating the source against an interval-local Green kernel.

**Must include:**

- The conceptual split between operator information and Green kernel evaluation coordinates.
- The statement that the Green kernel is one-dimensional and interval-local.
- The meaning of a Green operator as an integral source-to-solution map:
  \[
  \mathcal G_{\mathrm{unit}}[f_{\mathrm{unit}}](t)
  =
  \int_0^1
  G_{\mathrm{unit}}(t,\eta)
  f_{\mathrm{unit}}(\eta)
  \,d\eta,
  \qquad
  v=\mathcal G_{\mathrm{unit}}[f_{\mathrm{unit}}].
  \]

**Optional / can omit:**

- Activation function details.
- Architecture layer counts.
- Exact reference-kernel diagnostics.

**Suggested visual:** A branch/trunk-style schematic in conceptual terms: fixed local operator profiles define the axial operator; \((t,\eta)\) defines the kernel evaluation point.

**Equations / notation:**

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

**Slide text draft:**

- Fixed local operator profiles define the axial differential operator.
- Kernel coordinates \((t,\eta)\) define where the Green response is evaluated.
- A Green operator is the source-to-solution map induced by integrating the source against a Green kernel.
- Here the operator acts on the normalized source profile on the unit interval.

**Speaker emphasis:** Avoid making GreenNet sound like a generic regressor.  It is learning an interval-local Green operator: an integral operator that maps a normalized source profile to a normalized solution profile.

## Slide 5 - GreenNet II: Analytic Green Structure and Learned Correction

**Title:** GreenNet II: Analytic Green Structure and Learned Correction

**Subtitle:** A hybrid kernel: analytic singular structure plus neural smooth correction

**Main claim:** GreenNet is not a black-box kernel approximator; it embeds analytic Green-function structure and learns only the remaining smooth correction.

**Must include:**

- The central formula:
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
- The three role statements:
  - \(A(t)G_0(t,\eta)\): Dirac-\(\delta\) jump structure.
  - \(B(t)(J_0-\frac12E)\): Heaviside-type cancellation.
  - \(E(t,\eta)M(t)R_\theta(t,\eta)\): learned smooth correction.
- The analytic ingredients that make the roles interpretable:
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
- The phrase "analytic-neural GreenNet" or "hybrid Green kernel."

**Optional / can omit:**

- Full distributional proof.
- Full piecewise formula for \(J_0\).
- Reaction-coefficient discussion.

**Suggested visual:** Three stacked or color-coded terms forming \(G_\theta\): analytic jump term, analytic cancellation term, neural correction term.

**Equations / notation:**

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

**Slide text draft:**

- \(G_0\) supplies the Dirac-\(\delta\) jump through \(\partial_t^2G_0=-\delta\).
- \(J_0\) is an antiderivative of \(G_0\), so \(J_0-\frac12E\) produces the \(\partial_tG_0\) structure needed for Heaviside cancellation.
- \(E M R_\theta\): learns the remaining smooth correction.

**Speaker emphasis:** This slide carries the word "hybrid" in the talk title.  Say explicitly that \(G_0\) creates the singular jump, \(J_0\) is its antiderivative, and \(S=J_0-\frac12E\) provides the cancellation structure.  Since \(E\) is linear in \(t\), \(\partial_t^2E=0\), so \(\partial_t^2S=\partial_tG_0\).  Do not derive the full distributional proof.

## Slide 6 - GreenNet III: Source-to-Solution Supervision

**Title:** GreenNet III: Source-to-Solution Supervision

**Subtitle:** The kernel is trained through reconstruction, not direct kernel labels

**Main claim:** GreenNet is supervised by whether the learned operator action reconstructs target solutions from sources, rather than by pointwise labels for an exact Green kernel.

**Must include:**

- The learned operator action used for supervision:
  \[
  v_\theta(t)
  =
  \int_0^1
  G_\theta(t,\eta)f_{\mathrm{unit}}(\eta)\,d\eta.
  \]
- The target relation:
  \[
  v_\theta(t)\approx v(t).
  \]
- The statement that source-to-solution reconstruction is the learning signal.

**Optional / can omit:**

- Gaussian Process target construction details.
- Normalization details.
- Exact training objective notation if it crowds the slide.

**Suggested visual:** A source profile enters the Green kernel integration and produces a reconstructed solution profile; compare against a target solution curve.

**Equations / notation:**

\[
v_\theta(t)
=
\int_0^1
G_\theta(t,\eta)f_{\mathrm{unit}}(\eta)\,d\eta,
\qquad
v_\theta(t)\approx v(t).
\]

**Slide text draft:**

- Do not supervise the kernel pointwise.
- Supervise the operator action: source profile \(\rightarrow\) solution profile.
- This trains GreenNet as a solution operator on normalized intervals.

**Speaker emphasis:** This slide prevents a common misunderstanding: the model learns a Green operator through its action on sources, not by requiring exact Green labels everywhere.

## Slide 7 - CouplingNet I: Source-Conditioned Directional Split

**Title:** CouplingNet I: Source-Conditioned Directional Split

**Subtitle:** Learn how the forcing should be divided between axial directions

**Main claim:** CouplingNet supplies the multi-dimensional coupling missing from independent axial Green operators by learning a balanced directional source split.

**Must include:**

- The split balance:
  \[
  \phi+\psi=f.
  \]
- The role distinction: GreenNet provides axial inverses; CouplingNet provides source decomposition.
- The phrase "source-conditioned" to emphasize dependence on the input forcing.

**Optional / can omit:**

- Branch internals.
- Primary/transverse convection detail.
- Trunk terminology unless needed for Q&A.

**Suggested visual:** A full source field \(f\) splits into two fields \(\phi\) and \(\psi\), each feeding a directional Green reconstruction.

**Equations / notation:**

\[
\phi+\psi=f.
\]

**Slide text draft:**

- The source is multi-dimensional.
- Axial Green operators need directional source components.
- CouplingNet learns \((\phi,\psi)\) so the split remains physically balanced.

**Speaker emphasis:** Make clear that CouplingNet is not predicting the solution directly.  It predicts the directional source components that make Green reconstruction possible.

## Slide 8 - CouplingNet II: Branches and Local Context for Split Prediction

**Title:** CouplingNet II: Branches and Local Context for Split Prediction

**Subtitle:** Source, operator, axial position, and transverse boundary context shape the split

**Main claim:** CouplingNet predicts the split source from the forcing profile together with local operator and geometry/context information.

**Must include:**

- Source information: the forcing profile conditions the split.
- Fixed local operator context: diffusion, convection, and reaction affect how the source should be divided.
- Vector convection context: \(\mathbf b=(b_x,b_y)\) contributes primary and transverse transport information.
- Geometry/transverse context: the interval location and transverse boundary context influence the local split.
- Axial local trunk: pointwise variation along the primary interval.
- Pointwise transverse trunk: local transverse boundary context.

**Optional / can omit:**

- Implementation names.
- Tensor sizes.
- Config fields.
- Detailed branch equations.

**Suggested visual:** Conceptual context diagram with four inputs feeding the split predictor: source profile, operator coefficients, axial coordinate, transverse boundary context.

**Equations / notation:** Optional conceptual notation:

\[
(\text{source},\ \text{operator},\ \text{axial context},\ \text{transverse context})
\longrightarrow
(\phi,\psi).
\]

**Slide text draft:**

- Source branch: what forcing is being split?
- Fixed operator context: what local PDE is acting on the interval?
- Axial and transverse context: where is this point relative to local boundaries?
- Output: a source split compatible with Green reconstruction.

**Speaker emphasis:** Keep this slide conceptual.  The goal is to show why CouplingNet has enough context to predict a physically meaningful split, not to explain implementation surfaces.

## Slide 9 - CouplingNet III: Projection and Green Reconstruction

**Title:** CouplingNet III: Projection and Green Reconstruction

**Subtitle:** Enforce balance, reconstruct two solutions, and average them

**Main claim:** Projection enforces the PDE source balance, and the projected split fields are passed through Green reconstructions to form the final prediction.

**Must include:**

- Projection/balance message:
  \[
  \phi+\psi=f.
  \]
- Directional reconstructions:
  \[
  u_\phi=G_x[\phi],
  \qquad
  u_\psi=G_y[\psi].
  \]
- Final prediction:
  \[
  u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi).
  \]

**Optional / can omit:**

- Raw split fields before projection.
- Projection derivation.
- Pointwise transverse trunk detail; it belongs on Slide 8 if used.

**Suggested visual:** Pipeline diagram: source split \(\rightarrow\) projection \(\rightarrow\) two Green reconstructions \(\rightarrow\) averaged final solution.

**Equations / notation:**

\[
\phi+\psi=f,
\qquad
u_\phi=G_x[\phi],
\qquad
u_\psi=G_y[\psi],
\qquad
u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi).
\]

**Slide text draft:**

- Projection enforces \(\phi+\psi=f\).
- GreenNet maps each split source to a represented solution.
- Final solution is the average of two axial reconstructions.

**Speaker emphasis:** This slide connects the two models into a solver.  The key is that CouplingNet creates a balanced split and GreenNet performs structured inversion.

## Slide 10 - Energy-Norm Error Bound Proposition

**Title:** Energy-Norm Error Bound Proposition

**Subtitle:** Split consistency is tied to final solution error

**Main claim:** Under structural assumptions, the energy of the disagreement between the two represented solutions bounds the final solution energy error.

**Must include:**

- Split energy definition:
  \[
  \mathcal{E}_{\mathrm{split}}
  =
  \|u_\phi-u_\psi\|_a^2.
  \]
- Error-bound statement:
  \[
  \|u_{\mathrm{pred}}-u_*\|_a
  \le
  \frac{C_E}{2}
  \sqrt{\mathcal{E}_{\mathrm{split}}}.
  \]
- The condition: exact or controlled Green reconstruction assumptions.

**Optional / can omit:**

- Proof.
- Full admissibility discussion.
- Perturbation derivation; mention verbally only if needed.

**Suggested visual:** A triangle or bracket diagram showing \(u_\phi\), \(u_\psi\), their average \(u_{\mathrm{pred}}\), and the reference \(u_*\), with \(\mathcal{E}_{\mathrm{split}}\) controlling the average error.

**Equations / notation:**

\[
\mathcal{E}_{\mathrm{split}}
=
\|u_\phi-u_\psi\|_a^2,
\qquad
\|u_{\mathrm{pred}}-u_*\|_a
\le
\frac{C_E}{2}
\sqrt{\mathcal{E}_{\mathrm{split}}}.
\]

**Slide text draft:**

- Split consistency is not only a diagnostic.
- Under structural assumptions, it bounds the final energy error.
- Learned-Green errors add perturbation terms, so GreenNet accuracy still matters.

**Speaker emphasis:** State the claim clearly but conditionally.  Do not oversell it as an unconditional guarantee for every learned model.

## Slide 11 - Numerical Evidence I: GreenNet Reconstruction Quality

**Title:** Numerical Evidence I: GreenNet Reconstruction Quality

**Subtitle:** The normalized Green operator reconstructs interval solutions from sources

**Main claim:** GreenNet learns an axial source-to-solution map that reconstructs representative interval solutions accurately.

**Must include:**

- A representative interval source-to-solution reconstruction.
- The relation \(v_\theta(t)\approx v(t)\).
- One interval-level visual, such as target vs reconstructed solution or a Green kernel heatmap with reconstruction error.
- A concise statement that this evidence validates the axial inverse before CouplingNet is applied.

**Optional / can omit:**

- Per-interval metric tables.
- Exact dataset-generation detail.
- Large collections of interval examples.

**Suggested visual:** A two- or three-panel interval plot: source profile, target vs reconstructed solution, and reconstruction error.  A kernel heatmap can replace the source profile if it better supports the result.

**Equations / notation:**

\[
v_\theta(t)
=
\int_0^1G_\theta(t,\eta)f_{\mathrm{unit}}(\eta)\,d\eta,
\qquad
v_\theta(t)\approx v(t).
\]

**Slide text draft:**

- GreenNet evidence is interval-level.
- The learned kernel maps source profiles to solution profiles.
- This validates the axial inverse used later by CouplingNet.

**Speaker emphasis:** Keep this evidence separate from full solution plots.  The claim is specifically about GreenNet as a normalized one-dimensional inverse operator.

## Slide 12 - Numerical Evidence II: CouplingNet Solution Reconstruction

**Title:** Numerical Evidence II: CouplingNet Solution Reconstruction

**Subtitle:** The learned source split produces a consistent multi-dimensional solution

**Main claim:** CouplingNet's learned split, combined with Green reconstruction, matches the reference solution and keeps the two represented solutions consistent.

**Must include:**

- Reference solution panel.
- Predicted solution panel.
- Signed error panel.
- One split-consistency diagnostic, such as \(u_\phi-u_\psi\) or an energy-consistency trend.

**Optional / can omit:**

- Multiple coefficient families on the same slide.
- Full metric tables.
- Extra histograms or error distributions; move to backup if needed.

**Suggested visual:** Four-panel layout: reference, prediction, signed error, split mismatch.  If space is tight, use three panels and mention split consistency verbally.

**Equations / notation:** Optional:

\[
u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi),
\qquad
u_{\mathrm{pred}}-u_*,
\qquad
u_\phi-u_\psi.
\]

**Slide text draft:**

- Reference and prediction agree at the solution-field level.
- Signed error highlights where reconstruction is hardest.
- Split mismatch checks whether the two axial reconstructions agree.

**Speaker emphasis:** This is the final solver evidence.  Tie the split diagnostic back to the energy proposition from the previous slide.

## Slide 13 - Contributions and Takeaway

**Title:** Contributions and Takeaway

**Subtitle:** A structured reduced operator learner for elliptic PDEs

**Main claim:** The framework combines axial reduction, analytic-neural Green learning, source-conditioned coupling, and an energy-bound interpretation.

**Must include:**

- Four contribution bullets:
  - Axial reduction of multi-dimensional elliptic operator learning.
  - Analytic-neural GreenNet with singular/cancellation structure.
  - Source-conditioned CouplingNet with local/context-aware split prediction.
  - Energy-norm error-bound interpretation.
- One final sentence in MOR language.

**Optional / can omit:**

- Future work.
- Detailed limitations.
- Extra numerical claims not shown in the evidence slides.

**Suggested visual:** A compact four-part contribution graphic or a final pipeline summary.

**Equations / notation:** Optional repeat:

\[
\text{Axial Green operators} + \text{source coupling} \Rightarrow
\text{energy-consistent reconstruction}.
\]

**Slide text draft:**

- Reduce: multi-dimensional elliptic learning to axial Green operators.
- Structure: analytic Green kernel plus learned smooth correction.
- Couple: learn a balanced source split using source/operator/local context.
- Interpret: split energy connects to final energy error.

**Speaker emphasis:** End with the contribution, not with implementation.  The final message should be memorable for a MOR audience.

## Slide 14 - Backup / Q&A Prompt

**Title:** Backup / Q&A Prompt

**Subtitle:** Details available if the audience asks

**Main claim:** The backup material should support likely technical questions without being required for the main talk.

**Must include:**

- A list of backup candidates rather than a crowded main slide.
- Candidate 1: longer Dirac/Heaviside derivation sketch.
- Candidate 2: extra GreenNet reconstruction examples.
- Candidate 3: extra CouplingNet split-consistency or energy behavior figure.
- Candidate 4: Green reconstruction perturbation explanation.
- Candidate 5: connected-interval detail only if geometry questions arise.

**Optional / can omit:**

- Any backup item that is not ready by slide creation time.
- Full proof text.
- Implementation details.

**Suggested visual:** A clean menu of backup topics, or one selected backup derivation if the final deck needs a concrete backup slide.

**Equations / notation:** Optional, depending on selected backup.  A derivation backup may include:

\[
\partial_t^2G_0(t,\eta)=-\delta(t-\eta).
\]

**Slide text draft:**

- Analytic Green wrapping derivation.
- Extra GreenNet reconstruction evidence.
- Extra CouplingNet split-consistency evidence.
- Perturbed Green reconstruction interpretation.
- Geometry details only if asked.

**Speaker emphasis:** Treat this as Q&A support.  Do not spend main-talk time here unless a question makes it relevant.

## Core Phrase Bank

- **GreenNet:** GreenNet is not a black-box kernel approximator; it embeds analytic singular and cancellation structure and learns the remaining smooth correction.
- **CouplingNet:** CouplingNet learns the source split using source, operator, axial, and transverse context so axial Green reconstructions are consistent.
- **Energy:** The split energy is an error-bound quantity under structural assumptions, not only a diagnostic.
- **Geometry:** Geometry enters through non-unit axial intervals and pull-back scaling, not through a geometry-generation story.
- **Evidence:** GreenNet evidence shows interval reconstruction quality; CouplingNet evidence shows full solution reconstruction quality.
- **MOR framing:** The method reduces the operator representation while preserving Green-operator structure.
