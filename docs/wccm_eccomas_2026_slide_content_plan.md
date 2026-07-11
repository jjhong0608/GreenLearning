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

**Subtitle:** A two-dimensional elliptic solver built from coupled axial Green operators and a learned directional source split

**Main claim:** The method transforms a multi-dimensional elliptic problem into normalized axial Green subproblems plus a learned directional source split.

**Must include:**

- Talk title, conference, minisymposium, presenter, and affiliation.
- One-sentence thesis: axial Green reduction plus learned directional source split.
- The phrase "elliptic problems" or "elliptic PDEs" so the technical scope is immediate.

**Optional / can omit:**

- Detailed affiliation blocks if they crowd the title slide.
- Method diagram; save it for Slide 2 unless a compact visual is available.
- Long motivation text.

**Suggested visual:** Minimal title slide with a quiet schematic: a two-dimensional domain feeding into axial one-dimensional operators and then a reconstructed solution.

**Equations / notation:** None required.

**Slide text draft:**

- Transform a multi-dimensional elliptic problem into axial Green subproblems.
- Learn the directional source split that makes the axial reconstructions consistent.
- Combine analytic Green structure with neural correction.

**Speaker emphasis:** Open with the main thesis in one sentence.  The audience should know immediately that this is an axial Green-operator approach for a two-dimensional elliptic solver, not a geometry-generation talk and not a black-box neural solver.

**Animation plan:** No animation.  Keep the title slide static so the talk starts cleanly.  Do not use fragments, Auto-Animate, or decorative motion here; let Slide 2 carry the first animated reduction sequence.

## Slide 2 - Axial Reduction of Elliptic Operators

**Title:** Axial Reduction of Elliptic Operators

**Subtitle:** Replace a global source-to-solution map with structured one-dimensional Green reconstructions

**Main claim:** For a fixed heterogeneous elliptic operator, the reduction changes the source-to-solution representation from one global map to axial Green operators plus learned coupling.

**Must include:**

- A compact elliptic PDE statement:
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
- The direct source-to-solution map under one fixed operator context:
  \[
  \mathcal S_{a,\mathbf b,c}: f\mapsto u,
  \qquad
  u=\mathcal S_{a,\mathbf b,c}[f].
  \]
- A short, visually secondary caption only: "Operator fixed; source varies."
- The full-operator learning challenge: one global source-to-solution representation is high-dimensional and geometry-sensitive.
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
- The reduction framing: reduce operator complexity while preserving PDE structure.

**Optional / can omit:**

- Long related-work positioning.
- Detailed connected-interval construction.
- Implementation or data-layout detail.

**Suggested visual:** A compact contrast layout: the source-to-solution map \(\mathcal S_{a,\mathbf b,c}:f\mapsto u\) is defined once near the top; the left card then says "one global source-to-solution representation", the right card says "directional 1-D operators + learned coupling", and the physical \(L_x,L_y\) split sits below.  Avoid a separate bottom flow row if it causes vertical crowding.

**Equations / notation:**

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
\mathcal S_{a,\mathbf b,c}: f\mapsto u,
\qquad
u=\mathcal S_{a,\mathbf b,c}[f].
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

- Define the source-to-solution map once:
  \[
  \mathcal S_{a,\mathbf b,c}: f\mapsto u.
  \]
- Direct global representation is high-dimensional and geometry-sensitive.
- Operator fixed; source varies.
- The convection coefficient is a vector field, \(\mathbf b=(b_x,b_y)\).
- The boundary condition is homogeneous Dirichlet, matching the Green inverse used later.
- Axial reduction starts by viewing the fixed 2D operator through directional 1D operators.
- Reduced objects: one-dimensional axial Green operators.
- Missing ingredient: learned coupling of the multi-dimensional source.

**Speaker emphasis:** This slide must answer why axial reduction is useful and what is reduced, without over-framing the method as traditional MOR.  Show the physical \(L_x,L_y\) split here, but leave unit-interval normalization and scaling for Slide 4.

**Anchor sentence:** Include the early talk thesis as a compact callout after the \(L_x,L_y\) split:

> GreenNet supplies line-wise Green inverses; CouplingNet learns the source split that turns them into a 2D elliptic solver.

**Animation plan:** Use Auto-Animate with progressive reveal.  The goal is to guide attention through the reduction logic, not to decorate the slide.

- **Initial state:** Show the title, the compact homogeneous Dirichlet PDE, and the source-to-solution operator
  \[
  \mathcal S_{a,\mathbf b,c}: f\mapsto u.
  \]
- **Click 1:** Reveal the direct global representation challenge with the phrase
  "one global source-to-solution representation" and "high-dimensional and
  geometry-sensitive."
- **Click 2:** Reveal the physical directional split \(L_x,L_y\) and \(L_xu+L_yu=f\).
- **Click 3:** Emphasize the reduced-view card and the physical \(L_x,L_y\) split:
  \[
  \text{fixed 2D elliptic operator}
  \rightarrow
  \text{directional 1D operators }L_x,L_y
  \rightarrow
  \text{axial Green reconstructions}+\text{learned coupling}.
  \]
- **Optional pacing guard:** Do not add a separate bottom flow row unless the slide has enough vertical space; keep "learned coupling" highlighted in the reduced-view card rather than as a separate fragment.
- **PDF fallback:** The handout PDF should show the homogeneous Dirichlet PDE, direct-map challenge, \(L_x,L_y\) split, and reduction diagram together.  The presentation PDF may keep the animated states as separate fragments.

## Slide 3 - From a 2D Elliptic Problem to Coupled Axial Green Solves

**Title:** From a 2D Elliptic Problem to Coupled Axial Green Solves

**Subtitle:** Graphic abstract: slice, solve line-wise, learn the coupling, reconstruct the field

**Main claim:** The full computational story can be read visually as a sequence from the fixed two-dimensional PDE to axial slices, line-wise GreenNet inverses, learned source coupling, and a reconstructed two-dimensional field.

**Must include:**

- A figure-first layout with four large visual stages:
  1. A two-dimensional forcing \(f(x,y)\) drawn on a general domain \(\Omega\), not a rectangular patch and not a one-dimensional curve.  Do not label it as a "source field" and do not draw contour/axial-looking lines inside this first domain.
  2. Axial interval intersections inside the same general domain.
  3. GreenNet as a line-wise inverse with a directly rendered KaTeX math card centered on the kernel integration formula.  Use a generic line-source profile \(\rho\), not \(f\), so the visual is not confused with the original two-dimensional forcing.
  4. CouplingNet as directional coupling: \(f\) is split into \(\phi,\psi\), the split sources pass through \(G_x,G_y\), and the paths merge into a two-dimensional solution \(u\).  Do not use "average" on this graphic-abstract slide.
- Minimal text labels inside the visual stages, not paragraph explanations.
- A bottom takeaway:
  ```text
  Axial GreenNet provides line-wise inverses; CouplingNet learns the directional source split that couples them into a 2D solution.
  ```
- The stage order:
  \[
  \text{2D forcing}
  \rightarrow
  \text{axial slices}
  \rightarrow
  \text{line-wise GreenNet}
  \rightarrow
  \text{coupled 2D solution}.
  \]
- A visible reference to \(\phi+\psi=f\) in the final coupled-field stage.
- A final stage that is understandable as coupling, not just as a generic solution surface.

**Optional / can omit:**

- Full physical directional split equations; they are on Slide 2.
- Pull-back/scaling equations; they are on Slide 4.
- GreenNet analytic wrapping; it is on Slide 6.
- CouplingNet branch/trunk details; they are on Slide 10.
- Energy-bound interpretation; it is on Slide 12.
- Complex geometry construction or disconnected interval bookkeeping.

**Suggested visual:** A deck-native graphic abstract, not a screenshot.  Use four cards connected by arrows: a compact smooth non-rectangular domain with scalar forcing \(f(x,y)\), a larger axial interval card, a GreenNet KaTeX math card showing kernel integration with a generic line source profile \(\rho\), and a larger CouplingNet KaTeX math card showing split-to-\(G_x,G_y\)-to-\(u\).  The CouplingNet card should make \(\phi+\psi=f\) and "source split couples line-wise inverses" visually dominant.  The visual should be readable even if the speaker says only one or two sentences.  Use high-resolution PNG panels only as a fallback if the direct KaTeX cards become unreadable in Quarto/Reveal export.

**Equations / notation:**

\[
\text{2D forcing}
\rightarrow
\text{axial slices}
\rightarrow
\text{line-wise GreenNet}
\rightarrow
\text{coupled 2D solution}.
\]

\[
\phi+\psi=f.
\]

\[
v(t)=\int_0^1G_\theta(t,\eta)\rho(\eta)\,d\eta.
\]

\[
f\xrightarrow{\mathrm{CouplingNet}}(\phi,\psi),
\qquad
\phi\xrightarrow{G_x}u_\phi,
\qquad
\psi\xrightarrow{G_y}u_\psi
\Rightarrow u.
\]

**Slide text draft:**

- Start from a two-dimensional forcing \(f(x,y)\) on \(\Omega\).
- Slice the domain into axial intervals.
- Use GreenNet as the line-wise Green kernel inverse \(f\mapsto v\).
- Use CouplingNet to learn the directional source split and combine the axial reconstructions into \(u\).

**Speaker emphasis:** Treat this as a visual roadmap.  It should answer "how does the two-dimensional computation flow?" before the technical slides begin.  Do not explain the analytic Green formula, CouplingNet branches, or energy proof here.

**Animation plan:** Use light staged reveal and keep the slide image-centered.

- **Initial state:** Show only the first stage: 2D forcing \(f(x,y)\) on a general domain.
- **Click 1:** Reveal axial slices and the first arrow.
- **Click 2:** Reveal line-wise GreenNet and the second arrow.
- **Click 3:** Reveal the coupled 2D field and the third arrow.
- **Click 4:** Reveal the bottom takeaway.
- **Auto-Animate:** Optional.  If used, keep it limited to small movement or emphasis of the flow; the slide should not depend on animation to be understandable.
- **PDF fallback:** The handout PDF should show all four stages, arrows, and the bottom takeaway together.

## Slide 4 - Unit-Interval Pull-Back and Operator Scaling

**Title:** Unit-Interval Pull-Back and Operator Scaling

**Subtitle:** Physical length is absorbed into normalized operator coefficient profiles and source profile

**Main claim:** Non-unit physical intervals are mapped to \([0,1]\), but their length remains in the transformed one-dimensional operator.

**Must include:**

- The generic physical one-dimensional operator after the directional split:
  \[
  s\in[s_0,s_1],
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
- The pull-back map:
  \[
  s=s_0+Lt,\qquad v(t)=u(s_0+Lt).
  \]
- The visual definition of interval length:
  \[
  L=s_1-s_0.
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
- The normalized conservative-form unit equation:
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
- The sentence: "The interval is normalized, but the physical length is not discarded."
- The clarification that \(b_\parallel\) is the primary convection component:
  \(b_\parallel=b_x\) on an \(x\)-directed interval and
  \(b_\parallel=b_y\) on a \(y\)-directed interval.

**Optional / can omit:**

- Chain-rule derivation.
- Full \(L_x,L_y\) split repetition; that belongs on Slide 2.
- Geometry generation.
- Interval bookkeeping or disconnected interval discussion.

**Suggested visual:** A physical 1D operator on \([s_0,s_1]\) mapped to a unit-interval operator on \([0,1]\).  Label the physical interval endpoints as \(s_0,s_1\), mark the length as \(L=s_1-s_0\), and reveal the unit interval \(0\to1\) below it during the coordinate-normalization step.

**Equations / notation:**

\[
s\in[s_0,s_1],
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

\[
s=s_0+Lt,\qquad L=s_1-s_0,
\qquad v(t)=u(s_0+Lt),\qquad t\in[0,1].
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

\[
b_\parallel=b_x\ \text{on an \(x\)-directed interval},
\qquad
b_\parallel=b_y\ \text{on a \(y\)-directed interval}.
\]

**Slide text draft:**

- Slide 2 defines the physical directional split.
- Slide 4 normalizes each resulting physical 1D operator.
- Pull each physical axial interval back to \([0,1]\).
- The interval length reappears in the normalized operator coefficient profiles and source profile.
- The normalized equation has the same conservative one-dimensional structure on the unit interval.
- The vector convection field contributes through the primary axial component.
- This transformed operator is the one GreenNet learns.

**Speaker emphasis:** This is the technical bridge from non-square/non-unit geometry to a shared normalized Green learning problem.  Do not repeat the full \(L_x,L_y\) split or derive the chain rule; show the physical 1D operator, the pull-back map, and the scaling rule.

**Animation plan:** Use Auto-Animate for the interval morphing and fragments for equations/callouts.  The goal is to show that normalization changes the coordinate domain but preserves physical length through the transformed operator.

- **Initial state:** Show the physical 1D operator \(\mathcal L_{\mathrm{phys}}u=f_{\mathrm{phys}}\) with \(s\in[s_0,s_1]\) on a visual interval labeled \(s_0\) and \(s_1\).  Mark the physical interval length as \(L=s_1-s_0\).  Use the speaker line: "After the directional split, each axial interval carries a physical 1D operator."
- **Click 1:** Reveal the unit interval \(0\to1\) below the physical interval and show the pull-back map in the same fragment, so the equation appears only when the normalized coordinate is visible:
  \[
  s=s_0+Lt,\qquad v(t)=u(s_0+Lt),\qquad t\in[0,1].
  \]
  This is a fragment-based interval transformation rather than an additional slide, so the main slide count does not change.
- **Click 2:** Reveal and emphasize the callout:
  "The interval is normalized, but the physical length is not discarded."
  Use \(L\) as the visual accent that moves from the physical interval into the scaling callout.  Keep this callout visually separated from the interval card so it reads as an independent takeaway rather than a caption.
- **Click 3:** Reveal the full scaling rule as one block:
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
- **Click 4:** Reveal the normalized conservative-form unit equation and the \(b_\parallel\) clarification:
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
- **No initial \(b_\parallel\) note:** Do not show the \(b_\parallel\) clarification before the scaling/normalized-equation stage.
- **PDF fallback:** The handout PDF should show the physical operator with \(s\in[s_0,s_1]\), physical endpoint labels \(s_0,s_1\), \(L=s_1-s_0\), the unit interval \(0\to1\), pull-back map, "length is not discarded" callout, scaling rule, normalized equation, and \(b_\parallel\) note together.  The presentation PDF may keep the animated states as separate fragments.

## Slide 5 - GreenNet I: Normalized Axial Green Operator

**Title:** GreenNet I: Normalized Axial Green Operator

**Subtitle:** Learn one-dimensional Green kernels, not a global two-dimensional kernel

**Main claim:** GreenNet represents the normalized axial Green operator: a source-to-solution map induced by integrating the source against a local unit-interval Green kernel.

**Must include:**

- The conceptual split between operator information and Green kernel evaluation coordinates.
- The statement that the Green kernel is one-dimensional and local to the normalized unit interval.
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
- A short contrast strip before the operator-action visual:
  ```text
  Previous step: normalize the operator. This step: learn its Green inverse.
  ```

**Optional / can omit:**

- Activation function details.
- Architecture layer counts.
- Exact reference-kernel diagnostics.

**Suggested visual:** A branch/trunk-style schematic in conceptual terms: axial coefficient profiles define each local 1D operator; \((t,\eta)\) defines the kernel evaluation point.  The center object should show \(G_{\mathrm{unit}}(t,\eta)\) with the label "kernel integral operator."

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

- Axial coefficient profiles define each local 1D operator.
- Kernel coordinates \((t,\eta)\) define where the Green kernel is evaluated.
- A Green operator is the source-to-solution map induced by integrating the source against a Green kernel.
- Here the operator acts on the normalized source profile on the unit interval.

**Speaker emphasis:** Avoid making GreenNet sound like a generic regressor.  It is learning a local unit-interval Green kernel whose integral action maps a normalized source profile to a normalized solution profile.

**Animation plan:** Use light Auto-Animate continuity plus a 3-click progressive reveal.  The goal is to make the Green operator action visible as source \(\rightarrow\) kernel integral \(\rightarrow\) solution, without introducing the analytic kernel formula from Slide 6 or the supervision objective from Slide 7.

- **Initial state:** Keep the unit interval visual from Slide 4 and show only the contrast strip and the statement:
  "Green operator = source-to-solution map."
- **Click 1:** Reveal the normalized source profile \(f_{\mathrm{unit}}(\eta)\) on the \(\eta\)-axis.
- **Click 2:** Reveal the local unit-interval kernel panel \(G_{\mathrm{unit}}(t,\eta)\), preferably as a heatmap or matrix-like panel, together with the operator action:
  \[
  \mathcal G_{\mathrm{unit}}[f_{\mathrm{unit}}](t)
  =
  \int_0^1
  G_{\mathrm{unit}}(t,\eta)
  f_{\mathrm{unit}}(\eta)
  \,d\eta.
  \]
- **Click 3:** Reveal the output solution profile \(v(t)\) on the \(t\)-axis and emphasize:
  \[
  v=\mathcal G_{\mathrm{unit}}[f_{\mathrm{unit}}].
  \]
- **Static note:** Keep "one-dimensional and local unit-interval" visible throughout the slide.
- **Pacing guard:** Do not reveal \(G_\theta=E M R_\theta+B(J_0-\frac12E)+A G_0\) here; that is Slide 6.  Do not discuss training loss here; that is Slide 7.
- **PDF fallback:** The handout PDF should show the source profile, kernel panel, integral action, output solution profile, and local unit-interval note together.  The presentation PDF may keep the three reveal states as fragments.

## Slide 6 - GreenNet II: Analytic Green Structure and Learned Correction

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
  - \(B(t)(J_0-\frac12E)\): Heaviside cancellation.
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
  Keep \(A(t)\) and \(B(t)\) coefficient factor definitions out of the main slide and move them to Backup A.
  \[
  E(t,\eta)=t\eta(1-\eta),
  \qquad
  M(t)=1-t.
  \]
- The phrase "analytic-neural GreenNet" or "hybrid Green kernel."

**Optional / can omit:**

- Full distributional proof.
- Full piecewise formula for \(J_0\).
- Reaction-coefficient discussion.

**Suggested visual:** Three stacked or color-coded terms forming \(G_\theta\): Dirac delta structure, Heaviside cancellation, and learned smooth correction.  Use the bottom whitespace for a compact takeaway strip rather than leaving an empty lower region.

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
E(t,\eta)=t\eta(1-\eta),
\qquad
M(t)=1-t.
\]

**Slide text draft:**

- Do not learn the Green singularity from scratch.
- \(G_0\) supplies the Dirac-\(\delta\) jump through \(\partial_t^2G_0=-\delta\).
- \(J_0\) is an antiderivative of \(G_0\), so \(J_0-\frac12E\) produces the \(\partial_tG_0\) structure needed for Heaviside cancellation.
- \(E M R_\theta\): learns the remaining smooth correction.

**Speaker emphasis:** This slide carries the word "hybrid" in the talk title.  Start from the submitted-abstract motivation: variable-coefficient Green kernels are rarely available in closed form, and source-point singularity plus homogeneous boundary constraints are hard for a plain neural network to learn from scratch.  Say explicitly that \(G_0\) creates the singular jump and flux-jump behavior, \(J_0\) is its antiderivative, and \(S=J_0-\frac12E\) provides the cancellation structure.  Since \(E\) is linear in \(t\), \(\partial_t^2E=0\), so \(\partial_t^2S=\partial_tG_0\).  Do not derive the full distributional proof.

**Animation plan:** Use three same-heading Auto-Animate states rather than a single dense slide.  The goal is to make a dense formula readable by decomposing it into a structural thesis, built-in analytic mechanisms, and learned smooth correction.  This should feel like a guided technical argument, not a proof slide.

- **State 1:** Show the thesis only, with a compact motivation line: "Do not learn the Green singularity from scratch."
  \[
  G_\theta
  =
  \text{Dirac delta structure}
  +
  \text{Heaviside cancellation}
  +
  \text{learned smooth correction}.
  \]
  The visible role cards should be labeled "Dirac delta structure", "Heaviside cancellation", and "Learned smooth correction."  Keep \(\delta\) in the displayed derivative identity rather than in all-caps card text.  Add a bottom takeaway strip: "Built-in singular and cancellation structure reduces what the neural network must learn."
- **State 2:** Reveal the full formula with the analytic terms emphasized and the learned term muted:
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
  In the same state, reveal the minimal base-kernel facts:
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
  Use the speaker line: "\(A(t)G_0(t,\eta)\) supplies the Dirac-\(\delta\) jump and flux-jump structure."
  Also reveal the antiderivative/cancellation identities:
  \[
  \partial_tJ_0(t,\eta)=G_0(t,\eta),
  \qquad
  S(t,\eta)=J_0(t,\eta)-\frac12E(t,\eta),
  \qquad
  \partial_t^2S(t,\eta)=\partial_tG_0(t,\eta).
  \]
  Use the speaker line: "\(J_0\) is an antiderivative of \(G_0\), so \(S\) creates the structure needed for Heaviside cancellation."
  Add a bottom takeaway strip: "The analytic terms handle the Green-function singularity before learning."
- **State 3:** Emphasize the \(E(t,\eta)M(t)R_\theta(t,\eta)\) term, mute the analytic terms, and reveal only the envelope definitions:
  \[
  E(t,\eta)=t\eta(1-\eta),
  \qquad
  M(t)=1-t.
  \]
  Use the speaker line: "The neural network learns the remaining smooth correction, while the \(E(t,\eta)M(t)\) envelope keeps that correction boundary-compatible."
  Add a bottom takeaway strip: "The neural network learns only the smooth residual, with boundary-compatible envelope factors."
- **Backup handoff:** Move the \(A(t)\), \(B(t)\) coefficient factor definitions and the operator-application interpretation to Backup A.
- **Pacing guard:** Do not show the full piecewise formula for \(J_0\), \(A/B\) coefficient-factor details, or the weak/distributional proof.  Keep the slide focused on role decomposition and boundary-compatible learned correction.
- **PDF fallback:** The handout PDF should show the full formula, three role labels, \(G_0\) facts, \(J_0/S\) identities, and envelope definitions together.  The presentation PDF may keep the three Auto-Animate states as separate rendered states.

## Slide 7 - GreenNet III: Source-to-Solution Supervision

**Title:** GreenNet III: Source-to-Solution Supervision

**Subtitle:** The kernel is trained through reconstruction, not direct kernel labels

**Main claim:** GreenNet is supervised by whether the learned operator action reconstructs target solutions from sources, rather than by pointwise labels for an exact Green kernel.

**Must include:**

- The GP target-solution construction:
  \[
  w(t)\sim\mathcal{GP}(0,k_\ell),
  \]
  \[
  v(t)=w(t)-\bigl((1-t)w(0)+t\,w(1)\bigr),
  \qquad
  v(0)=v(1)=0.
  \]
- The source generated from the target solution:
  \[
  f_{\mathrm{unit}}=\mathcal L_{\mathrm{unit}}v.
  \]
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
- The expected reconstruction loss:
  \[
  \mathcal J_{\mathrm{Green}}
  \sim
  \mathbb E
  \left[
  \int_0^1
  |v_\theta(t)-v(t)|^2\,dt
  \right].
  \]
- The statement that source-to-solution reconstruction, not pointwise kernel labels, is the learning signal.
- The final takeaway:
  "Training checks whether the learned Green operator maps sources generated from target solutions back to those target solutions."

**Optional / can omit:**

- Gaussian Process covariance details beyond \(k_\ell\).
- Normalization details.
- Quadrature or sampling implementation detail.

**Suggested visual:** A four-stage visual: GP smooth profile \(\rightarrow\) boundary-compatible target \(v\) and source generated from that target solution \(f_{\mathrm{unit}}\) \(\rightarrow\) learned Green reconstruction \(v_\theta\) \(\rightarrow\) target/prediction overlay with reconstruction loss.

**Equations / notation:**

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
G_\theta(t,\eta)f_{\mathrm{unit}}(\eta)\,d\eta,
\qquad
v_\theta(t)\approx v(t).
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

**Slide text draft:**

- No pointwise Green-kernel labels.
- Sample a smooth profile with a Gaussian Process and remove the endpoint interpolant to enforce homogeneous boundary values.
- Generate the source by applying the unit operator to the target solution.
- Supervise the operator action: source profile \(\rightarrow\) solution profile.
- Train GreenNet with expected source-to-solution reconstruction loss.

**Speaker emphasis:** This slide prevents a common misunderstanding: the model learns a Green operator through its action on sources, not by requiring exact Green labels everywhere.  Say explicitly that \(w\) is the GP sample, \(v\) is the boundary-compatible target solution obtained by subtracting the endpoint interpolant, and \(f_{\mathrm{unit}}\) is generated from \(v\) by the unit operator.

**Animation plan:** Use light Auto-Animate continuity from Slide 5 plus progressive reveal.  The goal is to show where the supervised learning pair comes from and how the reconstruction loss trains the learned Green operator.

- **Initial state:** Show the contrast callout:
  "No pointwise Green-kernel labels."
- **Click 1:** Reveal the GP target construction:
  \[
  w(t)\sim\mathcal{GP}(0,k_\ell),
  \]
  \[
  v(t)=w(t)-\bigl((1-t)w(0)+t\,w(1)\bigr),
  \qquad
  v(0)=v(1)=0.
  \]
  Use the speaker line: "\(w\) is a smooth GP sample; \(v\) is the boundary-compatible target solution."
- **Click 2:** Reveal the source generated from the target solution:
  \[
  f_{\mathrm{unit}}=\mathcal L_{\mathrm{unit}}v.
  \]
  Use the speaker line: "The source is generated by applying the normalized unit operator to the target solution."
- **Click 3:** Reveal the learned Green reconstruction:
  \[
  v_\theta(t)
  =
  \int_0^1
  G_\theta(t,\eta)
  f_{\mathrm{unit}}(\eta)\,d\eta.
  \]
- **Click 4:** Reveal the target/prediction overlay, the expected reconstruction loss, and the final takeaway:
  \[
  \mathcal J_{\mathrm{Green}}
  \sim
  \mathbb E
  \left[
  \int_0^1
  |v_\theta(t)-v(t)|^2\,dt
  \right].
  \]
  Final takeaway: "Training checks whether the learned Green operator maps sources generated from target solutions back to those target solutions."
- **Static note:** Keep "source-to-solution supervision, not kernel-label supervision" visible throughout the slide.
- **Pacing guard:** Do not expand the GP covariance, normalization, or quadrature details.  The slide should explain the learning signal, not become a dataset-generation derivation.
- **PDF fallback:** The handout PDF should show the GP sample, endpoint-interpolant removal, source generated from the target solution, learned reconstruction, target/prediction overlay, final takeaway, and expected loss together.  The presentation PDF may keep the four reveal states as fragments.

## Slide 8 - Numerical Evidence I: GreenNet Kernel Approximation

**Title:** Numerical Evidence I: GreenNet Kernel Approximation

**Subtitle:** Capturing the singular Green structure on an axial interval

**Main claim:** The learned GreenNet kernel captures the singular Green structure for a reaction-free convection-diffusion line problem, and the signed error is not concentrated near the singular diagonal.

**Status note:** This evidence slide appears immediately after GreenNet supervision and before CouplingNet is introduced.  The slide validates the line-wise Green kernel first; CouplingNet then answers how these line-wise inverses are coupled into a 2D solver.

**Must include:**

- A compact reaction-free convection-diffusion problem setup strip:
  \[
  \Omega=\{x^2+y^2<0.5^2\},
  \qquad
  -\nabla\cdot(a\nabla u)+\mathbf b\cdot\nabla u=f,
  \qquad
  c=0,
  \]
  \[
  a(x,y)=1+\frac12\sin(2\pi x)\sin(2\pi y),
  \qquad
  \mathbf b(x,y)=
  \left(
  \frac12\sin(\pi x)\cos(\pi y),
  -\frac12\cos(\pi x)\sin(\pi y)
  \right).
  \]
- A compact axial-line context tag:
  \[
  y\text{-directed interval at }x=-0.25,
  \qquad
  L=0.866.
  \]
  This identifies the physical line without exposing the artifact interval id.
- Reference Green kernel heatmap \(G_{\mathrm{ref}}(t,\eta)\).
- Learned Green kernel heatmap \(G_\theta(t,\eta)\) with the same color scale as the reference heatmap.
- Signed error heatmap \(G_\theta-G_{\mathrm{ref}}\) with a zero-centered diverging scale.
- Fixed-\(\eta\) slice at \(\eta=0.75\) showing reference and learned curves.
- A compact diagnostic callout: kernel relative error, slice relative error, normalized-coordinate diagonal-band definition, diagonal error mass / area, and diagonal mean / off-band mean.
- The final takeaway sentence:
  "The learned kernel captures the singular Green structure, and the signed error is not concentrated near the singular diagonal."

**Optional / can omit:**

- The visible `Kernel-level evidence` label; the slide title, placement, and notes already identify this as GreenNet validation.
- Per-interval metric tables.
- Exact dataset-generation detail.
- Training and artifact context such as sampler mode, epoch count, batch size, device, or checkpoint path.
- Large collections of interval examples.

**Suggested visual:** Three reveal states using separated assets from `checkpoints/Disk_CD/green/artifacts`: first show the problem strip, axial-line context tag, reference kernel, learned kernel, signed error, and diagnostics; then keep the line context, signed error, and diagnostics visible while enlarging the fixed-\(\eta=0.75\) slice; then reveal the takeaway.

**Equations / notation:**

\[
G_\theta(t,\eta)\approx G_{\mathrm{ref}}(t,\eta),
\qquad
G_\theta(t,\eta)-G_{\mathrm{ref}}(t,\eta).
\]

\[
\eta=0.75,
\qquad
|t-\eta|\le 5/128=0.0391.
\]

**Slide text draft:**

- Reaction-free convection-diffusion disk problem.
- Axial line: \(y\)-directed interval at \(x=-0.25\), \(L=0.866\).
- Reference kernel, learned kernel, and signed error.
- Fixed-\(\eta=0.75\) slice.
- The learned kernel captures the singular Green structure.
- The signed error is not concentrated near the singular diagonal.

**Speaker emphasis:** Keep this evidence separate from full solution plots.  The claim is specifically about the axial Green kernel learned by GreenNet, not the final CouplingNet solution.  This slide should end the GreenNet block: after validating the line-wise inverse, the talk moves to CouplingNet and asks how the source should be split across those inverses.

**Animation plan:** Use three Reveal.js states with Auto-Animate continuity.

- **State 1:** Show reference kernel, learned kernel, signed error heatmap, and diagnostics.
- **State 2:** Keep signed error and diagnostics visible while revealing a much larger fixed-\(\eta=0.75\) slice.
- **State 3:** Keep the state-2 layout and reveal the takeaway sentence.
- **Pacing guard:** Do not show the legacy composite panel.  Do not reveal all elements as separate small fragments; the slide should use the three logical states above.
- **PDF fallback:** The presentation PDF may show the three states as separate rendered pages.  The handout version should preserve the heatmaps, enlarged fixed-\(\eta\) slice, diagnostic card, and takeaway.

## Slide 9 - CouplingNet I: Directional Source Split

**Title:** CouplingNet I: Directional Source Split

**Subtitle:** Learn how the forcing is divided between axial Green reconstructions

**Main claim:** CouplingNet learns the directional source split, assigning the forcing to horizontal and vertical line-wise flux-divergence/source components that feed the axial Green reconstructions.

**Must include:**

- The split map:
  \[
  f\longmapsto(\phi,\psi).
  \]
- The directional source definitions:
  \[
  \phi=L_xu,
  \qquad
  \psi=L_yu.
  \]
- The compact physical split operators:
  \[
  L_xu
  =
  -\partial_x(a\partial_xu)
  +
  b_x\partial_xu
  +
  \frac12cu,
  \]
  \[
  L_yu
  =
  -\partial_y(a\partial_yu)
  +
  b_y\partial_yu
  +
  \frac12cu.
  \]
- The split balance:
  \[
  \phi+\psi=f.
  \]
- The final takeaway: CouplingNet learns line-wise flux-divergence/source components that couple the axial Green reconstructions.

**Optional / can omit:**

- Branch internals.
- Primary/transverse convection detail.
- Trunk terminology unless needed for Q&A.
- The word "source-conditioned"; keep that framing for the branch/context explanation on Slide 10.

**Suggested visual:** Ghosted \(G_x[\cdot]\) and \(G_y[\cdot]\) boxes first ask what source each inverse should receive, then the full source \(f\) splits into \(\phi\) and \(\psi\) with compact operator definitions and the final balance \(\phi+\psi=f\).

**Equations / notation:**

\[
f\longmapsto(\phi,\psi),
\qquad
\phi=L_xu,
\qquad
\psi=L_yu,
\]
\[
L_xu=-\partial_x(a\partial_xu)+b_x\partial_xu+\frac12cu,
\qquad
L_yu=-\partial_y(a\partial_yu)+b_y\partial_yu+\frac12cu,
\]
\[
\phi+\psi=f.
\]

**Slide text draft:**

- Axial Green operators need directional source components.
- \(\phi\) is the \(x\)-directed source and \(\psi\) is the \(y\)-directed source.
- CouplingNet learns line-wise flux-divergence/source components that couple the axial Green reconstructions.

**Speaker emphasis:** Make clear that CouplingNet is not predicting the solution directly.  It predicts horizontal and vertical line-wise flux-divergence, or source, components that make Green reconstruction possible.  The exact split would be defined by the reference solution, but the slide can write \(u\) to keep the notation light.  Contrast the learning signals: GreenNet is supervised by one-dimensional source-to-solution reconstruction pairs, while CouplingNet is trained without reference-solution or split labels through balance, Green reconstruction, and split-energy consistency.

**Animation plan:** Use Auto-Animate plus a simple 4-click progressive reveal.  This slide is the transition from GreenNet to CouplingNet, so the animation should explain why a second network is needed without introducing branch internals or reconstruction details.

- **Initial state:** Show ghosted directional Green inverses,
  \[
  G_x[\cdot],
  \qquad
  G_y[\cdot],
  \]
  with the question:
  "What source should each axial inverse receive?"
- **Click 1:** Reveal the split map:
  \[
  f\longrightarrow(\phi,\psi).
  \]
  Use the speaker line: "The forcing is multi-dimensional, but the Green operators are directional."
- **Click 2:** Reveal the directional source definitions:
  \[
  \phi=L_xu,
  \qquad
  \psi=L_yu,
  \]
  together with compact \(L_x\) and \(L_y\) formulas.  Use a warm/cool color pairing for \(\phi\) and \(\psi\) so the two directions remain visually distinct in later slides.
- **Click 3:** Reveal the balance constraint prominently:
  \[
  \phi+\psi=f.
  \]
  Use the speaker line: "The split is learned, but it must remain physically balanced."
- **Click 4:** Reveal the final takeaway: "CouplingNet learns line-wise flux-divergence/source components that couple the axial Green reconstructions."
- **Pacing guard:** Do not introduce source/coefficient/geometry branches here; that belongs on Slide 10.  Do not introduce projection, \(u_\phi\), \(u_\psi\), or \(u_{\mathrm{pred}}\); those belong on Slide 11.
- **PDF fallback:** The handout PDF should show the ghosted \(G_x,G_y\) context, the full source \(f\), the split \((\phi,\psi)\), the directional source definitions, the balance equation, and the final takeaway together.  The presentation PDF may keep the four reveal states as fragments.

## Slide 10 - CouplingNet II: Branches and Local Context for Split Prediction

**Title:** CouplingNet II: Branches and Local Context for Split Prediction

**Subtitle:** Branch nets encode profiles; trunk nets encode pointwise coordinates

**Main claim:** CouplingNet predicts the split source by combining branch-net features for profiles and line geometry with trunk-net features for pointwise axial and transverse coordinates.

**Must include:**

- Branch nets:
  - Source profile: the sample-varying forcing profile.
  - Coefficient profiles: diffusion, primary/transverse convection, and reaction along axial lines.
  - Line-geometry structure: interval length, endpoints, and transverse placement.
- Trunk nets:
  - Axial local trunk:
    \[
    t_{\parallel}
    \]
    is the pointwise coordinate along the primary axial interval.
  - Pointwise transverse trunk:
    \[
    t_{\perp}
    \]
    is the pointwise coordinate on the transverse interval through the same physical point.
- Final message as an underbrace takeaway:
  \[
  \underbrace{\text{profiles and line geometry}}_{\text{branch nets}}
  \quad+\quad
  \underbrace{\text{pointwise coordinates}}_{\text{trunk nets}}
  \quad\Rightarrow\quad
  \text{directional source split}.
  \]

**Optional / can omit:**

- Implementation names.
- Tensor sizes.
- Config fields.
- Detailed branch equations.
- The phrase "fixed operator context" on the visible slide.

**Suggested visual:** Two grouped columns feeding a central split predictor.  The left group is "Branch nets" with source profile, coefficient profiles, and line-geometry structure.  The right group is "Trunk nets" with \(t_{\parallel}\) and \(t_{\perp}\).  A bottom takeaway strip uses the remaining space.

**Equations / notation:** Optional conceptual notation:

\[
(\text{branch features},\ \text{trunk features})
\longrightarrow
(\phi,\psi).
\]
\[
t_{\parallel}: \text{primary axial coordinate},
\qquad
t_{\perp}: \text{transverse coordinate through the same point}.
\]
\[
\underbrace{\text{profiles and line geometry}}_{\text{branch nets}}
\quad+\quad
\underbrace{\text{pointwise coordinates}}_{\text{trunk nets}}
\quad\Rightarrow\quad
\text{directional source split}.
\]

**Slide text draft:**

- Branch nets: source profile, coefficient profiles, and line-geometry structure.
- Trunk nets: axial local coordinate \(t_{\parallel}\) and pointwise transverse coordinate \(t_{\perp}\).
- Output: a directional source split compatible with Green reconstruction.

**Speaker emphasis:** Keep this slide conceptual.  The goal is to show how CouplingNet separates profile-level information from pointwise coordinate information.  Avoid saying "fixed operator context" on the visible slide.  Use the submitted-abstract framing: the coefficient field is fixed for the problem, but its axial profiles vary from line to line; the same operator-learning model is reused across lines.

**Animation plan:** Use Auto-Animate with a central split-predictor node and two grouped reveal stages.  The goal is to make the message explicit: branch nets encode profiles and line geometry, while trunk nets encode pointwise coordinates.

- **Initial state:** Show a central predictor node with
  \[
  ? \longrightarrow (\phi,\psi)
  \]
  and the prompt:
  "What information is needed to predict the split?"
- **Click 1:** Reveal the Branch nets group:
  source profile, coefficient profiles, and line-geometry structure.
  Use the speaker line:
  "These branches encode function profiles and the line-level geometry of the local problem."
- **Click 2:** Reveal the Trunk nets group:
  \(t_{\parallel}\) and \(t_{\perp}\).
  Explain that \(t_{\parallel}\) gives pointwise variation along the primary axial interval, while \(t_{\perp}\) carries transverse boundary proximity at the same physical point.
- **Click 3:** Reveal the final conceptual map and takeaway:
  \[
  (\text{branch features},\ \text{trunk features})
  \longrightarrow
  (\phi,\psi).
  \]
- **Static note:** Keep the branch-net/trunk-net distinction visible in the final state with the underbrace takeaway:
  \[
  \underbrace{\text{profiles and line geometry}}_{\text{branch nets}}
  +
  \underbrace{\text{pointwise coordinates}}_{\text{trunk nets}}
  \Rightarrow
  \text{directional source split}.
  \]
- **Pacing guard:** Do not introduce projection, \(u_\phi\), \(u_\psi\), or \(u_{\mathrm{pred}}\); those belong on Slide 11.  Do not add implementation names, tensor sizes, config fields, or detailed branch equations.
- **PDF fallback:** The handout PDF should show both branch-net and trunk-net groups plus the final conceptual map.  The presentation PDF may keep the two groups and the takeaway as fragments.

## Slide 11 - CouplingNet III: Projection and Green Reconstruction

**Title:** CouplingNet III: Projection and Green Reconstruction

**Subtitle:** Project in physical split variables, then reconstruct and average

**Main claim:** A raw CouplingNet split is physically projected into a balanced split, then two axial Green reconstructions are averaged into the final prediction.

**Must include:**

- Raw split produced by CouplingNet:
  \[
  (\phi_{\mathrm{raw}},\psi_{\mathrm{raw}})
  \]
  with the visible note that it is not guaranteed balanced.
- Projection-in-physical-variables balance message:
  \[
  r=f-(\phi_{\mathrm{raw}}+\psi_{\mathrm{raw}}),
  \qquad
  \phi=\phi_{\mathrm{raw}}+\frac12r,
  \qquad
  \psi=\psi_{\mathrm{raw}}+\frac12r.
  \]
  \[
  \phi+\psi=f.
  \]
- The note that projection is applied in physical split variables, not as a unit-interval normalization step.
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
- A compact bottom pipeline:
  \[
  (\phi_{\mathrm{raw}},\psi_{\mathrm{raw}})
  \xrightarrow{\text{projection}}
  (\phi,\psi),\ \phi+\psi=f
  \xrightarrow{G_x,G_y}
  (u_\phi,u_\psi)
  \xrightarrow{\text{average}}
  u_{\mathrm{pred}}.
  \]

**Optional / can omit:**

- Projection derivation.
- Unit-to-physical and physical-to-unit conversion details.
- Pointwise transverse trunk detail; it belongs on Slide 10 if used.

**Suggested visual:** Two-stage solver pipeline.  Stage 1 is balance projection in physical variables: raw split \(\rightarrow\) projection step \(\rightarrow\) balanced split.  Stage 2 is Green reconstruction: \(x\)-represented solution, \(y\)-represented solution, and final average.

**Equations / notation:**

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
u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi).
\]

**Slide text draft:**

- CouplingNet first proposes a raw directional split.
- Projection splits the residual symmetrically in physical variables so that \(\phi+\psi=f\).
- GreenNet reconstructs two represented solutions from the balanced split.
- The solver prediction is the average of the two axial reconstructions.

**Speaker emphasis:** This slide connects the two models into a solver.  The key is that the raw split is not trusted as-is: it is projected in physical variables before Green reconstruction.  Do not expand the unit-to-physical conversion details; the slide only needs the principle that balance is imposed before the two represented solutions are averaged.

**Animation plan:** Use a two-stage pipeline reveal.  Auto-Animate is useful for keeping the title and stage layout stable while fragments reveal the calculation.  Do not show standalone arrows before their boxes; each arrow should reveal with the destination box it points to.

- **Initial state:** Start from the raw split produced by the previous CouplingNet context slide:
  \[
  (\phi_{\mathrm{raw}},\psi_{\mathrm{raw}}).
  \]
- **Click 1:** Reveal the raw split card and the note "not guaranteed balanced".
- **Click 2:** Reveal the first arrow with the projection block and the residual:
  \[
  r=f-(\phi_{\mathrm{raw}}+\psi_{\mathrm{raw}}).
  \]
  Keep "physical split variables" explicit in the subtitle or stage label.
- **Click 3:** Reveal the second arrow with the projection formula and balanced split:
  \[
  \phi=\phi_{\mathrm{raw}}+\frac12r,
  \qquad
  \psi=\psi_{\mathrm{raw}}+\frac12r,
  \qquad
  \phi+\psi=f.
  \]
- **Click 4:** Reveal the Green reconstruction stage:
  \[
  u_\phi=G_x[\phi],
  \qquad
  u_\psi=G_y[\psi].
  \]
- **Click 5:** Reveal the final average and compact bottom pipeline:
  \[
  u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi).
  \]
- **Static note:** Keep "physical balance first, axial Green reconstruction second" visible in the final state.
- **Pacing guard:** Do not discuss the energy bound here; that belongs on Slide 12.  Do not re-open branch/context details from Slide 10.
- **PDF fallback:** The handout PDF should show the full pipeline, the projection note, the balance equation, the two Green reconstructions, and the averaged prediction together.

## Slide 12 - Energy-Norm Error Bound Proposition

**Title:** Energy-Norm Error Bound Proposition

**Subtitle:** Small split energy controls the final energy error

**Main claim:** Under structural assumptions, the split-energy loss is not only a diagnostic; it bounds the final solution energy error.

**Must include:**

- Energy norm definition:
  \[
  \|v\|_a^2
  =
  \int_\Omega a(x)|\nabla v(x)|^2\,dx,
  \qquad
  a(x)>0.
  \]
  State on the slide that \(a(x)\) is the diffusion coefficient.
- Split-energy loss:
  \[
  \mathcal{E}_{\mathrm{split}}
  =
  \|u_\phi-u_\psi\|_a^2.
  \]
- Reference solution definition:
  \[
  \mathcal L u_*=f,
  \qquad
  u_*|_{\partial\Omega}=0.
  \]
- Error-bound statement:
  \[
  \|u_{\mathrm{pred}}-u_*\|_a
  \le
  \frac{C_E}{2}
  \sqrt{\mathcal{E}_{\mathrm{split}}}.
  \]
- \(C_E\) definition as a short visible label: stability constant of the fixed elliptic operator.
- The condition, revealed last: exact or controlled Green reconstruction and \(H_0^1(\Omega)\)-admissible represented solutions.

**Optional / can omit:**

- Proof.
- Full admissibility discussion.
- Perturbation derivation; mention verbally only if needed.

**Suggested visual:** A triangle or bracket diagram showing \(u_\phi\), \(u_\psi\), their average \(u_{\mathrm{pred}}\), and the reference \(u_*\), with \(\mathcal{E}_{\mathrm{split}}\) controlling the average error.

**Equations / notation:**

\[
\|v\|_a^2
=
\int_\Omega a(x)|\nabla v(x)|^2\,dx,
\qquad
a(x)>0.
\]

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
\frac{C_E}{2}
\sqrt{\mathcal{E}_{\mathrm{split}}}.
\]

\[
C_E:\ \text{stability constant of the fixed elliptic operator}.
\]

**Slide text draft:**

- The two Green reconstructions produce two represented solutions.
- The energy norm is diffusion-weighted; \(a(x)\) is the diffusion coefficient.
- Their energy disagreement defines the split-energy loss.
- \(u_*\) is the reference solution of the full elliptic problem.
- Under structural assumptions, this loss bounds the final prediction error.
- Learned-Green errors appear as perturbation terms, so GreenNet accuracy remains important.

**Speaker emphasis:** Define \(a(x)\), \(u_*\), and \(C_E\) before showing the bound.  State the claim clearly but conditionally.  Do not oversell it as an unconditional guarantee for every learned model.

**Animation plan:** Use a staged proposition reveal with a simple geometry-of-the-argument diagram.  Avoid proof variables and keep the slide focused on the error-bound claim.

- **Initial state:** Carry over \(u_\phi\), \(u_\psi\), and \(u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi)\) from Slide 11 as a triangle or bracket diagram.
- **Click 1:** Reveal the energy norm:
  \[
  \|v\|_a^2=\int_\Omega a(x)|\nabla v(x)|^2\,dx,
  \qquad
  a(x)>0.
  \]
  Add the small label "\(a(x)\): diffusion coefficient."
- **Click 2:** Highlight the bracket between \(u_\phi\) and \(u_\psi\), then reveal
  \[
  \mathcal{E}_{\mathrm{split}}=\|u_\phi-u_\psi\|_a^2.
  \]
- **Click 3:** Add the reference solution \(u_*\) and an error arrow from \(u_{\mathrm{pred}}\) to \(u_*\).
  Define
  \[
  \mathcal L u_*=f,
  \qquad
  u_*|_{\partial\Omega}=0.
  \]
- **Click 4:** Reveal the proposition:
  \[
  \|u_{\mathrm{pred}}-u_*\|_a
  \le
  \frac{C_E}{2}
  \sqrt{\mathcal{E}_{\mathrm{split}}}.
  \]
- **Click 5:** Reveal the assumption footer:
  "Conditional on exact/controlled Green reconstruction and \(H_0^1(\Omega)\)-admissible represented solutions."
- **Static note:** Keep the \(C_E\) label visible in the final state: "stability constant of the fixed elliptic operator."
- **Pacing guard:** Do not include the proof, \(q_c,q_x,q_y\), or the full perturbation bound on the main slide.  Mention verbally that learned Green reconstruction errors add perturbation terms if time permits.
- **PDF fallback:** The handout PDF should show the diffusion-weighted energy norm, split-energy definition, \(u_*\) definition, \(C_E\) label, final error bound, assumption footer, and the \(u_\phi,u_\psi,u_{\mathrm{pred}},u_*\) diagram together.

## Slide 13 - Numerical Evidence II: CouplingNet Solution Reconstruction

**Title:** Numerical Evidence II: CouplingNet Solution Reconstruction

**Subtitle:** Quantile-selected samples show source, reference, prediction, and signed error

**Main claim:** Solver-level evidence: quantile-selected samples show that the learned directional source split supports 2D solution reconstruction across the observed relative-error range, not only on a single favorable case.

**Status note:** Use the separated field assets generated by `plot_wccm_coupling_evidence_panel.py` from `checkpoints/Disk_CDR/coupling/artifacts` as the current canonical CouplingNet evidence visuals.  The deck assembles the assets in a slide-native table so field panels remain free of titles, axes, and colorbars.

**Must include:**

- A compact two-line problem setup strip:
  \[
  \begin{aligned}
  \Omega=\{x^2+y^2<0.5^2\},\qquad
  &-\nabla\cdot(a\nabla u)+\mathbf b\cdot\nabla u+c u=f,\\
  a=1+\frac12\sin(2\pi x)\sin(2\pi y),\qquad
  &\mathbf b=
  \left(
  \frac12\sin(\pi x)\cos(\pi y),
  -\frac12\cos(\pi x)\sin(\pi y)
  \right),\quad
  c=\frac12\left(1+\frac12\cos(2\pi x)\cos(2\pi y)\right).
  \end{aligned}
  \]
- Five columns selected by relative solution error: `min`, `q25`, `q50`, `q75`, and `max`.
- Four rows of field panels: `Source`, `Reference`, `Prediction`, and `Signed error`.
- Field panels with no internal title, axis, colorbar, sample id, or artifact-specific text.
- A separate slide-native metric card showing relative solution error and split-energy loss for the five selected columns.
- A short scale note: `Scales: per-sample source/solution; shared signed-error scale.`
- No separate `Solver-level evidence` pill; the slide title, placement after the solver pipeline, and problem/evidence layout identify it as coupled solver evidence.

**Optional / can omit:**

- Multiple coefficient families on the same slide.
- Per-panel colorbars and axis labels.
- Sample ids and artifact-specific file stems.
- Flux-divergence target/error panels; move these to backup if needed.
- Extra histograms, full metric distributions, or split-mismatch field panels.

**Preferred layout:** A 5-by-4 field matrix assembled in Quarto/Reveal.js from separated PNG assets, with row labels on the left, quantile labels on top, and a separate right-side metric card.

**Fallback layout:** If the 5-by-4 matrix becomes too dense after final figure changes, keep the `q50` column as a larger four-panel main slide and move the full quantile matrix to backup.

**Selection criteria:** Use the selected samples exported under the relative-solution-error quantile policy.  This avoids a best-case-only visual and shows how source, reference, prediction, and signed error behave across the selected error distribution.

**Equations / notation:** Optional speaker notation:

\[
u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi),
\qquad
u_{\mathrm{pred}}-u_*,
\qquad
u_\phi-u_\psi.
\]

**Slide text draft:**

- Columns are selected by relative solution error quantiles.
- Rows compare source, reference solution, prediction, and signed error.
- Relative solution error ranges from \(2.33\%\) to \(6.96\%\) across the selected columns.
- Scales: per-sample source/solution; shared signed-error scale.
- Quantile-selected samples show that the learned directional source split supports 2D solution reconstruction across the observed relative-error range, not only on a single favorable case.

**Speaker emphasis:** This is the full solver evidence for the convection-diffusion-reaction disk run.  Do not explain every panel; use the rows and columns to make the visual contract clear, then point to the signed-error row and metric card.  Mention \(u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi)\) verbally if the audience needs the link back to projection and Green reconstruction.  If needed, contrast with GreenNet Evidence: GreenNet uses a reaction-free convection-diffusion kernel validation case because reference axial Green kernels are available there, while this slide evaluates the coupled CDR solver.

**Animation plan:** Keep the source row and the right-side metric card visible when the slide appears.  Reveal the `Reference`, `Prediction`, and `Signed error` rows one row at a time, then reveal the takeaway sentence.  Do not reveal individual panels independently; row-wise reveal keeps the labels and five quantile examples synchronized.  Do not use heavy Auto-Animate; the numerical figure should remain the focus.

**PDF fallback:** The handout PDF should show the 5-by-4 matrix, the metric card, and the concise interpretation together.  Because panel titles, axes, and colorbars are removed from the images, row/column labels must remain visible in the slide-native layout.

## Slide 14 - Takeaway: Coupled Axial Green Solvers

**Title:** Takeaway: Coupled Axial Green Solvers

**Subtitle:** GreenNet learns line-wise Green kernels; CouplingNet learns the directional source split.

**Main claim:** A 2D elliptic problem is solved through axial Green inversions and a learned, balance-preserving source decomposition.

**Must include:**

- Final thesis sentence:
  "A 2D elliptic problem is solved through axial Green inversions and a learned, balance-preserving source decomposition."
- Four contribution bullets:
  - Axial Green kernels.
  - Analytic structure.
  - Source split as unsupervised directional split learning.
  - Energy bound for unsupervised split consistency.
- A closing sentence:
  "GreenNet supplies line-wise Green inverses; CouplingNet learns the source split that turns them into a 2D elliptic solver."

**Optional / can omit:**

- Future work.
- Detailed limitations.
- Extra numerical claims not shown in the evidence slides.

**Suggested visual:** A compact four-part contribution graphic or a final pipeline summary with four tags: axial Green kernels, analytic structure, source split, and energy bound.

**Equations / notation:** Optional repeat:

\[
\text{Axial Green inversions}
+\text{balance-preserving source split}
\Rightarrow
\text{2D elliptic solution}.
\]

**Slide text draft:**

- A 2D elliptic problem is solved through axial Green inversions and a learned, balance-preserving source decomposition.
- Axial Green kernels: normalize axial intervals and learn line-wise inverse kernels.
- Analytic structure: encode singular Green behavior before neural correction.
- Source split: learn phi/psi without reference-solution or split labels.
- Energy bound: connect unsupervised split consistency to final solution error under assumptions.
- GreenNet supplies line-wise Green inverses; CouplingNet learns the source split that turns them into a 2D elliptic solver.

**Speaker emphasis:** End with the solver decomposition, not with implementation.  The final message should be memorable for a MOR audience: GreenNet supplies line-wise Green inverses, and CouplingNet learns the source split that turns them into a 2D elliptic solver.  State that the reference solution is not used to train CouplingNet; reference solutions are used only for evaluation.  CouplingNet is trained through balance, Green reconstruction, and split consistency, with split-energy consistency as the optimization loss.

**Animation plan:** Use a light progressive reveal.  This slide should feel stable and conclusive rather than technical.

- **Initial state:** Show only the final thesis sentence as a left-aligned technical callout:
  "A 2D elliptic problem is solved through axial Green inversions and a learned, balance-preserving source decomposition."
- **Click 1:** Reveal "Axial Green kernels".
- **Click 2:** Reveal "Analytic structure".
- **Click 3:** Reveal "Source split".
- **Click 4:** Reveal "Energy bound".
- **Final state:** Reveal a wide bottom closing banner:
  "GreenNet supplies line-wise Green inverses; CouplingNet learns the source split that turns them into a 2D elliptic solver."
- **Pacing guard:** Do not add new equations, numerical metrics, proof details, or future-work lists on this slide.
- **Layout guard:** Use taller four-block cards and a wide closing banner so the slide does not leave unused lower whitespace.
- **PDF fallback:** The handout PDF should show the thesis sentence, four contribution blocks, and the closing GreenNet/CouplingNet role banner together.

## Slide 15 - Backup / Q&A Menu

**Title:** Backup / Q&A Menu

**Subtitle:** Details available if the audience asks

**Main claim:** Backup material should be available for likely technical questions, but it should not be part of the 11-12 minute main talk.

**Must include:**

- Ready Backup A: Dirac/Heaviside derivation sketch.
- Ready Backup B: imperfect Green reconstruction perturbation.
- Ready Backup C: connected-interval pull-back detail.
- Deferred figure-dependent backup: extra GreenNet reconstruction examples.
- Deferred figure-dependent backup: extra CouplingNet split-consistency or energy behavior figure.

**Optional / can omit:**

- Any numerical backup item before final figures are selected.
- Full proof text on the menu slide.
- Implementation details.

**Suggested visual:** A clean two-column menu: "Ready without figures" and "Deferred until numerical figures are available."

**Equations / notation:** No equation is required on the menu slide.  Keep equations on the concrete backup slides.

**Slide text draft:**

- Ready if asked: analytic wrapping derivation.
- Ready if asked: perturbed Green reconstruction interpretation.
- Ready if asked: connected-interval pull-back detail.
- Deferred: extra GreenNet and CouplingNet evidence after figure selection.

**Speaker emphasis:** Treat this as Q&A support.  Do not spend main-talk time here unless a question makes it relevant.

**Animation plan:** Use light fragments only.

- **Initial state:** Show the slide title and the "Ready without figures" heading.
- **Click 1:** Reveal Backup A.
- **Click 2:** Reveal Backup B.
- **Click 3:** Reveal Backup C.
- **Click 4:** Reveal the deferred figure-dependent backup heading and the two deferred evidence candidates.
- **Pacing guard:** Do not turn the menu into another technical slide.
- **PDF fallback:** Show all ready and deferred backup items at once.

## Backup A - Dirac/Heaviside Derivation Sketch

**Title:** Backup A: Dirac/Heaviside Derivation Sketch

**Subtitle:** Why the analytic Green wrapping has three terms

**Main claim:** The analytic terms build the Green kernel's singular and cancellation structure before the neural network learns the remaining smooth correction.

**Must include:**

- A heading that frames the explanation:
  ```text
  Operator application creates two effects.
  ```
- The analytic-neural GreenNet composition:

  \[
  G_\theta(t,\eta)
  =
  E(t,\eta)M(t)R_\theta(t,\eta)
  +
  B(t)\left(J_0(t,\eta)-\frac12E(t,\eta)\right)
  +
  A(t)G_0(t,\eta).
  \]

- The Dirac-\(\delta\) structure identity:

  \[
  \partial_t^2G_0(t,\eta)=-\delta(t-\eta).
  \]

- The antiderivative relation:

  \[
  \partial_tJ_0(t,\eta)=G_0(t,\eta).
  \]

- The cancellation helper:

  \[
  S(t,\eta)=J_0(t,\eta)-\frac12E(t,\eta),
  \qquad
  \partial_t^2S(t,\eta)=\partial_tG_0(t,\eta).
  \]
- The coefficient factors, kept here rather than on the main GreenNet II slide:
  \[
  A(t)=\frac{1}{a_{\mathrm{unit}}(t)},
  \qquad
  B(t)=
  \frac{
  a'_{\mathrm{unit}}(t)+b_{\parallel,\mathrm{unit}}(t)
  }{
  a_{\mathrm{unit}}(t)^2
  }.
  \]

**Optional / can omit:**

- Full piecewise formula for \(J_0\).
- Full distributional proof.
- Implementation details.

**Suggested visual:** Three operator-action blocks labeled "Effect 1: Dirac jump", "Effect 2: Heaviside leftover", and "Analytic compensation", plus a compact coefficient-factor strip for \(A(t)\) and \(B(t)\).

**Equations / notation:**

\[
A(t)G_0(t,\eta)
\quad\rightarrow\quad
\text{Dirac-\(\delta\) jump structure}.
\]

\[
B(t)S(t,\eta)
\quad\rightarrow\quad
\text{Heaviside cancellation}.
\]

\[
E(t,\eta)M(t)R_\theta(t,\eta)
\quad\rightarrow\quad
\text{learned smooth correction}.
\]

**Slide text draft:**

- Operator application creates two effects: the needed Dirac jump and an induced Heaviside-type leftover.
- \(A(t)G_0(t,\eta)\) supplies the Dirac-\(\delta\) jump structure.
- \(J_0\) is an antiderivative of \(G_0\), so \(S=J_0-\frac12E\) generates the \(\partial_tG_0\) structure needed for Heaviside cancellation.
- \(B(t)S(t,\eta)\) cancels the Heaviside contribution before \(E(t,\eta)M(t)R_\theta(t,\eta)\) learns the remaining smooth residual.

**Speaker emphasis:** This slide answers "why is GreenNet not just a black-box kernel?"  Keep the explanation role-based rather than proof-heavy.

**Animation plan:** Use a three-step fragment reveal.

- **Initial state:** Show only the full \(G_\theta\) composition.
- **Click 1:** Highlight \(A(t)G_0(t,\eta)\) and reveal \(\partial_t^2G_0=-\delta\).
- **Click 2:** Highlight \(B(t)S(t,\eta)\) and reveal \(\partial_t^2S=\partial_tG_0\).
- **Click 3:** Highlight \(E(t,\eta)M(t)R_\theta(t,\eta)\) as the learned smooth correction.
- **Pacing guard:** Do not derive every distributional derivative unless the audience explicitly asks.
- **PDF fallback:** Show all three role blocks and identities together.

## Backup B - Imperfect Green Reconstruction Perturbation

**Title:** Backup B: Imperfect Green Reconstruction Perturbation

**Subtitle:** What changes when the learned Green inverse is not exact?

**Main claim:** The split-energy bound remains meaningful with learned Green errors, but it must include perturbation terms for directional mismatch and common bias.

**Must include:**

- Approximate reference reconstructions:

  \[
  G_x[\phi_*]=u_*+\varepsilon_x,
  \qquad
  G_y[\psi_*]=u_*+\varepsilon_y.
  \]

- Split energy:

  \[
  \mathcal{E}_{\mathrm{split}}
  =
  \|u_\phi-u_\psi\|_a^2.
  \]

- Perturbed final error bound:

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

**Optional / can omit:**

- Full theorem proof.
- Admissibility proof.
- Detailed coercivity constant derivation.

**Suggested visual:** A two-channel error diagram: \(\varepsilon_x-\varepsilon_y\) as directional mismatch and \(\varepsilon_x+\varepsilon_y\) as common Green bias.  Use two side-by-side cards so the audience can see that mismatch and bias are different perturbation channels.

**Equations / notation:**

\[
\varepsilon_x-\varepsilon_y
\quad\rightarrow\quad
\text{perturbs split consistency}.
\]

\[
\varepsilon_x+\varepsilon_y
\quad\rightarrow\quad
\text{common Green bias}.
\]

**Slide text draft:**

- If Green reconstruction is exact, the split-energy term controls the final solution energy error.
- If the learned Green reconstructions are imperfect, directional mismatch enters through \(\varepsilon_x-\varepsilon_y\).
- A common bias \(\varepsilon_x+\varepsilon_y\) can be invisible to split consistency, so GreenNet accuracy remains part of the story.

**Speaker emphasis:** This is the honest limitation slide.  Energy consistency is an error-bound mechanism under controlled Green reconstruction, not a claim that every learned Green error disappears.

**Animation plan:** Reveal the perturbation channels sequentially.

- **Initial state:** Show approximate reconstructions \(G_x[\phi_*]\) and \(G_y[\psi_*]\).
- **Click 1:** Reveal the exact split-energy term.
- **Click 2:** Reveal the mismatch term \(\|\varepsilon_x-\varepsilon_y\|_a\).
- **Click 3:** Reveal the common-bias term \(\|\varepsilon_x+\varepsilon_y\|_a\).
- **Pacing guard:** Avoid proof variables and keep the interpretation focused on the two perturbation channels.
- **PDF fallback:** Show the full bound and the two-channel interpretation together.

## Backup C - Connected-Interval Pull-Back Detail

**Title:** Backup C: Connected-Interval Pull-Back Detail

**Subtitle:** How non-square domains become normalized one-dimensional Green problems

**Main claim:** Non-square geometry is handled by mapping each connected physical axial interval to a unit interval and scaling the operator accordingly.

**Must include:**

- The connected physical interval and pull-back:

  \[
  I=[s_0,s_1],
  \qquad
  s=s_0+Lt,
  \qquad
  t\in[0,1].
  \]

- The operator scaling:

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

**Optional / can omit:**

- Geometry generation algorithm.
- File or schema details.
- Mesh construction and sample generation workflow.

**Suggested visual:** A deck-native SVG of a non-square domain slice that produces two separated connected intervals \(I_1\) and \(I_2\), with an outside-domain gap between them.  Show that each connected interval is an independent 1D Green problem before showing the pull-back/scaling formulas.

**Equations / notation:**

\[
\text{connected interval}
\quad\rightarrow\quad
\text{independent 1D Dirichlet Green problem}.
\]

\[
\text{disconnected intervals are not merged}.
\]

**Slide text draft:**

- A non-square domain can produce one or more connected intervals along a physical axial slice.
- Each connected interval is treated as an independent one-dimensional Dirichlet Green problem.
- Connected intervals are not merged across outside-domain gaps, because merging would create artificial information flow through regions outside the domain.
- Geometry is not the main story; it motivates unit-interval normalization and scaling.

**Speaker emphasis:** Use this only if the audience asks about geometry.  The main talk should keep complex geometry lightweight and focus on the operator reduction.

**Animation plan:** Use a simple three-step reveal.

- **Initial state:** Show the physical axial slice and one connected interval.
- **Click 1:** Reveal the pull-back \(s=s_0+Lt\).
- **Click 2:** Reveal the unit-interval scaling rules.
- **Click 3:** Reveal the warning that disconnected intervals are not merged.
- **Pacing guard:** Do not discuss geometry extraction or data generation.
- **PDF fallback:** Show the interval map and scaling rule together.

## Core Phrase Bank

- **GreenNet:** GreenNet is not a black-box kernel approximator; it embeds analytic singular and cancellation structure and learns the remaining smooth correction.
- **CouplingNet:** CouplingNet learns the source split using source, operator, axial, and transverse context so axial Green reconstructions are consistent.
- **Energy:** The split energy is an error-bound quantity under structural assumptions, not only a diagnostic.
- **Geometry:** Geometry enters through non-unit axial intervals and pull-back scaling, not through a geometry-generation story.
- **Evidence:** GreenNet evidence shows axial Green-kernel structure and signed error; CouplingNet evidence shows full solution reconstruction quality.
- **MOR framing:** The method reduces the operator representation while preserving Green-operator structure.
