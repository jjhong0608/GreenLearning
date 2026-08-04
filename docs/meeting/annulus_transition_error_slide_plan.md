# Annulus Transition Error Meeting Slide Plan

## Document Status

- **Purpose:** 공동연구자 회의에서 Annulus transition error의 원인 가설과
  정량 진단 결과를 공유하기 위한 Quarto 슬라이드 작성 계획
- **Current stage:** 20장 Quarto Reveal.js deck 구현 및 layout QA 단계
- **Implemented deck:**
  `docs/meeting/annulus_transition_error/annulus_transition_error.qmd`,
  presentation SCSS, frozen-artifact Plotly assets, rendered HTML
- **Primary audience:** 이전 회의에서 Annulus transition error 현상을 이미
  확인한 공동연구자
- **Language:** visible slide text는 영어, speaker notes는 한국어
- **Target length:** main slides 18장, backup slides 2장, 총 20장

## Active Revision Contract

- Slide 2는 \(2.19\times\) line-length ratio와 \(4.80\times\)
  response-scale ratio를 분리해 표시하고, geometry/pipeline/equation/takeaway
  block의 높이를 조정해 하단 공간을 균형 있게 사용한다.
- Slide 4의 네 formulation 이름은 `Original physical symmetric`,
  `Length-aware variants`, `Response-space`, `Physical symmetric`으로 고정한다.
  Length-aware card에는 \(d_0\), \(\kappa\),
  \(d_{\mathrm{RPS}}=d_0+\kappa d\)의 정의와 의미를 함께 표시한다.
- Slide 5의 세 rule은 `Signal`, `Sample adaptive`, `Operator aware`를 같은
  행 구조로 비교하고, 하단에는
  \((\phi,\psi,u_\phi,u_\psi)\)가 frozen이라는 공통 invariant를 둔다.
- Slide 11의 weak-prediction 식은 sidebar 전용 two-line formula card를 쓰며,
  Poisson/CDR result-field Plotly colorbar title은 bar 오른쪽에 배치한다.
- 발표용 visible text와 한국어 speaker notes에서는 formulation을 version
  번호나 model-contract 번호로 부르지 않는다. 내부 checkpoint contract,
  builder metadata, frozen numerical arrays는 변경하지 않는다.
- Slide 15의 exact-balance/response-consistency 수식은 유지하고, Slide 16은
  fixed tangent correction의 실제 설정과 response-mismatch ratio를 명시한다.
- Poisson `coupling18`과 CDR `coupling8`의 frozen tangent artifact를 각각
  Slide 17/18에서 설명한다. 두 슬라이드는 mismatch 감소와 final evaluation
  metric을 분리하며 symmetric-trained baseline 대비 causal improvement를
  주장하지 않는다.
- 20장 구조, 50개 fragment, numerical values와 frozen-artifact provenance를
  유지한다. 1600x900 및 1280x720의 final/intermediate states에서 overflow,
  overlap, page error, external request가 없어야 한다.

## Meeting Goal

이번 transition block의 목적은 관찰된 error field를 다시 소개하는 것이 아니라,
다음 질문에 대한 현재의 답을 정리하는 것이다.

> Why does the reconstruction error concentrate near the annulus transition
> even though the physical domain and reference solution are continuous?

회의에서 전달할 핵심 결론은 다음과 같다.

> At the annulus transition, the axial representation changes from two short
> independent intervals to one long interval. The predicted physical
> directional source split does not provide sufficient inverse-\(L^2\)
> compensation for the resulting response-scale change. Pointwise balance
> projection enforces \(\phi+\psi=f\), but it does not enforce cross-line
> regularity of \(L_x^2\phi\) and \(L_y^2\psi\). The source-split error is
> therefore amplified during pull-back and propagated by both exact and learned
> Green reconstruction.

이 결론은 완전히 증명된 theorem이 아니라, 현재 정량 진단으로 가장 강하게
지지되는 **working hypothesis**로 제시한다.

## Agreed Scope

### Include

- 이전 회의에서 확인한 transition-localized directional-source 및 solution
  error의 짧은 recap
- Annulus inner-boundary transition에서 axial connected intervals가 바뀌는 구조
- 인접 split line과 one-segment line 사이의 \(L\), \(L^2\) scale 변화
- Physical directional-source error에서 unit-response error로 이어지는 변환
- Exact Green과 learned Green을 사용한 stagewise error diagnosis
- 현재 근거로 배제할 수 있는 원인
- 확인된 사실과 working hypothesis의 명시적인 구분
- Transition error를 줄이기 위해 시험한 projection/output-space formulation과
  현재 physical-symmetric formulation까지의 변화
- Equal mean을 대체하는 세 post-hoc estimator의 공통 구조와 서로 다른 weight
  information contract: geometry-only compact C2, mismatch-detected seam C2,
  local full-PDE weak-residual reliability
- Frozen Poisson `coupling15`에서 fixed presets로 수행한 four-estimator clean
  comparison과 transition/trace diagnostics
- Frozen CDR `coupling5`에서 동일한 estimator ordering이 유지되는지 확인한
  broader-operator consistency comparison
- Local weak-residual reliability를 최종 estimator로 사용한 Poisson/CDR의
  representative directional-source, reconstruction, signed-error field와 전체
  50-sample accuracy summary
- CDR physical coefficient fields (a(x,y)), \(\mathbf b(x,y)\), \(c(x,y)\)와
  coefficient-independent Poisson context
- Frozen Green response에서 계산한 fixed diagonal gain으로 physical source
  projection을 condition하는 Green-Response Preconditioning
- Symmetric-balanced source에서 balance plane을 따라 이동하는 fixed tangent
  correction과 Poisson `coupling18`/CDR `coupling8`의 50-sample response-mismatch
  audit

### Exclude

- Pre-projection fuser의 구조, residual/absolute/off 비교 및 관련 artifact
- \(u_\phi,u_\psi\in H_0^1(\Omega)\) 가정의 완화 가능성
- 승인된 Slide 5-18 이외의 새로운 loss, projection, network architecture 또는
  regularization 제안
- Learned reconstruction weight/gate, Multi-Orientation Axial Charts 및 full
  Green-response matrix solve
- Training-time relative split consistency, weak operator closure 및 관련
  `coupling6` loss ablation. Slides 5-14의 local weak residual은 학습 loss가
  아니라 frozen reconstruction을 결합하는 post-hoc reliability indicator로만
  포함
- Self-trace gluing, transition-only cross-axis carrier, null-space 분석 및
  general boundary-energy 개발 과정
- SOAP/AdamW optimizer 비교
- Dataset-size 및 source-generation ablation
- 이번 transition diagnosis와 직접 관련 없는 Disk 및 기타 CDR 결과. Annulus
  CDR `coupling5`의 fixed four-estimator comparison과 weak-result field summary만
  포함

제외 항목은 main slide와 backup slide 모두에 넣지 않는다. 이후 별도 회의
주제로 다룰 수 있다.

## Evidence Hierarchy

슬라이드에서는 주장마다 근거 수준을 구분한다.

### Quantitatively Verified

- Inner radius:

  \[
  \rho=0.2.
  \]

- Nearest split and one-segment axial coordinates:

  \[
  |x|,|y|=0.1953125
  \quad\text{and}\quad
  |x|,|y|=0.203125.
  \]

- Mean segment lengths:

  \[
  L_{\mathrm{split}}\approx0.4172,
  \qquad
  L_{\mathrm{single}}\approx0.9138.
  \]

- Length and length-squared ratios:

  \[
  \frac{L_{\mathrm{single}}}{L_{\mathrm{split}}}
  \approx2.19,
  \qquad
  \frac{L_{\mathrm{single}}^2}{L_{\mathrm{split}}^2}
  \approx4.80.
  \]

- Unit-interval and physical-interval Green integration agree to float64
  round-off.
- The exact Green reconstruction preserves the same transition pattern seen
  with the learned GreenNet.
- GreenNet approximation error and target-source closure error are too small to
  explain the observed transition pattern.
- The projected physical source satisfies the pointwise balance constraint
  within numerical tolerance.
- The optional FEniCSx directional targets are numerical diagnostics rather than
  exact balance labels. For sample 47, the predicted balance residual has
  maximum magnitude \(6.66\times10^{-16}\), while the target balance residual
  has RMS \(9.89\times10^{-3}\) and maximum magnitude \(2.64\times10^{-1}\).
- The historical sample-47 arrays exactly reproduce the stored projected
  physical field through

  \[
  \phi
  =
  \frac12
  \left(
  f+\frac{P_0}{L_x^2}-\frac{Q_0}{L_y^2}
  \right),
  \]

  where \(P_0=\texttt{raw\_unit\_phi}\) and
  \(Q_0=\texttt{raw\_unit\_psi}\). The maximum discrepancy from the stored
  `phi` array is \(4.44\times10^{-16}\). The Slide 2 transformation is
  therefore the verified historical calculation, not a current-code contract
  projected backward onto an older checkpoint.
- On the frozen Poisson `Annulus_poisson/coupling15` 50-sample test set, equal
  mean, geometry-only compact C2, mismatch-detected seam C2, and local
  weak-residual reliability have mean `rel_sol` values `5.569825%`,
  `5.470165%`, `5.312818%`, and `4.860579%`. Their changes versus equal mean
  are `-1.789%`, `-4.614%`, and `-12.734%`, with `35/50`, `44/50`, and `50/50`
  sample wins. The corresponding transition-zone RMS changes are `-3.412%`,
  `-6.954%`, and `-8.360%`; transition trace-jump RMS changes are `-45.308%`,
  `-23.553%`, and `-48.280%`.
- On the frozen CDR `annulus_CDR/coupling5` 50-sample test set, the same four
  estimators have mean `rel_sol` values `5.156246%`, `5.054177%`, `4.943157%`,
  and `4.564722%`. Their changes versus equal mean are `-1.980%`, `-4.133%`,
  and `-11.472%`, with `36/50`, `41/50`, and `49/50` sample wins. The
  corresponding transition-zone RMS changes are `-4.225%`, `-7.391%`, and
  `-9.870%`; transition trace-jump RMS changes are `-44.956%`, `-22.425%`, and
  `-48.785%`.
- For the final local weak-residual estimator, the 50-sample mean/median
  `rel_sol` values are `4.860579%`/`4.822652%` for Poisson and
  `4.564722%`/`4.205909%` for CDR. The unchanged directional-source diagnostic
  mean `rel_flux` values are `17.098821%` and `17.898558%`, respectively.
- The result-field representatives are the existing selected q50-role samples:
  Poisson sample 0 and CDR sample 9. Their weak `rel_sol` values are
  `4.824260%` and `4.573527%`. Their physical balance residuals have maximum
  magnitudes `2.22e-16` and `4.02e-16`, respectively.
- The result slides must load `phi`, `psi`, `u_phi`, and `u_psi` from the
  standard selected artifact and load the final prediction from
  `u_weak_residual_reliability` in the weak-comparison archive. The standard
  artifact field `u_pred` is the equal mean and is not the weak result.
- Poisson uses the fixed CDR exploratory presets without a Poisson parameter
  sweep. The presentation order is Poisson first for conceptual isolation of
  geometry, followed by CDR for broader-operator consistency; it must not be
  described as the chronological tuning order or as independent cross-domain
  validation.

### Strongly Supported Interpretation

- The predicted physical source-split error does not decrease by the
  approximately \(1/4.80\) factor needed to compensate for the response-scale
  jump.
- The first strong transition contrast appears after applying the mandatory
  \(L^2\) response pull-back.
- Pointwise balance projection cannot remove the error because it preserves the
  directional difference mode and has no neighboring-line regularity
  constraint.
- Local full-PDE weak defects provide a stronger reliability signal than the
  tested geometry-only and mismatch-seam rules on both frozen Annulus
  checkpoints, without reading a known transition coordinate or reference
  target.
- The identical estimator ordering under Pure Poisson and CDR supports an
  equation-family-consistent signal on this fixed Annulus geometry.

### Not Yet Proven

- Whether the transition sensitivity is unavoidable for every model using the
  current connected-interval parameterization
- The relative contributions of discrete axial sampling, model approximation,
  and the source-split identifiability problem
- Which modification will remove the transition error without introducing a
  new bias
- Whether the weak-residual reliability improvement transfers to Disk,
  multi-hole, and nonconvex domains or remains specific to this Annulus
  geometry
- Whether the striped weak-residual weight field preserves sufficient
  transverse regularity under geometry and grid refinement

## Narrative Flow

The first three main slides follow a single causal chain:

\[
\boxed{
\text{axial representation change}
\;\longrightarrow\;
\text{insufficient inverse-response compensation}
\;\longrightarrow\;
L^2\text{-scaled error amplification}
\;\longrightarrow\;
\text{transition-localized solution error}
}
\]

The narrative must distinguish the following statements:

1. The physical annulus and reference solution are not discontinuous.
2. The connected-interval representation changes sharply at the tangent slice.
3. The \(L^2\) scaling is mathematically required by the unit-interval pull-back.
4. The scaling reveals and amplifies insufficient compensation in the predicted
   physical directional split; it is not an arbitrary numerical correction.
5. Exact and learned Green operators propagate essentially the same pattern.

Slide 4 then changes from diagnosis to method history:

\[
\boxed{
\text{physical symmetric}
\;\longrightarrow\;
\text{length-aware variants}
\;\longrightarrow\;
\text{response-space projection}
\;\longrightarrow\;
\text{current physical-symmetric}
}
\]

This final slide must explain which projection/output-space hypotheses were
tested and retired. It must not expand into a complete loss or admissibility
development history.

Slides 5-18 then separate method definition, method construction, estimator comparison,
final-field evidence, tangent correction, and its measured response audit:

\[
\boxed{
\text{three fixed post-hoc blend rules}
\;\longrightarrow\;
\text{three detailed weight constructions}
\;\longrightarrow\;
\text{Poisson clean comparison}
\;\longrightarrow\;
\text{CDR consistency check}
\;\longrightarrow\;
\text{Poisson weak-result fields/errors}
\;\longrightarrow\;
\text{CDR weak-result coefficients/fields/errors}
\;\longrightarrow\;
\text{symmetric balance-plane tangent correction}
\;\longrightarrow\;
\text{Poisson/CDR fixed-step response audit}
}
\]

Slide 5 defines the common partition-of-unity estimator and the three fixed
weight contracts without using result metrics. Slides 6-8 then explain the
geometry-only compact C2, mismatch-detected seam C2, and local weak-residual
reliability constructions one at a time, using the same sensor-to-weight visual
grammar. Slide 9 presents Poisson first to isolate geometry/reconstruction
behavior from convection and reaction. Slide 10 then shows that the same estimator ordering persists for CDR on the
same Annulus geometry and explicitly separates cross-equation consistency from
cross-domain generalization. Slides 9 and 10 are measured frozen-checkpoint,
no-retraining experiments that change only the final reconstruction estimator.
Slides 11-12 present the final Poisson weak-result fields and signed errors, while
Slides 13-14 repeat the result contract for CDR and add its physical
coefficients. These four slides describe the computed result rather than
diagnosing the transition. Slide 15 separates exact source balance from
directional response consistency. Slide 16 then starts from that
symmetric-balanced source and introduces one fixed Jacobi-preconditioned
tangent step inside the balance plane. Slides 17-18 then report the directly
measured pre/post response mismatch for Poisson and CDR, together with the final
evaluation metrics. These result slides establish that the fixed tangent step
reduces its own mismatch objective on all 50 test samples in both equations;
they do not establish a causal solution-quality gain over a separately trained
symmetric baseline.

## Main Slide Plan

### Slide 1: Transition Structure Appears Before and After Reconstruction

**Purpose**

Recall the previously shared \(u_{\mathrm{pred}}\) error, separate it into the
two directional reconstruction paths, and show whether the same transition
structure is already visible in the directional-source errors. Slide 1
establishes the observed stage-to-stage pattern; it does not claim by itself
that the source split is the unique cause.

**Provisional title**

> Transition Structure Appears in Directional Sources and Reconstructions

**Locked artifact source**

- Artifact root:
  `checkpoints/Annulus_poisson/coupling/artifacts`
- Selected sample: sample 47
- Artifact role: maximum-\(\mathrm{rel\_sol}\) selected test sample
- Recorded artifact metric:

  \[
  \mathrm{rel\_sol}=0.1139417139.
  \]

- Raw artifact:
  `checkpoints/Annulus_poisson/coupling/artifacts/data/selected_raw_arrays.npz`
- Required sample-47 arrays:
  `coords_valid`, `phi_error`, `psi_error`, `u_phi_error`,
  `u_pred_error`, and `u_psi_error`

The composite is an artifact-only visualization. It must load these arrays
directly with `numpy.load(..., allow_pickle=False)` and must not instantiate a
current CouplingNet, load the historical checkpoint, rerun inference, or pass
through the current config parser. This keeps the figure independent of later
checkpoint and model-contract changes.

**Field definitions**

- Directional-source errors:

  \[
  e_\phi=\phi-\phi_*,
  \qquad
  e_\psi=\psi-\psi_*.
  \]

- Directional and averaged solution errors:

  \[
  e_{u_\phi}=u_\phi-u_*,
  \qquad
  e_{u_{\mathrm{pred}}}=u_{\mathrm{pred}}-u_*,
  \qquad
  e_{u_\psi}=u_\psi-u_*,
  \]

  with

  \[
  u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi).
  \]

**Composite figure layout**

Use one Plotly figure with a two-row, five-panel final state:

```text
Directional-source errors
┌────────────────────────────┐  ┌────────────────────────────┐
│        phi - phi_*         │  │        psi - psi_*         │
└────────────────────────────┘  └────────────────────────────┘
              |                               |
              v                               v

Solution-reconstruction errors
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│    u_phi - u_*   │ │   u_pred - u_*  │ │    u_psi - u_*   │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

- The top row has two equal-width panels and communicates the directional
  source-component mismatch before Green reconstruction.
- The bottom row has three equal-width panels ordered as
  \(u_\phi\), \(u_{\mathrm{pred}}\), and \(u_\psi\).
- The left and right columns preserve the path correspondence
  \(e_\phi\rightarrow e_{u_\phi}\) and
  \(e_\psi\rightarrow e_{u_\psi}\).
- The center bottom panel emphasizes that the averaged prediction retains
  spatial structure from both directional paths.
- Use a 16:9 canvas, initially \(1600\times900\), with equal physical aspect
  ratio in every panel.
- Fix every panel to the same physical coordinate range, initially
  \([-0.54,0.54]\times[-0.54,0.54]\).
- Show the `x` and `y` labels once at the figure level rather than repeating
  them in every panel. Keep only the tick labels needed to read the common
  spatial coordinates.
- HTML hover must report `x`, `y`, the mathematical field name, and the signed
  error value.

**Color contract**

- Use a zero-centered diverging `RdBu` scale, with negative values red, zero
  white, and positive values blue.
- The two directional-source panels share one `coloraxis` and one colorbar.
- The three solution-error panels share a separate `coloraxis` and one
  colorbar.
- Do not share a color range between the source and solution rows because they
  have different physical meanings and numerical scales.
- For the locked sample-47 artifact, use the full maximum absolute value within
  each row:

  \[
  C_{\mathrm{source}}
  =
  \max(\lVert e_\phi\rVert_\infty,\lVert e_\psi\rVert_\infty)
  =
  1.2686799967,
  \]

  \[
  C_{\mathrm{solution}}
  =
  \max(
  \lVert e_{u_\phi}\rVert_\infty,
  \lVert e_{u_{\mathrm{pred}}}\rVert_\infty,
  \lVert e_{u_\psi}\rVert_\infty)
  =
  2.4034501199\times10^{-3}.
  \]

- Use ranges
  \([-C_{\mathrm{source}},C_{\mathrm{source}}]\) and
  \([-C_{\mathrm{solution}},C_{\mathrm{solution}}]\). Do not apply percentile
  clipping on this diagnostic slide.

**Numerical audit retained in the planning record**

| Field | RMS | Maximum absolute error |
|---|---:|---:|
| \(\phi-\phi_*\) | \(1.07148\times10^{-1}\) | \(1.26534\) |
| \(\psi-\psi_*\) | \(1.06736\times10^{-1}\) | \(1.26868\) |
| \(u_\phi-u_*\) | \(5.18985\times10^{-4}\) | \(2.40345\times10^{-3}\) |
| \(u_{\mathrm{pred}}-u_*\) | \(2.90215\times10^{-4}\) | \(1.20400\times10^{-3}\) |
| \(u_\psi-u_*\) | \(4.92351\times10^{-4}\) | \(2.22369\times10^{-3}\) |

This table is for figure validation and speaker preparation. It should not be
placed as a second dense element on Slide 1.

**Transition annotation**

- Mark the four inner-boundary cardinal points in slide-native overlay:

  \[
  (\pm0.2,0),
  \qquad
  (0,\pm0.2).
  \]

- Apply the markers to the center \(u_{\mathrm{pred}}\) panel only, rather than
  repeating them over all five panels.
- Use small open markers and one compact label, `axial transition at r=0.2`.
  Do not draw four full guide lines across every subplot.

**Target-field interpretation**

The FEniCSx \(\phi_*,\psi_*\) arrays are optional numerical directional targets,
not exact balance labels. The prediction satisfies

\[
\phi+\psi=f
\]

to round-off, but the stored target has a nonzero numerical balance residual.
Consequently,

\[
e_\phi+e_\psi
=
f-(\phi_*+\psi_*).
\]

For sample 47, the predicted balance residual has maximum magnitude
\(6.66\times10^{-16}\), while the target balance residual has RMS
\(9.89\times10^{-3}\) and maximum magnitude \(2.64\times10^{-1}\). Therefore
the top-row fields may be used to locate mismatch against the FEniCSx
directional targets, but they must not be presented as evidence that the
prediction violates the balance projection. Put this qualification in the
speaker notes and use the compact visible label `against numerical FEniCSx
targets`.

**Planned output assets**

- Interactive presentation asset:
  `assets/annulus_transition_sample47_error_matrix.html`
- Static/offline and handout asset:
  `assets/annulus_transition_sample47_error_matrix.pdf`
- Do not create or use a PNG version for the meeting slide.
- The HTML must be a standalone Plotly document. The PDF must be rendered from
  the same figure specification so the two formats have identical panel order,
  color limits, labels, and annotations.
- Retain the five existing individual HTML/PDF figures only as provenance and
  visual cross-checks; do not embed five independent iframes in Quarto.

**Minimal on-slide text**

- Row labels: `Directional-source errors` and `Solution-reconstruction errors`
- Formula:

  \[
  u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi).
  \]

- Provenance label: `Sample 47 | max selected rel. sol. error`
- Compact target qualifier: `phi_* and psi_*: numerical FEniCSx targets`

**Takeaway**

> Transition-localized structure is visible in both directional-source errors
> and persists in both axial reconstructions. Averaging reduces the global
> error but does not eliminate the transition pattern.

**Speaker emphasis**

- This phenomenon was already shared in the previous meeting.
- Do not repeat the complete training setup.
- The \(u_\phi\) error emphasizes horizontal transition structure and the
  \(u_\psi\) error emphasizes vertical transition structure.
- The top row shows that a related pattern is visible before reconstruction,
  but does not by itself prove a unique causal mechanism.
- Do not interpret the numerical target imbalance as a failure of the
  prediction's balance projection.
- Transition location and stage-to-stage propagation, not aggregate relative
  error, are the focus.

**Reveal plan**

1. Show the complete five-panel composite so the directional paths can be
   compared immediately.
2. Add the four cardinal markers on the \(u_{\mathrm{pred}}\) panel and reveal
   the takeaway sentence.

Do not use a five-step panel-by-panel reveal. The simultaneous spatial
comparison is the purpose of this slide, and the complete final state must
remain legible in the PDF fallback.

### Slide 2: Line Length Enters Projection and Pull-Back

**Purpose**

Define the exact transformation used by the historical run:

\[
\text{raw reference output}
\longrightarrow
\text{physical raw proposal}
\longrightarrow
\text{physical balance projection}
\longrightarrow
\text{reference-interval source}.
\]

This slide explains both where \(L_x,L_y\) enter and why the two directional
raw outputs are coupled before Green reconstruction.

**Provisional title**

> Where Line Length Enters Projection and Pull-Back

**Geometry context**

For \(|y|<\rho\):

\[
\Omega\cap\{y=\bar y\}
=
[-x_{\mathrm{out}},-x_{\mathrm{in}}]
\cup
[x_{\mathrm{in}},x_{\mathrm{out}}],
\]

\[
L_{\mathrm{split}}
=
x_{\mathrm{out}}-x_{\mathrm{in}}.
\]

For \(|y|>\rho\):

\[
\Omega\cap\{y=\bar y\}
=
[-x_{\mathrm{out}},x_{\mathrm{out}}],
\]

\[
L_{\mathrm{single}}
=
2x_{\mathrm{out}}.
\]

The drawing should show that the physical annulus remains smooth while the
connected-interval representation changes from two interval problems to one.

Use the measured neighboring-line values as a compact geometry callout:

\[
0.1953125:
\quad
L_{\mathrm{split}}\approx0.4172,
\]

\[
0.203125:
\quad
L_{\mathrm{single}}\approx0.9138.
\]

\[
\frac{L_{\mathrm{single}}}{L_{\mathrm{split}}}
\approx2.19,
\qquad
\frac{L_{\mathrm{single}}^2}{L_{\mathrm{split}}^2}
\approx4.80.
\]

Use the \(4.80\times\) value as the dominant visual number.

**Physical and reference coordinates**

For the horizontal connected segment through \((x,y)\),

\[
x=x_L(y)+L_x(y)t_x,
\qquad
t_x=\frac{x-x_L(y)}{L_x(y)}.
\]

For the vertical connected segment,

\[
y=y_B(x)+L_y(x)t_y,
\qquad
t_y=\frac{y-y_B(x)}{L_y(x)}.
\]

The coordinate map identifies a physical point with one local coordinate on
each of its two connected axial intervals. The value scaling below is a
separate operation required by the second-order operator pull-back.

**Stage 1: historical raw reference outputs**

Use notation that is independent of the current model contract:

\[
P_0
=
\texttt{raw\_unit\_phi},
\qquad
Q_0
=
\texttt{raw\_unit\_psi}.
\]

\(P_0,Q_0\) are the historical CouplingNet outputs in the
reference-interval directional-source scale. Do not label them as the balanced
physical \(\phi,\psi\).

**Stage 2: convert to physical raw source proposals**

\[
p_0=\frac{P_0}{L_x^2},
\qquad
q_0=\frac{Q_0}{L_y^2}.
\]

This is the pre-projection conversion. It changes the field units used by the
projection; it does not move the point to a different physical location.

**Stage 3: symmetric projection in physical source space**

\[
d_0=p_0-q_0,
\]

\[
\phi=\frac{f+d_0}{2},
\qquad
\psi=\frac{f-d_0}{2}.
\]

The projection is pointwise at the same physical \((x,y)\) and satisfies

\[
\phi+\psi=f,
\qquad
\phi-\psi=d_0
=
\frac{P_0}{L_x^2}
-
\frac{Q_0}{L_y^2}.
\]

It enforces physical source balance while preserving the physical raw
difference mode.

**Stage 4: pull balanced sources back to the reference intervals**

\[
\Phi=L_x^2\phi,
\qquad
\Psi=L_y^2\psi.
\]

\(\Phi,\Psi\) are the source terms supplied to the unit-interval Green
reconstructions. There is no additional length multiplication inside the
normalized Green integral:

\[
u_\phi(t_x)
=
\int_0^1
G_x^{\mathrm{unit}}(t_x,\eta)\Phi(\eta)\,d\eta,
\]

\[
u_\psi(t_y)
=
\int_0^1
G_y^{\mathrm{unit}}(t_y,\eta)\Psi(\eta)\,d\eta.
\]

**Compact pipeline**

\[
\boxed{
(P_0,Q_0)
\overset{/L_x^2,/L_y^2}{\longrightarrow}
(p_0,q_0)
\overset{\phi+\psi=f}{\longrightarrow}
(\phi,\psi)
\overset{\times L_x^2,\times L_y^2}{\longrightarrow}
(\Phi,\Psi)
}
\]

**Expanded formula: the key transition-sensitive coupling**

Substituting the physical projection into the pull-back gives

\[
\Phi
=
\frac12
\left[
L_x^2f
+
P_0
-
\frac{L_x^2}{L_y^2}Q_0
\right],
\]

\[
\Psi
=
\frac12
\left[
L_y^2f
-
\frac{L_y^2}{L_x^2}P_0
+
Q_0
\right].
\]

These formulas must be the final mathematical reveal on Slide 2. They show
that:

- the horizontal reference source \(\Phi\) depends on both \(P_0\) and the
  vertical raw output \(Q_0\);
- the vertical reference source \(\Psi\) depends on both \(Q_0\) and the
  horizontal raw output \(P_0\);
- cross-axis ratios \(L_x^2/L_y^2\) and \(L_y^2/L_x^2\) enter explicitly; and
- an abrupt change in either axial length can change both reconstructed paths,
  even though the physical projection remains exactly balanced.

**Historical formula audit**

The Slide 1 historical artifact stores no explicit output-scaling metadata, so
the formula was verified from the arrays rather than inferred from current
source code. For sample 47,

\[
\max
\left|
\phi_{\mathrm{stored}}
-
\frac12
\left(
f+\frac{P_0}{L_x^2}-\frac{Q_0}{L_y^2}
\right)
\right|
=
4.44\times10^{-16}.
\]

Keep this value in speaker notes or backup material. It does not need to occupy
main-slide space.

**Visual composition**

- Use a top grid with two flexible slice cards and a dedicated ratio card of
  approximately 260 px. Display \(4.80\times\) as the dominant response-scale
  value and `line-length ratio: 2.19x` as a separate short row; do not place the
  full ratio formula in the narrow card.
- Use the central area for four connected math stages:
  \(P_0,Q_0\), \(p_0,q_0\), \(\phi,\psi\), and \(\Phi,\Psi\).
- Use consistent colors by space: neutral for raw reference, one physical-space
  accent for \(p_0,q_0,\phi,\psi\), and a reference-space accent for
  \(\Phi,\Psi\).
- Put `projection in physical source space` directly above the projection
  stage.
- Use the expanded \(\Phi,\Psi\) equations as the final reveal, with the two
  cross-axis length ratios visually emphasized.
- Increase the slide-local card heights and vertical spacing for the geometry,
  pipeline, expanded equations, and takeaway so the final state uses the lower
  slide area without entering the slide-number or progress region.
- Do not show current code names, checkpoint-version numbers, or model
  architecture on the slide.

**Takeaway**

> Physical balance is enforced after dividing by the two directional
> length-squared scales, and the balanced components are then pulled back with
> those scales. The resulting reference sources contain explicit cross-axis
> length ratios.

**Speaker emphasis**

- The physical geometry is continuous.
- The abrupt quantity is the discrete connected-interval response
  parameterization.
- The projection is performed in physical source space, not directly on
  \(P_0,Q_0\).
- Dividing by \(L^2\) before projection and multiplying by \(L^2\) after
  projection are different parts of one coordinate-and-unit transformation;
  they are not mutually cancelling scalar operations because the projection
  couples both directional fields.
- \(L^2\) is required by the second-order operator pull-back and must not be
  presented as an optional heuristic.

**Reveal plan**

1. Show the two-short-interval slice.
2. Replace or compare it with the one-long-interval slice.
3. Reveal the measured \(2.19\times\) length and \(4.80\times\)
   length-squared ratios.
4. Reveal the four-stage compact pipeline through the physical projection.
5. Expand \(\Phi,\Psi\) and highlight \(L_x^2/L_y^2\) and
   \(L_y^2/L_x^2\).

### Slide 3: The Strong Contrast Appears at the \(L^2\) Pull-Back

**Purpose**

Apply the Slide 2 transformation to the errors and use stagewise diagnostics to
identify where the transition contrast becomes strong. Slide 3 supplies the
measurement; it does not repeat the full raw/projection derivation.

**Provisional title**

> Locating the Transition Error in the Reconstruction Pipeline

**Error transformation**

\[
e_\phi=\phi-\phi_*,
\qquad
e_\psi=\psi-\psi_*.
\]

Pull-back maps the physical directional-source errors to reference-source
errors:

\[
E_\Phi=L_x^2e_\phi,
\qquad
E_\Psi=L_y^2e_\psi.
\]

The corresponding exact-Green source-error responses are

\[
\delta u_\phi^{\mathrm{exact}}
=
\mathcal G_x^{\mathrm{exact}}[E_\Phi],
\qquad
\delta u_\psi^{\mathrm{exact}}
=
\mathcal G_y^{\mathrm{exact}}[E_\Psi].
\]

Use the following slide pipeline:

\[
(e_\phi,e_\psi)
\longrightarrow
\bigl(L_x^2e_\phi,L_y^2e_\psi\bigr)
\longrightarrow
\bigl(\delta u_\phi^{\mathrm{exact}},
\delta u_\psi^{\mathrm{exact}}\bigr)
\longrightarrow
\bigl(e_{u_\phi}^{\mathrm{learned}},
e_{u_\psi}^{\mathrm{learned}}\bigr).
\]

**Primary quantitative table: sample 47**

| Stage | x/\(\phi\) path | y/\(\psi\) path |
|---|---:|---:|
| Projected physical source error | 0.797 | 1.159 |
| \(L^2\)-scaled response error | 3.822 | 5.559 |
| Exact-Green source response | 1.517 | 1.630 |
| Learned-Green solution error | 1.530 | 1.609 |

The entries are one-segment-line to split-line RMS ratios. The table caption
must state this explicitly.

**Exact/learned decomposition retained for interpretation**

For each directional path, the diagnostic separates the learned total error
into

\[
\text{learned total error}
=
\text{exact response to source-split error}
+
\text{exact target closure}
+
\text{learned-minus-exact Green contribution}.
\]

This decomposition is not a new training loss. It uses reference fields only
for evaluation and is included to determine whether the transition pattern is
created before or inside Green reconstruction.

**Candidate diagnostic assets**

- Primary HTML:
  `checkpoints/Annulus_poisson/coupling4/length_response_diagnostics/figures/sample_0047_sample_000047/source_stages.html`
- Matching PDF:
  `checkpoints/Annulus_poisson/coupling4/length_response_diagnostics/figures/sample_0047_sample_000047/source_stages.pdf`
- Primary HTML:
  `checkpoints/Annulus_poisson/coupling4/length_response_diagnostics/figures/sample_0047_sample_000047/reconstruction_decomposition.html`
- Matching PDF:
  `checkpoints/Annulus_poisson/coupling4/length_response_diagnostics/figures/sample_0047_sample_000047/reconstruction_decomposition.pdf`
- `checkpoints/Annulus_poisson/coupling4/length_response_diagnostics/diagnosis_report.md`

The final slide should either use the compact numerical table or a newly
generated slide-native stagewise bar chart. Do not place the existing composite
diagnostic figure on the slide without checking readability at 16:9.

**Ruled-out causes**

- GreenNet approximation error
- Unit-versus-physical integration mismatch
- Quadrature or target-source closure error
- Failure of the pointwise balance constraint

**Working hypothesis**

> The physical source-split error does not provide the inverse-\(L^2\)
> compensation needed across the segment transition. The mandatory pull-back
> amplifies this error, and the Green operators propagate it.

**Speaker emphasis**

- A moderate physical source error is not sufficient near a large response-scale
  change.
- \(E_\Phi=L_x^2e_\phi\) and \(E_\Psi=L_y^2e_\psi\) are the exact error
  transformations corresponding to Slide 2.
- Balance projection enforces \(\phi+\psi=f\), but not cross-line regularity of
  \(L_x^2\phi\) and \(L_y^2\psi\).
- Exact and learned Green results are nearly identical for the transition
  structure.
- Present the conclusion as a working hypothesis supported by diagnostics, not
  as a completed impossibility theorem.

**Reveal plan**

1. Start from the physical errors \(e_\phi,e_\psi\) already shown on Slide 1.
2. Apply \(E_\Phi=L_x^2e_\phi\), \(E_\Psi=L_y^2e_\psi\) and reveal the
   measured jump.
3. Reveal the exact-Green response and then the learned-Green response.
4. Show the ratio table and highlight the first strong contrast at the
   pull-back stage.
5. End with exact/learned agreement and the working-hypothesis sentence.

### Slide 4: Projection Strategies Tested for the Annulus Transition

**Purpose**

Summarize how the CouplingNet projection and output space changed
while attempting to reduce the annulus transition error. This is a compact
method-history slide, not a new causal proof and not a complete account of every
training-loss experiment.

**Provisional title**

> Projection Strategies Tested for the Annulus Transition

**Provisional subtitle**

> Exact balance was preserved in every formulation, but the
> transition-localized error remained.

**Scope boundary**

Include only:

- the original physical-symmetric reference-response pipeline;
- geometry-weighted and response-preconditioned length-aware variants;
- response-space projection;
- the current physical-symmetric formulation;
- one compact note that transition-specific edge weighting was tested and
  removed.

Exclude from the slide, backup slides, and speaker notes:

- every pre-projection fuser variant;
- relative split consistency and weak operator closure;
- self-trace gluing and cross-axis carrier objectives;
- null-space, boundary-anchor, and canonical boundary-energy development;
- the \(H_0^1(\Omega)\)-assumption relaxation discussion;
- optimizer, source-count, and dataset-generation changes.

**Four-stage timeline**

Use a single horizontal timeline with four cards. Retired cards use a muted
neutral color; the current physical-symmetric card uses the only strong accent.

#### Stage 1: Original physical-symmetric projection

\[
(P_0,Q_0)
\overset{/L_x^2,/L_y^2}{\longrightarrow}
(p_0,q_0)
\overset{\phi+\psi=f}{\longrightarrow}
(\phi,\psi)
\overset{\times L_x^2,\times L_y^2}{\longrightarrow}
(\Phi,\Psi).
\]

- Projection is symmetric in physical directional-source space.
- Pointwise balance is exact.
- The transition seam remains after the reference-response pull-back.

#### Stage 2: Length-aware projection variants

Combine the geometry-weighted and response-preconditioned attempts in one card.
Do not put both complete derivations on the slide. Use the RPS difference update
as the representative formula:

\[
d_{\mathrm{RPS}}
=
d_0+\kappa d.
\]

The card must define every term, not only the final update:

\[
\sigma_x=L_x^2,
\qquad
\sigma_y=L_y^2,
\qquad
d=p-q,
\]

\[
d_0=
\frac{\sigma_y-\sigma_x}{\sigma_x+\sigma_y}f,
\qquad
\kappa=
\frac{4\sigma_x\sigma_y}{(\sigma_x+\sigma_y)^2}.
\]

Here, \(d_0\) is the equal-response baseline when the learned raw difference is
zero, and \(\kappa\in(0,1]\) attenuates that raw difference as the two line
response scales become imbalanced.

- Geometry-weighted direct/swapped rules redistributed the balanced split using
  \(L_x^2,L_y^2\).
- RPS attempted to provide the inverse-response compensation needed across the
  line-length jump.
- The `coupling4` diagnostic found partial compensation, but the strong
  transition contrast still appeared at the mandatory \(L^2\) pull-back.
- Status label: `tested / retired`.

Geometry-weighted post-hoc comparisons are not presented as decisive trained
ablations because their raw outputs came from checkpoints trained under another
projection rule.

#### Stage 3: Response-space projection

\[
\frac{\Phi}{L_x^2}
+
\frac{\Psi}{L_y^2}
=f.
\]

- The network predicted directional responses \(P,Q\), and projection was
  performed directly in response coordinates.
- No additional \(L^2\) multiplication followed projection.
- In the `coupling5` diagnostic, mean physical source errors decreased by about
  \(60\%\), and mean `rel_flux` changed from \(0.9996\) to \(0.4102\).
- Mean `rel_sol` did not improve: \(0.0856\rightarrow0.0868\).
- The remaining source error moved toward modes with greater elliptic inverse
  response, so a smaller source norm did not imply a smaller solution error.
- Status label: `tested / retired`.

These numbers compare existing physical-source and response-space runs, not a fully controlled
single-factor paired-seed ablation. Put this qualification in the speaker
notes.

#### Stage 4: Current physical-symmetric formulation

\[
p=\frac{P}{L_x^2},
\qquad
q=\frac{Q}{L_y^2},
\]

\[
\phi=\frac{f+p-q}{2},
\qquad
\psi=\frac{f-p+q}{2},
\]

\[
\Phi=L_x^2\phi,
\qquad
\Psi=L_y^2\psi.
\]

- The code returned to an explicit physical-source projection with an explicit
  reference-response pull-back.
- Pointwise cross-axis length context remains available to the shared
  transverse trunk.
- Retired response-space, RPS, and geometry-weighted modes are not active in
  the current formulation.
- Status label: `current`.

Do not explain the current training objective on this slide. Its purpose is to
close the projection-history loop, not to introduce the separate loss-history
discussion.

**Compact secondary note**

Place the following sentence in a narrow bottom strip or in speaker notes if
the slide becomes crowded:

> Transition-specific edge weighting was also tested, but local improvement
> did not translate into lower global solution error; it was removed.

Do not show the transition-weighted loss formula or any auxiliary loss metric.

**Visual composition**

```text
Original physical    Length-aware       Response-space      Current physical
symmetric            weighted / RPS     projection          physical symmetric
      |                    |                  |                    |
      v                    v                  v                    v
seam observed       partial correction   flux improved      retained baseline
                    seam remained         sol unchanged
```

- Use the same vertical grammar in all four cards: coordinate/length dependence,
  formulation content, observed outcome, and bottom-aligned status label. The
  length-aware card may use the extra height for the compact RPS definitions.
- Keep checkpoint names and detailed metric provenance in speaker notes.
- Connect the four cards with a single left-to-right arrow.
- Use no field heatmaps on this slide; Slide 1 already supplies spatial
  evidence.

**Takeaway**

> All projection variants enforced the source balance accurately. They changed
> how directional error was distributed, but none removed the transition
> pattern. Changing the projection space alone was therefore insufficient.

**Speaker emphasis**

- This is a record of hypotheses tested, not a claim that every change was an
  independently controlled ablation.
- Pointwise balance failure is not the explanation: every projection shown
  satisfies the intended balance constraint to numerical precision.
- Response-space projection improved source/flux metrics but not final solution
  error.
- The current physical-symmetric formulation was selected because its physical and reference
  roles are explicit; do not claim that it has already solved the transition
  error.

**Reveal plan**

1. Reveal the original physical-symmetric pipeline.
2. Add the length-aware geometry-weighted/RPS card.
3. Add the response-space card and its source/flux-versus-solution result.
4. Add the current physical-symmetric card and the final takeaway.

The final PDF state must show all four stages without relying on animation.

### Slide 5: Three Fixed Post-Hoc Cross-Axis Blend Rules

**Purpose**

Define the common post-reconstruction estimator once and separate the three
ways of constructing its partition-of-unity weights. This slide explains the
methods only; Slides 6-8 expand the three constructions and quantitative
Poisson/CDR evidence follows on Slides 9 and 10.

**Provisional title**

> Three Ways to Choose the Cross-Axis Reconstruction Weight

**Common estimator and invariants**

\[
u_{\mathrm{equal}}=\frac12(u_\phi+u_\psi),
\qquad
u_{\mathrm{blend}}=w_\phi u_\phi+w_\psi u_\psi,
\qquad
w_\phi+w_\psi=1.
\]

All three rules are applied after projection, pull-back, and directional Green
reconstruction:

\[
(\phi,\psi)
\longrightarrow
(\Phi,\Psi)
\longrightarrow
(u_\phi,u_\psi)
\longrightarrow
u_{\mathrm{blend}}.
\]

They leave \(\phi,\psi,u_\phi,u_\psi\), the exact source balance, GreenNet,
CouplingNet, and training unchanged. No rule uses `sol` or target `phi/psi` to
construct weights.

**Rule A: Geometry-only compact C2**

Known split/merge coordinates define transverse distances
\(d_\phi,d_\psi\). For ramp width \(\delta\), use

\[
B(s)=1-10s^3+15s^4-6s^5,
\quad 0\le s<1,
\qquad B(s)=0\quad(s\ge1),
\]

\[
\theta_{\mathrm{geom}}
=
\gamma\left[B(d_\psi/\delta)-B(d_\phi/\delta)\right],
\qquad
w_\phi=\frac{1+\theta_{\mathrm{geom}}}{2}.
\]

The rule is deterministic and sample-independent. It is exactly the equal mean
outside compact support, but it requires explicit topology-transition
metadata. The fixed comparison preset is \(\gamma=0.5\), \(\delta=4h\).

**Rule B: Mismatch-detected seam C2**

Use the frozen prediction mismatch

\[
m=u_\phi-u_\psi
\]

to build normalized horizontal- and vertical-edge jump profiles. Smooth each
one-dimensional profile, select at most two separated peaks per axis by
physical non-maximum suppression, and use only those detected coordinates as
the centers of the same compact C2 ramp. No known annulus transition
coordinate or segment length enters weight construction. The fixed comparison
preset is \(\gamma=0.3\), width \(12h\), one profile-smoothing step, and peak
threshold `0.25`.

**Rule C: Local weak-residual reliability**

For each frozen directional candidate, assemble the full local P1 weak defect

\[
R_\phi=R_x(u_\phi;\phi)+R_y(u_\phi;\psi),
\qquad
R_\psi=R_x(u_\psi;\phi)+R_y(u_\psi;\psi).
\]

With nodal mass \(m=m_x+m_y\), define

\[
\eta_\phi^2=\frac{R_\phi^2}{m+\varepsilon},
\qquad
\eta_\psi^2=\frac{R_\psi^2}{m+\varepsilon},
\]

\[
\theta_{\mathrm{weak}}
=
\gamma
\frac{\eta_\psi^2-\eta_\phi^2}
{\eta_\phi^2+\eta_\psi^2+2\,\mathrm{floor}},
\qquad
w_\phi=\frac{1+\theta_{\mathrm{weak}}}{2}.
\]

The rule uses predicted fields, projected directional sources, prescribed
coefficients, and axial P1 geometry. It uses no known transition coordinate
and requires no global matrix solve. This is a post-hoc reliability indicator,
not the excluded training-time weak-operator-closure loss. The fixed comparison
preset is \(\gamma=0.5\), two 50:50 graph-smoothing steps, and relative floor
`0.1`.

**Information contract**

| Rule | Known transition | Prediction-adaptive | Uses coefficients | Compact support |
| --- | ---: | ---: | ---: | ---: |
| Geometry C2 | yes | no | no | yes |
| Mismatch seam C2 | no | yes | no | yes |
| Weak residual | no | yes | yes | no |

The actual Slide 5 cards use the following three aligned rows so the comparison
can be read horizontally:

| Rule | Signal | Sample adaptive | Operator aware |
| --- | --- | ---: | ---: |
| Geometry C2 | known transition geometry | no | no |
| Mismatch seam C2 | edge jump of \(u_\phi-u_\psi\) | yes | no |
| Weak residual | local full-PDE defect | yes | yes |

**Visual composition**

Use three equal-width method cards under one common-estimator strip. Each card
contains one compact equation, the same three information rows, and a
bottom-aligned limitation pill. Add a full-width invariant band below the cards:

\[
(\phi,\psi,u_\phi,u_\psi)\ \text{remain frozen},
\qquad
u_{\mathrm{blend}}=w_\phi u_\phi+(1-w_\phi)u_\psi.
\]

Do not include result metrics or a large error heatmap on this slide.

**Takeaway**

> The three estimators share the same reconstruction contract; they differ only
> in whether trust comes from known topology, prediction mismatch, or local PDE
> consistency.

**Reveal plan**

1. Replace the equal mean with a partition-of-unity estimator.
2. Reveal geometry-only C2 and its topology requirement.
3. Add prediction-detected seam C2.
4. Add local weak-residual reliability and the common no-reference boundary.

### Slide 6: Geometry-Only Compact C2 Construction

**Purpose**

Explain how known split/merge topology is converted into a compact, smooth,
sample-independent partition weight. This slide contains no performance metric.

**Provisional title**

> Geometry-Only Compact C2: Encode Known Topology

**Construction**

Let \(\Gamma_\phi\) be the transverse coordinates where horizontal connected
interval multiplicity changes and \(\Gamma_\psi\) the corresponding vertical
set. Define

\[
d_\phi(x,y)=\operatorname{dist}(y,\Gamma_\phi),
\qquad
d_\psi(x,y)=\operatorname{dist}(x,\Gamma_\psi).
\]

Use the compact quintic bump

\[
B(s)=
\begin{cases}
1-10s^3+15s^4-6s^5,&0\le s<1,\\
0,&s\ge1,
\end{cases}
\]

and form

\[
\theta_{\mathrm{geom}}
=\gamma\left[B(d_\psi/\delta)-B(d_\phi/\delta)\right],
\qquad
w_\phi=\frac{1+\theta_{\mathrm{geom}}}{2},
\qquad
w_\psi=1-w_\phi.
\]

Because \(B(1)=B'(1)=B''(1)=0\), the correction joins the equal mean with C2
regularity at the support edge. The locked diagnostic preset is
\(\gamma=0.5\), \(\delta=0.03125=4h\), giving
\(w_\phi,w_\psi\in[0.25,0.75]\).

**Visual composition**

- Top invariant strip: known topology only, exact partition, sample independent.
- Left formula card: distances, compact bump, and weight equation.
- Right Plotly panel: \(B(d_\phi/\delta)\),
  \(B(d_\psi/\delta)\), \(\theta_{\mathrm{geom}}\), and \(w_\phi\).
- Bottom boundary: deterministic and exactly equal mean outside support, but
  explicit transition metadata is required.

**Locked artifact source**

- `checkpoints/Annulus_poisson/coupling15/compact_c2_cross_axis_blend/data/selected_fixed_smooth_blend_arrays.npz`
- `checkpoints/Annulus_poisson/coupling15/compact_c2_cross_axis_blend/summary.json`
- Presentation asset: `geometry_c2_method_sample0.html`

**Reveal plan**

1. Define transition families, distances, and the compact quintic bump.
2. Reveal influence fields, signed correction, and final weight map.
3. State the C2 support-edge guarantee and known-topology limitation.

### Slide 7: Mismatch-Detected Seam C2 Construction

**Purpose**

Show how prediction disagreement detects seam location and orientation while a
separate fixed C2 ramp controls the final weight shape. This separation prevents
raw mismatch noise from becoming the weight directly.

**Provisional title**

> Mismatch-Detected Seam C2: Detect, Then Blend

**Construction**

For the frozen reconstructions define

\[
m=u_\phi-u_\psi,
\qquad
\sigma_m=\left(\frac1N\sum_p m(p)^2\right)^{1/2}.
\]

On each existing axial edge \(e=(p_i,p_j)\), use

\[
J_e=
\frac{|m(p_j)-m(p_i)|}{\sigma_m+\varepsilon}
\frac{h_{\mathrm{ref}}}{\|p_j-p_i\|}.
\]

Aggregate edge jumps into axis profiles,

\[
P_x(k)=\left(\frac1{|E_x(k)|}\sum_{e\in E_x(k)}J_e^2\right)^{1/2},
\qquad
P_y(k)=\left(\frac1{|E_y(k)|}\sum_{e\in E_y(k)}J_e^2\right)^{1/2},
\]

then smooth and apply physical non-maximum suppression:

\[
\widehat\Gamma_x=\operatorname{NMS}(\mathcal S P_x),
\qquad
\widehat\Gamma_y=\operatorname{NMS}(\mathcal S P_y).
\]

Only the detected coordinates are transferred to the compact C2 profile:

\[
\theta_{\mathrm{seam}}=\gamma\left[
B\!\left(\frac{\operatorname{dist}(x,\widehat\Gamma_x)}{\delta}\right)
-B\!\left(\frac{\operatorname{dist}(y,\widehat\Gamma_y)}{\delta}\right)
\right],
\qquad
w_\phi=\frac{1+\theta_{\mathrm{seam}}}{2}.
\]

The locked preset is \(\gamma=0.3\), \(\delta=0.09375=12h\), one profile
smoothing step, relative peak threshold `0.25`, and at most two separated seams
per axis.

**Visual composition**

- Top invariant strip: prediction mismatch only, detect first, fixed smooth
  profile second, sample adaptive.
- Left formula card: normalized edge jumps, one-dimensional profiles, NMS.
- Right Plotly panel for frozen Poisson sample 0: mismatch field, both profiles
  with detected peaks, \(\theta_{\mathrm{seam}}\), and \(w_\phi\).
- Bottom boundary: no known topology or coefficient is required, but common
  candidate bias is invisible and oscillatory mismatch may create false seams.

**Locked artifact source**

- `checkpoints/Annulus_poisson/coupling15/weak_residual_reliability_blend_comparison/data/selected_weak_residual_blend_arrays.npz`
- `data/geometry/annulus_02_05_1_128.npz`
- Existing deterministic mismatch/seam helper; no model checkpoint is loaded.
- Presentation asset: `mismatch_seam_c2_method_sample0.html`

**Reveal plan**

1. Define normalized mismatch jumps and axis profiles.
2. Reveal detected seams, compact correction, and final weight map.
3. Separate disagreement localization from correctness estimation.

### Slide 8: Local Weak-Residual Reliability Construction

**Purpose**

Explain how each directional reconstruction is tested against the complete
split PDE and how the lower local defect receives more reconstruction weight.

**Provisional title**

> Local Weak-Residual Reliability: Trust the Better PDE Candidate

**Directional weak form**

For \(s\in\{x,y\}\),

\[
B_s(v,\chi)=\int_{\ell_s}
\left(a\,v_s\chi_s+b_s v_s\chi+\frac12c\,v\chi\right)\,ds,
\]

\[
R_s(v;\zeta_s)_i=B_s(v,\chi_i)-\langle\zeta_s,\chi_i\rangle,
\qquad
R(v)=R_x(v;\phi)+R_y(v;\psi).
\]

Both \(u_\phi\) and \(u_\psi\) are tested against both directional equations.
With \(m=m_x+m_y\), graph smoothing \(\mathcal S_{\mathrm{graph}}\), and
sample-relative floor \(\tau\),

\[
\eta_v^2=\mathcal S_{\mathrm{graph}}
\left(\frac{R(v)^2}{m+\varepsilon}\right),
\qquad v\in\{u_\phi,u_\psi\},
\]

\[
\tau=r_{\mathrm{floor}}
\frac{\langle\eta_\phi^2\rangle+\langle\eta_\psi^2\rangle}{2}
+\varepsilon,
\]

\[
\theta_{\mathrm{weak}}
=\gamma\frac{\eta_\psi^2-\eta_\phi^2}
{\eta_\phi^2+\eta_\psi^2+2\tau},
\qquad
w_\phi=\frac{1+\theta_{\mathrm{weak}}}{2}.
\]

The locked preset is \(\gamma=0.5\), two 50:50 graph-smoothing steps, and
`relative_floor=0.1`. Smaller \(\eta_\phi\) increases \(w_\phi\).

**Visual composition**

- Top invariant strip: full \(a,\mathbf b,c\) operator, axial P1
  gather/scatter, no reference target or global solve.
- Left formula card: directional bilinear form, full candidate residual, and
  mass-normalized local indicator.
- Right Plotly panel for frozen Poisson sample 0: log-displayed
  \(\eta_\phi^2\), \(\eta_\psi^2\), signed reliability, and \(w_\phi\).
- Bottom boundary: local PDE consistency is a reliability heuristic, not an
  a posteriori pointwise error bound; axial striping may remain.

**Locked artifact source**

- `checkpoints/Annulus_poisson/coupling15/weak_residual_reliability_blend_comparison/data/selected_weak_residual_blend_arrays.npz`
- `checkpoints/Annulus_poisson/coupling15/weak_residual_reliability_blend_comparison/summary.json`
- Presentation asset: `weak_residual_reliability_method_sample0.html`
- The frozen comparison and the optional production reconstruction now call
  the same local weak-residual reliability core. The slide keeps the frozen
  numerical artifact unchanged; this note prevents future drift between the
  displayed equation and the evaluator implementation.

**Reveal plan**

1. Define the complete directional weak residual for each candidate.
2. Reveal smoothed local indicators, signed reliability, and final weight map.
3. State the no-reference/no-global-solve contract and heuristic limitation.

### Slide 9: Poisson Geometry-First Comparison

**Purpose**

Evaluate all three fixed rules on Pure Poisson first, so convection and reaction
cannot confound the interpretation of the Annulus transition behavior.

**Provisional title**

> Poisson: Local PDE Reliability Gives the Strongest Post-Hoc Improvement

**Question**

> When coefficient complexity is removed, which final reconstruction blend
> best reduces global and transition-localized error?

**Frozen evaluation contract**

- Coupling checkpoint:
  `checkpoints/Annulus_poisson/coupling15/complex_coupling_model_best_energy.safetensors`
- Green checkpoint: `checkpoints/Annulus_poisson/green/model.safetensors`
- PDE coefficients: `coefficients/Pure_Poisson.py`
- Test set: 50 full-reference samples
- Training/checkpoint changes: none
- Poisson parameter sweep: disabled
- `sol` use: evaluation metrics only after every weight is fixed

**Four-estimator result**

| Estimator | Mean `rel_sol` | Change vs equal | Wins | Transition RMS | Trace-jump RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal mean | 5.569825% | baseline | 0/50 | baseline | baseline |
| Geometry-only C2 | 5.470165% | -1.789% | 35/50 | -3.412% | -45.308% |
| Mismatch seam C2 | 5.312818% | -4.614% | 44/50 | -6.954% | -23.553% |
| **Local weak residual** | **4.860579%** | **-12.734%** | **50/50** | **-8.360%** | **-48.280%** |

The paired-bootstrap 95% intervals for the weak rule are
`[-13.923%,-11.551%]` for mean `rel_sol`, `[-11.027%,-5.487%]` for transition
RMS, and `[-48.996%,-47.512%]` for trace-jump RMS.

**Locked evidence source**

- Artifact root:
  `checkpoints/Annulus_poisson/coupling15/weak_residual_reliability_blend_comparison`
- Aggregate figure: `figures/aggregate/four_way_rel_sol.html`
- Static fallback: `figures/aggregate/four_way_rel_sol.pdf`
- Full metrics: `metrics/per_sample_weak_residual_blend_comparison.csv`
- Selected audit arrays: `data/selected_weak_residual_blend_arrays.npz`
- Representative maximum-error sample: `sample_0047`

**Visual composition**

- Left two thirds: the existing four-way paired `rel_sol` scatter. Preserve the
  no-change diagonal and estimator colors.
- Right upper: four-row metric strip with `-12.734%` and `50/50` emphasized.
- Right lower: a compact sample-47 inset containing weak \(w_\phi\), equal-mean
  signed error, and weak-blend signed error. Do not place the complete 12-panel
  diagnostic on the slide.
- Status label: `frozen Poisson checkpoint / fixed presets / no retraining`.

**Interpretation boundary and preset provenance**

- Geometry C2 gives the most controlled topology-specific correction and
  suppresses trace jumps on all 50 samples.
- Mismatch seam C2 improves global and broad transition error more than
  geometry C2, but suppresses the trace jump less strongly.
- Weak residual is best on every Poisson sample, but its weights have axial
  stripe structure and maximum neighboring jump `0.302343`; metric improvement
  is not a proof of transverse regularity.
- The fixed presets originated in the prior CDR exploratory diagnostics and
  were transferred to Poisson without retuning. Poisson is shown first for
  conceptual clarity, not as the chronological tuning order.

**Takeaway**

> In the coefficient-simple problem, all three rules reduce the Annulus error,
> and local weak-residual reliability gives the strongest global result.

**Reveal plan**

1. Establish the frozen Poisson/no-retraining contract.
2. Reveal Geometry C2 and mismatch seam C2 relative to equal mean.
3. Reveal the weak-residual result and `50/50` wins.
4. End with the striped-weight and preset-provenance limitations.

### Slide 10: CDR Broader-Operator Consistency Check

**Purpose**

Show whether the Poisson estimator ordering persists when variable diffusion,
rotational convection, and reaction are present on the same Annulus geometry.

**Provisional title**

> CDR: The Same Estimator Ordering Persists

**Problem context**

\[
-\nabla\!\cdot(a\nabla u)+\mathbf b\!\cdot\nabla u+c\,u=f
\]

uses the Annulus CDR coefficient family and an independent indexed-GP source
set. The comparison remains post-hoc and changes only the final estimator.

**CDR four-estimator result**

| Estimator | Mean `rel_sol` | Change vs equal | Wins | Transition RMS | Trace-jump RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal mean | 5.156246% | baseline | 0/50 | baseline | baseline |
| Geometry-only C2 | 5.054177% | -1.980% | 36/50 | -4.225% | -44.956% |
| Mismatch seam C2 | 4.943157% | -4.133% | 41/50 | -7.391% | -22.425% |
| **Local weak residual** | **4.564722%** | **-11.472%** | **49/50** | **-9.870%** | **-48.785%** |

**Cross-equation comparison**

| Rule | Poisson `rel_sol` change | CDR `rel_sol` change |
| --- | ---: | ---: |
| Geometry C2 | -1.789% | -1.980% |
| Mismatch seam C2 | -4.614% | -4.133% |
| Weak residual | -12.734% | -11.472% |

Both equations give the same ordering:

\[
\text{equal mean}
\;>\;
\text{geometry C2}
\;>\;
\text{mismatch seam C2}
\;>\;
\text{weak residual},
\]

where `>` means larger mean relative solution error.

**Locked evidence source**

- Artifact root:
  `checkpoints/annulus_CDR/coupling5/weak_residual_reliability_blend_comparison`
- Aggregate figure: `figures/aggregate/four_way_rel_sol.html`
- Static fallback: `figures/aggregate/four_way_rel_sol.pdf`
- Full metrics: `metrics/per_sample_weak_residual_blend_comparison.csv`
- Selected audit arrays: `data/selected_weak_residual_blend_arrays.npz`

**Visual composition**

- Main visual: a three-row Poisson-versus-CDR grouped bar chart of relative
  `rel_sol` changes. Keep zero visible and use one consistent color per rule.
- Side card: the common estimator-order arrow and the CDR `49/50` weak wins.
- Bottom strip: transition RMS and trace-jump changes for both equations.
- Do not repeat the full CDR paired scatter unless requested during discussion;
  keep its HTML available for interactive inspection.
- Status label: `same Annulus geometry / independent sources / cross-equation`.

**Interpretation boundary**

- Poisson and CDR use independent source realizations but the same Annulus
  geometry; this is cross-equation consistency, not cross-domain validation.
- CDR supplied the exploratory preset provenance, so this slide is not an
  independently tuned validation set and the sweep must not be shown as model
  selection.
- The CDR weak weights remain axially striped with maximum neighboring jump
  `0.315646`.
- Disk, multi-hole, nonconvex, and grid-refinement tests remain required before
  production integration.

**Takeaway**

> The Poisson ranking survives coefficient complexity, supporting a general
> PDE-reliability signal on this Annulus geometry without yet proving
> cross-domain generalization.

**Reveal plan**

1. Add the CDR operator terms to the same Annulus geometry.
2. Reveal the CDR four-estimator table.
3. Place Poisson and CDR relative changes side by side.
4. End with the same-geometry and calibration-provenance limitations.

### Slide 11: Poisson Directional Sources and Weak-Residual Reconstruction

**Purpose**

Present the current Pure Poisson calculation as a numerical result rather than
as another transition diagnosis. Show the known source, the balanced physical
directional sources, both axial Green reconstructions, and the final local
weak-residual reliability blend for one representative test sample.

**Provisional title**

> Pure Poisson: Directional Sources and Weak-Residual Reconstruction

**Result contract**

The final estimator is

\[
u_{\mathrm{pred}}^{\mathrm{weak}}
=
w_\phi^{\mathrm{weak}}u_\phi
+
\left(1-w_\phi^{\mathrm{weak}}\right)u_\psi.
\]

The weak reliability rule changes only the final reconstruction blend. It does
not change the frozen-checkpoint fields \(\phi,\psi,u_\phi,u_\psi\), and it does
not read `sol`, `target_phi`, or `target_psi` when constructing the weight.

**Locked artifact sources**

- Standard artifact:
  `checkpoints/Annulus_poisson/coupling15/artifacts/data/selected_raw_arrays.npz`
- Weak-comparison artifact:
  `checkpoints/Annulus_poisson/coupling15/weak_residual_reliability_blend_comparison/data/selected_weak_residual_blend_arrays.npz`
- Selected representative: sample 0, the existing selected q50-role sample
- Standard artifact prefix: `sample_0000_sample_000000`
- Final weak prediction: `u_weak_residual_reliability` at selected sample id 0

The builder must not use the standard artifact `u_pred` as the final result;
that field is \(0.5(u_\phi+u_\psi)\), not the weak-residual blend.

**Field composition**

- Problem strip:

  \[
  -\Delta u=f,
  \qquad
  a=1,
  \quad
  \mathbf b=0,
  \quad
  c=0.
  \]

- Directional-source row: \(f\), \(\phi\), \(\psi\).
- Solution row: reference \(u\), \(u_\phi\), \(u_\psi\), and
  \(u_{\mathrm{pred}}^{\mathrm{weak}}\).
- Small method inset: \(w_\phi^{\mathrm{weak}}\), with
  \(w_\psi^{\mathrm{weak}}=1-w_\phi^{\mathrm{weak}}\).

**Representative and aggregate metrics**

For sample 0:

\[
\operatorname{rel\_sol}(u_\phi)=12.465237\%,
\qquad
\operatorname{rel\_sol}(u_\psi)=8.686395\%,
\]

\[
\operatorname{rel\_sol}
\left(u_{\mathrm{pred}}^{\mathrm{weak}}\right)
=4.824260\%,
\qquad
\|\phi+\psi-f\|_\infty=2.22\times10^{-16}.
\]

Across all 50 Poisson test samples, weak `rel_sol` has mean `4.860579%` and
median `4.822652%`. Put the sample-specific values in the main metric card and
the aggregate values in a narrow footer strip.

**Visual composition**

- Use a compact three-panel source row above a four-panel solution row.
- Give all source panels the same physical extent, but let \(f\) use its own
  scalar range while \(\phi\) and \(\psi\) share a source-component range.
- Make reference \(u\), \(u_\phi\), \(u_\psi\), and weak
  \(u_{\mathrm{pred}}\) share one solution color range.
- Use one colorbar per row rather than one per panel.
- Place the shared `f` and `u` colorbar titles to the right of the bar, with
  explicit title/tick font sizes, bar thickness, horizontal padding, and figure
  right margin. Apply the same layout to the CDR result-field figure.
- Put the weak-prediction equation in a dedicated sidebar card and break it into
  two aligned lines so its KaTeX and MathML boxes remain fully inside the card
  at both 1600x900 and 1280x720.
- Do not draw transition lines, cardinal markers, segment-length maps, or
  problem-diagnosis callouts.
- Status label:
  `frozen checkpoint / post-hoc final blend / no retraining`.

**Takeaway**

> The projected directional sources satisfy pointwise balance, and local weak
> reliability combines the two complementary axial reconstructions into the
> reported Poisson solution.

**Reveal plan**

1. Introduce the Poisson coefficient-simple context and source split.
2. Reveal the two directional Green reconstructions.
3. Add the weak reliability weight and final prediction.
4. Reveal representative and 50-sample metrics.

### Slide 12: Poisson Signed Errors and Accuracy

**Purpose**

Report the directional-source diagnostics and solution-reconstruction errors
for the same sample without returning to the transition-causality narrative.

**Provisional title**

> Pure Poisson: Signed Errors and Test-Set Accuracy

**Error fields**

Use signed differences throughout:

\[
e_\phi=\phi-\phi_*,
\qquad
e_\psi=\psi-\psi_*,
\]

\[
e_{u_\phi}=u_\phi-u,
\qquad
e_{u_\psi}=u_\psi-u,
\qquad
e_{u_{\mathrm{pred}}}
=u_{\mathrm{pred}}^{\mathrm{weak}}-u.
\]

The FEniCSx \(\phi_*,\psi_*\) fields are optional numerical directional
targets, not exact analytical balance labels. Use the visible qualifier
`against numerical FEniCSx directional targets`, and keep the full target
balance explanation in speaker notes.

**Representative metrics**

For sample 0:

| Field | Error RMS | Relative diagnostic |
| --- | ---: | ---: |
| \(\phi-\phi_*\) | `7.224871e-2` | `14.910623%` |
| \(\psi-\psi_*\) | `7.269581e-2` | `21.885865%` |
| \(u_\phi-u\) | `5.141676e-4` | `12.465237%` |
| \(u_\psi-u\) | `3.582975e-4` | `8.686395%` |
| \(u_{\mathrm{pred}}^{\mathrm{weak}}-u\) | `1.989916e-4` | `4.824260%` |

The standard combined directional-source metric is
`rel_flux=18.398244%` for sample 0 and has 50-sample mean `17.098821%`. Label
`rel_flux` as a directional-source diagnostic; weak blending does not change
it because \(\phi\) and \(\psi\) are unchanged.

**Visual composition**

- Top row: \(e_\phi\), \(e_\psi\).
- Bottom row: \(e_{u_\phi}\), \(e_{u_\psi}\),
  \(e_{u_{\mathrm{pred}}}\).
- Use one zero-centered symmetric `RdBu` range for the two source errors and a
  separate shared zero-centered symmetric range for all three solution errors.
- Put RMS values in compact panel subtitles and relative metrics in one side
  card; do not repeat five colorbars.
- Do not emphasize transition zones or attach causal annotations.

**Takeaway**

> The two directional solutions have different signed-error structures, while
> their weak-residual reliability blend gives the reported final Poisson
> accuracy.

**Reveal plan**

1. Show both directional-source diagnostic errors together.
2. Reveal the two directional solution errors on one shared scale.
3. Add the final weak-prediction error and metric card.

### Slide 13: CDR Coefficients, Sources, and Reconstruction

**Purpose**

Present the corresponding result for the broader convection-diffusion-reaction
operator, including the prescribed physical coefficients that define the
frozen Green and Coupling problem.

**Provisional title**

> CDR: Physical Coefficients, Directional Sources, and Reconstruction

**Problem context**

\[
-\nabla\!\cdot(a\nabla u)
+\mathbf b\!\cdot\nabla u
+c\,u=f.
\]

Coefficient figures are sample-independent physical fields evaluated directly
at `coords_valid`. They are not branch-interpolated, pulled back, or scaled by
segment length.

**Locked artifact sources**

- Standard artifact:
  `checkpoints/annulus_CDR/coupling5/artifacts/data/selected_raw_arrays.npz`
- Coefficient artifact:
  `checkpoints/annulus_CDR/coupling5/artifacts/data/coefficient_fields.npz`
- Weak-comparison artifact:
  `checkpoints/annulus_CDR/coupling5/weak_residual_reliability_blend_comparison/data/selected_weak_residual_blend_arrays.npz`
- Selected representative: sample 9, the existing selected q50-role sample
- Standard artifact prefix: `sample_0009_sample_000009`
- Final weak prediction: `u_weak_residual_reliability` at selected sample id 9

As on Slide 11, do not substitute the standard equal-mean artifact `u_pred` for
the weak result.

**Field composition**

- Narrow coefficient strip: diffusion \(a(x,y)\), convection vector
  \(\mathbf b(x,y)\), and reaction \(c(x,y)\).
- Render convection as a \(|\mathbf b|\) scalar background with deterministic
  quiver arrows. Keep separate \(b_x,b_y\) scalar fields for backup only.
- Directional-source row: \(f\), \(\phi\), \(\psi\).
- Solution row: reference \(u\), \(u_\phi\), \(u_\psi\), and
  \(u_{\mathrm{pred}}^{\mathrm{weak}}\).
- Small method inset: \(w_\phi^{\mathrm{weak}}\).

**Representative and aggregate metrics**

For sample 9:

\[
\operatorname{rel\_sol}(u_\phi)=11.989277\%,
\qquad
\operatorname{rel\_sol}(u_\psi)=10.485543\%,
\]

\[
\operatorname{rel\_sol}
\left(u_{\mathrm{pred}}^{\mathrm{weak}}\right)
=4.573527\%,
\qquad
\|\phi+\psi-f\|_\infty=4.02\times10^{-16}.
\]

Across all 50 CDR test samples, weak `rel_sol` has mean `4.564722%` and median
`4.205909%`.

**Visual composition**

- Treat the three coefficient plots as a shallow context strip rather than as
  three equal-sized dominant panels.
- Use `Viridis` for \(a\), \(c\), and \(|\mathbf b|\), with arrows over the
  convection-magnitude background.
- Match the source and solution panel contracts from Slide 11 exactly.
- Use two reveal states if needed for legibility: coefficient/source context
  first, then reference/directional/final solutions. The PDF fallback must
  preserve both states as readable pages rather than shrinking all panels.
- Do not add transition markers or compare coefficient complexity to an error
  mechanism on this slide.

**Takeaway**

> The same directional reconstruction and local-reliability combination
> produces the reported solution under variable diffusion, rotational
> convection, and reaction.

**Reveal plan**

1. Establish the physical coefficient fields.
2. Reveal the CDR source and balanced directional components.
3. Reveal both Green reconstructions and the weak final prediction.
4. Add representative and aggregate metrics.

### Slide 14: CDR Signed Errors and Accuracy

**Purpose**

Complete the CDR result with the same signed-error and metric grammar used for
Poisson, so the two equation families can be read without changing conventions.

**Provisional title**

> CDR: Signed Errors and Test-Set Accuracy

**Error fields**

Show the same five signed differences as Slide 12:

\[
\phi-\phi_*,
\qquad
\psi-\psi_*,
\qquad
u_\phi-u,
\qquad
u_\psi-u,
\qquad
u_{\mathrm{pred}}^{\mathrm{weak}}-u.
\]

Use the same FEniCSx directional-target qualifier and keep target/reference
fields evaluation-only.

**Representative metrics**

For sample 9:

| Field | Error RMS | Relative diagnostic |
| --- | ---: | ---: |
| \(\phi-\phi_*\) | `1.212461e-1` | `18.586352%` |
| \(\psi-\psi_*\) | `1.216139e-1` | `21.708287%` |
| \(u_\phi-u\) | `6.756257e-4` | `11.989277%` |
| \(u_\psi-u\) | `5.908865e-4` | `10.485543%` |
| \(u_{\mathrm{pred}}^{\mathrm{weak}}-u\) | `2.577296e-4` | `4.573527%` |

The standard combined directional-source metric is
`rel_flux=20.147319%` for sample 9 and has 50-sample mean `17.898558%`.

**Visual composition**

- Reuse the exact Slide 12 panel order, row-level shared scales, aspect ratio,
  typography, and metric-card placement.
- Determine CDR error limits independently from Poisson; do not force one
  cross-equation color range when the source realizations are independent.
- Keep the equation label and coefficient family in a small header strip, not
  inside individual panels.
- Avoid transition-zone statistics, trace-jump metrics, and diagnostic markers.

**Takeaway**

> Under the broader CDR operator, the two directional reconstructions remain
> complementary and their local weak-residual blend gives the reported final
> solution accuracy.

**Reveal plan**

1. Show the CDR directional-source diagnostic errors.
2. Reveal both directional solution errors.
3. Add the final weak-prediction error and the representative/full-test metric
   card.

### Slide 15: Exact Balance Does Not Imply Directional Response Consistency

**Purpose**

Reframe symmetric projection correctly. It is an exact Euclidean projection to
the physical-source balance plane, not a failed approximation. What it does not
determine is which feasible point on that plane makes the two directional Green
responses consistent.

**Title**

> Exact Balance Does Not Imply Directional Response Consistency

**Physical raw proposals and balance plane**

Convert the network responses to physical raw proposals,

\[
p=\frac{P}{L_x^2},
\qquad
q=\frac{Q}{L_y^2},
\qquad
d=p-q.
\]

The pointwise affine balance plane is

\[
\mathcal C_f=\{(\phi,\psi):\phi+\psi=f\}.
\]

The symmetric-balanced pair is

\[
\widetilde\phi=\frac{f+d}{2},
\qquad
\widetilde\psi=\frac{f-d}{2},
\]

so

\[
\widetilde\phi+\widetilde\psi=f,
\qquad
\widetilde\phi-\widetilde\psi=d.
\]

The common direction \((1,1)\) is normal to \(\mathcal C_f\), while the
difference direction \((1,-1)\) is tangent to it. Symmetric projection removes
the common-mode balance defect and preserves the raw difference mode.

**Directional response mismatch**

With the existing frozen directional response operators,

\[
\widetilde u_\phi=H_x\widetilde\phi,
\qquad
\widetilde u_\psi=H_y\widetilde\psi,
\]

\[
m_0=H_x\widetilde\phi-H_y\widetilde\psi.
\]

Here \(H_s=K_sW_sL_s^2\) contains the Green kernel, prescribed coefficient,
quadrature, evaluation position, and segment length. Keep these five factors
as the existing balanced 3+2 block grid.

Exact balance does not imply response consistency:

\[
\widetilde\phi+\widetilde\psi=f
\not\Longrightarrow
H_x\widetilde\phi=H_y\widetilde\psi.
\]

The existing canonical energy and optimizer already try to reduce directional
mismatch while training. The evaluated fixed tangent step makes one
response-informed movement explicit without leaving the feasible plane.

**Visual composition**

- Left card: physical raw proposals, \(\mathcal C_f\), symmetric projection,
  normal direction, and tangent direction.
- Center card: \(\widetilde u_\phi,\widetilde u_\psi,m_0\) and the five response
  factors.
- Right card: exact-balance implication that does not follow, plus the distinct
  roles of projection and canonical-energy training.
- Bottom takeaway: feasibility and response consistency are separate
  requirements.

**Takeaway**

> Exact source balance selects a feasible split, but not necessarily the
> response-consistent point on the balance plane.

**Reveal plan**

1. Start with the physical proposals and symmetric orthogonal projection.
2. Reveal the directional response operators and mismatch \(m_0\).
3. Reveal the remaining tangent freedom and the role of existing energy
   training.

### Slide 16: Tangent Correction Within the Balance Plane

**Purpose**

Start from the exact-balanced symmetric pair and take one fixed,
differentiable, response-gradient step in the feasible tangent direction. Do
not reintroduce the assumption that raw \(p,q\) are individually reliable
directional-source candidates.

**Title**

> Tangent Green-Response Correction Within the Balance Plane

**Feasible tangent family**

Parameterize every candidate by

\[
\phi(\delta)=\widetilde\phi+\delta,
\qquad
\psi(\delta)=\widetilde\psi-\delta.
\]

Then

\[
\phi(\delta)+\psi(\delta)=f
\qquad\text{for every }\delta.
\]

The update therefore changes only the balance-plane difference mode.

**Directional response objective and gradient**

Define

\[
J(\delta)
=
\frac12
\left\|
H_x(\widetilde\phi+\delta)
-H_y(\widetilde\psi-\delta)
\right\|_{M_\Omega}^2
=
\frac12\left\|m_0+(H_x+H_y)\delta\right\|_{M_\Omega}^2.
\]

At \(\delta=0\),

\[
g=\nabla J(0)=(H_x+H_y)^\top M_\Omega m_0.
\]

The forward actions construct \(m_0\), and the transpose action asks how each
source-point tangent change affects the full directional response mismatch.

**Fixed column-diagonal Jacobi preconditioner**

Reuse the fixed source-column gains only as a preconditioner:

\[
\gamma_{s,j}^2
=
\left[H_s^\top M_\Omega H_s\right]_{jj},
\qquad
G_j=\gamma_{x,j}^2+\gamma_{y,j}^2,
\]

\[
\overline G=\frac1N\sum_jG_j,
\qquad
D_j
=
G_j+(\lambda_{\mathrm{rel}}+\varepsilon_{\mathrm{rel}})\overline G.
\]

For the two displayed runs, use the fixed setting

\[
\eta=0.01,
\qquad
\lambda_{\mathrm{rel}}=0.01,
\]

and take one step,

\[
\delta_j=-\eta\frac{g_j}{D_j},
\qquad
\phi=\widetilde\phi+\delta,
\qquad
\psi=\widetilde\psi-\delta.
\]

The column diagonal scales the tangent gradient. It does not allocate the raw
balance residual \(f-p-q\), and there is no opposite-gain closed form.

**Local descent interpretation**

For \(v=-D^{-1}g\),

\[
\left.\frac{d}{d\epsilon}J(\epsilon v)\right|_{\epsilon=0}
=-g^\top D^{-1}g\leq0.
\]

This proves a local descent direction at the symmetric pair. The actual fixed
step is audited sample by sample with

\[
\rho_b
=
\frac{\|m_{\mathrm{post},b}\|_{M_\Omega}}
{\|m_{\mathrm{pre},b}\|_{M_\Omega}}.
\]

The condition \(\rho_b<1\) means that the finite tangent step reduced the
directly optimized response mismatch for sample \(b\). It does not mean that
the solution error decreased by the same fraction.

**Method boundaries**

- The symmetric-balanced source is the only tangent base.
- \(\eta=0.01\), \(\lambda_{\mathrm{rel}}=0.01\), and
  \(\varepsilon_{\mathrm{rel}}\) are fixed config scalars in the displayed
  Poisson/CDR runs.
- Geometry, prescribed coefficients, quadrature, and the frozen GreenNet are
  allowed; reference `sol/phi/psi` targets are not.
- Forward and transpose response actions are segment-local and differentiable.
- Learned step sizes, learned gates, row norms, global response matrices,
  full-Gram solves, and new loss terms are excluded.
- The tangent correction runs before the unchanged canonical energy and
  SOAP/AdamW optimization.
- \(\eta=0\) is exactly the symmetric-projection ablation.

**Visual composition**

- Top-left card: the symmetric-balanced tangent family and exact balance.
- Top-right card: fixed/reference-free/no-solve information contract.
- Middle-left card: \(m_0\), \(J(\delta)\), and \(g\).
- Middle-right card: \(\gamma_s^2\), \(D\), and the fixed tangent step.
- Bottom strip: exact balance, local descent direction, and unchanged training
  pipeline.

**Takeaway**

> First enforce balance exactly, then take one response-informed feasible
> descent step before the existing energy optimization.

**Reveal plan**

1. Show the symmetric-balanced tangent family and information contract.
2. Reveal the response mismatch objective and tangent gradient.
3. Reveal the fixed column-diagonal Jacobi denominator and update.
4. Reveal exact balance, local descent interpretation, the \(\rho_b\) audit,
   and the unchanged downstream training path.

### Slide 17: Poisson Fixed Tangent Result

**Title**

> Poisson: Fixed Tangent Correction Reduces Response Mismatch

**Frozen evidence**

- Artifact root: `checkpoints/Annulus_poisson/coupling18/artifacts`
- Representative role: q50, sample 41
- Fixed setting: \(\eta=0.01\), \(\lambda_{\mathrm{rel}}=0.01\)
- Presentation asset: `poisson_tangent_result_q50.html`

Use a 2x2 Plotly composite:

1. \(m_{\mathrm{pre}}\) on q50 sample 41.
2. \(m_{\mathrm{post}}\) on the same sample and the exact same zero-centered
   color range as panel 1.
3. Final signed \(u_{\mathrm{pred}}-u\) on an independent zero-centered color
   range.
4. All 50 per-sample post-versus-pre mismatch RMS values with the \(y=x\)
   reference line.

**Locked metrics**

- Mean \(\rho_b=0.350216\), hence mean response-mismatch reduction `65.0%`.
- `50/50` samples satisfy \(\rho_b<1\).
- Mean tangent correction / symmetric pair: `5.390%`.
- Mean equal-mean `rel_sol`: `3.660%`.
- Mean weak-final `rel_sol`: `3.405%`.
- Mean `rel_flux`: `12.487%`.

The 65.0% claim refers only to response mismatch. The `rel_sol` values describe
this trained pipeline and do not provide a causal tangent-versus-symmetric
comparison.

**Reveal plan**

1. Define the pre/post field comparison and \(\rho_b\).
2. Reveal the 50-sample reduction and tangent-correction magnitude.
3. Reveal final evaluation metrics and the causal-comparison limitation.

### Slide 18: CDR Fixed Tangent Result

**Title**

> CDR: The Same Tangent Mechanism Persists with Variable Coefficients

**Frozen evidence**

- Artifact root: `checkpoints/annulus_CDR/coupling8/artifacts`
- Representative role: q50, sample 33
- Fixed setting: \(\eta=0.01\), \(\lambda_{\mathrm{rel}}=0.01\)
- Presentation asset: `cdr_tangent_result_q50.html`

Use the same 2x2 panel order and visual grammar as Slide 17. The pre/post
mismatch panels share one CDR-specific zero-centered color range. Poisson and
CDR do not share color limits because they use different PDEs and independent
source realizations.

**Locked metrics**

- Mean \(\rho_b=0.370161\), hence mean response-mismatch reduction `63.0%`.
- `50/50` samples satisfy \(\rho_b<1\).
- Mean tangent correction / symmetric pair: `4.694%`.
- Mean equal-mean `rel_sol`: `3.232%`.
- Mean weak-final `rel_sol`: `2.970%`.
- Mean `rel_flux`: `13.677%`.

This is cross-equation evidence on the same Annulus geometry, not cross-domain
validation and not a paired Poisson-versus-CDR ranking.

**Reveal plan**

1. Reveal q50 CDR pre/post response mismatch.
2. Reveal the 50-sample reduction and correction magnitude.
3. Reveal final evaluation metrics and the same-geometry limitation.

### Result And Candidate Order

The final ten main slides must close with the following evidence and experimental
order:

1. Define the common post-reconstruction estimator and distinguish fixed
   geometry, prediction-mismatch, and local weak-residual weight information.
2. Use the frozen Poisson comparison as the first displayed result so the
   geometry/reconstruction effect is not confounded by convection or reaction.
3. Use the frozen CDR comparison to show that the same estimator ordering
   persists under a broader elliptic operator, while retaining the
   same-geometry and calibration-provenance limitations.
4. Present the final Poisson weak-result directional sources, reconstructions,
   signed errors, and representative/full-test metrics without transition
   diagnosis language.
5. Present the corresponding CDR coefficients, directional sources,
   reconstructions, signed errors, and metrics with the same visual contract.
6. Define the fixed tangent correction from the symmetric-balanced source and
   its direct response-mismatch ratio \(\rho_b\).
7. Present Poisson first, then CDR, showing the q50 pre/post fields, 50-sample
   paired mismatch audit, correction magnitude, and final evaluation metrics.
8. Treat the tangent results as direct objective validation. A causal
   solution-quality conclusion still requires a paired retraining experiment
   against the symmetric baseline.

The three post-hoc blend rules and the proposed response gains construct their
outputs without reference targets. Slide 5 defines the rules, Slides 6-8 explain
their construction, Slide 9 presents Poisson comparison evidence, Slide 10
presents CDR consistency, Slides 11-14 present the final computed fields and
evaluation-only errors, Slide 15 separates exact balance from directional
response consistency, Slide 16 defines the evaluated fixed
Jacobi-preconditioned tangent step, and Slides 17-18 present its Poisson/CDR
response-mismatch audits. Learned weights, learned
steps, learned gates, full response-matrix solves, and additional axial
orientations remain outside the meeting plan.

## Optional Backup Slides

### Backup A: Unit and Physical Green Integrals Are Equivalent

Show:

\[
\int_0^1
G_{\mathrm{unit}}(t,\eta)
\bigl(L^2f_{\mathrm{phys}}(\eta)\bigr)
\,d\eta
\]

and

\[
\int_{s_0}^{s_1}
G_{\mathrm{phys}}(s,\xi)
f_{\mathrm{phys}}(\xi)
\,d\xi,
\qquad
G_{\mathrm{phys}}=L\,G_{\mathrm{unit}}.
\]

Report the observed maximum absolute and relative differences:

\[
1.56\times10^{-17},
\qquad
7.18\times10^{-16}.
\]

**Purpose:** answer the question of whether reconstruction on the physical
interval would remove the transition pattern.

### Backup B: Exact Green Error Decomposition

For sample 47, show the RMS contribution table:

| Contribution | RMS | Fraction of learned total RMS |
|---|---:|---:|
| Learned total solution error | \(4.22665\times10^{-4}\) | 1.000 |
| Exact response to source-split error | \(4.20892\times10^{-4}\) | 0.9958 |
| Learned-minus-exact Green response | \(6.03\times10^{-9}\) | \(1.43\times10^{-5}\) |
| Target exact closure | \(5.41\times10^{-6}\) | 0.0128 |

**Purpose:** support the conclusion that GreenNet approximation is not the
dominant source of the observed transition error.

## Visual Design Direction

- Use a quiet technical meeting style rather than the conference-deck visual
  hierarchy.
- Prefer one dominant figure or schematic per slide.
- Use a neutral domain color, one accent for the split line, and one accent for
  the one-segment line.
- Reserve a diverging color scale for signed error fields.
- Make \(4.80\times\) the dominant number on Slide 2.
- Use the same colors for the physical-source, scaled-response, exact-Green, and
  learned-Green stages across Slide 3 and backup slides.
- Use one consistent accent per post-hoc rule on Slides 5-10: geometry C2,
  mismatch seam C2, and weak residual. Slides 6-8 use amber, coral, and teal
  construction panels; Slide 9's Poisson scatter and Slide 10's CDR comparison
  preserve those colors. Use a separate response-operator accent for Slides 15-16
  and do not reuse the retired-method color from Slide 4.
- Slides 11-14 must use one result grammar across both equations: physical
  fields on sequential scales, signed errors on zero-centered diverging scales,
  one colorbar per field group, identical panel order, identical domain extent,
  and one-to-one aspect ratio.
- Slides 11-14 must not contain transition markers, cardinal markers,
  segment-length maps, trace-jump metrics, or causal-diagnosis annotations.
- CDR coefficient figures use physical direct-at-`coords_valid` values. Render
  convection as a magnitude background with quiver arrows in the main slide;
  reserve separate `bx` and `by` panels for backup inspection.
- Avoid decorative animations. Use fragments only to expose one causal step at
  a time.
- For Slide 1, generate one five-panel Plotly composite directly from
  `checkpoints/Annulus_poisson/coupling/artifacts/data/selected_raw_arrays.npz`.
  Use its HTML as the presentation asset and the matching PDF as the
  static/offline fallback; do not create or use a PNG export.
- For other slides, prefer Plotly HTML for interactive meeting inspection and
  matching PDF assets for static/offline use. Any format change must be agreed
  before Quarto authoring.

Quarto Reveal.js supports fragments and `::: {.notes}` speaker notes. The final
implementation should use these features only after the slide content and
wording are approved:

- <https://quarto.org/docs/presentations/revealjs/>
- <https://quarto.org/docs/reference/formats/presentations/revealjs>

## Planned Quarto Structure After Content Approval

The implemented canonical location is:

```text
docs/meeting/annulus_transition_error/
├── annulus_transition_error.qmd
├── styles.scss
└── assets/
```

The `.qmd` uses:

- `format: revealjs`
- 16:9 layout
- speaker notes for the complete discussion script
- slide-local fragments rather than globally incremental lists
- static image fallbacks for every Plotly figure
- presentation HTML plus a handout-oriented PDF export check

## Asset Preparation Tasks After Approval

1. Add an artifact-only Plotly builder for the locked sample-47 arrays. It must
   create the specified five-panel HTML/PDF pair without importing or loading a
   current CouplingNet checkpoint.
2. Redraw the two-segment/one-segment geometry as a slide-native schematic.
3. Build the Slide 2 raw-reference to physical-projection to
   reference-pull-back pipeline with slide-native KaTeX and verify the expanded
   \(\Phi,\Psi\) equations against the historical sample-47 arrays.
4. Convert the stagewise ratio table into a compact slide-native table or
   Plotly bar chart.
5. Build the Slide 4 four-stage projection timeline with slide-native KaTeX,
   muted retired-state labels, and one current-formulation accent.
6. Verify the Slide 4 physical-source/response-space numerical comparison against the existing
   `coupling4` and `coupling5` diagnosis reports.
7. Build the Slide 5 three-card method schematic with the common estimator and
   the geometry, mismatch-seam, and weak-residual information contracts. Use no
   result metric on this slide.
8. Build Slide 6's geometry-only construction panel from the compact-C2 archive,
   including both compact influences, signed correction, and final weight.
9. Recompute Slide 7's mismatch seam fields from stored sample-0
   `u_phi/u_psi` and geometry metadata with the existing deterministic helper;
   do not load a checkpoint or run inference.
10. Build Slide 8's weak-reliability construction panel from the stored
    smoothed indicators, signed reliability, and final partition weight.
11. Build the Slide 9 Poisson four-way paired scatter from
   `checkpoints/Annulus_poisson/coupling15/weak_residual_reliability_blend_comparison`,
   and create a compact `w_phi/equal-error/weak-error` sample-47 inset from its
   selected raw NPZ. Do not show or run a Poisson parameter sweep.
12. Build the Slide 10 Poisson-versus-CDR grouped comparison from the two frozen
   summary/CSV pairs. Keep the CDR interactive four-way scatter available for
   discussion but do not duplicate it as the dominant slide visual.
13. Record in Slide 9-10 speaker notes that presets originated in the CDR
    exploratory diagnostics and were transferred to Poisson without retuning;
    presentation order is not tuning chronology.
14. Build the Slide 11 Poisson field composite from the standard selected
    artifact and weak-comparison archive for sample 0. The final panel must use
    `u_weak_residual_reliability`, never the standard equal-mean `u_pred`.
15. Build the Slide 12 Poisson signed-error composite from `phi_error`,
    `psi_error`, `u_phi-sol`, `u_psi-sol`, and
    `u_weak_residual_reliability-sol`, with the approved representative and
    aggregate metric cards.
16. Build the Slide 13 CDR coefficient/field composite for sample 9. Load
    `a`, `bx`, `by`, `b_magnitude`, `c`, and `quiver_indices` from
    `coefficient_fields.npz`; load the final solution from the weak-comparison
    archive.
17. Build the Slide 14 CDR signed-error composite using the same field order,
    target qualifier, color-scale policy, and metric-card layout as Slide 12.
18. Save the four result composites as matching standalone Plotly HTML and PDF
    assets, provisionally named
    `poisson_weak_result_fields_sample0`,
    `poisson_weak_result_errors_sample0`,
    `cdr_weak_result_fields_sample9`, and
    `cdr_weak_result_errors_sample9`.
19. Verify the tangent formulas against the production forward/transpose
    response actions and fixed Jacobi denominator. Lock the displayed setting
    to \(\eta=0.01\), \(\lambda_{\mathrm{rel}}=0.01\), and define \(\rho_b\).
20. Build the Slide 17/18 q50 composites from the frozen coupling18/coupling8
    NPZ/CSV/JSON only. Enforce shared pre/post mismatch color ranges, independent
    final-error ranges, 50-sample paired scatter, and SHA-256 provenance.
21. Crop or regenerate diagnostic figures without artifact-specific titles that
    compete with slide headings.
22. Copy only approved final assets into the meeting `assets/`
    directory.
23. Verify every numerical value against
   `length_response_diagnostics/summary.json`, metrics CSV files, and
   the relevant post-hoc geometry, seam, and weak-residual comparison
   `summary.json`/`diagnosis_report.md` files.

## Content Validation Checklist

- [ ] The previous observation is recalled in one slide without repeating the
      full earlier presentation.
- [ ] Slide 1 uses the sample-47 five-panel composite generated from
      `checkpoints/Annulus_poisson/coupling/artifacts/data/selected_raw_arrays.npz`,
      with matching HTML/PDF output and no PNG export.
- [ ] The composite contains `phi_error`, `psi_error`, `u_phi_error`,
      `u_pred_error`, and `u_psi_error` in the approved two-row order.
- [ ] The source row and solution row use separate zero-centered shared color
      ranges and exactly one colorbar per row.
- [ ] Every panel uses the same physical extent and a one-to-one aspect ratio.
- [ ] The figure builder does not load the historical CouplingNet checkpoint or
      current model/config contract.
- [ ] The FEniCSx target-balance qualification appears in the speaker notes.
- [ ] The physical domain is never described as discontinuous.
- [ ] Slide 2 distinguishes historical raw reference outputs \(P_0,Q_0\),
      physical raw proposals \(p_0,q_0\), balanced physical sources
      \(\phi,\psi\), and pulled-back reference sources \(\Phi,\Psi\).
- [ ] Slide 2 states that projection is pointwise in physical source space and
      verifies \(\phi+\psi=f\), \(\phi-\psi=p_0-q_0\).
- [ ] Slide 2 shows the full compact pipeline
      \((P_0,Q_0)\rightarrow(p_0,q_0)\rightarrow(\phi,\psi)
      \rightarrow(\Phi,\Psi)\).
- [ ] The expanded Slide 2 equations expose the cross-axis ratios
      \(L_x^2/L_y^2\) and \(L_y^2/L_x^2\).
- [ ] The historical formula audit reproduces stored `phi` with maximum error
      \(4.44\times10^{-16}\).
- [ ] \(L^2\) scaling is described as a required pull-back, not an optional
      correction.
- [ ] The \(2.19\times\) length and \(4.80\times\) length-squared ratios are
      sourced from the diagnostic summary.
- [ ] Slide 3 uses \(E_\Phi=L_x^2e_\phi\) and
      \(E_\Psi=L_y^2e_\psi\) before showing exact/learned Green responses.
- [ ] The sample-47 stagewise ratios are labeled as one-segment/split-line RMS
      ratios.
- [ ] Exact and learned Green results are not conflated.
- [ ] Verified observations and the working hypothesis are visually separated.
- [ ] Slide 4 contains exactly the four approved projection stages: original
      physical-symmetric, length-aware variants, response-space, and current
      physical-symmetric.
- [ ] Slide 4 states that all projection variants preserve the intended balance
      while none removes the transition pattern.
- [ ] Response-space source/flux improvement is not presented as solution-error
      improvement.
- [ ] Existing physical-source/response-space runs are not described as a fully controlled single-factor
      paired ablation.
- [ ] Slide 5 defines the common
      \(u_{\mathrm{blend}}=w_\phi u_\phi+w_\psi u_\psi\) estimator and exact
      partition \(w_\phi+w_\psi=1\).
- [ ] Slide 5 distinguishes all three fixed information contracts: known
      topology for Geometry C2, prediction mismatch for detected seam C2, and
      local full-PDE defect for weak reliability.
- [ ] Slide 5 states that all three rules are post-reconstruction, leave
      \(\phi,\psi,u_\phi,u_\psi\) unchanged, and use no reference target in
      weight construction.
- [ ] Slide 6 defines the topology-transition distances, compact quintic bump,
      exact partition, C2 support-edge condition, and known-topology limitation.
- [ ] Slide 6 uses the sample-independent compact-C2 archive and contains no
      performance metric.
- [ ] Slide 7 defines normalized mismatch edge jumps, one-dimensional RMS
      profiles, smoothing/NMS, and a separately prescribed compact C2 ramp.
- [ ] Slide 7 reconstructs sample-0 seam fields from frozen arrays and geometry
      metadata without model inference, coefficients, or reference data.
- [ ] Slide 8 defines
      \(R_\phi=R_x(u_\phi;\phi)+R_y(u_\phi;\psi)\) and
      \(R_\psi=R_x(u_\psi;\phi)+R_y(u_\psi;\psi)\), includes the full
      \(a,\mathbf b,c\) weak form, and separates the indicator from the excluded
      training-time weak-closure loss.
- [ ] Slide 8 states that the calculation uses axial P1 gather/scatter, not a
      global matrix solve, and uses no reference target.
- [ ] Slide 9 uses the frozen Poisson `coupling15` comparison and reports all
      four estimators, including weak mean `rel_sol=4.860579%`, `-12.734%`
      relative change, `50/50` wins, transition RMS `-8.360%`, and trace jump
      `-48.280%`.
- [ ] Slide 9 states that the Poisson run uses fixed CDR-origin presets with no
      Poisson sweep and that `sol` is evaluation-only.
- [ ] Slide 9 includes the weak axial-stripe/max-neighbor-jump `0.302343`
      limitation and does not claim structural regularity.
- [ ] Slide 10 reports the CDR four-way result, including weak mean
      `rel_sol=4.564722%`, `-11.472%`, `49/50` wins, transition RMS `-9.870%`,
      and trace jump `-48.785%`.
- [ ] Slide 10 presents the Poisson-versus-CDR change table and the identical
      estimator ordering without calling it cross-domain validation.
- [ ] Slide 10 states that the equations use independent sources on the same
      Annulus geometry and preserves the CDR calibration-provenance caveat.
- [ ] Slides 11-14 define the reported final prediction as
      \(u_{\mathrm{pred}}^{\mathrm{weak}}
      =w_\phi^{\mathrm{weak}}u_\phi
      +(1-w_\phi^{\mathrm{weak}})u_\psi\) and state that the blend leaves
      \(\phi,\psi,u_\phi,u_\psi\) unchanged.
- [ ] Slide 11 uses Poisson sample 0 and shows \(f,\phi,\psi,u,u_\phi,u_\psi\),
      weak \(u_{\mathrm{pred}}\), and a compact weak-weight inset.
- [ ] Slide 11 loads weak `u_pred` from `u_weak_residual_reliability` rather than
      the standard equal-mean artifact field `u_pred`.
- [ ] Slide 11 reports sample weak `rel_sol=4.824260%`, balance residual
      `2.22e-16`, and 50-sample mean/median weak `rel_sol`
      `4.860579%`/`4.822652%`.
- [ ] Slide 12 shows the two directional-source and three solution signed errors
      on separate row-shared symmetric scales and labels \(\phi_*,\psi_*\) as
      numerical FEniCSx directional targets.
- [ ] Slide 12 labels `rel_flux` as an unchanged directional-source diagnostic,
      not as a weak-blend metric.
- [ ] Slide 13 shows physical \(a\), \(\mathbf b\), and \(c\) coefficient
      context and CDR sample 9 fields using the same result contract as
      Poisson.
- [ ] Slide 13 uses a convection-magnitude background plus quiver arrows and
      keeps `bx/by` scalar figures outside the main slide.
- [ ] Slide 13 reports sample weak `rel_sol=4.573527%`, balance residual
      `4.02e-16`, and 50-sample mean/median weak `rel_sol`
      `4.564722%`/`4.205909%`.
- [ ] Slide 14 reuses the Slide 12 panel order and signed-error convention, while
      determining its row color limits independently for the unpaired CDR
      source realization.
- [ ] Slides 11-14 contain no transition markers, cardinal markers, line-length
      maps, trace-jump metrics, or causal problem statement.
- [ ] Slides 11-14 use reference `sol/target_phi/target_psi` only for displayed
      evaluation errors and metrics, never for weak-weight construction.
- [ ] Slide 15 defines the physical proposals \(p=P/L_x^2,q=Q/L_y^2\), the
      balance plane \(\mathcal C_f\), and the symmetric-balanced pair that
      preserves \(d=p-q\).
- [ ] Slide 15 identifies \((1,1)\) as the normal/common direction and
      \((1,-1)\) as the tangent/difference direction.
- [ ] Slide 15 defines \(m_0=H_x\widetilde\phi-H_y\widetilde\psi\), retains the
      five \(H_s\) factors, and distinguishes exact source balance from
      directional response consistency.
- [ ] Slide 16 parameterizes the feasible update as
      \(\phi=\widetilde\phi+\delta\),
      \(\psi=\widetilde\psi-\delta\), so \(\phi+\psi=f\) is structural.
- [ ] Slide 16 defines \(J(\delta)\),
      \(g=(H_x+H_y)^\top M_\Omega m_0\), the fixed Jacobi denominator \(D\),
      and \(\delta=-\eta D^{-1}g\).
- [ ] Slide 16 displays \(\eta=0.01\),
      \(\lambda_{\mathrm{rel}}=0.01\), and
      \(\rho_b=\|m_{\mathrm{post},b}\|/\|m_{\mathrm{pre},b}\|\).
- [ ] Slide 16 uses the column diagonal only to precondition the tangent
      gradient. Raw-residual allocation, opposite-gain correction, and the full
      coupled solve are absent.
- [ ] Slide 16 states only a local descent direction at \(\delta=0\), preserves
      the existing canonical-energy/optimizer path, and points to the measured
      finite-step audit without making a causal baseline claim.
- [ ] Slide 17 uses Poisson q50 sample 41, reports mean \(\rho_b=0.350216\),
      `65.0%` response-mismatch reduction, and `50/50` improved samples.
- [ ] Slide 18 uses CDR q50 sample 33, reports mean \(\rho_b=0.370161\),
      `63.0%` response-mismatch reduction, and `50/50` improved samples.
- [ ] Each tangent result asset gives pre/post mismatch one shared
      zero-centered range and gives `u_pred_error` an independent range.
- [ ] Slides 17-18 distinguish response-mismatch reduction from equal-mean and
      weak-final `rel_sol`, and do not claim causal improvement over a
      symmetric-trained checkpoint.
- [ ] Slides 5-14's blend construction and Slide 16's tangent construction are
      reference-free; displayed reference errors are evaluation-only, and
      learned weights, learned steps, learned gates, and target-fitted
      calibration are absent.
- [ ] Relative split consistency, training-time weak operator closure,
      self-trace gluing, cross-axis carrier, null-space analysis, and
      boundary-energy development are absent from main slides, backup slides,
      and speaker notes.
- [ ] Pre-projection fuser is absent.
- [ ] The \(H_0^1\) assumption-relaxation discussion is absent.
- [ ] Slides 9-14 are described as measured frozen-checkpoint post-hoc results.
      Slides 15-16 define the tangent method, and Slides 17-18 present its
      frozen trained-run diagnostics with explicit causal limitations.

## Locked Authoring Decisions

- The deck has eighteen main slides followed by Backup A/B.
- Visible slide text is English and speaker notes are Korean.
- Slide 15 is method-only, Slide 16 defines the evaluated fixed method, and
  Slides 17-18 present direct mismatch evidence plus evaluation metrics.
- Slide 16 uses the production fixed tangent contract: symmetric-balanced base,
  response forward/transpose gradient, column-diagonal Jacobi denominator, and
  one balance-preserving step.
- The displayed runs verify the tangent objective on 50/50 samples, but any
  causal solution-quality conclusion still requires paired retraining against
  a symmetric baseline.
