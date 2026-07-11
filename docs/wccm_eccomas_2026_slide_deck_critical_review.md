# WCCM-ECCOMAS 2026 Slide Deck Critical Review

이 문서는 현재 Quarto + Reveal.js deck
`docs/presentations/wccm_eccomas_2026/wccm_eccomas_2026.qmd`와 planning docs를 기준으로
WCCM-ECCOMAS 2026 발표 슬라이드를 최대한 비판적으로 평가한 revision note이다. 목적은
슬라이드를 비난하는 것이 아니라, 15분 발표에서 청중이 실제로 이해하고 기억해야 할 주장만
남기기 위해 위험한 부분을 미리 드러내는 것이다.

평가 기준은 다음이다.

- 발표 대상: WCCM-ECCOMAS 2026, `MS165 - Methods and Applications of Model Order Reduction`
- 시간 제약: 질의응답 포함 15분, 실제 발표 11-12분
- 핵심 주장:
  `A 2D elliptic problem is solved through axial Green inversions and a learned,
  balance-preserving source decomposition.`
- 발표 도구: Quarto + Reveal.js, fragments와 selected Auto-Animate 사용
- 현재 main deck: Slide 1-15, backup/Q&A slides

## Executive Assessment

현재 deck은 연구 내용을 담는 데에는 상당히 진전되어 있지만, 아직 conference-ready라고 보기에는
위험하다. 가장 큰 장점은 GreenNet, CouplingNet, energy-bound, numerical evidence를 모두 하나의
논리 흐름에 넣었다는 점이다. 그러나 발표 시간이 짧기 때문에 지금 상태 그대로는 청중이 “무엇이
새로운가”를 이해하기 전에 수식, architecture, evidence가 과밀하게 지나갈 가능성이 높다.

내 판단은 **almost-ready but not yet delivery-ready**이다. 즉, 기술적으로 필요한 내용은 대부분
들어갔지만, 발표용 메시지 압축과 시각적 hierarchy가 아직 충분하지 않다.

가장 큰 risk 5개는 다음이다.

1. **핵심 novelty가 분산되어 있다.** GreenNet analytic structure, CouplingNet source split,
   energy-bound, numerical evidence가 모두 중요하게 보이지만, 어떤 것이 “main contribution”인지
   15분 안에 선명하게 남기 어렵다.
2. **수식 밀도가 높은 slide가 많다.** 특히 Slide 4, Slide 6, Slide 7, Slide 11은 발표자가
   설명을 잘해도 청중이 동시에 읽기 어렵다.
3. **MOR audience framing이 약하다.** axial Green inversions가 왜 model reduction 관점의
   structured representation인지 더 직접적으로 말해야 한다.
4. **Numerical evidence는 좋아졌지만 해석 문장이 아직 방어적이다.** Figure는 설득력이 있지만
   “이 그림으로 정확히 무엇을 주장하는가”가 더 짧고 강해야 한다.
5. **Backup numbering과 main slide count가 불일치한다.** 현재 deck source에는 `Backup 14A`,
   planning doc에는 `Backup 15A`가 쓰인다. 발표자가 Q&A 중 backup을 찾을 때 혼란을 줄 수 있다.

## High-Priority Critiques

### 1. The talk still needs a sharper single-sentence contribution

**Issue:**  
현재 closing thesis는 좋다. 하지만 이 문장이 Slide 14에만 강하게 등장하고, 초반 Slide 2-3에서는
같은 문장이 덜 강하게 반복된다. 청중은 발표 초반 2분 안에 contribution frame을 잡아야 하는데,
현재는 “axial reduction”, “graphic abstract”, “pull-back”, “Green operator”가 연속해서 나오면서
main contribution sentence가 분산된다.

**Why it matters:**  
MOR minisymposium 청중은 새로운 neural architecture 세부보다 “어떤 reduced representation을
제안하는가”를 먼저 듣고 싶어 한다. 이 framing이 늦어지면 GreenNet analytic formula가 나오기 전에
발표의 기준점이 흔들린다.

**Recommended fix:**  
Slide 2 또는 Slide 3에 closing thesis의 짧은 버전을 고정 문장으로 넣는다.

```text
Axial Green operators provide line-wise inverses; CouplingNet learns the source split
that makes them a 2D solver.
```

이 문장은 Slide 14의 closing line과 거의 같아도 된다. 반복이 아니라 anchoring이다.

### 2. GreenNet analytic structure is strong but too easy to over-explain

**Issue:**  
Slide 6 `GreenNet II: Analytic Green Structure and Learned Correction`은 talk title의
“Hybrid Green's Function Learning”을 설명하는 핵심 slide이다. 그러나 현재 3-state Auto-Animate
구조는 수학적으로는 좋지만, 발표 시간이 15분인 상황에서는 한 state라도 길어지면 전체 talk를
압박한다.

**Why it matters:**  
청중이 \(G_0\), \(J_0\), \(A\), \(B\), \(E\), \(M\), \(R_\theta\)를 모두 이해해야 한다고 느끼면,
CouplingNet과 evidence로 넘어가기 전에 cognitive load가 커진다. 이 slide의 목적은 full derivation이
아니라 “singularity와 cancellation은 analytic하게 처리하고 smooth residual만 학습한다”는 메시지다.

**Recommended fix:**  
Slide 6의 visible text를 다음 3문장으로 더 강하게 정리한다.

- \(A(t)G_0(t,\eta)\): builds the Dirac-delta jump.
- \(B(t)(J_0-\frac12E)\): cancels the Heaviside contribution.
- \(E(t,\eta)M(t)R_\theta\): learns the smooth residual while preserving boundaries.

나머지 identity는 backup 15A로 보내도 된다. Main slide에서 \(J_0\) antiderivative identity는
유지하되, piecewise definitions는 절대 늘리지 않는 편이 좋다.

### 3. CouplingNet is clearer than before, but the word "coupling" still needs a sharper visual meaning

**Issue:**  
Slide 8-10은 CouplingNet을 세 단계로 설명한다. 구조는 맞지만, “coupling”이 구체적으로 무엇을
couple하는지 그림만 보면 즉시 보이지 않는다. Slide 8은 branch/trunk architecture이고, Slide 9는
projection/reconstruction pipeline이라서 두 slide 사이에 “source split is the coupling variable”이라는
문장이 더 강하게 필요하다.

**Why it matters:**  
GreenNet은 line-wise inverse를 제공한다. 그러나 line-wise inverse만으로는 2D solution이 되지 않는다.
발표의 핵심은 CouplingNet이 \(\phi,\psi\)를 통해 line problems를 서로 맞춘다는 점이다. 이 메시지가
약하면 CouplingNet이 그냥 neural postprocessor처럼 보일 수 있다.

**Recommended fix:**  
Slide 8 또는 Slide 9의 takeaway에 다음 형태의 문장을 넣는다.

```text
The learned variables are not the solution; they are the directional sources
that couple the two families of axial Green inverses.
```

### 4. Energy-bound slide is mathematically valuable but visually too theorem-like

**Issue:**  
Slide 11 `Energy-Norm Error Bound Proposition`은 중요한 주장을 담고 있지만, 현재는 proposition
content와 condition footer가 동시에 theorem 분위기를 만든다. 발표자가 proof를 하지 않을 예정이라면
슬라이드가 약간 “논문 본문 일부를 보여주는” 느낌이 날 수 있다.

**Why it matters:**  
MOR audience에게 error-bound는 강한 selling point이다. 하지만 조건부 theorem임을 너무 길게 설명하면
claim이 약해 보이고, 너무 짧게 말하면 과장처럼 들린다. 균형이 중요하다.

**Recommended fix:**  
Slide 11의 structure를 다음 3층으로 단순화한다.

1. Definition: \(\mathcal{E}_{\mathrm{split}}=\|u_\phi-u_\psi\|_a^2\)
2. Bound: \(\|u_{\mathrm{pred}}-u_*\|_a\le \frac{C_E}{2}\sqrt{\mathcal{E}_{\mathrm{split}}}\)
3. Condition strip: exact/controlled Green reconstruction and \(H_0^1\)-admissibility

현재보다 definition과 bound 사이의 visual distance를 줄이고, \(C_E\) 설명은 one-line note로
충분하다.

### 5. Numerical evidence slides need a stronger "claim-to-figure" mapping

**Issue:**  
Slide 12와 Slide 13의 figures는 상당히 좋아졌지만, 슬라이드 제목과 figure layout만으로는 “이
evidence가 어떤 claim을 검증하는가”가 즉시 보이지 않는다. Slide 12는 singular Green structure,
Slide 13은 full solver reconstruction을 보여주지만, 두 evidence가 서로 다른 검증 대상을 가진다는
차이가 더 선명해야 한다.

**Why it matters:**  
짧은 발표에서 numerical evidence는 복잡한 이론을 정당화하는 마지막 기회다. 청중은 각 plot을 자세히
읽지 않는다. 제목과 한 줄 claim만 보고 “무엇이 증명됐는지” 판단한다.

**Recommended fix:**  
각 evidence slide에 다음 claim label을 작게 넣는다.

- Slide 12: `Kernel-level evidence`
- Slide 13: `Solver-level evidence`

또는 subtitle에 반영한다.

```text
GreenNet kernel-level evidence: singular structure without diagonal-dominated error
CouplingNet solver-level evidence: reconstruction across relative-error quantiles
```

## Slide-by-Slide Review

### Slide 1 - Hybrid Green's Function Learning With Axial Reduction

**Issue:**  
Title은 강하지만, 첫 장에서 `Hybrid`가 무엇을 의미하는지 바로 알려주지 않는다.

**Why it matters:**  
청중은 “hybrid = analytic Green structure + learned correction”을 나중에야 알게 된다. 초반에는
hybrid가 FEM/neural hybrid인지, GreenNet/CouplingNet hybrid인지, analytic/neural hybrid인지 불명확하다.

**Recommended fix:**  
부제나 speaker opening에 다음 문장을 사용한다.

```text
Hybrid means analytic Green structure plus learned source coupling.
```

Visible text로 넣을 필요는 없지만, speaker opening에서는 반드시 말해야 한다.

### Slide 2 - Axial Reduction of Elliptic Operators

**Issue:**  
\(\mathcal S_{a,\mathbf b,c}:f\mapsto u\), \(L_x,L_y\), fixed operator framing이 모두 들어가
있어 수학적 출발점은 좋다. 하지만 “why this is model reduction”이 아직 말로만 존재한다.

**Why it matters:**  
MS165 청중은 reduction의 대상이 state dimension인지, operator family인지, solver decomposition인지
구분하려고 한다. 현재 slide는 reduction target을 수식으로 보여주지만 MOR language로 한 번 더
번역하지 않는다.

**Recommended fix:**  
다음 문장을 subtitle 또는 bottom note로 넣는다.

```text
Reduction target: replace one global 2D source-to-solution map by coupled 1D Green inversions.
```

### Slide 3 - From a 2D Elliptic Problem to Coupled Axial Green Solves

**Issue:**  
Graphic abstract 방향은 좋다. 그러나 네 개 card 모두가 같은 시각적 weight를 가지며, GreenNet card와
CouplingNet card의 수식은 여전히 작다.

**Why it matters:**  
이 slide는 청중의 mental model을 만드는 유일한 그림 중심 slide다. 여기서 흐름이 즉시 잡히지 않으면
이후 수식 slide들이 각각 따로 보인다.

**Recommended fix:**  
네 card 중 2번째 `Axial interval view`와 4번째 `CouplingNet`을 더 크게 만든다. 첫 번째 source
problem card는 조금 줄이고, 마지막 card에는 `\(\phi+\psi=f\)`를 더 크게 보이게 한다. 이 slide의
목표는 예쁜 flow가 아니라 “line-wise inverse만으로는 안 되고 source split이 필요하다”를 그림으로
보이는 것이다.

### Slide 4 - Unit-Interval Pull-Back and Operator Scaling

**Issue:**  
필요한 식이 모두 들어가 있지만, physical 1D operator, pull-back, scaling, normalized equation이 한
slide 안에서 모두 중요해 보인다.

**Why it matters:**  
청중은 chain rule을 따라가려다 GreenNet story를 놓칠 수 있다. 이 slide는 derivation slide가 아니라
normalization convention slide여야 한다.

**Recommended fix:**  
가장 큰 visible emphasis를 다음 한 줄에 둔다.

```text
The interval is normalized; the length \(L\) moves into the operator.
```

Physical/normalized full equation은 speaker가 필요할 때만 설명하고, scaling rule을 가장 먼저 기억하게
해야 한다.

### Slide 5 - GreenNet I: Normalized Axial Green Operator

**Issue:**  
Green operator action을 명확히 정의한다는 점은 좋다. 다만 Slide 4와 Slide 5가 모두 unit interval을
보여주기 때문에 audience가 “같은 이야기를 반복하나?”라고 느낄 수 있다.

**Why it matters:**  
Slide 4는 coordinate transform, Slide 5는 inverse operator이다. 이 차이가 명확하지 않으면 GreenNet의
필요성이 약해진다.

**Recommended fix:**  
Slide 5의 first reveal에 다음 contrast를 넣는다.

```text
Slide 4: normalize the operator.
Slide 5: learn its Green inverse.
```

### Slide 6 - GreenNet II: Analytic Green Structure and Learned Correction

**Issue:**  
이 slide는 기술적으로 가장 중요하지만 가장 위험하다. 수식이 설득력은 있지만, 시간과 cognitive load
모두를 많이 요구한다.

**Why it matters:**  
이 slide가 길어지면 CouplingNet과 evidence가 급하게 지나간다. 반대로 너무 짧게 지나가면 “hybrid”
claim이 충분히 증명되지 않는다.

**Recommended fix:**  
main talk에서는 45-60초 이상 쓰지 않는다. 말할 내용은 3개 역할만으로 고정한다.

```text
Dirac jump, Heaviside cancellation, smooth learned correction.
```

full distributional derivation은 Backup 15A로 넘긴다.

### Slide 7 - GreenNet III: Source-to-Solution Supervision

**Issue:**  
GP sample, endpoint correction, source generation, reconstruction loss가 모두 들어가 있다.
정확하지만, 짧은 발표에서는 dataset generation detail처럼 들릴 수 있다.

**Why it matters:**  
이 slide의 목적은 “exact kernel label이 아니라 operator action으로 학습한다”이다. GP construction은
그 목적을 뒷받침하는 보조 설명이다.

**Recommended fix:**  
visible text의 hierarchy를 다음 순서로 둔다.

1. No exact kernel labels.
2. Generate target solutions.
3. Apply the unit operator to get sources.
4. Train the Green operator action.

GP는 “smooth target generator”로만 짧게 언급한다.

### Slide 8 - CouplingNet I: Directional Source Split

**Issue:**  
\(\phi,\psi\) 정의가 들어간 것은 좋다. 그러나 CouplingNet이 “solution predictor”가 아니라
“source split predictor”라는 점을 더 반복해야 한다.

**Why it matters:**  
청중은 neural PDE solver를 보면 자연스럽게 \(f\mapsto u\) direct model을 떠올린다. 이 오해를
여기서 끊어야 한다.

**Recommended fix:**  
visible takeaway를 다음으로 더 직접화한다.

```text
CouplingNet predicts sources for Green inverses, not the solution directly.
```

### Slide 9 - CouplingNet II: Branches and Local Context for Split Prediction

**Issue:**  
Branch/trunk 분리는 잘 보이지만 중앙 dark box의 수식이 작고 시각적으로 heavy하다.

**Why it matters:**  
중앙 box는 slide의 시각적 중심인데, 그 안의 수식이 읽히지 않으면 청중은 좌우 card를 따로 읽게 된다.
그러면 branch/trunk fusion의 목적이 약해진다.

**Recommended fix:**  
중앙 수식을 줄이거나 제거하고, 대신 다음 문장을 크게 둔다.

```text
profiles + pointwise coordinates -> directional source split
```

상세 수식은 speaker note 또는 backup으로 충분하다.

### Slide 10 - CouplingNet III: Projection and Green Reconstruction

**Issue:**  
projection, Green reconstruction, average가 잘 정리되어 있지만, `projection`이 왜 필요한지 시각적으로
즉시 보이지 않는다.

**Why it matters:**  
이 slide는 CouplingNet raw output이 PDE balance를 만족하도록 만드는 핵심 단계다. 이 단계가 그냥
postprocessing처럼 보이면 method의 물리적 constraint가 약하게 전달된다.

**Recommended fix:**  
`projection` box 바로 아래에 다음 짧은 equation을 크게 둔다.

```text
\(\phi+\psi=f\)
```

그리고 `physical split variables`라는 표현은 speaker가 말하고, visible text는 `balance projection`
중심으로 둔다.

### Slide 11 - Energy-Norm Error Bound Proposition

**Issue:**  
현재 slide는 proposition으로서 매우 가치가 있지만, theorem claim과 assumptions가 좁은 공간 안에
있어 청중이 조건을 다 읽기 어렵다.

**Why it matters:**  
이 slide는 method의 credibility를 높이는 slide다. 조건부 bound라는 정직함은 유지하되, 청중이
bound 자체를 먼저 기억해야 한다.

**Recommended fix:**  
첫 click에서는 bound를 숨기지 말고, definition과 bound를 같은 visual group에 둔다. Assumption은
작은 footer로 유지한다. 또한 `exact/controlled Green reconstruction`은 말로 설명하고 visible text는
최소화한다.

### Slide 12 - Numerical Evidence I: GreenNet Kernel Approximation

**Issue:**  
Figure 선택은 적절하다. Reference kernel, learned kernel, signed error, fixed-\(\eta\) slice가
GreenNet claim과 잘 맞는다. 그러나 현재 screenshot 기준 하단 takeaway가 잘릴 위험이 있고, diagnostic
card의 일부 text는 발표장에서 작게 보일 가능성이 높다.

**Why it matters:**  
이 slide는 “GreenNet이 singular structure를 잘 잡는다”는 증거다. 하단 takeaway가 잘리면 가장 중요한
해석 문장이 약해진다.

**Recommended fix:**  
problem strip 또는 diagnostic card 높이를 줄이고, takeaway를 한 줄로 압축한다.

```text
The kernel captures the singular structure; error is not diagonal-dominated.
```

현재 문장은 정확하지만 길다.

### Slide 13 - Numerical Evidence II: CouplingNet Solution Reconstruction

**Issue:**  
5-by-4 matrix는 좋은 evidence layout이지만, metric card의 작은 글씨와 많은 panel 때문에 발표 중
눈길이 분산된다.

**Why it matters:**  
이 slide는 full solver evidence다. 청중은 한 번에 20개 field를 자세히 보지 않는다. row-by-row reveal은
좋지만, 최종 상태에서는 여전히 정보량이 많다.

**Recommended fix:**  
발표 중에는 `q50` 또는 `max` column 하나를 verbal focus로 지정한다. 예를 들어:

```text
I will focus on the max-error column; the other columns show this is not a cherry-picked case.
```

또한 metric card의 note는 더 짧게 줄인다.

```text
Scales: per-sample source/solution; shared signed-error scale.
```

### Slide 14 - Takeaway: Coupled Axial Green Solvers

**Issue:**  
Closing slide는 매우 좋아졌다. 다만 four blocks가 모두 같은 weight를 가지므로 발표의 마지막 기억이
네 개 bullet로 분산될 수 있다.

**Why it matters:**  
마지막 slide는 Q&A 직전의 memory anchor다. 한 문장이 남아야 한다.

**Recommended fix:**  
closing banner를 가장 강하게 읽고, four blocks는 보조 근거처럼 보이게 한다. 발표자는 마지막에
반드시 다음 문장으로 끝낸다.

```text
GreenNet supplies the line-wise inverses; CouplingNet supplies the missing source coupling.
```

### Slide 15 - Backup / Q&A Menu

**Issue:**  
Backup menu의 후보는 좋지만 numbering inconsistency가 있다. 현재 deck source에는 `Backup 14A`,
`Backup 14B`, `Backup 14C`가 보이고, planning doc에는 `Backup 15A`, `Backup 15B`,
`Backup 15C`가 보인다.

**Why it matters:**  
Q&A 중 backup slide를 찾을 때 발표자가 혼란스러울 수 있다. 또한 문서와 deck의 불일치는 최종
production 과정에서 실수를 만든다.

**Recommended fix:**  
backup naming을 `Backup A`, `Backup B`, `Backup C`로 바꾸는 것이 가장 안전하다. 숫자를 빼면
main slide count가 바뀌어도 유지된다.

### Backup 15A - Dirac/Heaviside Derivation Sketch

**Issue:**  
Q&A backup으로 적절하다. 하지만 main slide와 거의 같은 수식이 다시 나오면 질문자가 원하는
“왜?”에 충분히 답하지 못할 수 있다.

**Why it matters:**  
backup은 main slide의 반복이 아니라 질문 대응이어야 한다.

**Recommended fix:**  
`Operator application creates two effects`라는 heading을 넣고, Dirac term과 Heaviside term을
operator action 관점으로 짧게 나눈다.

### Backup 15B - Imperfect Green Reconstruction Perturbation

**Issue:**  
좋은 limitation slide다. 다만 \(\varepsilon_x+\varepsilon_y\)와
\(\varepsilon_x-\varepsilon_y\)의 의미를 그림으로 더 직관화할 수 있다.

**Why it matters:**  
이 slide는 “energy consistency가 만능은 아니다”라는 정직한 방어 논리다. 수식만으로는 질문자에게
방어적으로 보일 수 있다.

**Recommended fix:**  
두 perturbation channel을 다음처럼 표시한다.

```text
directional mismatch: visible to split energy
common bias: not removed by agreement alone
```

### Backup 15C - Connected-Interval Pull-Back Detail

**Issue:**  
복잡한 geometry 설명을 backup으로 둔 결정은 좋다. 그러나 현재 main talk에서 complex geometry를
가볍게 다루기 때문에, 이 backup은 질문이 오면 아주 중요해진다.

**Why it matters:**  
청중이 “non-square geometry에서 axial interval을 어떻게 정의하나?”라고 물으면, 이 slide가 method의
domain generality를 방어해야 한다.

**Recommended fix:**  
한 개의 non-square domain slice 그림을 크게 넣고, disconnected intervals are not merged를 시각적으로
보여준다. 수식보다 그림이 먼저여야 한다.

## Cross-Cutting Review

### Narrative

**Issue:**  
현재 narrative는 논리적으로는 맞지만, `GreenNet analytic structure`와 `CouplingNet directional split`
중 어느 것이 primary contribution인지 slide sequence만으로는 약간 분산된다.

**Why it matters:**  
15분 발표에서는 contribution hierarchy가 선명해야 한다.

**Recommended fix:**  
다음 hierarchy를 명시한다.

1. Core representation: axial Green inversions.
2. GreenNet: learns line-wise Green kernels with analytic structure.
3. CouplingNet: learns source split that couples line-wise inverses.
4. Energy bound: explains why split agreement matters.

### Mathematics

**Issue:**  
수식은 정확하지만 많다. 특히 Slide 4, 6, 7, 11이 모두 수식 중심이므로 중반부가 heavy해질 수 있다.

**Why it matters:**  
청중이 수식을 처리하는 동안 speaker message를 놓칠 수 있다.

**Recommended fix:**  
각 수식 slide마다 “one equation to remember”를 지정한다.

- Slide 4: scaling rule
- Slide 5: Green operator integral
- Slide 6: \(G_\theta = analytic + learned\)
- Slide 7: reconstruction loss
- Slide 11: energy error bound

### Visual Design

**Issue:**  
전체적으로 clean하지만, 일부 slide는 card가 많고 text가 작다. Evidence slides는 특히 화면 크기와
강의장 projector 품질에 민감하다.

**Why it matters:**  
WCCM 같은 큰 학회장에서는 작은 수식과 metric text가 거의 보이지 않을 수 있다.

**Recommended fix:**  
16:9 projector에서 뒷자리 기준으로 읽어야 하는 text만 남긴다. Diagnostic metric은 speaker가 말할
숫자만 크게 남기고, note성 설명은 speaker notes로 보낸다.

### Animation

**Issue:**  
Animation strategy는 적절하지만 click budget이 누적될 위험이 있다. Slide 5-11 사이에 technical reveal이
많아지면 발표자가 timing을 잃기 쉽다.

**Why it matters:**  
15분 발표에서 클릭이 많으면 말의 rhythm이 끊긴다.

**Recommended fix:**  
각 technical slide의 max click count를 미리 정한다.

- Slide 5: 2 clicks
- Slide 6: 3 clicks
- Slide 7: 3 clicks
- Slide 8: 3 clicks
- Slide 9: 2 clicks
- Slide 10: 2 clicks
- Slide 11: 2 clicks
- Evidence slides: row/element reveal만 사용

### Evidence

**Issue:**  
현재 evidence는 claim과 꽤 잘 맞는다. 하지만 Slide 12는 kernel approximation, Slide 13은 solution
reconstruction이라는 구분을 청중이 놓칠 수 있다.

**Why it matters:**  
GreenNet과 CouplingNet evidence가 섞이면 “kernel이 좋아서 solution도 좋다”라는 단순한 주장으로
오해될 수 있다.

**Recommended fix:**  
Slide 12에는 `Kernel-level evidence`, Slide 13에는 `Solver-level evidence` label을 작게 넣는다.

## Prioritized Revision Plan

### 반드시 고칠 것

- Backup numbering을 `Backup A/B/C` 또는 `Backup 15A/15B/15C`로 통일한다.
- Slide 12 takeaway clipping을 해결한다.
- Slide 13 metric note를 더 짧게 줄인다.
- Slide 2 또는 Slide 3에 contribution sentence를 더 선명하게 반복한다.
- Slide 6에서 main-talk 설명 범위를 3 역할로 제한한다.

### 가능하면 고칠 것

- Slide 3 graphic abstract에서 GreenNet/CouplingNet card의 수식 크기를 키운다.
- Slide 8 중앙 dark box의 수식을 줄이고, branch/trunk fusion message를 크게 한다.
- Slide 11의 proposition visual hierarchy를 definition -> bound -> condition 순서로 단순화한다.
- Evidence slides에 `Kernel-level evidence`와 `Solver-level evidence` label을 넣는다.
- Slide 14 four blocks의 visual weight를 조금 낮추고 closing banner를 더 우선시한다.

### 유지해도 되는 것

- Quarto + Reveal.js 방향.
- Static PNG/PDF-ready evidence assets를 main deck에 사용하는 결정.
- GreenNet analytic structure를 main flow 독립 slide로 둔 결정.
- CouplingNet을 source split, branch/trunk context, projection/reconstruction으로 나눈 3-slide 구성.
- Energy-bound를 proposition으로 제시하되 conditional statement로 제한하는 방향.

## Final Judgment

현재 deck은 연구의 구성 요소를 모두 담고 있으며, 큰 방향은 맞다. 다만 지금 상태의 가장 큰 문제는
“내용 부족”이 아니라 “내용 과잉”이다. 발표자가 모든 slide를 설명하려고 하면 15분 안에 핵심 메시지가
흐려질 가능성이 높다. Revision의 목표는 내용을 더 추가하는 것이 아니라, 각 slide에서 하나의
기억할 문장만 남기는 것이다.

최종 발표에서 청중이 기억해야 할 문장은 다음 하나여야 한다.

```text
GreenNet supplies line-wise Green inverses; CouplingNet learns the source split
that turns them into a 2D elliptic solver.
```

이 문장이 Slide 2, Slide 3, Slide 14에서 반복되고, GreenNet/CouplingNet/evidence가 모두 이 문장을
뒷받침하도록 정렬되면 발표의 설득력이 크게 올라갈 것이다.

검토 확신도: **0.92**. 남은 불확실성은 수학적 정보 부족이 아니라 실제 발표자가 선호하는 delivery
style, 그리고 최종 numerical evidence figure가 현재 상태에서 더 바뀔 수 있다는 점이다.
