# Matrix-Free Tangent Subspace and Geometry Connectivity Meeting Deck Plan

## Summary

- 기존 Annulus 미팅 덱은 수정하지 않고, 새 후속 미팅 자료를 `docs/meeting/tangent_subspace_connectivity/`에 독립적인 Quarto Reveal.js 덱으로 작성한다.
- 발표 제목은 **“From Exact Balance to Geometry-Aware Response Alignment”**, 부제는 **“Matrix-Free Tangent Subspaces and Structural K-Connectivity”**로 고정한다.
- 슬라이드의 visible text와 수식 설명은 영어로 작성하고, 모든 슬라이드에 정확히 하나의 상세한 한국어 발표자 노트를 작성한다. 발표 시간 제한은 두지 않으며 노트에 시간 단축용 문구를 넣지 않는다.
- Reveal.js fragment는 한 번의 click마다 하나의 수학적 또는 논리적 단계만 추가한다. 발표자 노트의 `Click:` cue와 실제 fragment 순서를 일치시킨다. Quarto의 공식 [speaker notes](https://quarto.org/docs/presentations/revealjs/#speaker-notes)와 [fragment ordering](https://quarto.org/docs/presentations/revealjs/advanced.html#fragment-order) 방식을 따른다.
- 덱은 title을 포함한 **39개 main slide**로 구성하며 backup slide는 두지 않는다. Unit-square source-count 결과는 합의대로 본편 후반에 배치한다.
- 기존 model, training, projection, checkpoint, geometry/sample NPZ와 frozen 실험 결과는 변경하거나 다시 계산하지 않는다.

## Slide Content Contract

### 1. Motivation and Exact Balance: Slides 1–7

1. **Title:** 발표 질문을 source balance와 directional response alignment의 관계로 제시한다.
2. **Previous Observation:** 기존 Annulus sample 47의 \(\phi,\psi,u_\phi,u_\psi,u_{\mathrm{pred}}\) error를 이용해 transition-localized error를 한 장으로 복습한다.
3. **Exact Source Balance Is Not Equal Response:** \(\phi+\psi=f\)와 \(H_x\phi=H_y\psi\)가 서로 다른 조건임을 분리한다.
4. **Raw Output to Physical Proposals:** \(P,Q\rightarrow p=P/L_x^2,\;q=Q/L_y^2\)와 projection/pull-back 좌표계를 설명한다.
5. **Symmetric Projection Algebra:** \(\widetilde\phi=\tfrac12[f+(p-q)]\), \(\widetilde\psi=\tfrac12[f-(p-q)]\)를 유도하고 sum은 고정하며 difference mode를 보존함을 보인다.
6. **Geometry of the Balance Plane:** normal \((1,1)\), tangent \((1,-1)\), affine balance plane을 점별 그림으로 설명한다.
7. **Global Tangent Field:** \(\delta\in\mathbb R^P\), \(\phi=\widetilde\phi+\delta\), \(\psi=\widetilde\psi-\delta\)이며 \(\delta\)가 sample별·point별 field임을 설명한다.

### 2. Response Least Squares: Slides 8–13

8. \(H_x,H_y\)를 physical directional source-to-response operator로 정의한다.
9. \(m_0=H_x\widetilde\phi-H_y\widetilde\psi\), \(S=H_x+H_y\), \(m(\delta)=m_0+S\delta\)를 부호까지 유도한다.
10. \(J(\delta)=\tfrac12\|m_0+S\delta\|_{M_\Omega}^2\)와 physical mass inner product를 정의한다.
11. \(g(\delta)=S^\top M_\Omega(m_0+S\delta)\)와 \((g_0)_j=\langle Se_j,m_0\rangle_{M_\Omega}\)의 adjoint 의미를 설명한다.
12. \(A=S^\top M_\Omega S\), \(A_{ij}=\langle Se_i,Se_j\rangle_{M_\Omega}\)를 response Gram operator로 해석한다.
13. 이상적인 \(A\delta^\star=-g_0\)와 production의 matrix-free \(Az=S^\top M_\Omega(Sz)\)를 비교하고 global matrix/solve가 없음을 명시한다.

### 3. \(K=1\) and General \(K\): Slides 14–22

14. Production \(D=\gamma_x^2+\gamma_y^2+\lambda_{\mathrm{damp}}\)가 exact \(\operatorname{diag}(A)\)가 아니라 positive column-gain Jacobi surrogate임을 구분한다.
15. \(z_0=D^{-1}g_0\), \(v_0=Sz_0\), \(\delta(\eta)=-\eta z_0\)의 source/response 역할을 설명한다.
16. \(\eta^\star=\langle m_0,v_0\rangle_M/(\langle v_0,v_0\rangle_M+\varepsilon)\)를 미분으로 유도하고 “fixed 1D line에서만 exact”임을 강조한다.
17. \(K=1\)의 한계는 point support가 아니라 correction subspace가 한 방향뿐이라는 것임을 설명한다.
18. 첫 update 후 \(m_1\), \(g_1\), \(z_{1,\mathrm{raw}}\), \(v_{1,\mathrm{raw}}\)를 순차적으로 구성한다.
19. 목적함수와 일치하도록 response \(M_\Omega\)-inner product에서 orthogonalize하며 같은 combination을 source direction에도 적용함을 설명한다.
20. 두 번의 modified Gram-Schmidt, degenerate-direction fallback, \(Sz_k=v_k\) invariant를 설명한다.
21. \(c_k\), \(\delta_{k+1}=\delta_k-c_kz_k\), \(m_{k+1}=m_k-c_kv_k\), \(\delta_K=-\sum c_kz_k\)의 전체 recurrence를 단계별 animation으로 제시한다.
22. \(K\)를 “response-orthogonal tangent correction patterns의 수”로 정의하고, 모든 \(K\)에서 \(\phi_K+\psi_K=f\)가 정확히 유지됨을 정리한다.

### 4. Krylov Interpretation and Geometry Reach: Slides 23–32

23. 표준 \(\mathcal K_K(A,g_0)=\operatorname{span}\{g_0,Ag_0,\ldots,A^{K-1}g_0\}\)를 conceptual bridge로 소개하되, production basis는 preconditioned residual과 response-space MGS를 쓰는 **Krylov-like nested response subspace**라고 명시한다.
24. 실제 \(g_0\)는 dense할 수 있으므로 \(K=1\)을 pointwise-independent 계산으로 해석하지 않는다. Support와 independent correlation capacity를 분리한다.
25. Geometry visualization의 \(g_0=e_i\)는 실제 sample gradient가 아니라 operator reach를 보기 위한 localized canonical probe임을 설명한다.
26. 같은 connected horizontal/vertical axial segment를 공유하는 point graph와 \(d_L(i,j)\)를 정의한다.
27. 한 conceptual \(A=S^\top M_\Omega S\) action의 forward/adjoint mixing을 반영한 \(d_A=\lceil d_L/2\rceil\), \(K_{\mathrm{first}}=d_A+1\)을 설명한다.
28. \(C_i(K)\), \(C_{\mathrm{global}}(K)\), global/tail 99% geometry-only selection rule을 정의한다.
29. Unit-square와 Disk를 비교하여 두 geometry 모두 \(K=2\)에서 structural reach 100%임을 보인다.
30. Annulus representative seed의 \(K=1\ldots4\) reach \(0.00927\%,31.2106\%,98.2666\%,100\%\)를 hole-induced split-segment 구조와 연결한다.
31. Pentagram representative seed의 \(K=2,3,4\) reach \(83.6177\%,98.8189\%,99.8688\%\)와 extreme-tip full-reach 한계를 함께 보여준다.
32. Geometry reach가 structural accessibility는 설명하지만 PDE-specific optimal \(K\), 실제 production support 또는 numerical causality를 증명하지 않는다는 한계 슬라이드를 둔다.

### 5. Numerical Evidence and Protocol: Slides 33–39

33. Pentagram trained \(K=1,2,3,4\)의 `rel_sol`, `rel_u_phi`, `rel_u_psi`, `rel_flux`를 동일한 표와 error bars/paired sample distribution으로 비교한다.
34. Tangent core forward+backward \(141.4/211.8/282.2/361.8\) ms와 정확도 개선을 한 그래프에서 비교해 accuracy optimum \(K=4\), cost-quality knee \(K=3\)를 구분한다.
35. Geometry reach, spectral enrichment, trained accuracy가 서로 다른 evidence layer임을 설명하고 Pentagram 개선을 reach 하나로 인과 해석하지 않는다.
36. 해석을 “transition repair”에서 “balance-preserving general directional response alignment”로 확장한다.
37. Unit-square 4-seed fixed-2400-step source-count 실험 \(600/1200/2400/4800\)과 mean `rel_sol` \(0.4255/0.3931/0.3696/0.3505\%\)를 제시한다.
38. 이후 실험의 공통 source budget을 \(N_{\mathrm{train}}=4800\)으로 선택한 근거를 paired-seed 및 sample-win 결과와 함께 정리한다.
39. Exact algebra, structural proxy, empirical evidence를 한 장에서 분리한 뒤 최종 결론과 현재 주장 가능한 범위를 정리한다.

## Implementation Changes

- 새 content contract를 `docs/meeting/tangent_subspace_connectivity_slide_plan.md`에 작성한다. 각 slide의 영어 headline, visible claim, 수식, fragment 순서, 한국어 발표자 노트의 핵심 문장, frozen source provenance를 기록한다.
- 새 덱 디렉터리에 `tangent_subspace_connectivity.qmd`, `styles.scss`, `build_assets.py`, `qa_reveal.js`를 둔다. 기존 Annulus 덱의 1600×900 canvas, 1280×720 QA, local Plotly, MathML, linear navigation과 offline policy를 계승하되 스타일은 새 덱 전용 class로 분리한다.
- `build_assets.py`는 frozen NPZ/CSV/JSON/기존 offline HTML만 읽는다. model/checkpoint loading이나 inference를 하지 않으며, deck-local Plotly HTML/PNG와 `assets/manifest.json`에 source path, metric key, sample/seed, SHA-256를 기록한다.
- Annulus historical figure는 기존 frozen offline asset을 hash-verified deck-local copy로 보존한다. Geometry \(K\)-figures는 `geometry_k_connectivity_visualization`의 JSON/NPZ를 이용해 동일 color scale의 incremental K shell figure로 다시 구성한다.
- Pentagram \(K=1\ldots4\) 자산은 `tangent_topology_k_analysis`와 `coupling8/9/10/11` best-energy 결과만 사용한다. Unit-square 자산은 `training_size_analysis`의 frozen CSV/JSON을 사용한다.
- 수식·알고리즘 슬라이드는 이미지에 수식을 굽지 않고 QMD MathML과 HTML/CSS diagram으로 만든다. `EXACT ALGEBRA`, `PRODUCTION ALGORITHM`, `STRUCTURAL PROXY`, `EMPIRICAL RESULT` badge를 사용해 근거 수준을 구분한다.
- fragment animation은 `raw proposal → balance projection`, `mismatch → adjoint gradient → preconditioned direction → response direction`, `K=1→4 reach shell` 순서로 사용한다. 단순 장식 animation은 추가하지 않는다.
- 최종 산출물은 `tangent_subspace_connectivity.html`, deck-local assets/manifest, QA report와 screenshots다. PDF는 생성하거나 저장하지 않는다.
- 새 deck 사용법과 provenance는 `README.md`에 추가하고, 합의된 발표 해석과 검증 명령은 `docs/memory.md`에 기록한다. 기존 Annulus deck, slide plan, rendered HTML은 수정하지 않는다.

## Test Plan

- 새 asset test는 frozen source 존재 여부, SHA-256 provenance, exact metric extraction, local Plotly dependency, deterministic rebuild, checkpoint/model loading 부재를 검증한다.
- 새 deck test는 logical slide count 39, 영어 visible text, 슬라이드당 한국어 notes 하나, formula/sign contract, production-vs-proxy caveat, fragment-index uniqueness와 `Click:` cue 일치를 검증한다.
- Content tests는 \(S=H_x+H_y\), \(A=S^\top M_\Omega S\), \(D\neq\operatorname{diag}(A)\), \(\delta_K=-\sum c_kz_k\), exact balance, localized-seed caveat와 주요 numerical value를 고정한다.
- Quarto render 후 1600×900과 1280×720에서 모든 final state와 모든 intermediate fragment state를 검사한다. Overflow, overlap, page error, broken iframe, external request와 clipped formula가 없어야 한다.
- 검증 순서는 focused asset/deck pytest, asset rebuild, Quarto render, browser QA, 전체 `pytest test`, `ruff check`, non-mutating format check, `mypy src`, `git diff --check`로 고정한다.

## Rollback Strategy

- 새 덱은 독립 디렉터리와 독립 tests로 구성하므로 rollback은 새 content contract, deck directory, tests와 README/memory 항목만 제거하는 것이다.
- 기존 Annulus QMD/HTML/assets와 frozen checkpoint/analysis 결과는 rollback 대상에 포함하지 않는다.
- Interactive Plotly가 특정 viewport에서 안정적으로 배치되지 않으면 numerical content는 유지하고 해당 panel만 hash-verified static PNG로 교체한다.
- Frozen source와 슬라이드 numerical claim이 불일치하면 수치를 재계산하거나 checkpoint를 실행하지 않고 작업을 중단한다. 충돌한 metric key, source artifact와 최소 content-only 수정안을 보고한다.

## Assumptions and Confidence

- 이번 자료는 기존 Annulus 덱을 대체하지 않는 후속 독립 덱이다.
- Unit-square source-count 결과는 본편 Slides 37–38에 포함한다.
- Product/product_fuser screening, 네 수치 예제 전체 프로토콜, minimal-network/transverse-branch 실험과 unfinished Disk 결과는 포함하지 않는다.
- Geometry-only visualization은 production tangent direction의 literal support map으로 표현하지 않는다.
- 구현 계획 확신도는 **0.98**이다. 규칙이나 목적의 모호성은 없다. 남은 불확실성은 geometry reach와 trained accuracy 사이의 정량적 인과관계가 아직 분리 검증되지 않았다는 경험적 정보 부족이며, 덱은 이를 limitation으로 명시한다.

## Executable `/goal` Draft

```text
/goal

`/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry/PLAN.md`의
"Matrix-Free Tangent Subspace and Geometry Connectivity Meeting Deck Plan"을
기준 문서로 참고하여 후속 공동연구자 미팅용 Quarto Reveal.js 덱을 끝까지
구현한다.

완료는 다음 조건으로 검증한다.

- 기존 Annulus 미팅 덱을 수정하지 않고 별도의 tangent-subspace-connectivity
  QMD, SCSS, assets, HTML과 QA 구성을 만들 것,
- title을 포함한 39개 main slide가 계획된 순서와 수학적 서사를 유지할 것,
- 모든 visible slide text는 영어이고 각 slide에 정확히 하나의 상세한 한국어
  speaker note가 있을 것,
- symmetric balance plane, response least-squares objective, gradient, Hessian,
  K=1 exact line search와 K-dimensional matrix-free recurrence가 부호와
  source/response-space 의미까지 정확히 설명될 것,
- production preconditioned response subspace와 표준 Krylov interpretation,
  localized-seed geometry-only reach proxy를 서로 혼동하지 않을 것,
- Unit square, Disk, Annulus, Pentagram K-connectivity 결과와 Pentagram
  K=1..4 accuracy/cost 결과가 frozen source와 정확히 일치할 것,
- Unit-square 4-seed source-count 결과와 4800-source 선택 근거가 본편에
  포함될 것,
- exact algebra, production algorithm, structural proxy, empirical result가
  시각적으로 구분될 것,
- asset generation은 frozen NPZ/CSV/JSON/HTML만 읽고 model inference,
  checkpoint loading 또는 장기 계산을 수행하지 않을 것,
- 모든 asset에 source path, metric key와 SHA-256 provenance를 기록할 것,
- Reveal fragment 한 단계가 하나의 논리적 설명만 추가하고 한국어 Click cue와
  실제 fragment 순서가 일치할 것,
- 1600x900과 1280x720의 final 및 intermediate fragment browser QA에서
  overflow, overlap, clipped formula, page error, broken iframe과 external
  request가 없을 것,
- focused deck/assets tests, 전체 pytest, Ruff, mypy와 git diff check가
  통과할 것.

수정 범위는 새 meeting slide plan, 새 Quarto Reveal.js deck directory와
presentation-only asset builder, deck QA, 관련 tests, README 및 docs/memory.md로
제한한다.

Model, trainer, projection, tangent implementation, checkpoint, geometry/sample
NPZ, frozen experiment result, 기존 Annulus meeting deck와 numerical value는
변경하지 않는다. PDF는 생성하지 않는다.

각 구현 단계 후 가장 작은 asset/deck test를 먼저 실행하고, assets와 Quarto
HTML을 다시 생성한 뒤 두 viewport의 모든 fragment state를 검토한다.

Frozen source와 계획된 numerical claim 또는 code-consistent tangent formula를
동시에 유지할 수 없다면 작업을 중단하고 다음을 보고한다.

1. 충돌하는 formula, metric key, slide 또는 frozen artifact,
2. 영향을 받는 figure, speaker note와 provenance record,
3. frozen evidence와 기존 production semantics를 유지하는 가장 작은
   content/layout-only 수정안.
```
