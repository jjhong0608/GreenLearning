# Coupler Repo Memory

이 문서는 Coupler repo에서 사용자와 Codex가 합의한 연구 맥락, 모델 구조,
coefficient 의미, 실험 설계 기준, 논문용 데이터/figure 생성 기준을 기록한다.
`README.md`는 사용법, `AGENTS.md`는 작업 규칙, 이 문서는 연구/실험 의사결정
기억을 담당한다.

## Stable Project Context

- 이 repo의 중심 목표는 axial Green function 관점에서 2D PDE solution을
  재구성하는 GreenONet/CouplingNet 실험을 수행하는 것이다.
- GreenONet은 axial Green kernel 또는 그 근사를 학습/제공한다.
- CouplingNet은 axial decomposition을 이용해 solution/flux reconstruction을
  학습한다.
- 주요 PDE family는 Poisson, variable diffusion, diffusion-reaction,
  convection-diffusion이다.
- 논문용 결과 생성에서는 coefficient family, model variant, train/eval split,
  metric, figure 저장 규칙을 명확히 분리한다.

## Stable Modeling Decisions

- Convection coefficient는 scalar `b_fun`보다 vector form
  `bx_fun(x, y)`, `by_fun(x, y)`를 권장 API로 사용한다.
- Legacy coefficient file이 `b_fun(x, y)`만 제공하면
  `bx_fun = b_fun`, `by_fun = b_fun`으로 해석한다.
- 내부 tensor convention은 `b_vals[0] = b_x`, `b_vals[1] = b_y`를 유지한다.
  별도 `b_x`/`b_y` tensor field를 만들지 않는다.
- `branch_coefficient`는 diffusion/convection/reaction coefficient를 통합하는
  generic coefficient branch이다.
- `coupling_model.coefficient_terms` config가 diffusion, convection, reaction
  입력 여부를 제어한다.
- `coupling_model.branch_fusion.mode`는 CouplingNet branch feature 결합 방식을
  제어한다. 기본값 `product`는 기존 multiplicative fusion을 보존하고,
  실험 옵션 `product_fuser`는 active branch features와 그 component-wise product를
  concat한 뒤 learned fuser로 최종 branch representation을 만든다.
- `source_stencil_lift.enabled=true` path와 standard coefficient branch path는
  의미가 다르므로 섞지 않는다.
- `terminal.width`는 top-level runtime/logging config이며 Rich terminal wrapping을
  제어한다. `training.log` file output과는 별도 surface로 취급한다.

## Coefficient Families

- `Pure_Poisson.py`: constant diffusion, zero convection, zero reaction.
- `Sinusoidal_Diffusion_Only.py`: variable diffusion, zero convection,
  zero reaction.
- `Sinusoidal_Diffusion_Only_Ver2.py`: alternate smooth variable diffusion,
  zero convection, zero reaction.
- `Diffusion_Reaction_Ver2.py`: variable diffusion with reaction,
  zero convection.
- `Smooth_Variable_Diffusion_Reaction.py`: smooth variable diffusion with reaction,
  zero convection.
- `Convection_Diffusion_Reaction.py`: variable diffusion with convection and reaction.
- `Divergence_Free_Convection_Diffusion.py`: variable diffusion,
  divergence-free convection with amplitude `2.0`, zero reaction.
- 새 coefficient file을 추가할 때는 `a_fun`, `apx_fun`, `apy_fun`,
  `bx_fun`, `by_fun`, `c_fun`을 정의한다.

## Important Mathematical Checks

- `apx_fun`, `apy_fun`은 반드시 `a_fun`의 해석적 편미분과 일치해야 한다.
- Divergence-free convection은 `d bx / dx + d by / dy = 0`을 만족해야 한다.
- Reaction-free 문제는 `c_fun(x, y) = 0`이어야 한다.
- Green accuracy가 좋아도 solution reconstruction이 나쁠 수 있다. 이런 경우
  kernel 사용 경로, coefficient alignment, axis convention, normalization,
  quadrature/integration path를 우선 의심한다.
- Convection-diffusion 문제에서 `bx_fun`은 x-direction line/operator에,
  `by_fun`은 y-direction line/operator에 들어간다.

## GreenNet / GreenONet Notes

- GreenNet은 특정 coefficient problem에 대한 Green's function을 학습한다.
- 이 repo의 논문용 문제 설정에서는 2D coefficient field를 문제별로 고정한다.
- 하지만 GreenNet이 다루는 것은 full 2D Green's function 하나가 아니라 axial
  decomposition에서 생기는 1D Green's function들이다.
- 따라서 같은 2D coefficient problem 안에서도 각 x-axis/y-axis line이 보는
  1D operator coefficient slice가 달라질 수 있고, 각 축선의 Green's function도
  서로 다르다고 해석해야 한다.
- 현재 GreenNet 학습의 training/validation dataset은 저장된 외부 dataset을
  읽는 방식이 아니라 config의 sampling 설정으로 `ForwardSampler` 또는
  `BackwardSampler`가 즉석 생성한 `TrainingData`를 `AxialDataset`으로 감싸는
  방식이다.

## Model And Loss Notes

- `balance_projection.enabled=false`이면 CouplingNet은 projection 없이 raw
  output을 반환하는 path를 사용한다.
- `balance_loss`는 projection off일 때만 허용한다.
- `symmetric_boundary_loss`는 symmetric projection on일 때 boundary condition을
  raw difference mode에 학습시키기 위한 loss이다.
- `symmetric_boundary_loss` 계산은 `CouplingTrainer._symmetric_boundary_loss`에
  있으며, projection output이 아니라 `CouplingNet.raw_flux_at_coords`로 얻은
  raw flux를 사용한다. Phi boundary residual은
  `(phi_raw_boundary - psi_raw_endpoint) + f`, psi boundary residual은
  `(phi_raw_endpoint - psi_raw_boundary) - f`이고, 각 edge 방향으로 적분한 뒤
  `0.5 * (phi_loss + psi_loss)`로 합친다.
- `smooth_mask` projection은 mask option을 갖는다. 기본은 `quadratic`,
  추가 option으로 `sin`을 사용할 수 있다.
- `axis_1d_trunk.enabled=true`이면 shared 1D trunk를 사용하고, transverse 정보는
  boundary-aware sin/cos branch로 넣는다.
- Axis-1D trunk의 boundary-aware encoding은 raw coordinate `t`를 포함하지 않고
  `sin(n*pi*t)`, `cos(n*pi*t)` for `n=1..k`만 사용한다.
- Complex geometry mode는 `dataset.geometry_mode="complex"`일 때만 사용한다.
  기존 unit-square path는 기본값 `unit_square`로 보존한다.
- Complex geometry v1 입력 계약은 precomputed geometry `.npz`와 full-grid sample
  `.npz` 조합이다. geometry 추출 자체는 이 repo의 v1 구현 범위가 아니다.
- Complex geometry sample은 full-grid `rhs`, `sol`을 필수로 갖고, flux target은
  `phi`/`psi`를 우선 사용하며 legacy `uxx`/`uyy`를 fallback으로 해석한다.
  모든 full-grid array는 `[row=y, col=x]` convention으로 valid point에 gather한다.
- Complex geometry Green interval normalization은 `a_unit=a_phys`,
  `ap_unit=L*ap_phys`, `b_unit=L*b_phys`, `c_unit=L^2*c_phys`,
  `f_unit=L^2*f_phys`를 사용한다. Unit reconstruction에서 Green kernel에
  추가 segment length factor를 곱하지 않는다.
- Complex geometry GreenNet은 geometry `.npz`의 connected x/y segment를
  flat interval list `N=Sx+Sy`로 펼쳐 학습한다. 각 connected interval은
  독립 1D domain이며, 같은 fixed coordinate에 놓인 disconnected segment도
  합치지 않는다. Trunk coordinate는 항상 unit `(t, eta) in [0,1]^2`이고,
  `dataset.samples_per_line`은 complex GreenNet mode에서 connected interval당
  synthetic sample 수를 뜻한다.
- Complex GreenNet의 `training.green_quadrature.enabled=true`는
  reconstruction loss, train/validation `rel_sol`, `evaluate(...)`, complex Green
  artifact reconstruction에만 split Gauss-Legendre 적분을 적용한다. Fine source는
  `source_sampling_factor`로 생성하고 Gaussian source node에서 `linear` 또는
  natural cubic source interpolation을 선택해 사용한다. Default는 `linear`이고,
  `cubic`은 smooth source 실험용 opt-in이다. `rel_green`은 기존 uniform-grid
  diagnostic으로 유지하며, CouplingNet reconstruction/loss/evaluation에는 이 설정을
  적용하지 않는다.
- Complex CouplingNet은 unit-square CouplingNet처럼 source-conditioned model이다.
  Full-grid `rhs`를 valid point로 gather한 뒤 segment-local unit source branch로
  변환하고, `f_unit=L^2*f_phys` scaling과 segment별 unit L2 norm normalization을
  적용한다. Model raw unit output은 해당 segment source norm으로 다시 scale한다.
- Complex CouplingNet coefficient branch는 `coefficient_terms`에 따라 active
  `[a,b_primary,b_transverse,c]` 순서로 구성한다. `convection=true`이면 x/Phi
  path는 `[L_x*b_x, L_x*b_y]`, y/Psi path는 `[L_y*b_y, L_y*b_x]`를 넣는다.
  GreenNet과 Green reconstruction branch는 primary convection만 사용하며
  `[a,ap,b_primary,c]` contract를 유지한다. `a'`는 GreenONet reconstruction
  query용 branch에는 보관하지만 CouplingNet coefficient branch에는 넣지 않는다.
  Technical/comparison 문서에서도 이 구분을 유지한다: transverse convection은
  CouplingNet의 source-split prediction context이고 Green reconstruction coefficient가 아니다.
- Complex CouplingNet primary trunk는 항상 segment-local 1D `t`를 사용한다.
  Fixed-line transverse branch는 global geometry extent 기준으로 normalized
  `r_hat`을 만들고, `axis_1d_trunk.num_frequencies`와 `max_frequency`로
  Fourier encoding한다. Optional `axis_1d_trunk.transverse_trunk.enabled=true`
  path는 pointwise cross-axis local coordinate를 별도 trunk에 넣는다. x/Phi
  path는 primary `x_local_t`와 transverse `y_local_t`를 쓰고, y/Psi path는
  primary `y_local_t`와 transverse `x_local_t`를 쓴다. `fusion`은 `product` 또는
  `product_fuser`이고, disabled이면 이전 complex behavior를 보존한다.
  `trunk_positional_encoding`은 unit-square 2D trunk coordinate encoding이므로
  complex mode에서는 사용하지 않는다.
- Complex geometry mode에서는 `cross_consistency`, `smooth_mask`, `balance_loss`,
  `source_stencil_lift`, `green_response_feature`를 사용하지 않는다. Cross 관련
  key는 metric, log, artifact에 남기지 않는 것을 contract로 둔다.
- `docs/unit_square_vs_complex_geometry.md`는 unit-square core path와 complex
  geometry path의 canonical 비교 문서이다. 이 문서의 unit-square 설명은 complex
  extension과 비교하는 데 필요한 GreenNet/CouplingNet core structure만 다루며,
  complex mode에서 사용하지 않는 unit-square-only auxiliary option을 나열하지 않는
  writing convention을 유지한다.
- `docs/unit_square_vs_complex_geometry_math.md`는 코드 surface를 제거한
  수학적/architecture 중심 비교 문서이다. 이 문서는 branch, trunk, projection,
  Green reconstruction, domain representation 차이를 설명하는 데 집중하고,
  config/file/schema/tensor-size contract는 engineering note인
  `docs/unit_square_vs_complex_geometry.md`에 남긴다.
- `docs/complex_geometry_greennet_couplingnet_technical_report.md`는 학회 발표자료
  준비를 위한 complex-mode-only technical report이다. 이 문서는 unit-square 확장
  narrative 없이 complex geometry PDE, axial connected interval decomposition,
  GreenNet unit pull-back, source-conditioned CouplingNet, physical balance
  projection, Green reconstruction, training/evaluation 해석을 수식과 알고리즘
  중심으로 설명하며, 코드 surface와 complex mode에서 사용하지 않는 기능은 넣지 않는다.
- 이 technical report의 GreenNet section은 branch-trunk learned correction과
  analytic Green structure의 결합식을 포함한다. 수식은 코드 surface 없이
  \(R_\theta\), \(G_0\), \(J_0\), \(A\), \(B\), \(E\), \(M\) 중심으로 설명하고,
  reaction coefficient는 analytic factor가 아니라 learned correction을 통해 반영된다고
  기록한다.
- 같은 report의 GreenNet dataset 설명은 forward construction만 다룬다. Smooth
  target solution은 Gaussian Process sample로 설명하고, GreenNet supervision은 exact
  kernel label이 아니라 source-to-solution reconstruction target으로 기록한다. Fine
  source와 quadrature 설명은 이 dataset construction 설명에서 제외한다.
- 같은 report의 GreenNet analytic wrapping 설명은 \(A(t)G_0\)를
  Dirac-\(\delta\) property를 만드는 jump/singularity 구조로 해석하고,
  \(B(t)(J_0-\frac12E)\)를 그 과정에서 생기는 Heaviside-type contribution
  cancellation term으로 설명한다. Learned correction은 이 analytic singular/cancellation
  구조 이후에 남는 smooth residual을 담당한다고 기록한다.
- 같은 report의 CouplingNet 설명에서는 "local trunk" 대신 "axial local trunk" 용어를
  사용한다. Axial local trunk는 primary axial coordinate \(t_{\parallel}\)를 담당하고,
  pointwise transverse trunk는 transverse axial interval의 normalized coordinate
  \(t_{\perp}\)를 통해 pointwise boundary context를 전달한다. Transverse branch는
  global transverse placement만 담당하며 pointwise transverse boundary 정보를 대체하지
  않는다.
- 같은 report의 activation 설명에서는 rational activation을 fixed activation이 아니라
  learnable activation으로 설명한다. \(P_\alpha\)와 \(Q_\beta\)의 coefficient는
  현재 초기값에서 시작하지만 학습 중 업데이트되는 parameter이며, activation은 analytic
  Green wrapping이나 physical balance projection을 대체하지 않고 branch/trunk 내부의
  representation nonlinearity 역할을 한다.
- 같은 report는 energy analysis 문서를 단순 reference로 넘기지 않고, energy loss가
  final solution energy error를 bound한다는 핵심 의미를 독립적인 proposition 형식으로
  설명한다. Technical report 안에서는
  \(\mathcal{E}_{\mathrm{split}}=\|u_\phi-u_\psi\|_a^2\), \(H_0^1(\Omega)\) admissibility,
  source-linear weak inverse assumption,
  \(\|u_{\mathrm{pred}}-u_*\|_a\le\frac{C_E}{2}\sqrt{\mathcal{E}_{\mathrm{split}}}\) exact
  bound, imperfect Green reconstruction perturbation, common Green bias limitation을
  presentation-ready mathematical structural statement로 설명한다. 이 proposition은
  proof 대체가 아니며, proof-level theorem과 final solution error-bound corollaries는
  `docs/complex_geometry_energy_consistency_analysis.md`에 유지한다. Energy loss/error-bound
  notation은 \(\mathcal{E}_{\mathrm{split}}\)로 통일하고 이전 L-subscript-E 표기는 더 이상 쓰지 않는다.
- `docs/complex_geometry_energy_consistency_analysis.md`는 complex geometry energy
  consistency를 위한 continuous-domain Markdown analysis이다. 이 문서는
  connected-interval Green reconstruction을 전제로 하고, split operator는
  \(\frac12cu\) reaction split을 사용한다. Exact Green reconstruction theorem,
  exact/perturbed final solution error-bound corollaries, imperfect Green
  reconstruction perturbation을 모두 포함한다. Green reconstruction은 source에 대해 linear하다고 가정하며,
  approximate reconstruction perturbation에서는 최소한 source-difference identity를 요구한다.
  \(u_\phi,u_\psi,u_*\in H_0^1(\Omega)\) admissibility는 theorem assumption으로 두고
  별도 discussion에서 다룬다. 특히 represented solutions \(u_\phi,u_\psi\)의
  \(H_0^1(\Omega)\) membership은 interval endpoint zero에서 자동으로 나오지 않으며,
  connected-interval Green operators가 admissible source spaces에서
  \(H_0^1(\Omega)\)로 mapping된다는 조건으로 읽는다. Section 8은 transverse
  derivative, moving interval geometry, degenerating interval length를 별도
  regularity issue로 설명한다. \(L^2\)-consistency insufficiency는 local
  high-frequency example로 설명한다.
- `docs/wccm_eccomas_2026_presentation_outline.md`는 WCCM-ECCOMAS 2026,
  MS165 - Methods and Applications of Model Order Reduction 발표를 위한 영어 slide
  outline planning document이다. 발표 제목은 "Hybrid Green's Function Learning With
  Axial Reduction for Multi-Dimensional Elliptic Problems"로 둔다. 발표 흐름은 MOR
  motivation, axial reduction, unit-interval pull-back and scaling, GreenNet,
  CouplingNet, projection/reconstruction, Energy-Norm Error Bound Proposition,
  numerical evidence, conclusion 순서로 유지한다. Complex geometry는 axial line
  construction이 아니라 non-square/non-unit interval의 unit-interval normalization
  motivation으로만 가볍게 다룬다. GreenNet은 3-slide block으로 설명하며, analytic
  Green structure and learned correction은 main-flow 독립 slide로 둔다. Main slide에서는
  Dirac-\(\delta\) jump, Heaviside-type cancellation, learned smooth correction의 role
  decomposition만 설명하고, full Dirac/Heaviside derivation은 technical report 또는 backup
  discussion으로 남긴다. Code/config/schema/checkpoint/dataset generation detail은
  outline에서 제외한다.
- Circular complex geometry generator는 `cli/make_circular_geometry.py`이며
  center `(0, 0)` 고정, radius CLI option default `1.0`, `2 * radius / step_size`
  정수 조건을 사용한다. Grid interval은 `[-radius, radius]`이고, boundary grid point와
  degenerate boundary line은 제외하며, valid interior point가 있는 axial chord segment만
  저장한다. Reconstruction weight는 physical length가 아닌 segment-local unit coordinate
  기준 nonuniform trapezoid weight로 저장한다.
- Annulus complex geometry generator는 `cli/make_annular_geometry.py`이며
  2D PDE code path에서의 torus shape를 두 동심원 사이 영역으로 해석한다. Center는
  `(0, 0)` 고정이고, valid point는
  `inner_radius < sqrt(x^2+y^2) < outer_radius`인 strict interior point만 저장한다.
  Grid interval은 `[-outer_radius, outer_radius]`이고,
  `2 * outer_radius / step_size` 정수 조건을 사용한다. Inner hole을 지나거나 접하는
  axial line은 disconnected segment row로 나눠 저장해서 edge/reconstruction이 hole을
  가로지르지 않게 한다.
- FEniCSx complex sample generator는 optional `green_fenicsx` conda env에서만
  실행하는 path로 둔다. Main `green_net` training env와 `pyproject.toml`에는
  FEniCSx dependency를 섞지 않는다.
- Complex Coupling sample generator output `.npz`는 full-grid `rhs`, `sol`,
  `phi`, `psi` 배열을 저장하고, 모든 배열은 `[row=y, col=x]` convention을 따른다.
  `coords_valid` 밖의 full-grid 값은 `0.0`으로 채운다.
- FEniCSx sample generator의 `phi`/`psi`는
  `phi=-d_x(a d_x u)+b_x d_x u+0.5*c*u`,
  `psi=-d_y(a d_y u)+b_y d_y u+0.5*c*u`인 direction-split operator component이며,
  valid point에서 `phi + psi ~= rhs` balance residual을 summary에 기록한다.
- FEniCSx sample-level 병렬화는 spawned Python process worker로 독립 sample을
  나눠 생성하는 방식이다. MPI domain decomposition은 v1 범위가 아니며,
  `num_workers > 1`이면 `sample_seed_policy="indexed"`를 강제한다. Parent process만
  `generation_summary.json`을 작성하고, sample `.npz` schema는 `rhs`, `sol`,
  `phi`, `psi` 그대로 유지한다.
- FEniCSx Gmsh script input은 `build_domain(gmsh, context)`를 제공해야 하며,
  multi-surface disconnected domain에서는 valid point마다 들어갈 surface를
  `point_surface_tags`로 명시해야 한다. Script mode는 valid points를 mesh internal
  points로 embed하려고 시도하고, `.msh` mode는 vertex 보장을 기본 요구하지 않는다.
  `examples/unit_circle_gmsh.py`는 geometry `.npz`의 `radius` metadata를 읽어서
  non-unit circular geometry와 Gmsh disk mesh radius를 일치시킨다.
- Annulus FEniCSx sample generation은 `examples/annulus_gmsh.py`를 사용한다.
  이 script는 geometry `.npz`의 `inner_radius`와 `outer_radius` metadata를 읽고,
  Gmsh OCC cut으로 inner hole을 가진 single annulus surface를 반환한다. Surface가
  하나이므로 `point_surface_tags`는 필요 없고, 기존 valid-point embedding은 모든
  valid point를 같은 surface에 embed한다.
- Circular sample generation workflow는 `examples/unit_circle_gmsh.py`를 기본
  Gmsh domain script로 사용한다. Smoke default는 `h=0.25`, `mesh_size=0.035`,
  `solution_degree=3`, `target_degree=2`, `train=1`; small dataset default는
  `h=0.1`, `mesh_size=0.025`, same FEM degrees, `train/valid/test=32/8/8`이다.
  첫 coefficient는 `Pure_Poisson.py`로 고정한다. 현재 refined smoke는
  `phi + psi ~= rhs` residual `1e-2`를 통과했지만, small dataset default는
  max residual이 약 `3.7e-2`이므로 schema/loadability와 residual distribution 확인용으로
  먼저 사용하고, strict `1e-2` 품질 dataset이 필요하면 mesh/projection parameter를 더 조정한다.

## Experiment And Figure Planning

- 논문용 데이터 생성 전에 coefficient family, model variant, train/eval split,
  output directory, plot target을 먼저 확정한다.
- GreenNet 결과는 `rel_green`, `train_rel_sol`, `val_rel_sol`을 함께 본다.
- Complex GreenTrainer의 새 L-BFGS run은 Adam 단계와 같은 metric token을
  유지해 `train_rel_sol`, validation이 있으면 `val_rel_sol`, reference가 가능하면
  `rel_green`을 `training.log`에 기록한다. 과거 complex GreenNet run의 L-BFGS
  log에는 validation/`rel_green` token이 없을 수 있다.
- GreenNet 논문용 산출물은 run-level training metrics, per-line metrics,
  selected-line Green kernel data, selected-line coefficient slices,
  source-to-solution reconstruction data를 분리해서 저장하는 방향을 기본으로 한다.
- GreenNet `training.log`와 artifact summary의 `rel_green`은 exact/reference
  1D line Green kernel을 구성할 수 있는 경우에 유효하다. `c=0, b=0`이면
  diffusion reference를 쓰고, `c=0, b!=0`이면 convection-diffusion reference를
  쓴다. Reaction `c`가 nonzero이면 현재 exact/reference가 없으므로 invalid/skip
  처리한다.
- `ExactGreenFunction`의 기본 `forward()`, `__call__()`, `error()`는 diffusion-only
  reference로 유지한다. Reaction-free convection-diffusion line reference가
  필요하면 `convection_diffusion(b)`를 명시적으로 호출하고, x-lines에는 `bx`,
  y-lines에는 `by`의 line slice를 넘긴다. GreenNet `rel_green` 경로는 같은
  reference policy를 사용한다.
- Exact/reference Green kernel matrix convention은 `G[row=x, col=xi]`이다.
  Reconstruction은 마지막 dimension인 `xi` 방향으로 적분한다. Convection-diffusion
  kernel은 비대칭이므로 이 orientation을 해석식 테스트로 고정해야 한다.
- GreenNet problem selection의 현재 논의 기준은 `Pure_Poisson.py`와 두
  diffusion-only 문제를 `rel_green` 중심 주력 문제로 두고, reaction-free
  convection-diffusion 문제도 `rel_green_reference="convection_diffusion"`로
  비교할 수 있다. Reaction 포함 문제는 reconstruction 중심 보조/확장 산출물로
  다루는 것이다.
- GreenNet 논문용 산출물은 training loop에 묶기보다 checkpoint/config/dataset을
  다시 읽는 별도 재생성 script 또는 CLI로 만드는 방향을 선호한다.
- GreenNet이 주장하려는 내용이 fixed coefficient problem에서 학습한 Green
  function으로 같은 sampler/source distribution의 source term에 대한 solution을
  잘 재구성한다는 것이라면, 논문용 evaluation data는 별도 저장 dataset 없이도
  학습 때와 같은 config/sampler 설정으로 다시 생성해도 충분하다고 본다.
  다만 seed와 generation sequence는 metadata에 명시해서 training data 재사용과
  같은-distribution evaluation을 구분한다.
- GreenNet artifact exporter 구현은 새 CLI만 두기보다 dataset 재생성,
  Green kernel/reconstruction metric, fixed-`xi` slice 추출, Plotly multi-format
  저장을 작은 helper로 분리해 training loop와 독립시키는 방향이 좋다.
- GreenNet artifact exporter surface는 `cli/export_green_artifacts.py`이며,
  helper는 `src/greenonet/green_artifacts.py`와 `src/greenonet/plotly_io.py`에
  둔다. 기본 evaluation은 `validation_like`, 기본 seed는 `12345`, 기본 fixed
  `xi` fractions는 `0.25, 0.5, 0.75`이다. Device는 기본적으로 config의
  `training.device`를 따르며, 필요하면 CLI `--device`로 override한다.
- 논문용 산출물은 최소 산출물만 만드는 방향보다 가능한 한 많이 생성한 뒤,
  그중 논문에 적합한 결과와 figure를 선택하는 방향을 기본으로 한다.
- CouplingNet 논문용 run-level training curves는 `train loss`, `val loss`,
  `l2_consistency`, `energy_consistency`, `rel_flux`, `rel_sol`만 사용한다.
  Other auxiliary losses are ignored for paper-facing curves unless explicitly
  requested later.
- CouplingNet selected-sample figures are limited to source `f`, exact solution
  `u`, `u_pred`, `u_pred_x`, `u_pred_y`, signed solution errors
  `u_pred - u`, `u_pred_x - u`, `u_pred_y - u`, mismatch `u_pred_x - u_pred_y`,
  exact flux-divergences `phi`, `psi`, predicted flux-divergences `phi_pred`,
  `psi_pred`, signed flux-divergence errors `phi_pred - phi`, `psi_pred - psi`,
  and balance fields `phi + psi`, `f - phi - psi`. Error figures must be signed
  differences, not absolute values.
- CouplingNet selected-sample non-error comparison figures use shared color ranges
  within each selected sample for reference/prediction groups. Unit-square groups
  are `u/u_pred/u_pred_x/u_pred_y`, `phi/phi_pred`, and `psi/psi_pred`.
- Complex CouplingNet selected-sample artifact figures use valid-point scatter
  plots on `coords_valid`. The default fields are `rhs`, `sol`,
  `u_pred=0.5*(u_phi+u_psi)`, `u_phi`, `u_psi`, signed solution errors
  `u_pred - sol`, `u_phi - sol`, `u_psi - sol`, split mismatch `u_phi - u_psi`,
  projected physical `phi`/`psi`, and optional target `phi`/`psi` plus signed
  flux errors when sample flux targets are available. Complex error and mismatch
  scatter figures use zero-centered diverging colors. Complex non-error comparison
  groups are `sol/u_pred/u_phi/u_psi`, `target_phi/phi`, and `target_psi/psi`
  when flux targets exist; `rhs` and target-free flux diagnostics keep independent
  ranges.
- CouplingNet selected-sample flux-divergence figures should exclude boundary
  grid values. CouplingNet predictions use zero-padding at boundaries only for
  trapezoid-rule integration compatibility with boundary-zero Green functions;
  those padded boundary values are not meaningful flux-divergence predictions
  and should not be shown in paper-facing flux-divergence heatmaps.
- CouplingNet null-space and closure diagnostics are not part of the selected
  paper-facing figure set unless explicitly re-enabled later.
- CouplingNet 논문용 산출물에는 test-set aggregate metrics, balance/projection
  diagnostics, boundary diagnostics, coefficient/source context figures,
  ablation comparison tables, and raw metric/data archive를 유지한다.
- CouplingNet artifact 구현 시 current `CouplingEvaluator`는 일부 tensor
  computation을 재사용할 수 있지만, paper-facing selected figures는 별도 path
  또는 option으로 분리해야 한다. Current evaluator uses absolute errors and
  null/closure diagnostics, while the paper-facing set requires signed errors and
  excludes null/closure figures.
- CouplingNet paper artifact exporter surface는 `cli/export_coupling_artifacts.py`
  이며 helper는 `src/greenonet/coupling_artifacts.py`에 둔다. Test data는
  `dataset.test_path`의 `.npz`를 `CouplingDataset`으로 읽고, coefficient는 기본적으로
  `dataset.coefficient_functions_path`를 사용하며 `--coefficients`로 override한다.
  Device는 기본적으로 `coupling_training.device`를 따르고 `--device`가 우선한다.
- `plot_coupling_logs.py`는 paper-facing run-level curve 전용으로 유지한다. 출력은
  `loss`, `l2_consistency`, `energy_consistency`, `rel_flux`, `rel_sol` 5개만 생성하며,
  optional auxiliary loss curve는 debug용 `plot_logs.py`에서 다룬다.
- `plot_coupling_logs.py`는 complex CouplingNet의 `_log_epoch - epoch ... train/val ...`
  별도 줄 형식을 지원한다. Complex log는 `loss_energy_consistency`, `rel_sol`,
  `rel_flux`를 기록하고 cross metric을 남기지 않으므로, `loss_energy_consistency`는
  `energy_consistency` curve로 그리며 값이 없는 `l2_consistency` figure는 건너뛴다.
- `plot_coupling_logs.py`는 `--show-annotations` 옵션을 켜면 각 train/val trace의
  마지막 값과 최소값을 표시한다. 기본값은 꺼짐으로 유지한다. 값 표시는
  실제 curve point에 연결된 Plotly annotation으로 두고, last/min 위치에는 작은
  marker도 함께 찍는다. y-axis가 log-scale이므로 plotted marker 위치는
  `1e-16` floor로 맞추고, annotation y coordinate는 Plotly log-axis 규칙에 맞게
  `log10(plotted_y)`를 사용한다. Annotation이 켜진 figure는 마지막 epoch label이
  오른쪽에 놓일 수 있도록 x-axis 오른쪽 padding을 둔다.
- `plot_coupling_logs_aux_temp.py`는
  `checkpoints/Pure_Poisson/expreiments/raw_output_with_balance_loss/training.log`
  확인용 root-level 임시 script이다. Paper-facing default에는 넣지 않고,
  기존 5개 curve에 `balance_loss`, `symmetric_boundary_loss`만 추가해 본다.
- Best epoch과 final epoch을 분리해서 해석한다.
- Plot은 비교 실험 단위로 같은 scale, theme, output naming을 유지한다.
- Plotly 기반 논문용 figure는 나중에 수정하기 쉽도록 `html`, `png`, `pdf`와
  함께 Plotly figure spec `json`도 저장하는 것을 기본으로 한다. Static export가
  실패해도 `html`과 `json`은 저장해야 한다.
- `plot_green_logs.py`도 GreenNet log comparison figures를 `html`, `json`으로
  항상 저장하고, 가능한 경우 `png`, `pdf`도 함께 저장한다.
- `plot_complex_green_interval_metrics.py`는 complex GreenNet의
  `per_interval_metrics.csv` 시각화 전용 root-level script이다. pandas로 schema를
  검증하고 Plotly로 coordinate/length/distribution/chord-map figure를 만들며,
  `save_plotly_figure`를 통해 `html`, `json`, 가능한 경우 `png`, `pdf`를 함께
  저장한다.
- `plot_complex_coupling_sample_metrics.py`는 complex CouplingNet의
  `test_per_sample_metrics.csv` 시각화 전용 root-level script이다. pandas로
  sample-level schema를 검증하고 Plotly로 `rel_sol`, `rel_flux`, `loss`, log-loss,
  relative-metric distribution, `rel_sol`-`rel_flux` scatter, best/worst sample
  ranking figure를 만든다. CSV의 `rel_sol`/`rel_flux`는 raw fraction으로
  유지하고, figure에서는 `%` 단위(100배)로 표시한다.
- `plot_coupling_rel_sol_boxplots.py`는 workshop용 CouplingNet test result CSV
  비교 script이다. `checkpoints/For_Workshop/CouplingNetResults`의 네
  `*_per_sample_metrics.csv` 파일에서 `rel_sol`을 읽고 문제별 Plotly boxplot을
  `html/json/png/pdf` 네 형식으로 저장한다. 이 script는 점 표시 없이 박스 플롯만
  그리며 `rel_sol`은 `%` 단위(100배)로 표시한다.
  하위 분위수 필터를 위해 `--rel-sol-percentile` 옵션을 지원한다. 기본값 100은
  전체 샘플을 사용하고, 예를 들어 90을 주면 각 문제별 `rel_sol`에서 값이 낮은
  90%만 남겨 boxplot을 그린다.
- Figure 후보:
  - coefficient field visualization: `a`, `bx`, `by`, `c`
  - source/solution sample visualization
  - Green kernel heatmap per selected axial line
  - fixed-`xi` 1D Green function slices per selected axial line, with boundary
    values and diagonal/singularity behavior highlighted
  - training curves for GreenONet and CouplingNet
  - metric comparison table/bar chart
  - solution reconstruction and error heatmap
  - flux reconstruction and error heatmap

## Verification Defaults

- `.venv`가 없으면 `/home/jjhong0608/.conda/envs/green_net/bin/python`을 사용한다.
- Focused pytest를 먼저 실행하고, 이후 touched files에 대해 `ruff check`와
  필요한 `mypy`를 실행한다.
- Repo-wide `mypy src`는 2026-06 기준 통과 상태로 정리되었다. 새 변경 이후에는
  focused mypy뿐 아니라 가능한 경우 `mypy src` 전체 통과를 유지한다.
- Markdown 파일은 `ruff` 대상이 아니다.
- 이미 dirty worktree인 경우, 요청 범위와 무관한 변경은 건드리지 않는다.

## Open Planning Items

- 논문용 coefficient family 목록 확정.
- 각 family별 dataset 규모와 random seed 확정.
- GreenONet baseline config와 CouplingNet baseline config 확정.
- Ablation 목록 확정:
  - projection on/off
  - `balance_loss`
  - `symmetric_boundary_loss`
  - `smooth_mask` mask type
  - axis-1D trunk
  - coefficient term inputs
- Figure 목록과 paper section별 배치 확정.
- 결과 저장 directory naming convention 확정.
- Run log, metric CSV, generated figure를 어떤 형식으로 archive할지 확정.

## Update Policy

- Codex는 이 repo에서 답변하기 전에 이 `memory.md`를 먼저 참고한다.
- 답변 이후에는 새로 합의된 durable decision, 실험 기준, coefficient 의미,
  figure/data planning 기준이 생겼는지 판단하고 이 파일을 업데이트한다.
- 이 파일은 모든 작업 로그를 기록하는 곳이 아니다.
- 앞으로 반복해서 참조해야 하는 수학적 의미, API convention, 실험 설계 기준,
  figure planning 기준만 추가한다.
- 일회성 command output, transient error, 구현 세부 diff는 기록하지 않는다.
- 논문용 결과 생성 과정에서 확정된 dataset/figure/run naming은 이 파일에
  누적해서 갱신한다.
